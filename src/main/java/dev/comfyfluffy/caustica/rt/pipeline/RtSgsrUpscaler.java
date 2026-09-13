package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.SgsrParamsData;
import org.joml.Matrix4fc;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * Qualcomm Snapdragon Game Super Resolution 2 (SGSR2) as the reconstruction chain's upscaler: a three-pass
 * temporal upscaler that is pure compute, so unlike FSR it has no native runtime to be missing and runs on
 * any Vulkan device — which is what makes it the upscaler for mobile GPUs.
 *
 * <p>It owns the passes' working images and both ping-pong pairs (luma history at render resolution, output
 * history at display resolution) and decides when that history is invalid: the first frame after creation, a
 * resolution change, or an explicit request from the renderer (world reload, camera cut). Everything else is
 * the frame's: the render-resolution color the denoiser produced, the guides, the jitter the tracer applied,
 * and the display-resolution image to write.
 *
 * <p>Like every other reconstruction backend here, failures are latched rather than thrown: the caller keeps
 * a valid display-resolution image without it and falls back to its own resolve.
 */
public final class RtSgsrUpscaler {
    public static final RtSgsrUpscaler INSTANCE = new RtSgsrUpscaler();

    /**
     * The offset the YCoCg pack normalizes colour with. It belongs to the domain the stored colour is already
     * in, which pre-exposure centres near the metering key, so a constant of one is the offset upstream uses
     * and the one that keeps the pack's quantisation error near a fraction of a percent. Passing the exposure
     * the tracer applied instead would be off by its whole dynamic range and leave the unpacked colour
     * unusable. The constant is therefore tied to the stored domain being pre-exposure centred, which is the
     * exposure default; with pre-exposure switched off the stored colour is raw radiance and no constant can
     * match it.
     */
    private static final float TONE_MAP_OFFSET = 1.0f;
    /** Bounds for the near/far pair derived from the projection; outside them the derivation is unusable. */
    private static final float MIN_DERIVED_DISTANCE = 1.0e-4f;
    private static final float MAX_DERIVED_DISTANCE = 1.0e6f;
    /** Planes to report when the projection cannot supply a usable pair; the tracer reaches 10000 blocks. */
    private static final float FALLBACK_NEAR = 0.05f;
    private static final float FALLBACK_FAR = 10000.0f;

    private RtSgsrPipelines pipelines;
    private RtImage yCoCg;
    private RtImage motionDepthAlpha;
    private RtImage motionDepthClip;
    private RtImage lumaA;
    private RtImage lumaB;
    private RtImage historyA;
    private RtImage historyB;
    private int lumaRead;
    private int historyRead;
    private int renderWidth = -1;
    private int renderHeight = -1;
    private int displayWidth = -1;
    private int displayHeight = -1;
    private boolean resetHistory = true;
    private boolean failed;

    private RtSgsrUpscaler() {
    }

    /**
     * Marks the accumulated history unusable before the next frame, which makes the upscaler blend the current
     * frame alone. Called for world reloads, camera cuts and any render-state invalidation.
     */
    public void requestReset() {
        resetHistory = true;
    }

    /**
     * Clears a latched failure so the next frame tries again, restarting the history because the failed frames
     * left gaps in it. Called from the renderer's explicit recovery hook; nothing retries on its own, so a
     * device that genuinely cannot run these passes is not re-attempted every frame.
     */
    public void clearFailure() {
        if (failed) {
            failed = false;
            resetHistory = true;
            CausticaMod.LOGGER.info("SGSR2 failure latch cleared; retrying SGSR2");
        }
    }

    /** Create or resize this backend's resources. False means the caller keeps its own resolve. */
    public boolean ensure(int renderWidth, int renderHeight, int displayWidth, int displayHeight) {
        if (failed) {
            return false;
        }
        RtContext ctx = RtContext.currentOrNull();
        if (ctx == null) {
            return false;
        }
        try {
            if (pipelines == null) {
                pipelines = RtSgsrPipelines.create(ctx);
            }
            if (this.renderWidth != renderWidth || this.renderHeight != renderHeight
                    || this.displayWidth != displayWidth || this.displayHeight != displayHeight
                    || yCoCg == null) {
                // A resize is rare, and nothing in flight may still be reading the old history.
                ctx.waitIdle();
                destroyImages();
                yCoCg = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R32_UINT,
                        "sgsr ycocg " + renderWidth + "x" + renderHeight);
                motionDepthAlpha = ctx.createStorageImage(renderWidth, renderHeight,
                        VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "sgsr motion depth alpha " + renderWidth + "x" + renderHeight);
                motionDepthClip = ctx.createStorageImage(renderWidth, renderHeight,
                        VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "sgsr motion depth clip " + renderWidth + "x" + renderHeight);
                lumaA = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R32_UINT,
                        "sgsr luma a " + renderWidth + "x" + renderHeight);
                lumaB = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R32_UINT,
                        "sgsr luma b " + renderWidth + "x" + renderHeight);
                historyA = ctx.createStorageImage(displayWidth, displayHeight, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "sgsr history a " + displayWidth + "x" + displayHeight);
                historyB = ctx.createStorageImage(displayWidth, displayHeight, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "sgsr history b " + displayWidth + "x" + displayHeight);
                this.renderWidth = renderWidth;
                this.renderHeight = renderHeight;
                this.displayWidth = displayWidth;
                this.displayHeight = displayHeight;
                lumaRead = 0;
                historyRead = 0;
                resetHistory = true;
                CausticaMod.LOGGER.info("SGSR2 upscaling active: {}x{} -> {}x{}", renderWidth, renderHeight,
                        displayWidth, displayHeight);
            }
            return true;
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("SGSR2 setup failed; the reconstruction chain keeps its own resolve", t);
            return false;
        }
    }

    /**
     * Record the three passes, writing the display-resolution result into {@code out}.
     *
     * <p>{@code color} is the render-resolution scene-linear image the denoiser produced and {@code alpha} the
     * material guide whose emissive flag stands in for the reactive mask (this renderer has no opaque-only
     * buffer; emissive pixels are the ones that would otherwise smear across frames).
     */
    public boolean dispatch(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage velocity, RtImage alpha,
                            RtImage out, float jitterX, float jitterY, Matrix4fc viewToClip) {
        if (failed || pipelines == null || yCoCg == null) {
            return false;
        }
        try {
            RtImage lumaIn = lumaRead == 0 ? lumaA : lumaB;
            RtImage lumaOut = lumaRead == 0 ? lumaB : lumaA;
            RtImage historyIn = historyRead == 0 ? historyA : historyB;
            RtImage historyOut = historyRead == 0 ? historyB : historyA;
            boolean reset = resetHistory;
            SgsrParamsData params = new SgsrParamsData(renderWidth, renderHeight, displayWidth, displayHeight,
                    jitterX, jitterY, TONE_MAP_OFFSET, cameraFovHorizontal(viewToClip), nearPlane(viewToClip),
                    farPlane(viewToClip), viewToClip.m22(), viewToClip.m32(), reset ? 1 : 0);
            pipelines.record(cmd, color, depth, velocity, alpha, yCoCg, motionDepthAlpha, motionDepthClip,
                    lumaIn, lumaOut, historyIn, historyOut, out, params);
            lumaRead = 1 - lumaRead;
            historyRead = 1 - historyRead;
            resetHistory = false;
            return true;
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("SGSR2 upscaling failed; the reconstruction chain falls back to a blit", t);
            return false;
        }
    }

    /**
     * SGSR2's horizontal fov term, {@code tan(fovY/2) * renderWidth / renderHeight}. The projection's [1][1]
     * is {@code 1 / tan(fovY/2)}, so the tangent comes straight out of the matrix.
     */
    private float cameraFovHorizontal(Matrix4fc viewToClip) {
        float cotangent = viewToClip.m11();
        float tangent = Math.abs(cotangent) > 1.0e-6f ? 1.0f / cotangent : 1.0f;
        return tangent * renderWidth / (float) renderHeight;
    }

    /**
     * Near and far plane distances implied by the projection, ordered by magnitude.
     *
     * <p>The guide's depth is this renderer's reversed-Z: 1 at the near plane and 0 at the far plane. The
     * conversion to SGSR2's standard depth needs both bounds, and which depth value means "near" depends on
     * the projection's convention, so the two candidates are simply sorted — a camera's near plane is always
     * the closer of the two. A projection where the mapping degenerates (an infinite far plane, for example)
     * falls back to the distance the tracer itself can reach.
     */
    private float nearPlane(Matrix4fc viewToClip) {
        float[] bounds = depthBounds(viewToClip);
        return bounds[0];
    }

    private float farPlane(Matrix4fc viewToClip) {
        float[] bounds = depthBounds(viewToClip);
        return bounds[1];
    }

    /**
     * Near and far plane distances for the given projection, ordered by magnitude, or the tracer's own reach
     * when the mapping degenerates. Reversed-Z puts the near plane at guide depth 1 and the far plane at guide
     * depth 0, so the far distance is undefined for an infinite far plane and the pair then has to come from
     * somewhere else. Public because the conversion is the one piece of this backend that is pure arithmetic
     * and worth testing without a device.
     */
    public static float[] depthBounds(Matrix4fc viewToClip) {
        float m22 = viewToClip.m22();
        float m32 = viewToClip.m32();
        float atNear = distanceAtDepth(m32, m22, 1.0f);
        float atFar = distanceAtDepth(m32, m22, 0.0f);
        float near = Math.min(atNear, atFar);
        float far = Math.max(atNear, atFar);
        if (!(near >= MIN_DERIVED_DISTANCE) || !(far > near * 1.001f) || !(far <= MAX_DERIVED_DISTANCE)) {
            return new float[] {FALLBACK_NEAR, FALLBACK_FAR};
        }
        return new float[] {near, far};
    }

    /**
     * The same distance mapping the world and denoise shaders use on the depth guide. A depth that maps to no
     * finite distance (guide depth 0 under an infinite far plane) reports infinity, which makes the caller
     * above reject the derivation instead of clamping it into a plausible-looking number.
     */
    private static float distanceAtDepth(float m32, float m22, float depth) {
        float denom = depth + m22;
        if (Math.abs(denom) < 1.0e-6f) {
            return Float.POSITIVE_INFINITY;
        }
        return Math.abs(m32 / denom);
    }

    public void destroy() {
        if (pipelines != null) {
            pipelines.destroy();
            pipelines = null;
        }
        destroyImages();
        failed = false;
        resetHistory = true;
    }

    private void destroyImages() {
        for (RtImage image : new RtImage[] {yCoCg, motionDepthAlpha, motionDepthClip, lumaA, lumaB, historyA,
                historyB}) {
            if (image != null) {
                image.destroy();
            }
        }
        yCoCg = null;
        motionDepthAlpha = null;
        motionDepthClip = null;
        lumaA = null;
        lumaB = null;
        historyA = null;
        historyB = null;
        renderWidth = -1;
        renderHeight = -1;
        displayWidth = -1;
        displayHeight = -1;
    }
}

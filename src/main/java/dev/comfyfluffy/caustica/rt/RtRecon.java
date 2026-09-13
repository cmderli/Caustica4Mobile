package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.ffx.FsrUpscaler;
import dev.comfyfluffy.caustica.rt.pipeline.RtDenoisePipeline;
import dev.comfyfluffy.caustica.rt.pipeline.RtSgsrUpscaler;
import dev.comfyfluffy.caustica.rt.pipeline.RtDlssRr;
import org.joml.Matrix4fc;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * Owns the reconstruction stage: turning the noisy render-resolution trace and its guide buffers into the
 * display-resolution image the exposure/bloom/display chain consumes. One place decides which backend
 * runs, so the render resolution, the guide requirements and the dispatch cannot disagree with each
 * other.
 *
 * <p>The backends differ in what they cost and what they assume. DLSS Ray Reconstruction is a vendor
 * feature that denoises and upscales in one pass and carries its own temporal history; the built-in
 * backend is a spatial filter chain that runs anywhere and has no history at all. {@code off} keeps the
 * reference path — trace at 1:1 and blit — which is also where every backend lands when it cannot run:
 * every method here is failure-tolerant and reports "not done" rather than throwing, because the caller
 * already has a valid display-res image without it.
 */
public final class RtRecon {
    public static final RtRecon INSTANCE = new RtRecon();

    public enum Denoiser {
        OFF,
        DLSS_RR,
        NRD,
        BUILTIN,
    }

    public enum Upscaler {
        /** The backend writes display resolution itself (DLSS Ray Reconstruction), or nothing upscales. */
        NONE,
        FSR,
        /** Snapdragon Game Super Resolution 2: temporal, and pure compute, so it runs where FSR cannot. */
        SGSR,
        BILINEAR,
    }

    // Render/display ratios per reconstruction quality step, in FfxApiUpscaleQualityMode order:
    // native-aa, quality, balanced, performance, ultra-performance.
    private static final float[] QUALITY_RATIOS = {1.0f, 1.5f, 1.7f, 2.0f, 3.0f};
    // Filters want their inputs aligned, and the widening taps stay whole when the render size is.
    private static final int RENDER_ALIGNMENT = 8;

    private RtDenoisePipeline denoise;
    private RtImage denoisePing;
    private RtImage denoisePong;
    private int denoiseRenderWidth = -1;
    private int denoiseRenderHeight = -1;
    private int denoiseDisplayWidth = -1;
    private int denoiseDisplayHeight = -1;
    private boolean denoiseFailed;
    // The path tracer's split radiance lanes. Only the NRD backend consumes them; they are set every
    // frame from the composite because the guide images are recreated on resize.
    private RtImage gDiffLane;
    private RtImage gSpecLane;

    private RtRecon() {
    }

    /**
     * The denoiser this frame should use. Resolved from config on every call rather than cached, so a
     * config edit takes effect on the next frame; each step is a config read plus a latched capability or
     * failure check.
     */
    public Denoiser denoiser() {
        return switch (CausticaConfig.Rt.Reconstruction.DENOISER.get()) {
            case "dlss-rr" -> RtDlssRr.enabled() ? Denoiser.DLSS_RR : Denoiser.OFF;
            case "nrd" -> nrd() ? Denoiser.NRD : Denoiser.OFF;
            case "builtin" -> denoiseFailed ? Denoiser.OFF : Denoiser.BUILTIN;
            case "off" -> Denoiser.OFF;
            // auto: the vendor feature that denoises and upscales in one pass, else the temporal denoiser,
            // else the spatial filter chain that has no native dependency at all. A device that cannot host
            // NGX never reaches DLSS_RR here, and a machine without the NRD shim never reaches NRD.
            default -> RtDlssRr.enabled() ? Denoiser.DLSS_RR
                    : (nrd() ? Denoiser.NRD : (denoiseFailed ? Denoiser.OFF : Denoiser.BUILTIN));
        };
    }

    private static boolean nrd() {
        return RtNrd.INSTANCE.isAvailable();
    }

    /**
     * How the render-resolution result reaches display resolution. DLSS Ray Reconstruction denoises and
     * upscales in a single feature, so it owns the upscale and this returns {@link Upscaler#NONE} there.
     * {@code auto} prefers FSR where its runtime exists and SGSR2 everywhere else, so a missing FSR runtime
     * still leaves a temporal upscaler in charge rather than the built-in resolve.
     */
    public Upscaler upscaler() {
        if (denoiser() == Denoiser.DLSS_RR) {
            return Upscaler.NONE;
        }
        String configured = CausticaConfig.Rt.Reconstruction.UPSCALER.get();
        if ("bilinear".equals(configured)) {
            return Upscaler.BILINEAR;
        }
        if ("sgsr".equals(configured)) {
            return Upscaler.SGSR;
        }
        // auto: FSR's runtime is a Windows-only native library, so every other platform lands on SGSR2 —
        // temporal and merely a compute pass, which beats the bilinear resolve it replaces.
        return FsrUpscaler.INSTANCE.isAvailable() ? Upscaler.FSR : Upscaler.SGSR;
    }

    /** True when DLSS Ray Reconstruction owns the super-resolution mode ladder. */
    public boolean superResolutionIsDlss() {
        return denoiser() == Denoiser.DLSS_RR;
    }

    /**
     * True when a reconstruction pass will produce the display-res image; false means the caller must blit.
     * A denoiser always qualifies; without one, only FSR does — a bare bilinear resolve would reproduce
     * exactly what the blit already does.
     */
    public boolean isActive() {
        return denoiser() != Denoiser.OFF
                || (upscaler() != Upscaler.BILINEAR && upscaler() != Upscaler.NONE);
    }

    /**
     * Render resolution to trace at for the given display size, or {@code null} for 1:1 (nothing to
     * reconstruct). DLSS answers from its own quality table through NGX; every other combination derives
     * it from the configured super-resolution mode.
     */
    public int[] renderSize(int displayWidth, int displayHeight) {
        if (denoiser() == Denoiser.DLSS_RR) {
            return RtDlssRr.INSTANCE.queryOptimalRenderSize(displayWidth, displayHeight);
        }
        if (!isActive()) {
            return null;
        }
        return new int[] {
                alignUp(Math.round(displayWidth / qualityRatio()), displayWidth),
                alignUp(Math.round(displayHeight / qualityRatio()), displayHeight)};
    }

    private static float qualityRatio() {
        int quality = CausticaConfig.Rt.Reconstruction.QUALITY.value();
        return QUALITY_RATIOS[Math.clamp(quality, 0, QUALITY_RATIOS.length - 1)];
    }

    private static int alignUp(int value, int limit) {
        int aligned = Math.min(((value + RENDER_ALIGNMENT - 1) / RENDER_ALIGNMENT) * RENDER_ALIGNMENT, limit);
        return Math.max(aligned, 1);
    }

    /**
     * Everything the render-size decision depends on, folded into one value: which denoiser and upscaler
     * are active (a switch must rebuild even at an unchanged window size) and their quality knob.
     */
    public int renderSizeKey() {
        int quality = superResolutionIsDlss()
                ? RtDlssRr.quality()
                : CausticaConfig.Rt.Reconstruction.QUALITY.value();
        return (denoiser().ordinal() << 16) | (upscaler().ordinal() << 8) | (quality & 0xff);
    }

    /** Create or resize the backend's own resources. False means "not available; blit instead". */
    public boolean ensure(VkCommandBuffer cmd, int renderWidth, int renderHeight,
                          int displayWidth, int displayHeight) {
        return switch (denoiser()) {
            case DLSS_RR -> RtDlssRr.INSTANCE.ensureFeature(cmd.address(), renderWidth, renderHeight,
                    displayWidth, displayHeight);
            // The NRD chain reuses the built-in pipeline's bilinear resolve (or FSR) to reach display
            // resolution, so both backends need that pipeline and its scratch pair.
            case NRD -> ensureDenoise(renderWidth, renderHeight, displayWidth, displayHeight)
                    && RtNrd.INSTANCE.ensure(renderWidth, renderHeight, displayWidth, displayHeight);
            case BUILTIN -> ensureDenoise(renderWidth, renderHeight, displayWidth, displayHeight);
            // No denoiser: only an upscaler has anything to set up, and both read the raw trace directly, so
            // the built-in pipeline and its scratch pair are not needed.
            case OFF -> upscaler() == Upscaler.FSR
                    ? FsrUpscaler.INSTANCE.ensure(renderWidth, renderHeight, displayWidth, displayHeight,
                            CausticaConfig.Rt.Reconstruction.QUALITY.value())
                    : upscaler() == Upscaler.SGSR
                            && RtSgsrUpscaler.INSTANCE.ensure(renderWidth, renderHeight, displayWidth,
                                    displayHeight);
        };
    }

    private boolean ensureDenoise(int renderWidth, int renderHeight, int displayWidth, int displayHeight) {
        if (denoiseFailed) {
            return false;
        }
        RtContext ctx = RtContext.get();
        if (ctx == null) {
            return false;
        }
        try {
            if (denoise == null) {
                denoise = RtDenoisePipeline.create(ctx);
            }
            if (denoisePing == null || denoiseRenderWidth != renderWidth || denoiseRenderHeight != renderHeight
                    || denoiseDisplayWidth != displayWidth || denoiseDisplayHeight != displayHeight) {
                // A resize is rare and nothing may still be reading the old scratch pair.
                ctx.waitIdle();
                destroyDenoiseImages();
                denoisePing = ctx.createStorageImage(renderWidth, renderHeight,
                        VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "denoise scratch " + renderWidth + "x" + renderHeight);
                denoisePong = ctx.createStorageImage(renderWidth, renderHeight,
                        VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "denoise scratch " + renderWidth + "x" + renderHeight);
                denoiseRenderWidth = renderWidth;
                denoiseRenderHeight = renderHeight;
                denoiseDisplayWidth = displayWidth;
                denoiseDisplayHeight = displayHeight;
            }
            return true;
        } catch (Throwable t) {
            denoiseFailed = true;
            CausticaMod.LOGGER.error("Built-in denoiser setup failed; RT composite falls back to a blit", t);
            return false;
        }
    }

    /** Denoise {@code color} and write the display-res result into {@code out}. False means "not done". */
    public boolean evaluate(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage motion,
                            RtImage diffuseAlbedo, RtImage specularAlbedo, RtImage normals,
                            RtImage specularMotion, RtImage out,
                            int renderWidth, int renderHeight, int displayWidth, int displayHeight,
                            float jitterX, float jitterY, Matrix4fc worldToView, Matrix4fc viewToClip,
                            Matrix4fc worldPrevToWorld) {
        return switch (denoiser()) {
            case DLSS_RR -> RtDlssRr.INSTANCE.evaluate(cmd.address(), color, depth, motion, diffuseAlbedo,
                    specularAlbedo, normals, specularMotion, out, renderWidth, renderHeight,
                    displayWidth, displayHeight, jitterX, jitterY, worldToView, viewToClip);
            case NRD -> recordNrd(cmd, color, depth, motion, normals, diffuseAlbedo, specularAlbedo, out,
                    renderWidth, renderHeight, displayWidth, displayHeight, jitterX, jitterY, worldToView,
                    viewToClip, worldPrevToWorld);
            case BUILTIN -> recordDenoise(cmd, color, depth, motion, normals, diffuseAlbedo, specularAlbedo,
                    out, renderWidth, renderHeight, displayWidth, displayHeight, jitterX, jitterY, viewToClip);
            case OFF -> recordUpscaleOnly(cmd, color, depth, motion, diffuseAlbedo, out, renderWidth,
                    renderHeight, displayWidth, displayHeight, jitterX, jitterY, viewToClip);
        };
    }

    /**
     * No denoiser, but FSR still upscales the raw trace. Without it there is nothing to record: the
     * composite's linear blit already covers the 1:1 and stretched cases, so reporting "not done" is what
     * keeps that path in charge.
     */
    private boolean recordUpscaleOnly(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage motion,
                                      RtImage diffuseAlbedo, RtImage out, int renderWidth, int renderHeight,
                                      int displayWidth, int displayHeight, float jitterX, float jitterY,
                                      Matrix4fc viewToClip) {
        if (upscaler() == Upscaler.BILINEAR || upscaler() == Upscaler.NONE) {
            return false;
        }
        try {
            if (!ensureOffPathUpscaler(renderWidth, renderHeight, displayWidth, displayHeight)) {
                return false;
            }
            return dispatchUpscaler(cmd, color, depth, motion, diffuseAlbedo, out, renderWidth, renderHeight,
                    displayWidth, displayHeight, jitterX, jitterY, viewToClip);
        } catch (Throwable t) {
            CausticaMod.LOGGER.error("Upscaling of the raw trace failed; falling back to a blit", t);
            return false;
        }
    }

    private boolean ensureOffPathUpscaler(int renderWidth, int renderHeight, int displayWidth,
                                          int displayHeight) {
        if (upscaler() == Upscaler.FSR) {
            return FsrUpscaler.INSTANCE.ensure(renderWidth, renderHeight, displayWidth, displayHeight,
                    CausticaConfig.Rt.Reconstruction.QUALITY.value());
        }
        return RtSgsrUpscaler.INSTANCE.ensure(renderWidth, renderHeight, displayWidth, displayHeight);
    }

    /**
     * NRD denoises the renderer's split lanes in place and hands back a render-res image, which then takes
     * the same upscale as the built-in denoiser's output. The lane guides are written into the same
     * scratch image the built-in path uses, so the two backends share one upscale path.
     */
    private boolean recordNrd(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage motion,
                              RtImage normals, RtImage diffuseAlbedo, RtImage specularAlbedo, RtImage out,
                              int renderWidth, int renderHeight, int displayWidth, int displayHeight,
                              float jitterX, float jitterY, Matrix4fc worldToView, Matrix4fc viewToClip,
                              Matrix4fc worldPrevToWorld) {
        if (denoise == null || denoisePing == null || denoisePong == null) {
            return false;
        }
        try {
            // The upscale below reuses the built-in pipeline's resolve, whose descriptors are written only
            // here and in the built-in path. Without this bind the set stays undefined and the resolve reads
            // whatever the descriptor memory happened to contain.
            denoise.setImages(color, denoisePing, denoisePong, out, depth, normals, diffuseAlbedo, specularAlbedo);
            if (!RtNrd.INSTANCE.evaluate(cmd, depth, motion, normals, diffuseAlbedo, specularAlbedo,
                    gDiffLane, gSpecLane, denoisePing, renderWidth, renderHeight, jitterX, jitterY,
                    worldToView, viewToClip, worldPrevToWorld)) {
                return false;
            }
            if (upscaler() != Upscaler.BILINEAR && upscaler() != Upscaler.NONE
                    && dispatchUpscaler(cmd, denoisePing, depth, motion, diffuseAlbedo, out, renderWidth,
                            renderHeight, displayWidth, displayHeight, jitterX, jitterY, viewToClip)) {
                return true;
            }
            denoise.recordResolve(cmd, renderWidth, renderHeight, displayWidth, displayHeight,
                    viewToClip.m22(), viewToClip.m32());
            return true;
        } catch (Throwable t) {
            denoiseFailed = true;
            CausticaMod.LOGGER.error("NRD denoising failed; RT composite falls back to a blit", t);
            return false;
        }
    }

    private boolean recordDenoise(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage motion,
                                  RtImage normals, RtImage diffuseAlbedo, RtImage specularAlbedo, RtImage out,
                                  int renderWidth, int renderHeight, int displayWidth, int displayHeight,
                                  float jitterX, float jitterY, Matrix4fc viewToClip) {
        if (denoise == null || denoisePing == null || denoisePong == null) {
            return false;
        }
        try {
            denoise.setImages(color, denoisePing, denoisePong, out, depth, normals, diffuseAlbedo, specularAlbedo);
            if (upscaler() != Upscaler.BILINEAR && upscaler() != Upscaler.NONE) {
                // Both upscalers need the spatial iterations' result, not the raw trace.
                denoise.recordIterations(cmd, renderWidth, renderHeight, displayWidth, displayHeight,
                        viewToClip.m22(), viewToClip.m32());
                if (dispatchUpscaler(cmd, denoisePing, depth, motion, diffuseAlbedo, out, renderWidth,
                        renderHeight, displayWidth, displayHeight, jitterX, jitterY, viewToClip)) {
                    return true;
                }
            }
            denoise.record(cmd, renderWidth, renderHeight, displayWidth, displayHeight,
                    viewToClip.m22(), viewToClip.m32());
            return true;
        } catch (Throwable t) {
            denoiseFailed = true;
            CausticaMod.LOGGER.error("Built-in denoiser failed; RT composite falls back to a blit", t);
            return false;
        }
    }

    /**
     * Record the configured upscaler over a render-resolution image. False leaves the caller on its own
     * resolve or blit. FSR is a native feature that reprojects from the projection itself; SGSR2 is shader
     * passes that take the same guides the denoiser used plus the emissive guide as its reactive mask.
     */
    private boolean dispatchUpscaler(VkCommandBuffer cmd, RtImage source, RtImage depth, RtImage motion,
                                     RtImage diffuseAlbedo, RtImage out, int renderWidth, int renderHeight,
                                     int displayWidth, int displayHeight, float jitterX, float jitterY,
                                     Matrix4fc viewToClip) {
        return switch (upscaler()) {
            case SGSR -> RtSgsrUpscaler.INSTANCE.ensure(renderWidth, renderHeight, displayWidth, displayHeight)
                    && RtSgsrUpscaler.INSTANCE.dispatch(cmd, source, depth, motion, diffuseAlbedo, out,
                            jitterX, jitterY, viewToClip);
            case FSR -> FsrUpscaler.INSTANCE.dispatch(cmd, source, depth, motion, out,
                    renderWidth, renderHeight, displayWidth, displayHeight,
                    jitterX, jitterY, viewToClip.m11(), viewToClip.m22(), viewToClip.m32());
            case NONE, BILINEAR -> false;
        };
    }

    /**
     * Invalidate the upscalers' accumulated history. Called when the renderer's state is invalidated (world
     * reload, resize, resource reload): reprojecting history across a discontinuity is exactly the ghosting
     * a temporal upscaler has to be told about.
     */
    public void requestUpscalerReset() {
        RtSgsrUpscaler.INSTANCE.requestReset();
    }

    /** Drop a latched upscaler failure so the next frame retries it; the renderer's recovery hook calls this. */
    public void clearUpscalerFailure() {
        RtSgsrUpscaler.INSTANCE.clearFailure();
    }

    /** The path tracer's per-lobe radiance lanes, needed only while the NRD backend is active. */
    public void setRadianceLanes(RtImage diffLane, RtImage specLane) {
        this.gDiffLane = diffLane;
        this.gSpecLane = specLane;
    }

    /**
     * Release every backend's resources. All of them are released unconditionally: the active backend can
     * change between frames, and one that stayed torn down while inactive would keep its device objects
     * alive past the device that owns them.
     */
    public void destroy() {
        RtDlssRr.INSTANCE.destroy();
        RtNrd.INSTANCE.destroy();
        FsrUpscaler.INSTANCE.destroy();
        RtSgsrUpscaler.INSTANCE.destroy();
        if (denoise != null) {
            denoise.destroy();
            denoise = null;
        }
        destroyDenoiseImages();
        denoiseFailed = false;
    }

    private void destroyDenoiseImages() {
        if (denoisePing != null) {
            denoisePing.destroy();
            denoisePing = null;
        }
        if (denoisePong != null) {
            denoisePong.destroy();
            denoisePong = null;
        }
        denoiseRenderWidth = -1;
        denoiseRenderHeight = -1;
        denoiseDisplayWidth = -1;
        denoiseDisplayHeight = -1;
    }
}

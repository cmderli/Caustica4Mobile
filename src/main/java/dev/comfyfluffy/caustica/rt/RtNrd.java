package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vulkan.VulkanDevice;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.mixin.GpuDeviceAccessor;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.nrd.NrdLibrary;
import dev.comfyfluffy.caustica.rt.pipeline.RtNrdPipelines;
import net.fabricmc.loader.api.FabricLoader;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkInstance;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * NRD denoising backend for devices that cannot run DLSS Ray Reconstruction. The path tracer splits its
 * radiance into diffuse and specular lanes carrying a hit distance each; this class packs them into NRD's
 * input layout, lets the {@code caustica_nrd} shim run the denoiser, and recombines the result into the
 * render-resolution image that the built-in resolve (or FSR) then brings to display resolution.
 *
 * <p>Both ends of that chain demodulate by the material guides (see {@code nrdDemodFactor} in
 * recon_common.slang): NRD has no albedo input, so a radiance signal that still carries its material factor
 * comes back with neighbouring texels averaged into each other, which smears surface detail.
 *
 * <p>NRD is a vendor-neutral compute denoiser but its SDK is native, so the shim is loaded the way the NGX
 * natives are: a small library extracted next to the game. Everything here is best-effort — when the shim
 * is absent or refuses the device, {@link #isAvailable()} stays false and the caller keeps the built-in
 * denoiser, so a missing native never takes the frame down.
 */
public final class RtNrd {
    public static final RtNrd INSTANCE = new RtNrd();

    private static final BundledNative NATIVE = new BundledNative("caustica-nrd", "caustica_nrd");
    /** Written as worldPrevToWorld on the first frame, where there is no history to reproject. */
    private static final Matrix4f ZERO_MATRIX = new Matrix4f().zero();
    /** The shim's denoiser ids, in the order its ABI documents them. */
    private static final int DENOISER_REBLUR_DIFFUSE_SPECULAR = 0;
    private static final int DENOISER_RELAX_DIFFUSE_SPECULAR = 1;
    // Depth range NRD normalizes against. The tracer's rays reach 10000 world units, so the range has to sit
    // above that: NRD skips every pixel whose view-Z magnitude leaves the range, and it never writes its
    // output for a pixel it skips, so a range below the tracer's reach would leave holes in that geometry.
    // The front end derives its no-hit view-Z from this value, keeping the sky outside it by construction.
    private static final float DENOISING_RANGE = 20000.0f;
    // NRD's default ReblurSettings::hitDistanceParameters. The front end normalizes with the same triple and
    // the shim builds the denoiser with these defaults; changing one means changing all of them together.
    private static final float HIT_DIST_A = 3.0f;
    private static final float HIT_DIST_B = 0.1f;
    private static final float HIT_DIST_C = 20.0f;

    private NrdLibrary lib;
    private RtNrdPipelines pipelines;
    private RtImage normalRoughness;
    private RtImage viewZ;
    private RtImage diffIn;
    private RtImage specIn;
    private RtImage diffOut;
    private RtImage specOut;

    private int renderWidth = -1;
    private int renderHeight = -1;
    private int contextWidth = -1;
    private int contextHeight = -1;
    private int frameIndex;

    private boolean probeAttempted;
    private boolean available;
    private boolean failed;

    private final Matrix4f clipToView = new Matrix4f();
    private final Matrix4f viewToWorld = new Matrix4f();
    // Last frame's world->view rotation, and this frame's world->previous-view matrix derived from it.
    private final Matrix4f prevWorldToView = new Matrix4f();
    private final Matrix4f worldToViewPrev = new Matrix4f();
    private boolean hasPrevWorldToView;
    private final Matrix4f worldPrevToWorldInverse = new Matrix4f();

    private RtNrd() {
    }

    /**
     * Whether this system can run the NRD backend. Probed once by locating the shim; loading and instance
     * creation happen on first use, so a machine without the native pays one filesystem probe per session.
     */
    public boolean isAvailable() {
        if (!probeAttempted) {
            probeAttempted = true;
            // Present is not the same as loadable: Android's linker refuses a library outside the app's
            // namespace, which is what the external game directory is, so every candidate is tried and the
            // first one that opens wins.
            Path lastSeen = null;
            Throwable lastFailure = null;
            for (Path shim : NATIVE.candidates(CausticaConfig.Nrd.PATH.get())) {
                if (!Files.isRegularFile(shim)) {
                    continue;
                }
                lastSeen = shim;
                try {
                    lib = NrdLibrary.load(shim);
                    available = true;
                    CausticaMod.LOGGER.info("NRD denoising available (shim {}, {})", shim, denoiserLabel());
                    break;
                } catch (Throwable t) {
                    lastFailure = t;
                }
            }
            if (!available && lastSeen == null) {
                CausticaMod.LOGGER.info(
                        "NRD unavailable: {} not found (bundled natives, run/natives, or -Dcaustica.nrd.path)",
                        NATIVE.libraryName());
            } else if (!available) {
                CausticaMod.LOGGER.warn("NRD shim at {} could not be loaded; using the built-in denoiser",
                        lastSeen, lastFailure);
            }
        }
        return available;
    }

    /** Create or resize the denoiser for this resolution pair. False means the caller keeps its own path. */
    public boolean ensure(int renderWidth, int renderHeight, int displayWidth, int displayHeight) {
        if (!isAvailable() || failed) {
            return false;
        }
        RtContext ctx = RtContext.get();
        if (ctx == null) {
            return false;
        }
        try {
            if (pipelines == null) {
                pipelines = RtNrdPipelines.create(ctx);
            }
            if (this.renderWidth != renderWidth || this.renderHeight != renderHeight
                    || contextWidth != displayWidth || contextHeight != displayHeight || diffIn == null) {
                ctx.waitIdle();
                destroyImages();
                // NRD reads and writes its inputs in place (REBLUR patches the motion vectors), so every
                // image it touches comes from createStorageImage, which sets storage and sampled usage.
                normalRoughness = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R8G8B8A8_UNORM,
                        "nrd normal roughness " + renderWidth + "x" + renderHeight);
                viewZ = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R32_SFLOAT,
                        "nrd view z " + renderWidth + "x" + renderHeight);
                diffIn = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "nrd diffuse in " + renderWidth + "x" + renderHeight);
                specIn = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "nrd specular in " + renderWidth + "x" + renderHeight);
                diffOut = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "nrd diffuse out " + renderWidth + "x" + renderHeight);
                specOut = ctx.createStorageImage(renderWidth, renderHeight, VK10.VK_FORMAT_R16G16B16A16_SFLOAT,
                        "nrd specular out " + renderWidth + "x" + renderHeight);
                this.renderWidth = renderWidth;
                this.renderHeight = renderHeight;
                contextWidth = displayWidth;
                contextHeight = displayHeight;
                frameIndex = 0;
                if (!createInstance(displayWidth, displayHeight)) {
                    return false;
                }
            }
            return true;
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("NRD setup failed; RT composite falls back to the built-in denoiser", t);
            return false;
        }
    }

    /**
     * Which NRD denoiser the shim builds. REBLUR is the one this backend was written against, but its
     * Blur/HistoryFix/PostBlur passes issue quad operations from compute shaders, which Vulkan only allows
     * with {@code VK_KHR_compute_shader_derivatives} and its {@code computeDerivativeGroupQuads} feature —
     * absent on e.g. Adreno. RELAX uses no quad operations at all (verified against the SPIR-V the shim
     * embeds), so a device without that extension gets RELAX instead of no denoiser.
     */
    static int denoiserId(boolean computeShaderDerivatives) {
        return computeShaderDerivatives ? DENOISER_REBLUR_DIFFUSE_SPECULAR : DENOISER_RELAX_DIFFUSE_SPECULAR;
    }

    private int denoiserId() {
        return denoiserId(RtDeviceBringup.nrdDerivativesCapable());
    }

    /** True when the selected denoiser expects the normalized hit-distance convention REBLUR defines. */
    private boolean normalizeHitDistances() {
        return denoiserId() == DENOISER_REBLUR_DIFFUSE_SPECULAR;
    }

    private String denoiserLabel() {
        return denoiserId() == DENOISER_REBLUR_DIFFUSE_SPECULAR ? "REBLUR"
                : "RELAX (no " + RtDeviceBringup.NRD_DERIVATIVES_EXTENSION + " on this device)";
    }

    private boolean createInstance(int displayWidth, int displayHeight) {
        if (!(((GpuDeviceAccessor) RenderSystem.getDevice()).caustica$getBackend() instanceof VulkanDevice device)) {
            return false;
        }
        long getDeviceProcAddr;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkInstance instance = device.vkDevice().getPhysicalDevice().getInstance();
            getDeviceProcAddr = VK10.vkGetInstanceProcAddr(instance, stack.ASCII("vkGetDeviceProcAddr"));
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = arena.allocate(NrdLibrary.CREATE_INFO_BYTES, 8);
            NrdLibrary.putCreateInfo(info, device.vkDevice().address(),
                    device.vkDevice().getPhysicalDevice().address(), getDeviceProcAddr,
                    renderWidth, renderHeight, denoiserId());
            int rc = lib.create(info);
            if (rc != NrdLibrary.OK) {
                failed = true;
                CausticaMod.LOGGER.error("caustica_nrd_create failed: {}", NrdLibrary.describeReturn(rc));
                return false;
            }
            CausticaMod.LOGGER.info("NRD denoiser created: {}x{} render, {}x{} display, {}",
                    renderWidth, renderHeight, displayWidth, displayHeight, denoiserLabel());
            return true;
        }
    }

    /**
     * Record the whole NRD chain: pack, denoise, recombine. {@code worldPrevToWorld} maps the previous
     * frame's world space into this frame's; this renderer works in camera-relative world space, so the two
     * differ by a pure camera translation that the view matrices cannot express.
     */
    public boolean evaluate(VkCommandBuffer cmd, RtImage depth, RtImage motion, RtImage normals,
                            RtImage diffuseAlbedo, RtImage specularAlbedo,
                            RtImage laneDiff, RtImage laneSpec, RtImage out,
                            int renderWidth, int renderHeight,
                            float jitterX, float jitterY, Matrix4fc worldToView, Matrix4fc viewToClip,
                            Matrix4fc worldPrevToWorld) {
        if (!isAvailable() || failed || pipelines == null || diffIn == null) {
            return false;
        }
        try {
            pipelines.setImages(normals, depth, diffuseAlbedo, specularAlbedo, laneDiff, laneSpec,
                    normalRoughness, viewZ, diffIn, specIn, diffOut, specOut, out);
            RtGpuProfile.INSTANCE.markStart(cmd, RtGpuProfile.Stage.NRD_FRONTEND);
            pipelines.recordFrontend(cmd, renderWidth, renderHeight, viewToClip.m22(), viewToClip.m32(),
                    HIT_DIST_A, HIT_DIST_B, HIT_DIST_C, DENOISING_RANGE, normalizeHitDistances());
            RtGpuProfile.INSTANCE.markEnd(cmd, RtGpuProfile.Stage.NRD_FRONTEND);

            viewToClip.invert(clipToView);
            worldToView.invert(viewToWorld);
            // NRD needs the previous frame's world->view matrix to carry the camera delta, because this
            // renderer's world space moves with the camera. A static point's previous-frame world position
            // is its current one mapped back through worldPrevToWorld, so the previous view matrix is
            // "previous rotation times that inverse" — which is what the cached rotation alone cannot
            // express, and what makes history land on the right pixel while the camera moves.
            if (hasPrevWorldToView) {
                worldPrevToWorld.invert(worldPrevToWorldInverse);
                worldToViewPrev.set(prevWorldToView).mul(worldPrevToWorldInverse);
            } else {
                worldToViewPrev.set(worldToView);
            }
            // The shim records NRD's own dispatches into this command buffer, so the whole denoiser is one
            // envelope around that call.
            RtGpuProfile.INSTANCE.markStart(cmd, RtGpuProfile.Stage.NRD_CORE);
            boolean denoised = denoise(cmd, motion, jitterX, jitterY, worldToView, viewToClip,
                    worldPrevToWorld);
            RtGpuProfile.INSTANCE.markEnd(cmd, RtGpuProfile.Stage.NRD_CORE);
            if (!denoised) {
                return false;
            }

            RtGpuProfile.INSTANCE.markStart(cmd, RtGpuProfile.Stage.NRD_BACKEND);
            pipelines.recordBackend(cmd, renderWidth, renderHeight);
            RtGpuProfile.INSTANCE.markEnd(cmd, RtGpuProfile.Stage.NRD_BACKEND);
            return true;
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("NRD denoising failed; RT composite falls back to the built-in denoiser", t);
            return false;
        }
    }

    private boolean denoise(VkCommandBuffer cmd, RtImage motion, float jitterX, float jitterY,
                            Matrix4fc worldToView, Matrix4fc viewToClip, Matrix4fc worldPrevToWorld) {
        Matrix4fc previousView = hasPrevWorldToView ? worldToViewPrev : null;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment frame = arena.allocate(NrdLibrary.FRAME_INFO_BYTES, 8);
            // The packed lanes are NRD's inputs and the denoised lanes are separate images: NRD's
            // description treats IN_* and OUT_* as distinct resources, so they are never aliased.
            NrdLibrary.putFrameInfo(frame, cmd.address(), frameIndex++, jitterX, jitterY, DENOISING_RANGE,
                    viewToClip, clipToView, worldToView, viewToWorld,
                    viewZ.image, viewZ.view,
                    normalRoughness.image, normalRoughness.view,
                    motion.image, motion.view,
                    diffIn.image, diffIn.view,
                    specIn.image, specIn.view,
                    diffOut.image, diffOut.view,
                    specOut.image, specOut.view,
                    worldPrevToWorld, previousView == null ? ZERO_MATRIX : previousView);
            // Remembered for the next frame's worldToViewPrev. A frame that fails latches the backend off
            // entirely, so there is no later frame for a stale value to matter in.
            prevWorldToView.set(worldToView);
            hasPrevWorldToView = true;
            int rc = lib.denoise(frame);
            if (rc != NrdLibrary.OK) {
                failed = true;
                CausticaMod.LOGGER.error("caustica_nrd_denoise failed: {}", NrdLibrary.describeReturn(rc));
                return false;
            }
            return true;
        }
    }

    public void destroy() {
        if (lib != null) {
            try {
                lib.destroyInstance();
            } catch (Throwable t) {
                CausticaMod.LOGGER.warn("caustica_nrd_destroy failed", t);
            }
        }
        if (pipelines != null) {
            pipelines.destroy();
            pipelines = null;
        }
        destroyImages();
        lib = null;
        probeAttempted = false;
        available = false;
        failed = false;
        frameIndex = 0;
    }

    private void destroyImages() {
        normalRoughness = destroyImage(normalRoughness);
        viewZ = destroyImage(viewZ);
        diffIn = destroyImage(diffIn);
        specIn = destroyImage(specIn);
        diffOut = destroyImage(diffOut);
        specOut = destroyImage(specOut);
        renderWidth = -1;
        renderHeight = -1;
        contextWidth = -1;
        contextHeight = -1;
    }

    private static RtImage destroyImage(RtImage image) {
        if (image != null) {
            image.destroy();
        }
        return null;
    }
}

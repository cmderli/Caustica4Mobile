package dev.comfyfluffy.caustica.rt.ffx;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vulkan.VulkanDevice;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.mixin.GpuDeviceAccessor;
import dev.comfyfluffy.caustica.rt.BundledNative;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.accel.RtImage;


import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkInstance;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * FSR 3.1 temporal upscaler over the AMD FidelityFX API, used as the upscaling half of the built-in
 * reconstruction chain: the denoiser filters at render resolution and this brings the result to display
 * resolution with reprojection, instead of the bilinear resolve.
 *
 * <p>It is strictly optional. The runtime is a vendor-neutral Vulkan library but is not bundled with the
 * mod, so everything here is best-effort: when the library or a usable context is absent,
 * {@link #isAvailable()} stays false and the chain keeps its own resolve. The library is loaded lazily
 * and the failure is latched, so a missing runtime costs one filesystem probe per session, not one per
 * frame.
 */
public final class FsrUpscaler {
    public static final FsrUpscaler INSTANCE = new FsrUpscaler();

    private static final BundledNative NATIVE = new BundledNative("caustica-fsr", "amd_fidelityfx_vk");
    private static final long RESET_WAIT_TIMEOUT_NS = 200_000_000L;

    private FfxLibrary lib;
    private FfxUpscaleContext context;
    private boolean probeAttempted;
    private boolean available;
    private boolean failed;
    private boolean resetHistory;
    private long lastFrameNanos;

    private int contextRenderWidth = -1;
    private int contextRenderHeight = -1;
    private int contextDisplayWidth = -1;
    private int contextDisplayHeight = -1;
    private int contextQuality = Integer.MIN_VALUE;

    private FsrUpscaler() {
    }

    /**
     * Whether this system can run FSR at all. Probed once; the probe loads the library and asks it for the
     * upscale provider versions, which validates both the FFM plumbing and that the library really is the
     * FidelityFX runtime rather than a same-named file.
     */
    public boolean isAvailable() {
        if (probeAttempted) {
            return available;
        }
        probeAttempted = true;
        // Every candidate is tried: a runtime that exists but cannot be opened (Android's linker refuses
        // the external game directory) must not stop the search from reaching one that can.
        Path lastSeen = null;
        Throwable lastFailure = null;
        for (Path library : NATIVE.candidates(CausticaConfig.Fsr.PATH.get())) {
            if (!Files.isRegularFile(library)) {
                continue;
            }
            lastSeen = library;
            try {
                FfxLibrary loaded = FfxLibrary.load(library);
                var versions = FfxUpscaleContext.queryAvailableVersions(loaded);
                if (!versions.isEmpty()) {
                    lib = loaded;
                    available = true;
                    CausticaMod.LOGGER.info("FSR upscaling available ({} version {})", library,
                            versions.get(versions.size() - 1).name());
                    break;
                }
                CausticaMod.LOGGER.warn("FSR runtime at {} exposes no upscale provider", library);
            } catch (Throwable t) {
                lastFailure = t;
            }
        }
        if (!available && lastSeen == null) {
            CausticaMod.LOGGER.info("FSR unavailable: {} not found (bundled natives, run/natives, or -Dcaustica.fsr.path)",
                    NATIVE.libraryName());
        } else if (!available) {
            CausticaMod.LOGGER.warn("FSR runtime at {} could not be loaded; upscaling stays on the built-in resolve",
                    lastSeen, lastFailure);
        }
        return available;
    }

    /**
     * Create or resize the upscale context for this resolution pair. Returns false when FSR cannot run,
     * which leaves the caller on its own resolve path.
     */
    public boolean ensure(int renderWidth, int renderHeight, int displayWidth, int displayHeight, int quality) {
        if (!isAvailable() || failed) {
            return false;
        }
        if (context != null && contextRenderWidth == renderWidth && contextRenderHeight == renderHeight
                && contextDisplayWidth == displayWidth && contextDisplayHeight == displayHeight
                && contextQuality == quality) {
            return true;
        }
        if (!(((GpuDeviceAccessor) RenderSystem.getDevice()).caustica$getBackend() instanceof VulkanDevice device)) {
            return false;
        }
        try {
            releaseContext();
            // HIGH_DYNAMIC_RANGE: the input is scene-linear ACEScg with values above 1, and the exposure is
            // already applied by the tracer, so FSR must not estimate one. DEPTH_INVERTED: the depth guide
            // is reversed-Z. DEBUG_CHECKING makes the runtime report API misuse through the log callback
            // instead of silently producing a wrong image.
            int flags = FfxUpscaleContext.FLAG_HIGH_DYNAMIC_RANGE
                    | FfxUpscaleContext.FLAG_DEPTH_INVERTED
                    | FfxUpscaleContext.FLAG_DEBUG_CHECKING;
            long getDeviceProcAddr;
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkInstance instance = device.vkDevice().getPhysicalDevice().getInstance();
                getDeviceProcAddr = VK10.vkGetInstanceProcAddr(instance, stack.ASCII("vkGetDeviceProcAddr"));
            }
            context = FfxUpscaleContext.create(lib,
                    device.vkDevice().address(),
                    device.vkDevice().getPhysicalDevice().address(),
                    getDeviceProcAddr,
                    renderWidth, renderHeight,
                    displayWidth, displayHeight,
                    flags, true);
            contextRenderWidth = renderWidth;
            contextRenderHeight = renderHeight;
            contextDisplayWidth = displayWidth;
            contextDisplayHeight = displayHeight;
            contextQuality = quality;
            resetHistory = true;
            CausticaMod.LOGGER.info("FSR upscale context created: {}x{} -> {}x{}", renderWidth, renderHeight,
                    displayWidth, displayHeight);
            return true;
        } catch (Throwable t) {
            failed = true;
            releaseContext();
            CausticaMod.LOGGER.error("FSR context creation failed; upscaling stays on the built-in resolve", t);
            return false;
        }
    }

    /**
     * Record the upscale: {@code color} (the denoised render-res image), {@code depth} and {@code motion}
     * are reprojected into {@code out} at display resolution. Jitter is the sub-pixel camera offset already
     * applied to the primary ray, in render pixels; the depth guide is reversed-Z, so the camera planes are
     * passed in FSR's swapped convention.
     */
    public boolean dispatch(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage motion, RtImage out,
                            int renderWidth, int renderHeight, int displayWidth, int displayHeight,
                            float jitterX, float jitterY, float projectionM11, float projectionM22,
                            float projectionM32) {
        if (context == null || failed) {
            return false;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            float frameMs = nextFrameMs();
            CameraPlanes planes = cameraPlanes(projectionM22, projectionM32);
            context.dispatchUpscale(new FfxUpscaleContext.DispatchParams(
                    cmd.address(),
                    new FfxUpscaleContext.Resource(color.image, FfxUpscaleContext.FORMAT_R16G16B16A16_FLOAT,
                            renderWidth, renderHeight, FfxUpscaleContext.USAGE_READ_ONLY,
                            FfxUpscaleContext.STATE_COMMON),
                    new FfxUpscaleContext.Resource(depth.image, FfxUpscaleContext.FORMAT_R32_FLOAT,
                            renderWidth, renderHeight, FfxUpscaleContext.USAGE_DEPTHTARGET,
                            FfxUpscaleContext.STATE_COMMON),
                    new FfxUpscaleContext.Resource(motion.image, FfxUpscaleContext.FORMAT_R16G16_FLOAT,
                            renderWidth, renderHeight, FfxUpscaleContext.USAGE_READ_ONLY,
                            FfxUpscaleContext.STATE_COMMON),
                    new FfxUpscaleContext.Resource(out.image, FfxUpscaleContext.FORMAT_R16G16B16A16_FLOAT,
                            displayWidth, displayHeight, FfxUpscaleContext.USAGE_UAV,
                            FfxUpscaleContext.STATE_UNORDERED_ACCESS),
                    jitterX, jitterY,
                    // Our motion vectors are already in render-pixel units and jitter-free, so the scale is 1
                    // in both axes (the flag for display-res MVs stays unset: they are render-res).
                    1.0f, 1.0f,
                    renderWidth, renderHeight,
                    displayWidth, displayHeight,
                    frameMs, resetHistory,
                    planes.near(), planes.far(), fovY(projectionM11),
                    0));
            resetHistory = false;
            return true;
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("FSR dispatch failed; upscaling stays on the built-in resolve", t);
            return false;
        }
    }

    private float nextFrameMs() {
        long now = System.nanoTime();
        float frameMs = lastFrameNanos == 0 ? 16.6f
                : Math.clamp((now - lastFrameNanos) / 1_000_000.0f, 0.1f, 200.0f);
        lastFrameNanos = now;
        return frameMs;
    }

    /** FSR's naming under the DEPTH_INVERTED flag: {@code near} carries the far-plane distance. */
    private record CameraPlanes(float near, float far) {
    }

    /**
     * Near and far plane distances recovered from the projection matrix. Under reversed-Z FSR wants
     * {@code cameraNear} to carry the far-plane distance and vice versa, so the returned pair is already
     * swapped for the DEPTH_INVERTED flag this context is created with.
     */
    private static CameraPlanes cameraPlanes(float projectionM22, float projectionM32) {
        float far = Math.abs(projectionM22) < 1.0e-9f ? 1.0e9f : -projectionM32 / projectionM22;
        float near = far / (1.0f + projectionM22) * projectionM22;
        if (!(far > 0.0f) || !Float.isFinite(far)) {
            far = 1.0e9f;
        }
        if (!(near > 0.0f) || !Float.isFinite(near) || near > far) {
            near = 0.05f;
        }
        return new CameraPlanes(far, near);
    }

    private static float fovY(float projectionM11) {
        return projectionM11 > 1.0e-6f ? (float) (2.0 * Math.atan(1.0 / projectionM11)) : (float) Math.toRadians(70.0);
    }

    public void destroy() {
        releaseContext();
        lib = null;
        probeAttempted = false;
        available = false;
        failed = false;
        lastFrameNanos = 0;
    }

    private void releaseContext() {
        if (context == null) {
            return;
        }
        try {
            // No frame may still be reading the resources the context owns.
            RtContext ctx = RtContext.currentOrNull();
            if (ctx != null) {
                ctx.waitIdle();
            }
            context.close();
        } catch (Throwable t) {
            CausticaMod.LOGGER.warn("FSR context release failed", t);
        }
        context = null;
        contextRenderWidth = -1;
        contextRenderHeight = -1;
        contextDisplayWidth = -1;
        contextDisplayHeight = -1;
        contextQuality = Integer.MIN_VALUE;
    }

}

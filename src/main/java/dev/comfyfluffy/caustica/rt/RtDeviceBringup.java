package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.vulkan.VulkanBackend;
import com.mojang.blaze3d.vulkan.VulkanPhysicalDevice;
import com.mojang.blaze3d.vulkan.init.VulkanFeature;
import com.mojang.blaze3d.vulkan.init.VulkanPNextStruct;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VKCapabilitiesDevice;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkPhysicalDeviceAccelerationStructureFeaturesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceAccelerationStructurePropertiesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures2;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties2;
import org.lwjgl.vulkan.VkPhysicalDeviceRayTracingPipelineFeaturesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceRayTracingPipelinePropertiesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceRayQueryFeaturesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan12Features;
import org.lwjgl.vulkan.VkPhysicalDeviceOpacityMicromapFeaturesEXT;
import org.lwjgl.vulkan.VkPhysicalDeviceOpacityMicromapPropertiesEXT;
import org.lwjgl.vulkan.VkPhysicalDevicePresentIdFeaturesKHR;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.spongepowered.asm.mixin.injection.invoke.arg.Args;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
import static org.lwjgl.vulkan.KHRDeferredHostOperations.VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPositionFetch.VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRRayTracingPositionFetch.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR;
import static org.lwjgl.vulkan.KHRRayQuery.VK_KHR_RAY_QUERY_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRRayQuery.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
import static org.lwjgl.vulkan.EXTOpacityMicromap.VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME;
import static org.lwjgl.vulkan.EXTOpacityMicromap.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT;
import static org.lwjgl.vulkan.EXTOpacityMicromap.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT;
import static org.lwjgl.vulkan.NVLowLatency2.VK_NV_LOW_LATENCY_2_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRPresentId.VK_KHR_PRESENT_ID_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRPresentId.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;

/**
 * RT device bring-up. Enables the hardware ray-tracing device extensions and their
 * feature structs on vanilla's Blaze3D device at {@code vkCreateDevice} time.
 *
 * <p>Vanilla assembles a {@code VkPhysicalDeviceFeatures2} pNext chain from the
 * {@code Set<VulkanFeature>} (arg2) via {@code VulkanFeature.set} →
 * {@code findOrCreateStructInPNextChain} (dedup by sType), so {@code bufferDeviceAddress}
 * merges into the existing {@code VkPhysicalDeviceVulkan12Features} struct and the two
 * KHR structs are created fresh. BDA / descriptor-indexing / SPIR-V 1.4 are core on the
 * 1.4 device, so only three extension <i>names</i> are needed; the rest are feature enables.
 *
 * <p>Extension names are added to the device extension list separately; feature structs are added here.
 * Both are gated on the selected device actually supporting RT; if not, nothing is added
 * and the device comes up exactly as vanilla. {@code caustica.rt} is read once here, at
 * {@code vkCreateDevice} time, before the device exists — flipping it later at runtime cannot add
 * device features to an already-created device, so a config change only takes effect on restart.
 */
public final class RtDeviceBringup {
    public static boolean enabledByProperty() {
        return CausticaConfig.Rt.ENABLED.value();
    }

    /**
     * The device extensions every RT backend needs (BDA/descriptor-indexing/SPIR-V 1.4 are core on 1.4).
     * {@code ray_tracing_position_fetch} lets a hit read triangle vertex positions in object space — the
     * builtin {@code gl_HitTriangleVertexPositionsEXT} in a closest-hit stage, the ray-query
     * intersection query in an inline traversal — for the normal-map TBN, avoiding a positions buffer
     * plumbed through the geometry tables. LabPBR normal maps depend on it on both backends.
     * {@code ray_query} is the inline-traversal backend itself (the whole world pass on devices without a
     * ray-tracing pipeline, and the world-overlay pass's {@code rayQueryEXT} occlusion test on devices
     * with one).
     */
    public static final List<String> RT_BASE_EXTENSIONS = List.of(
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME);

    /**
     * The ray-tracing pipeline extension. Preferred when present — the SBT gives each hit record its own
     * shader and EXT SER can reorder invocations — but never required: without it the world pass runs on
     * {@link WorldBackend#RAY_QUERY}, which traverses the same acceleration structures from compute.
     */
    public static final String RT_PIPELINE_EXTENSION = VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME;

    /**
     * OPTIONAL RT extensions: enabled only when the selected device supports them AND the gate is on, but
     * never required — a device lacking them still comes up RT-capable (unlike {@link #RT_BASE_EXTENSIONS}, whose
     * absence disables RT entirely). {@code VK_EXT_opacity_micromap} (any-hit opt, lever C): per-triangle
     * opacity micromaps let the hardware skip the alpha-test hit record on fully-opaque/transparent cutout
     * micro-triangles, so it runs only on the foliage silhouette. Hardware-accelerated on RTX 40-series and
     * Blackwell; absent / software elsewhere, and meaningless to the ray-query backend, which has no hit
     * records for the hardware to skip — hence optional.
     */
    public static final List<String> OPTIONAL_RT_EXTENSIONS = List.of(
            VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);

    /**
     * NVIDIA Reflex. {@code VK_NV_low_latency2} adds no feature bits (function-only extension). Bundled with
     * {@code VK_KHR_present_id}: Reflex's latency markers carry a {@code presentID} that only correlates with
     * a specific present call when that present's {@code vkQueuePresentKHR} chains a matching
     * {@code VkPresentIdKHR} — which needs its own device feature bit (unlike low_latency2, which is
     * function-only).
     */
    public static final List<String> REFLEX_EXTENSIONS = List.of(
            VK_NV_LOW_LATENCY_2_EXTENSION_NAME, VK_KHR_PRESENT_ID_EXTENSION_NAME);

    /**
     * The two NVX device extensions the NGX runtime needs on the device. Named here so the capability
     * latch below and the device-extension request list in {@code VulkanBackendMixin} cannot drift apart
     * on which extensions make a device NGX-capable.
     */
    public static final String NGX_BINARY_IMPORT_EXTENSION = "VK_NVX_binary_import";
    public static final String NGX_IMAGE_VIEW_HANDLE_EXTENSION = "VK_NVX_image_view_handle";

    /**
     * NRD's REBLUR blur/history passes call {@code QuadReadAcrossX/Y} from compute shaders, which Vulkan
     * only allows when this extension and its {@code computeDerivativeGroupQuads} feature are enabled. It
     * is not core in 1.4, so the name has to be requested and the feature chained explicitly. RELAX needs
     * neither.
     */
    public static final String NRD_DERIVATIVES_EXTENSION = "VK_KHR_compute_shader_derivatives";
    // VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR { sType, pNext, quads, linear }. Hand-laid out
    // because this LWJGL build predates the extension and has no generated binding for it.
    private static final int DERIVATIVES_FEATURES_S_TYPE = 1_000_201_000;
    private static final int DERIVATIVES_FEATURES_BYTES = 24;
    private static final int DERIVATIVES_QUADS_OFFSET = 16;

    private static volatile boolean rtRequested;
    private static volatile boolean ngxCapable; // NVIDIA device advertising the NVX extensions NGX needs
    private static volatile boolean nrdDerivativesCapable; // NRD_DERIVATIVES_EXTENSION enabled + supported
    private static volatile SerBackend serBackend = SerBackend.NONE;
    private static volatile WorldBackend worldBackend = WorldBackend.NONE;
    private static volatile boolean ommEnabled; // VK_EXT_opacity_micromap actually enabled on the device
    private static volatile boolean reflexEnabled; // VK_NV_low_latency2 actually enabled on the device
    private static volatile boolean presentIdEnabled; // VK_KHR_present_id actually enabled on the device
    private static volatile boolean wideLinesEnabled; // VkPhysicalDeviceFeatures.wideLines actually enabled
    private static volatile float maxLineWidth = 1.0f; // device's lineWidthRange[1]; 1.0 unless wideLinesEnabled
    private static volatile int overlayMsaaSamples = VK10.VK_SAMPLE_COUNT_1_BIT; // capped to the device's framebufferColorSampleCounts
    private static volatile int maxOpacity4StateSubdivisionLevel;
    private static volatile int computeQueueFamilyIndex = -1;
    private static volatile int computeQueueIndex = -1;
    private static boolean loggedUnavailable;
    private static boolean loggedSerBindingsMissing;

    /**
     * The LWJGL binding class for {@code VK_EXT_ray_tracing_invocation_reorder}. Minecraft ships one
     * LWJGL, but Android launchers substitute their own: the launcher's LWJGL 3.3.6 has no bindings for
     * this extension at all. A direct reference to it is resolved while this class initializes, so its
     * absence surfaces as a {@code NoClassDefFoundError} before any device exists and takes the whole
     * client down. Naming it as a string and reading its constants reflectively turns that into "SER is
     * unavailable on this LWJGL build".
     */
    private static final String SER_FEATURES_CLASS =
            "org.lwjgl.vulkan.VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT";
    private static final String SER_EXTENSION_NAME = "VK_EXT_ray_tracing_invocation_reorder";
    // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT. The sType is an ABI
    // constant, so it stays valid even when the LWJGL class that would have named it is absent.
    private static final int SER_FEATURES_S_TYPE = 1_000_581_000;

    private static final VulkanPNextStruct AS_FEATURES_STRUCT = new VulkanPNextStruct(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            VkPhysicalDeviceAccelerationStructureFeaturesKHR.SIZEOF);
    private static final VulkanPNextStruct RT_PIPELINE_FEATURES_STRUCT = new VulkanPNextStruct(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR.SIZEOF);
    private static final VulkanPNextStruct POSITION_FETCH_FEATURES_STRUCT = new VulkanPNextStruct(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
            VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR.SIZEOF);
    private static final VulkanPNextStruct RAY_QUERY_FEATURES_STRUCT = new VulkanPNextStruct(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
            VkPhysicalDeviceRayQueryFeaturesKHR.SIZEOF);
    private static final VulkanPNextStruct SER_EXT_FEATURES_STRUCT = optionalStruct(
            SER_FEATURES_CLASS, SER_FEATURES_S_TYPE);
    private static final VulkanPNextStruct OMM_FEATURES_STRUCT = new VulkanPNextStruct(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT,
            VkPhysicalDeviceOpacityMicromapFeaturesEXT.SIZEOF);
    private static final VulkanPNextStruct PRESENT_ID_FEATURES_STRUCT = new VulkanPNextStruct(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
            VkPhysicalDevicePresentIdFeaturesKHR.SIZEOF);

    private static final VulkanFeature BUFFER_DEVICE_ADDRESS_FEATURE = new VulkanFeature(
            VulkanBackend.VK12_FEATURES_STRUCT, "bufferDeviceAddress",
            VkPhysicalDeviceVulkan12Features.BUFFERDEVICEADDRESS);
    private static final VulkanFeature RUNTIME_DESCRIPTOR_ARRAY_FEATURE = new VulkanFeature(
            VulkanBackend.VK12_FEATURES_STRUCT, "runtimeDescriptorArray",
            VkPhysicalDeviceVulkan12Features.RUNTIMEDESCRIPTORARRAY);
    private static final VulkanFeature SAMPLED_IMAGE_NON_UNIFORM_FEATURE = new VulkanFeature(
            VulkanBackend.VK12_FEATURES_STRUCT, "shaderSampledImageArrayNonUniformIndexing",
            VkPhysicalDeviceVulkan12Features.SHADERSAMPLEDIMAGEARRAYNONUNIFORMINDEXING);
    private static final VulkanFeature DESCRIPTOR_PARTIALLY_BOUND_FEATURE = new VulkanFeature(
            VulkanBackend.VK12_FEATURES_STRUCT, "descriptorBindingPartiallyBound",
            VkPhysicalDeviceVulkan12Features.DESCRIPTORBINDINGPARTIALLYBOUND);
    private static final VulkanFeature SAMPLED_IMAGE_UPDATE_AFTER_BIND_FEATURE = new VulkanFeature(
            VulkanBackend.VK12_FEATURES_STRUCT, "descriptorBindingSampledImageUpdateAfterBind",
            VkPhysicalDeviceVulkan12Features.DESCRIPTORBINDINGSAMPLEDIMAGEUPDATEAFTERBIND);
    private static final VulkanFeature SHADER_INT64_FEATURE = new VulkanFeature(
            VulkanBackend.VK10_FEATURES_STRUCT, "shaderInt64", VkPhysicalDeviceFeatures.SHADERINT64);
    private static final VulkanFeature ACCELERATION_STRUCTURE_FEATURE = new VulkanFeature(
            AS_FEATURES_STRUCT, "accelerationStructure",
            VkPhysicalDeviceAccelerationStructureFeaturesKHR.ACCELERATIONSTRUCTURE);
    private static final VulkanFeature RAY_TRACING_PIPELINE_FEATURE = new VulkanFeature(
            RT_PIPELINE_FEATURES_STRUCT, "rayTracingPipeline",
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR.RAYTRACINGPIPELINE);
    private static final VulkanFeature POSITION_FETCH_FEATURE = new VulkanFeature(
            POSITION_FETCH_FEATURES_STRUCT, "rayTracingPositionFetch",
            VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR.RAYTRACINGPOSITIONFETCH);
    private static final VulkanFeature RAY_QUERY_FEATURE = new VulkanFeature(
            RAY_QUERY_FEATURES_STRUCT, "rayQuery", VkPhysicalDeviceRayQueryFeaturesKHR.RAYQUERY);
    private static final VulkanFeature SER_EXT_FEATURE = optionalFeature(
            SER_EXT_FEATURES_STRUCT, SER_FEATURES_CLASS, "RAYTRACINGINVOCATIONREORDER",
            "rayTracingInvocationReorder(EXT)");
    private static final VulkanFeature OMM_FEATURE = new VulkanFeature(
            OMM_FEATURES_STRUCT, "micromap", VkPhysicalDeviceOpacityMicromapFeaturesEXT.MICROMAP);
    private static final VulkanFeature PRESENT_ID_FEATURE = new VulkanFeature(
            PRESENT_ID_FEATURES_STRUCT, "presentId", VkPhysicalDevicePresentIdFeaturesKHR.PRESENTID);
    private static final VulkanFeature WIDE_LINES_FEATURE = new VulkanFeature(
            VulkanBackend.VK10_FEATURES_STRUCT, "wideLines", VkPhysicalDeviceFeatures.WIDELINES);

    /**
     * The pNext struct for a feature whose LWJGL binding class this client may not have, or null when it
     * is absent (see {@link #SER_FEATURES_CLASS}). Null means "this build cannot express the feature",
     * which callers treat exactly like a device that does not support it.
     */
    private static VulkanPNextStruct optionalStruct(String bindingClass, int sType) {
        int size = optionalStaticInt(bindingClass, "SIZEOF", 0);
        return size > 0 ? new VulkanPNextStruct(sType, size) : null;
    }

    /** The VulkanFeature for {@code struct}, or null when its binding class (or member offset) is absent. */
    private static VulkanFeature optionalFeature(VulkanPNextStruct struct, String bindingClass,
                                                 String memberOffset, String name) {
        if (struct == null) {
            return null;
        }
        int offset = optionalStaticInt(bindingClass, memberOffset, -1);
        return offset >= 0 ? new VulkanFeature(struct, name, offset) : null;
    }

    /** A static int constant of an LWJGL binding class, or {@code fallback} when the class is absent. */
    static int optionalStaticInt(String className, String field, int fallback) {
        try {
            Class<?> binding = Class.forName(className, false, RtDeviceBringup.class.getClassLoader());
            return binding.getField(field).getInt(null);
        } catch (Throwable ignored) {
            return fallback;
        }
    }

    private static final List<VulkanFeature> REQUIRED_RT_FEATURES = List.of(
            BUFFER_DEVICE_ADDRESS_FEATURE,
            RUNTIME_DESCRIPTOR_ARRAY_FEATURE,
            SAMPLED_IMAGE_NON_UNIFORM_FEATURE,
            DESCRIPTOR_PARTIALLY_BOUND_FEATURE,
            SAMPLED_IMAGE_UPDATE_AFTER_BIND_FEATURE,
            SHADER_INT64_FEATURE,
            ACCELERATION_STRUCTURE_FEATURE,
            POSITION_FETCH_FEATURE,
            RAY_QUERY_FEATURE);

    /**
     * Which traversal the world pass runs on. Both backends trace the same acceleration structures and
     * share every shader below the entry points; they differ in who owns traversal. The ray-tracing
     * pipeline dispatches raygen/any-hit/closest-hit/miss stages through an SBT and can reorder
     * invocations (EXT SER); the ray-query backend inline-traverses from one compute thread per pixel and
     * emulates the any-hit records with a candidate loop, which is what makes the renderer run on devices
     * that expose {@code VK_KHR_ray_query} without {@code VK_KHR_ray_tracing_pipeline}.
     */
    public enum WorldBackend {
        NONE("none", null, null),
        // The ray-tracing pipeline's entry points come from SerBackend, which knows whether the ordinary
        // or the reordered indirect shader was compiled in; only the ray-query backend names its compute
        // shaders here.
        RAY_TRACING_PIPELINE("ray tracing pipeline", null, null),
        RAY_QUERY("ray query", "primary_rq.comp.spv", "indirect_rq.comp.spv");

        final String label;
        final String rayQueryPrimaryShader;
        final String rayQueryMainShader;

        WorldBackend(String label, String rayQueryPrimaryShader, String rayQueryMainShader) {
            this.label = label;
            this.rayQueryPrimaryShader = rayQueryPrimaryShader;
            this.rayQueryMainShader = rayQueryMainShader;
        }
    }

    private enum SerBackend {
        NONE("none", null, "primary.rgen.spv", "indirect.rgen.spv"),
        EXT("EXT", SER_EXTENSION_NAME, "primary.rgen.spv", "indirect_ser.rgen.spv");

        final String label;
        final String extensionName;
        final String worldPrimaryRaygenShader;
        final String worldRaygenShader;

        SerBackend(String label, String extensionName, String worldPrimaryRaygenShader,
                   String worldRaygenShader) {
            this.label = label;
            this.extensionName = extensionName;
            this.worldPrimaryRaygenShader = worldPrimaryRaygenShader;
            this.worldRaygenShader = worldRaygenShader;
        }
    }

    private record FeatureSupport(List<String> missingRequired, boolean rayTracingPipeline,
                                  SerBackend serBackend,
                                  boolean omm, boolean presentId, boolean wideLines) {
        boolean supportsRt() {
            return missingRequired.isEmpty();
        }
    }

    private RtDeviceBringup() {
    }

    /** True once we have augmented a device creation to request RT (extensions + features). */
    public static boolean rtRequested() {
        return rtRequested;
    }

    /** The traversal backend the world pass runs on; {@link WorldBackend#NONE} until device creation. */
    public static WorldBackend worldBackend() {
        return worldBackend;
    }

    /** True when the world pass inline-traverses from compute instead of using a ray-tracing pipeline. */
    public static boolean worldUsesRayQuery() {
        return worldBackend == WorldBackend.RAY_QUERY;
    }

    /**
     * True when the selected device can host the NGX/DLSS runtime: NVIDIA silicon advertising both NVX
     * extensions. Latched at device creation so every DLSS feature can be gated before it touches
     * {@code NgxRuntime} — on a device that merely lacks them, NGX initialization fails inside the
     * feature's first use, and a feature that lets that failure escape as an exception takes the whole RT
     * composite down with it. The vendor check is not redundant with the extension check: a driver can
     * advertise the NVX entry points without a usable NGX stack behind them, and NGX's own init is
     * best-effort, so this only ever narrows what is attempted.
     */
    public static boolean ngxCapable() {
        return ngxCapable;
    }

    /**
     * True when the device was created with {@code VK_KHR_compute_shader_derivatives} and its quad
     * derivative feature. NRD's REBLUR cannot run without it, so the NRD backend checks this before it
     * touches the shim; RELAX would not need it.
     */
    public static boolean nrdDerivativesCapable() {
        return nrdDerivativesCapable;
    }

    /**
     * The indirect/bounce pass entry point for the selected backend. The ray-tracing pipeline names its
     * raygen shader here (the SER variant when the extension is enabled), the ray-query backend the
     * compute shader dispatched over the render target.
     */
    public static String worldRaygenShader() {
        return worldBackend == WorldBackend.RAY_QUERY
                ? WorldBackend.RAY_QUERY.rayQueryMainShader : serBackend.worldRaygenShader;
    }

    /** The primary/guide pass entry point for the selected backend. */
    public static String worldPrimaryRaygenShader() {
        return worldBackend == WorldBackend.RAY_QUERY
                ? WorldBackend.RAY_QUERY.rayQueryPrimaryShader : serBackend.worldPrimaryRaygenShader;
    }

    /** Ray-query compute entry points for ReSTIR PT. */
    public static final String RAY_QUERY_RESTIR_TEMPORAL_SHADER = "restir_temporal_rq.comp.spv";
    public static final String RAY_QUERY_RESTIR_SPATIAL_SHADER = "restir_spatial_rq.comp.spv";

    /** ReSTIR PT path-generation + temporal-reuse entry point for the selected backend. */
    public static String worldRestirTemporalShader() {
        return worldBackend == WorldBackend.RAY_QUERY
                ? RAY_QUERY_RESTIR_TEMPORAL_SHADER : "restir_temporal.rgen.spv";
    }

    /** ReSTIR PT spatial-reuse + final-shading entry point for the selected backend. */
    public static String worldRestirSpatialShader() {
        return worldBackend == WorldBackend.RAY_QUERY
                ? RAY_QUERY_RESTIR_SPATIAL_SHADER : "restir_spatial.rgen.spv";
    }

    public static boolean serExtEnabled() {
        return serBackend == SerBackend.EXT;
    }

    /** True if {@code VK_EXT_opacity_micromap} was enabled on the device (gate on + device support). */
    public static boolean ommEnabled() {
        return ommEnabled;
    }

    /** True if {@code VK_NV_low_latency2} (Reflex) was enabled on the device (gate on + device support). */
    public static boolean reflexEnabled() {
        return reflexEnabled;
    }

    /** True if {@code VK_KHR_present_id} was enabled on the device (needed for Reflex marker/present correlation). */
    public static boolean presentIdEnabled() {
        return presentIdEnabled;
    }

    /** Hardware limit for 4-state opacity micromaps, populated by {@link #probe(VkDevice)}. */
    public static int maxOpacity4StateSubdivisionLevel() {
        return maxOpacity4StateSubdivisionLevel;
    }

    /** True if {@code VkPhysicalDeviceFeatures.wideLines} was enabled on the device (world-overlay thick
     *  lines, e.g. the block outline, use this instead of a screen-space quad when available). */
    public static boolean wideLinesEnabled() {
        return wideLinesEnabled;
    }

    /** The device's max native line width (raster {@code lineWidthRange[1]}); 1.0 if wideLines isn't
     *  enabled (Vulkan mandates exactly 1.0 in that case). Callers must clamp their desired width to this. */
    public static float maxLineWidth() {
        return maxLineWidth;
    }

    /** {@code VK_SAMPLE_COUNT_4_BIT} capped down to whatever the device's {@code framebufferColorSampleCounts}
     *  actually advertises (2x, or 1x/no MSAA on the rare device that lacks even that) — no device feature to
     *  enable, just a raster/framebuffer property, unlike {@link #wideLinesEnabled()}. World-overlay passes
     *  that need edge AA (e.g. the block outline's native wide line) use this as their pipeline's
     *  {@code rasterizationSamples}. */
    public static int overlayMsaaSamples() {
        return overlayMsaaSamples;
    }

    /** Whether device creation reserved a queue handle exclusively for Caustica terrain work. */
    public static boolean computeQueueReserved() {
        return computeQueueFamilyIndex >= 0 && computeQueueIndex >= 0;
    }

    public static int computeQueueFamilyIndex() {
        if (!computeQueueReserved()) {
            throw new IllegalStateException("Caustica compute queue was not reserved");
        }
        return computeQueueFamilyIndex;
    }

    public static int computeQueueIndex() {
        if (!computeQueueReserved()) {
            throw new IllegalStateException("Caustica compute queue was not reserved");
        }
        return computeQueueIndex;
    }

    /**
     * Reserve one additional physical queue at device-creation time. Minecraft's queue-family map only
     * requests handles for its graphics/compute/transfer queues; fetching a higher queue index without first
     * increasing the matching {@link VkDeviceQueueCreateInfo#queueCount()} would be invalid. Prefer a
     * compute-only family, but add a dedicated compute family when that leaves the Minecraft queues
     * untouched and has a free physical slot.
     */
    public static void reserveComputeQueue(VkDeviceCreateInfo deviceCreateInfo,
                                           VulkanPhysicalDevice physicalDevice, MemoryStack stack) {
        computeQueueFamilyIndex = -1;
        computeQueueIndex = -1;
        if (!rtRequested) {
            return;
        }

        VkDeviceQueueCreateInfo.Buffer requestedQueues = deviceCreateInfo.pQueueCreateInfos();
        if (requestedQueues == null) {
            CausticaMod.LOGGER.warn("Caustica RT disabled: Vulkan device creation has no queue requests");
            return;
        }

        var count = stack.callocInt(1);
        VK10.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.vkPhysicalDevice(), count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.calloc(count.get(0), stack);
        VK10.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.vkPhysicalDevice(), count, families);

        int[] requestedCounts = new int[families.capacity()];
        for (int i = 0; i < requestedQueues.capacity(); i++) {
            VkDeviceQueueCreateInfo request = requestedQueues.get(i);
            int family = request.queueFamilyIndex();
            if (family >= 0 && family < requestedCounts.length) {
                requestedCounts[family] = request.queueCount();
            }
        }

        int selectedFamily = -1;
        int selectedScore = Integer.MAX_VALUE;
        for (int family = 0; family < families.capacity(); family++) {
            VkQueueFamilyProperties properties = families.get(family);
            int flags = properties.queueFlags();
            if ((flags & VK10.VK_QUEUE_COMPUTE_BIT) == 0
                    || requestedCounts[family] >= properties.queueCount()) {
                continue;
            }
            // Compute-only families win; fewer capabilities then fewer already-requested handles break ties.
            int score = ((flags & VK10.VK_QUEUE_GRAPHICS_BIT) != 0 ? 1_000 : 0)
                    + Integer.bitCount(flags) * 10 + requestedCounts[family];
            if (score < selectedScore) {
                selectedFamily = family;
                selectedScore = score;
            }
        }

        if (selectedFamily < 0) {
            CausticaMod.LOGGER.warn(
                    "Caustica RT disabled: no spare compute-capable Vulkan queue is available");
            return;
        }

        int queueIndex = requestedCounts[selectedFamily];
        VkDeviceQueueCreateInfo matchingRequest = null;
        for (int i = 0; i < requestedQueues.capacity(); i++) {
            VkDeviceQueueCreateInfo request = requestedQueues.get(i);
            if (request.queueFamilyIndex() == selectedFamily) {
                matchingRequest = request;
                break;
            }
        }

        if (matchingRequest != null) {
            var oldPriorities = matchingRequest.pQueuePriorities();
            var priorities = stack.callocFloat(queueIndex + 1);
            if (oldPriorities != null) {
                for (int i = 0; i < queueIndex; i++) {
                    priorities.put(i, oldPriorities.get(i));
                }
            }
            matchingRequest.pQueuePriorities(priorities);
        } else {
            VkDeviceQueueCreateInfo.Buffer expanded = VkDeviceQueueCreateInfo.calloc(requestedQueues.capacity() + 1, stack);
            for (int i = 0; i < requestedQueues.capacity(); i++) {
                VkDeviceQueueCreateInfo source = requestedQueues.get(i);
                expanded.get(i)
                        .sType(source.sType())
                        .pNext(source.pNext())
                        .flags(source.flags())
                        .queueFamilyIndex(source.queueFamilyIndex())
                        .pQueuePriorities(source.pQueuePriorities());
            }
            expanded.get(requestedQueues.capacity())
                    .sType$Default()
                    .queueFamilyIndex(selectedFamily)
                    .pQueuePriorities(stack.callocFloat(1));
            deviceCreateInfo.pQueueCreateInfos(expanded);
        }

        computeQueueFamilyIndex = selectedFamily;
        computeQueueIndex = queueIndex;
        VkQueueFamilyProperties selected = families.get(selectedFamily);
        CausticaMod.LOGGER.info(
                "Reserved Caustica compute queue family={} index={} (family queues={}, flags=0x{})",
                selectedFamily, queueIndex, selected.queueCount(), Integer.toHexString(selected.queueFlags()));
    }

    /** Optional extensions whose extension and feature requirements are both supported. */
    private static List<String> supportedOptionalExtensions(VulkanPhysicalDevice physicalDevice,
                                                             FeatureSupport support) {
        List<String> supported = new ArrayList<>();
        if (support.omm) {
            supported.add(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);
        }
        if (reflexRequested() && physicalDevice.hasDeviceExtension(VK_NV_LOW_LATENCY_2_EXTENSION_NAME)) {
            supported.add(VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
            if (support.presentId) {
                supported.add(VK_KHR_PRESENT_ID_EXTENSION_NAME);
            }
        }
        return supported;
    }

    private static boolean ommRequested() {
        return CausticaConfig.Rt.Omm.ENABLED.value();
    }

    /** True when the config forces the ray-query traversal even on a device that could run the pipeline. */
    private static boolean rayQueryForced() {
        return "ray-query".equals(CausticaConfig.Rt.Traversal.BACKEND.get());
    }

    private static boolean reflexRequested() {
        return CausticaConfig.Rt.Reflex.ENABLED.value();
    }

    /** Query every feature boolean Caustica might enable in one complete Features2 chain. */
    private static FeatureSupport queryFeatureSupport(VulkanPhysicalDevice physicalDevice) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceFeatures2 available = VkPhysicalDeviceFeatures2.calloc(stack).sType$Default();
            for (VulkanFeature feature : REQUIRED_RT_FEATURES) {
                feature.struct().findOrCreateStructInPNextChain(available, stack);
            }

            // The pipeline feature is queried only when its extension is present and wanted, so a
            // ray-query-only device — and a device forced onto the ray-query backend — reports it as
            // absent rather than as a missing requirement. SER and OMM are gated on the same answer,
            // because both ride on records only a ray-tracing pipeline dispatches.
            boolean hasPipelineExt = !rayQueryForced()
                    && physicalDevice.hasDeviceExtension(RT_PIPELINE_EXTENSION);
            if (hasPipelineExt) {
                RAY_TRACING_PIPELINE_FEATURE.struct().findOrCreateStructInPNextChain(available, stack);
            }
            // SER needs the extension, the pipeline that dispatches its records, and LWJGL bindings for
            // its feature struct — the last of which an Android launcher's LWJGL may not ship.
            boolean hasSerExt = hasPipelineExt && SER_EXT_FEATURE != null
                    && physicalDevice.hasDeviceExtension(SER_EXTENSION_NAME);
            if (hasSerExt) {
                SER_EXT_FEATURE.struct().findOrCreateStructInPNextChain(available, stack);
            }

            boolean queryOmm = ommRequested() && hasPipelineExt
                    && physicalDevice.hasDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);
            if (queryOmm) {
                OMM_FEATURE.struct().findOrCreateStructInPNextChain(available, stack);
            }
            boolean queryPresentId = reflexRequested()
                    && physicalDevice.hasDeviceExtension(VK_NV_LOW_LATENCY_2_EXTENSION_NAME)
                    && physicalDevice.hasDeviceExtension(VK_KHR_PRESENT_ID_EXTENSION_NAME);
            if (queryPresentId) {
                PRESENT_ID_FEATURE.struct().findOrCreateStructInPNextChain(available, stack);
            }
            WIDE_LINES_FEATURE.struct().findOrCreateStructInPNextChain(available, stack);

            VK12.vkGetPhysicalDeviceFeatures2(physicalDevice.vkPhysicalDevice(), available);

            List<String> missing = new ArrayList<>();
            for (VulkanFeature feature : REQUIRED_RT_FEATURES) {
                if (!feature.get(available)) {
                    missing.add(feature.name());
                }
            }
            boolean rayTracingPipeline = hasPipelineExt && RAY_TRACING_PIPELINE_FEATURE.get(available);
            SerBackend supportedSer = hasSerExt && SER_EXT_FEATURE.get(available) ? SerBackend.EXT
                    : SerBackend.NONE;
            return new FeatureSupport(missing, rayTracingPipeline, supportedSer,
                    queryOmm && OMM_FEATURE.get(available),
                    queryPresentId && PRESENT_ID_FEATURE.get(available),
                    WIDE_LINES_FEATURE.get(available));
        }
    }

    private static String firstUnsupportedExtension(VulkanPhysicalDevice physicalDevice) {
        for (String ext : RT_BASE_EXTENSIONS) {
            if (!physicalDevice.hasDeviceExtension(ext)) {
                return ext;
            }
        }
        return null;
    }

    /** Standalone path: add RT extension names to the (mutable) arg0 list. */
    public static void addExtensions(List<String> augmentedExtensions, VulkanPhysicalDevice physicalDevice) {
        // Latched before the RT gates below: whether this device can host NGX has nothing to do with
        // whether it can host ray tracing, and the DLSS features are gated on it either way.
        ngxCapable = "NVIDIA".equals(physicalDevice.vendorName())
                && physicalDevice.hasDeviceExtension(NGX_BINARY_IMPORT_EXTENSION)
                && physicalDevice.hasDeviceExtension(NGX_IMAGE_VIEW_HANDLE_EXTENSION);
        if (!ngxCapable) {
            CausticaMod.LOGGER.info(
                    "NGX unavailable on [{}]: needs an NVIDIA device exposing {} and {}; DLSS features stay disabled",
                    physicalDevice.deviceName(), NGX_BINARY_IMPORT_EXTENSION, NGX_IMAGE_VIEW_HANDLE_EXTENSION);
        }
        if (!enabledByProperty() || firstUnsupportedExtension(physicalDevice) != null) {
            return;
        }
        FeatureSupport support = queryFeatureSupport(physicalDevice);
        if (!support.supportsRt()) {
            return;
        }
        for (String ext : RT_BASE_EXTENSIONS) {
            if (!augmentedExtensions.contains(ext)) {
                augmentedExtensions.add(ext);
            }
        }
        // The pipeline extension is added when the device advertises it and the ray-query backend is not
        // forced; without it the world pass falls back to ray query instead of disabling RT.
        if (support.rayTracingPipeline && !rayQueryForced()
                && !augmentedExtensions.contains(RT_PIPELINE_EXTENSION)) {
            augmentedExtensions.add(RT_PIPELINE_EXTENSION);
        }
        // Requested, and the feature chained, only when the driver actually supports it: adding the name
        // for an unsupported extension would fail vkCreateDevice.
        nrdDerivativesCapable = physicalDevice.hasDeviceExtension(NRD_DERIVATIVES_EXTENSION)
                && queryDerivativesSupport(physicalDevice);
        if (nrdDerivativesCapable && !augmentedExtensions.contains(NRD_DERIVATIVES_EXTENSION)) {
            augmentedExtensions.add(NRD_DERIVATIVES_EXTENSION);
            CausticaMod.LOGGER.info("Enabling {} for the NRD denoiser", NRD_DERIVATIVES_EXTENSION);
        }
        String serExtension = support.serBackend.extensionName;
        if (serExtension != null && !augmentedExtensions.contains(serExtension)) {
            augmentedExtensions.add(serExtension);
        }
        for (String ext : supportedOptionalExtensions(physicalDevice, support)) {
            if (!augmentedExtensions.contains(ext)) {
                augmentedExtensions.add(ext);
            }
        }
    }

    /** Add the RT VulkanFeatures to arg2 after the matching extension names have been requested. */
    @SuppressWarnings("unchecked")
    public static void addFeatures(Args args, VulkanPhysicalDevice physicalDevice) {
        if (!enabledByProperty()) {
            return;
        }
        rtRequested = false;
        serBackend = SerBackend.NONE;
        worldBackend = WorldBackend.NONE;
        ommEnabled = false;
        reflexEnabled = false;
        presentIdEnabled = false;
        wideLinesEnabled = false;
        maxLineWidth = 1.0f;
        String missingExtension = firstUnsupportedExtension(physicalDevice);
        if (missingExtension != null) {
            if (!loggedUnavailable) {
                loggedUnavailable = true;
                CausticaMod.LOGGER.warn("Ray tracing unavailable: device [{}] lacks extension {}",
                        physicalDevice.deviceName(), missingExtension);
            }
            return;
        }

        FeatureSupport support = queryFeatureSupport(physicalDevice);
        if (!support.supportsRt()) {
            if (!loggedUnavailable) {
                loggedUnavailable = true;
                CausticaMod.LOGGER.warn("Ray tracing unavailable: device [{}] lacks features {}",
                        physicalDevice.deviceName(), support.missingRequired);
            }
            return;
        }

        Set<VulkanFeature> features = new HashSet<>((Set<VulkanFeature>) args.get(2));
        // Core features merge into vanilla's VK10/VK12 structs; extension features create their matching
        // pNext structs. Every boolean here was verified by queryFeatureSupport above.
        features.addAll(REQUIRED_RT_FEATURES);
        // The pipeline feature is only in the required set when its extension is being enabled; a
        // ray-query-only device must not request a feature it does not have.
        WorldBackend supportedBackend = support.rayTracingPipeline
                ? WorldBackend.RAY_TRACING_PIPELINE : WorldBackend.RAY_QUERY;
        if (support.rayTracingPipeline) {
            features.add(RAY_TRACING_PIPELINE_FEATURE);
        }
        // Optional: EXT shader execution reordering. The feature struct is only expressible when this
        // client's LWJGL has its bindings, which is what queryFeatureSupport already folded into the
        // backend choice — SerBackend.EXT therefore implies SER_EXT_FEATURE is present.
        if (support.serBackend == SerBackend.EXT) {
            features.add(SER_EXT_FEATURE);
        } else if (SER_EXT_FEATURE == null && physicalDevice.hasDeviceExtension(SER_EXTENSION_NAME)
                && !loggedSerBindingsMissing) {
            loggedSerBindingsMissing = true;
            CausticaMod.LOGGER.info(
                    "EXT shader execution reordering unavailable: this client's LWJGL has no bindings for {} "
                            + "(the world pass still runs, without SER)", SER_EXTENSION_NAME);
        }

        // Optional: wideLines (core VK10 feature, no extension). Lets the world-overlay pass (block
        // outline) draw a real thick native line via a raster pipeline's lineWidth / VK_DYNAMIC_STATE_LINE
        // _WIDTH instead of a screen-space quad. Its absence must not disable RT — the overlay falls back
        // to whatever the device's mandated lineWidth (1.0) allows.
        wideLinesEnabled = support.wideLines;
        if (wideLinesEnabled) {
            features.add(WIDE_LINES_FEATURE);
            maxLineWidth = physicalDevice.vkPhysicalDeviceProperties().limits().lineWidthRange(1);
        }

        // World-overlay MSAA (block outline edge AA): a framebuffer property, not a feature — no
        // VulkanFeature entry needed, just cap the desired 4x down to what the device actually supports.
        int colorSampleCounts = physicalDevice.vkPhysicalDeviceProperties().limits().framebufferColorSampleCounts();
        if ((colorSampleCounts & VK10.VK_SAMPLE_COUNT_4_BIT) != 0) {
            overlayMsaaSamples = VK10.VK_SAMPLE_COUNT_4_BIT;
        } else if ((colorSampleCounts & VK10.VK_SAMPLE_COUNT_2_BIT) != 0) {
            overlayMsaaSamples = VK10.VK_SAMPLE_COUNT_2_BIT;
        } else {
            overlayMsaaSamples = VK10.VK_SAMPLE_COUNT_1_BIT;
        }

        // Optional: opacity micromaps (any-hit opt). Only when the gate is on AND the device advertises the
        // extension — its absence must not disable RT, so it is kept out of the mandatory feature set above.
        ommEnabled = support.omm;
        if (ommEnabled) {
            features.add(OMM_FEATURE);
        }

        // Optional: NVIDIA Reflex (VK_NV_low_latency2). Function-only extension, no feature struct to add.
        reflexEnabled = reflexRequested() && physicalDevice.hasDeviceExtension(VK_NV_LOW_LATENCY_2_EXTENSION_NAME);

        // Optional: VK_KHR_present_id (presentID<->present correlation for Reflex markers). Its absence must
        // not disable Reflex sleep/pacing itself — only marker correlation degrades.
        presentIdEnabled = reflexEnabled && support.presentId;
        if (presentIdEnabled) {
            features.add(PRESENT_ID_FEATURE);
        }

        args.set(2, features);

        rtRequested = true;
        worldBackend = supportedBackend;
        serBackend = support.serBackend;
        String optionalSuffix = supportedOptionalExtensions(physicalDevice, support).stream()
                .map(ext -> " + " + ext).reduce("", String::concat);
        CausticaMod.LOGGER.info(
                "Ray tracing: world pass on {} + extensions {}{}{}{} + features [bufferDeviceAddress, "
                        + "accelerationStructure, rayQuery{}, SER={}"
                        + (wideLinesEnabled ? ", wideLines(max=" + maxLineWidth + ")" : "")
                        + (ommEnabled ? ", opacityMicromap" : "") + "] + overlayMsaa=" + overlayMsaaSamples + "x on [{}]",
                worldBackend.label, RT_BASE_EXTENSIONS,
                support.rayTracingPipeline ? " + " + RT_PIPELINE_EXTENSION : "",
                serBackend.extensionName == null ? "" : " + " + serBackend.extensionName,
                optionalSuffix,
                support.rayTracingPipeline ? ", rayTracingPipeline" : "",
                serBackend.label, physicalDevice.deviceName());
    }

    /**
     * Post-creation verification: confirm the RT entry points actually loaded on the new device and log
     * the acceleration-structure limits (plus the SBT limits when the world pass runs on a ray-tracing
     * pipeline). If this logs "OK", the device truly came up RT-capable.
     */
    public static void probe(VkDevice device) {
        if (!rtRequested) {
            CausticaMod.LOGGER.info("Ray tracing not requested; skipping RT probe");
            maxOpacity4StateSubdivisionLevel = 0;
            return;
        }
        try {
            VKCapabilitiesDevice caps = device.getCapabilities();
            boolean rtPipeline = caps.vkCreateRayTracingPipelinesKHR != 0L;
            boolean asBuild = caps.vkCmdBuildAccelerationStructuresKHR != 0L;
            boolean traceRays = caps.vkCmdTraceRaysKHR != 0L;
            // The ray-query backend never creates a ray-tracing pipeline or calls vkCmdTraceRaysKHR, and the
            // loader only resolves those entry points for an enabled extension — so they are required
            // exactly when the selected backend uses them.
            boolean pipelinesOk = worldBackend != WorldBackend.RAY_TRACING_PIPELINE
                    || (rtPipeline && traceRays);
            if (!(pipelinesOk && asBuild)) {
                CausticaMod.LOGGER.error(
                        "RT extensions enabled but entry points missing (rtPipeline={}, asBuild={}, traceRays={}) — RT bring-up FAILED",
                        rtPipeline, asBuild, traceRays);
                return;
            }
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps =
                        VkPhysicalDeviceAccelerationStructurePropertiesKHR.calloc(stack).sType$Default();
                // The ray-tracing pipeline properties struct belongs to an extension a ray-query-only device
                // was never asked for, so it stays out of the chain there.
                VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps =
                        VkPhysicalDeviceRayTracingPipelinePropertiesKHR.calloc(stack).sType$Default();
                // Chain the OMM properties only when the feature is enabled (else the driver would ignore an
                // unrecognized struct, but keeping the chain clean matches the enabled feature set).
                VkPhysicalDeviceOpacityMicromapPropertiesEXT ommProps = null;
                if (ommEnabled) {
                    ommProps = VkPhysicalDeviceOpacityMicromapPropertiesEXT.calloc(stack).sType$Default();
                    asProps.pNext(ommProps.address());
                }
                VkPhysicalDeviceProperties2 props2 = VkPhysicalDeviceProperties2.calloc(stack).sType$Default();
                if (worldBackend == WorldBackend.RAY_TRACING_PIPELINE) {
                    rtProps.pNext(asProps.address());
                    props2.pNext(rtProps.address());
                } else {
                    props2.pNext(asProps.address());
                }
                VK12.vkGetPhysicalDeviceProperties2(device.getPhysicalDevice(), props2);

                CausticaMod.LOGGER.info(
                        "RT bring-up OK — world pass on {}; maxAS geometry/instance/primitive = {}/{}/{}",
                        worldBackend.label,
                        asProps.maxGeometryCount(), asProps.maxInstanceCount(), asProps.maxPrimitiveCount());
                if (worldBackend == WorldBackend.RAY_TRACING_PIPELINE) {
                    CausticaMod.LOGGER.info(
                            "SBT limits — shaderGroupHandleSize={}, shaderGroupBaseAlignment={}, maxRayRecursionDepth={}",
                            rtProps.shaderGroupHandleSize(), rtProps.shaderGroupBaseAlignment(),
                            rtProps.maxRayRecursionDepth());
                }
                if (ommProps != null) {
                    maxOpacity4StateSubdivisionLevel = ommProps.maxOpacity4StateSubdivisionLevel();
                    CausticaMod.LOGGER.info(
                            "Opacity micromaps enabled — maxSubdivisionLevel 4-state={}, 2-state={}",
                            ommProps.maxOpacity4StateSubdivisionLevel(), ommProps.maxOpacity2StateSubdivisionLevel());
                } else {
                    maxOpacity4StateSubdivisionLevel = 0;
                }
            }
            if (reflexEnabled) {
                boolean sleepMode = caps.vkSetLatencySleepModeNV != 0L;
                boolean sleep = caps.vkLatencySleepNV != 0L;
                boolean marker = caps.vkSetLatencyMarkerNV != 0L;
                boolean timings = caps.vkGetLatencyTimingsNV != 0L;
                if (sleepMode && sleep && marker && timings) {
                    CausticaMod.LOGGER.info(
                            "Reflex (VK_NV_low_latency2) entry points loaded — presentId={}", presentIdEnabled);
                } else {
                    CausticaMod.LOGGER.error(
                            "Reflex extension enabled but entry points missing (sleepMode={}, sleep={}, marker={}, timings={})",
                            sleepMode, sleep, marker, timings);
                    reflexEnabled = false;
                }
            }
        } catch (Throwable t) {
            // A probe must never break device creation.
            CausticaMod.LOGGER.error("RT probe threw; continuing without RT", t);
        }
    }

    /** Reads {@code computeDerivativeGroupQuads} through a hand-built feature struct. */
    private static boolean queryDerivativesSupport(VulkanPhysicalDevice physicalDevice) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long derivatives = stack.nmalloc(8, DERIVATIVES_FEATURES_BYTES);
            MemoryUtil.memSet(derivatives, 0, DERIVATIVES_FEATURES_BYTES);
            MemoryUtil.memPutInt(derivatives, DERIVATIVES_FEATURES_S_TYPE);
            VkPhysicalDeviceFeatures2 query = VkPhysicalDeviceFeatures2.calloc(stack).sType$Default();
            query.pNext(derivatives);
            VK12.vkGetPhysicalDeviceFeatures2(physicalDevice.vkPhysicalDevice(), query);
            return MemoryUtil.memGetInt(derivatives + DERIVATIVES_QUADS_OFFSET) != 0;
        }
    }

    /**
     * Append the compute-shader-derivatives feature struct to the device create info's pNext chain, which
     * by then already carries vanilla's {@code VkPhysicalDeviceFeatures2}. This is done by hand rather than
     * through vanilla's feature-set abstraction because this LWJGL build has no binding for the struct, and
     * a hand-built chain is the only way to control its bytes exactly.
     *
     * <p>The tail is found by walking {@code pNext}, which every Vulkan struct exposes at offset 8, so the
     * walk needs no knowledge of the struct types already in the chain. The struct lives on the caller's
     * stack, which stays alive until {@code vkCreateDevice} returns.
     */
    public static void chainDerivativesFeatures(VkDeviceCreateInfo createInfo, MemoryStack stack) {
        if (!nrdDerivativesCapable) {
            return;
        }
        long struct = stack.nmalloc(8, DERIVATIVES_FEATURES_BYTES);
        MemoryUtil.memSet(struct, 0, DERIVATIVES_FEATURES_BYTES);
        MemoryUtil.memPutInt(struct, DERIVATIVES_FEATURES_S_TYPE);
        MemoryUtil.memPutInt(struct + DERIVATIVES_QUADS_OFFSET, 1);

        long node = createInfo.pNext();
        if (node == 0L) {
            createInfo.pNext(struct);
            return;
        }
        while (true) {
            long next = MemoryUtil.memGetLong(node + 8);
            if (next == 0L) {
                MemoryUtil.memPutLong(node + 8, struct);
                return;
            }
            node = next;
        }
    }
}

package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.NrdBackendPushData;
import dev.comfyfluffy.caustica.rt.gen.NrdFrontendPushData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * The two host passes around the NRD denoiser: a front end that packs this renderer's guide buffers into
 * NRD's input layout, and a back end that unpacks the denoised lanes back into pre-exposed scene-linear
 * radiance. NRD itself runs between them, dispatched by the caustica_nrd shim.
 *
 * <p>They are separate pipelines rather than one because the shim records its own dispatches between the
 * two, and the back end must not read a lane before the denoiser has written it.
 */
public final class RtNrdPipelines {
    private static final String FRONTEND_SHADER = "/caustica/shaders/pipelines/nrd_frontend/main.comp.spv";
    private static final String BACKEND_SHADER = "/caustica/shaders/pipelines/nrd_backend/main.comp.spv";

    private final RtContext ctx;
    private final long frontendSetLayout;
    private final long frontendPool;
    private final long frontendSet;
    private final long frontendLayout;
    private final long frontendPipeline;
    private final long backendSetLayout;
    private final long backendPool;
    private final long backendSet;
    private final long backendLayout;
    private final long backendPipeline;
    private final long sampler;

    private long boundNormal = -1;
    private long boundDepth = -1;
    private long boundAlbedo = -1;
    private long boundSpecAlbedo = -1;
    private long boundLaneDiff = -1;
    private long boundLaneSpec = -1;
    private long boundNormalRoughness = -1;
    private long boundViewZ = -1;
    private long boundNrdDiffIn = -1;
    private long boundNrdSpecIn = -1;
    private long boundNrdDiffOut = -1;
    private long boundNrdSpecOut = -1;
    private long boundCombined = -1;
    private boolean destroyed;

    private RtNrdPipelines(RtContext ctx, long frontendSetLayout, long frontendPool, long frontendSet,
                           long frontendLayout, long frontendPipeline, long backendSetLayout, long backendPool,
                           long backendSet, long backendLayout, long backendPipeline, long sampler) {
        this.ctx = ctx;
        this.frontendSetLayout = frontendSetLayout;
        this.frontendPool = frontendPool;
        this.frontendSet = frontendSet;
        this.frontendLayout = frontendLayout;
        this.frontendPipeline = frontendPipeline;
        this.backendSetLayout = backendSetLayout;
        this.backendPool = backendPool;
        this.backendSet = backendSet;
        this.backendLayout = backendLayout;
        this.backendPipeline = backendPipeline;
        this.sampler = sampler;
    }

    public static RtNrdPipelines create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, handle), "vkCreateSampler(nrd)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "nrd linear sampler");

            // Front end: six reads (guides, both lanes, both material guides) and four writes (NRD's packed
            // inputs).
            VkDescriptorSetLayoutBinding.Buffer frontendBindings =
                    VkDescriptorSetLayoutBinding.calloc(NRD_FRONTEND_BINDING_COUNT, stack);
            sampled(frontendBindings, NRD_FRONTEND_NORMAL);
            sampled(frontendBindings, NRD_FRONTEND_DEPTH);
            sampled(frontendBindings, NRD_FRONTEND_DIFF_IN);
            sampled(frontendBindings, NRD_FRONTEND_SPEC_IN);
            storage(frontendBindings, NRD_FRONTEND_NORMAL_ROUGHNESS_OUT);
            storage(frontendBindings, NRD_FRONTEND_VIEWZ_OUT);
            storage(frontendBindings, NRD_FRONTEND_DIFF_OUT);
            storage(frontendBindings, NRD_FRONTEND_SPEC_OUT);
            sampled(frontendBindings, NRD_FRONTEND_ALBEDO);
            sampled(frontendBindings, NRD_FRONTEND_SPEC_ALBEDO);
            long frontendSetLayout = createSetLayout(ctx, vk, stack, frontendBindings, "nrd front end");
            // The front end reads six images and writes four; the pool has to match the layout exactly.
            long frontendPool = createPool(ctx, vk, stack, 1, 4, 6, "nrd front end");
            long frontendSet = allocateSet(ctx, vk, stack, frontendPool, frontendSetLayout, "nrd front end");
            long frontendLayout = createPipelineLayout(ctx, vk, stack, frontendSetLayout,
                    NrdFrontendPushData.BYTE_SIZE, "nrd front end");
            long frontendPipeline = createPipeline(ctx, vk, stack, FRONTEND_SHADER, frontendLayout, "nrd front end");

            // Back end: the two denoised lanes in, the un-denoised lanes, the guides NRD skips sky with, the
            // material guides that put the demodulated factor back, and the combined render-res image out.
            // Every binding the shader declares has to be listed here: a descriptor the shader reads but the
            // layout omits is undefined at run time.
            VkDescriptorSetLayoutBinding.Buffer backendBindings =
                    VkDescriptorSetLayoutBinding.calloc(NRD_BACKEND_BINDING_COUNT, stack);
            sampled(backendBindings, NRD_BACKEND_DIFF_IN);
            sampled(backendBindings, NRD_BACKEND_SPEC_IN);
            sampled(backendBindings, NRD_BACKEND_RAW_DIFF);
            sampled(backendBindings, NRD_BACKEND_RAW_SPEC);
            sampled(backendBindings, NRD_BACKEND_NORMAL);
            sampled(backendBindings, NRD_BACKEND_DEPTH);
            storage(backendBindings, NRD_BACKEND_OUTPUT);
            sampled(backendBindings, NRD_BACKEND_ALBEDO);
            sampled(backendBindings, NRD_BACKEND_SPEC_ALBEDO);
            long backendSetLayout = createSetLayout(ctx, vk, stack, backendBindings, "nrd back end");
            long backendPool = createPool(ctx, vk, stack, 1, 1, 8, "nrd back end");
            long backendSet = allocateSet(ctx, vk, stack, backendPool, backendSetLayout, "nrd back end");
            long backendLayout = createPipelineLayout(ctx, vk, stack, backendSetLayout,
                    NrdBackendPushData.BYTE_SIZE, "nrd back end");
            long backendPipeline = createPipeline(ctx, vk, stack, BACKEND_SHADER, backendLayout, "nrd back end");

            return new RtNrdPipelines(ctx, frontendSetLayout, frontendPool, frontendSet, frontendLayout,
                    frontendPipeline, backendSetLayout, backendPool, backendSet, backendLayout,
                    backendPipeline, sampler);
        }
    }

    private static void sampled(VkDescriptorSetLayoutBinding.Buffer bindings, int binding) {
        bindings.get(binding).binding(binding)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
    }

    private static void storage(VkDescriptorSetLayoutBinding.Buffer bindings, int binding) {
        bindings.get(binding).binding(binding)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
    }

    private static long createSetLayout(RtContext ctx, VkDevice vk, MemoryStack stack,
                                        VkDescriptorSetLayoutBinding.Buffer bindings, String label) {
        VkDescriptorSetLayoutCreateInfo info =
                VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(bindings);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateDescriptorSetLayout(vk, info, null, handle),
                "vkCreateDescriptorSetLayout(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, handle.get(0),
                label + " descriptor set layout");
        return handle.get(0);
    }

    /** Descriptor counts are explicit: deriving them from the binding total silently undersizes the pool. */
    private static long createPool(RtContext ctx, VkDevice vk, MemoryStack stack, int maxSets,
                                   int storageImages, int samplers, String label) {
        VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(2, stack);
        sizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(maxSets * storageImages);
        sizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(maxSets * samplers);
        VkDescriptorPoolCreateInfo info = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType$Default().maxSets(maxSets).pPoolSizes(sizes);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateDescriptorPool(vk, info, null, handle), "vkCreateDescriptorPool(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, handle.get(0), label + " descriptor pool");
        return handle.get(0);
    }

    private static long allocateSet(RtContext ctx, VkDevice vk, MemoryStack stack, long pool, long setLayout,
                                    String label) {
        VkDescriptorSetAllocateInfo info = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType$Default().descriptorPool(pool).pSetLayouts(stack.longs(setLayout));
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkAllocateDescriptorSets(vk, info, handle), "vkAllocateDescriptorSets(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, handle.get(0), label + " descriptor set");
        return handle.get(0);
    }

    private static long createPipelineLayout(RtContext ctx, VkDevice vk, MemoryStack stack, long setLayout,
                                             int pushBytes, String label) {
        VkPushConstantRange.Buffer range = VkPushConstantRange.calloc(1, stack);
        range.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pushBytes);
        VkPipelineLayoutCreateInfo info = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType$Default().pSetLayouts(stack.longs(setLayout)).pPushConstantRanges(range);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreatePipelineLayout(vk, info, null, handle), "vkCreatePipelineLayout(" + label + ")");
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, handle.get(0), label + " pipeline layout");
        return handle.get(0);
    }

    private static long createPipeline(RtContext ctx, VkDevice vk, MemoryStack stack, String shader,
                                       long pipelineLayout, String label) {
        long module = loadModule(vk, stack, shader, label);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " shader module");
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                .module(module).pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
        info.get(0).sType$Default().stage(stage).layout(pipelineLayout);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, info, null, handle),
                "vkCreateComputePipelines(" + label + ")");
        VK10.vkDestroyShaderModule(vk, module, null);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, handle.get(0), label + " compute pipeline");
        return handle.get(0);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String shader, String label) {
        byte[] bytes;
        try (InputStream input = RtNrdPipelines.class.getResourceAsStream(shader)) {
            if (input == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + shader);
            }
            bytes = input.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + shader, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo info = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, info, null, module), "vkCreateShaderModule(" + label + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }

    /** Point both passes at this frame's images. Views are cached, so the steady state writes nothing. */
    public void setImages(RtImage guideNormal, RtImage guideDepth, RtImage guideAlbedo, RtImage guideSpecAlbedo,
                          RtImage laneDiff, RtImage laneSpec, RtImage normalRoughness, RtImage viewZ,
                          RtImage diffIn, RtImage specIn, RtImage diffOut, RtImage specOut, RtImage combined) {
        boolean unchanged = boundNormal == guideNormal.view && boundDepth == guideDepth.view
                && boundAlbedo == guideAlbedo.view && boundSpecAlbedo == guideSpecAlbedo.view
                && boundLaneDiff == laneDiff.view && boundLaneSpec == laneSpec.view
                && boundNormalRoughness == normalRoughness.view && boundViewZ == viewZ.view
                && boundNrdDiffIn == diffIn.view && boundNrdSpecIn == specIn.view
                && boundNrdDiffOut == diffOut.view && boundNrdSpecOut == specOut.view
                && boundCombined == combined.view;
        if (unchanged) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(19, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(19, stack);
            int index = 0;
            index = writeSampled(stack, images, writes, index, frontendSet, NRD_FRONTEND_NORMAL, guideNormal.view);
            index = writeSampled(stack, images, writes, index, frontendSet, NRD_FRONTEND_DEPTH, guideDepth.view);
            index = writeSampled(stack, images, writes, index, frontendSet, NRD_FRONTEND_DIFF_IN, laneDiff.view);
            index = writeSampled(stack, images, writes, index, frontendSet, NRD_FRONTEND_SPEC_IN, laneSpec.view);
            index = writeStorage(stack, images, writes, index, frontendSet,
                    NRD_FRONTEND_NORMAL_ROUGHNESS_OUT, normalRoughness.view);
            index = writeStorage(stack, images, writes, index, frontendSet, NRD_FRONTEND_VIEWZ_OUT, viewZ.view);
            index = writeStorage(stack, images, writes, index, frontendSet, NRD_FRONTEND_DIFF_OUT, diffIn.view);
            index = writeStorage(stack, images, writes, index, frontendSet, NRD_FRONTEND_SPEC_OUT, specIn.view);
            // The material guides both ends demodulate with, so the pair cancels texel for texel.
            index = writeSampled(stack, images, writes, index, frontendSet, NRD_FRONTEND_ALBEDO, guideAlbedo.view);
            index = writeSampled(stack, images, writes, index, frontendSet, NRD_FRONTEND_SPEC_ALBEDO,
                    guideSpecAlbedo.view);
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_DIFF_IN, diffOut.view);
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_SPEC_IN, specOut.view);
            // The tracer's own lanes and the guides, so the back end can rebuild the sky pixels NRD skips.
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_RAW_DIFF, laneDiff.view);
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_RAW_SPEC, laneSpec.view);
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_NORMAL, guideNormal.view);
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_DEPTH, guideDepth.view);
            index = writeStorage(stack, images, writes, index, backendSet, NRD_BACKEND_OUTPUT, combined.view);
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_ALBEDO, guideAlbedo.view);
            index = writeSampled(stack, images, writes, index, backendSet, NRD_BACKEND_SPEC_ALBEDO,
                    guideSpecAlbedo.view);
            VK10.vkUpdateDescriptorSets(ctx.vk(), VkWriteDescriptorSet.create(writes.address(), index), null);
        }
        boundNormal = guideNormal.view;
        boundDepth = guideDepth.view;
        boundAlbedo = guideAlbedo.view;
        boundSpecAlbedo = guideSpecAlbedo.view;
        boundLaneDiff = laneDiff.view;
        boundLaneSpec = laneSpec.view;
        boundNormalRoughness = normalRoughness.view;
        boundViewZ = viewZ.view;
        boundNrdDiffIn = diffIn.view;
        boundNrdSpecIn = specIn.view;
        boundNrdDiffOut = diffOut.view;
        boundNrdSpecOut = specOut.view;
        boundCombined = combined.view;
    }

    private int writeSampled(MemoryStack stack, VkDescriptorImageInfo.Buffer images,
                             VkWriteDescriptorSet.Buffer writes, int index, long set, int binding, long view) {
        images.get(index).imageView(view).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        writes.get(index).sType$Default().dstSet(set).dstBinding(binding)
                .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
        return index + 1;
    }

    private int writeStorage(MemoryStack stack, VkDescriptorImageInfo.Buffer images,
                             VkWriteDescriptorSet.Buffer writes, int index, long set, int binding, long view) {
        images.get(index).imageView(view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        writes.get(index).sType$Default().dstSet(set).dstBinding(binding)
                .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
        return index + 1;
    }

    /** Pack this renderer's guides and lanes into NRD's inputs. */
    public void recordFrontend(VkCommandBuffer cmd, int width, int height, float projM22, float projM32,
                               float hitDistA, float hitDistB, float hitDistC, float denoisingRange,
                               boolean normalizeHitDistances) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "nrd front end")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, frontendPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, frontendLayout, 0,
                    stack.longs(frontendSet), null);
            ByteBuffer push = stack.malloc(NrdFrontendPushData.BYTE_SIZE);
            new NrdFrontendPushData(width, height, projM22, projM32, hitDistA, hitDistB, hitDistC,
                    denoisingRange, normalizeHitDistances ? 1 : 0).write(push);
            VK10.vkCmdPushConstants(cmd, frontendLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    /** Recombine the denoised lanes into the render-res image the upscaler consumes. */
    public void recordBackend(VkCommandBuffer cmd, int width, int height) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "nrd back end")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, backendPipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, backendLayout, 0,
                    stack.longs(backendSet), null);
            ByteBuffer push = stack.malloc(NrdBackendPushData.BYTE_SIZE);
            new NrdBackendPushData(width, height, 0.0f, 0.0f).write(push);
            VK10.vkCmdPushConstants(cmd, backendLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, frontendPipeline, null);
        VK10.vkDestroyPipeline(vk, backendPipeline, null);
        VK10.vkDestroySampler(vk, sampler, null);
        VK10.vkDestroyPipelineLayout(vk, frontendLayout, null);
        VK10.vkDestroyPipelineLayout(vk, backendLayout, null);
        VK10.vkDestroyDescriptorPool(vk, frontendPool, null);
        VK10.vkDestroyDescriptorPool(vk, backendPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, frontendSetLayout, null);
        VK10.vkDestroyDescriptorSetLayout(vk, backendSetLayout, null);
        destroyed = true;
    }
}

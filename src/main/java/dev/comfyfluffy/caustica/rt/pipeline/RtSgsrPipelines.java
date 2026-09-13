package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.SgsrParamsData;
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
import java.util.Arrays;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * The three Qualcomm SGSR2 compute passes: {@code convert} and {@code activate} at render resolution, then
 * {@code upscale} at display resolution. Each pass owns one descriptor set, rewritten only when the images
 * behind it change, and all three share one push-constant block.
 *
 * <p>The passes are strictly ordered inside a frame and each reads what the previous one wrote, so the caller
 * records them back to back. The history images they ping-pong are the caller's, because only the caller knows
 * when a frame's history stops being valid.
 */
public final class RtSgsrPipelines {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/sgsr/";
    private static final int THREADS = 8;

    private final RtContext ctx;
    private final long linearSampler;
    private final long nearestSampler;
    private final Pass convert;
    private final Pass activate;
    private final Pass upscale;

    private RtSgsrPipelines(RtContext ctx, long linearSampler, long nearestSampler, Pass convert,
                            Pass activate, Pass upscale) {
        this.ctx = ctx;
        this.linearSampler = linearSampler;
        this.nearestSampler = nearestSampler;
        this.convert = convert;
        this.activate = activate;
        this.upscale = upscale;
    }

    public static RtSgsrPipelines create(RtContext ctx) {
        long linear = createSampler(ctx, "sgsr linear sampler", true);
        // YCoCg and the luma history are integer textures: filtering them is not defined, so every fetch
        // from them goes through the nearest sampler.
        long nearest = createSampler(ctx, "sgsr nearest sampler", false);
        return new RtSgsrPipelines(ctx, linear, nearest,
                new Pass(ctx, linear, nearest, "convert.comp.spv", "sgsr convert",
                        new int[] {SGSR_CONVERT_COLOR, SGSR_CONVERT_DEPTH, SGSR_CONVERT_VELOCITY,
                                SGSR_CONVERT_ALPHA},
                        new boolean[] {true, true, true, true},
                        new int[] {SGSR_CONVERT_YCOCG_OUT, SGSR_CONVERT_MOTION_OUT}),
                new Pass(ctx, linear, nearest, "activate.comp.spv", "sgsr activate",
                        new int[] {SGSR_ACTIVATE_LUMA_IN, SGSR_ACTIVATE_MOTION_IN, SGSR_ACTIVATE_YCOCG_IN},
                        new boolean[] {false, true, false},
                        new int[] {SGSR_ACTIVATE_MOTIONCLIP_OUT, SGSR_ACTIVATE_LUMA_OUT}),
                new Pass(ctx, linear, nearest, "upscale.comp.spv", "sgsr upscale",
                        new int[] {SGSR_UPSCALE_HISTORY_IN, SGSR_UPSCALE_MOTIONCLIP_IN, SGSR_UPSCALE_YCOCG_IN},
                        new boolean[] {true, true, false},
                        new int[] {SGSR_UPSCALE_HISTORY_OUT, SGSR_UPSCALE_SCENE_OUT}));
    }

    /**
     * Records the three passes for one frame. {@code lumaRead}/{@code historyRead} are this frame's history
     * inputs; the caller swaps both pairs after this returns, because the passes write the other half.
     */
    public void record(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage velocity, RtImage alpha,
                       RtImage yCoCg, RtImage motionDepthAlpha, RtImage motionDepthClip,
                       RtImage lumaRead, RtImage lumaWrite, RtImage historyRead, RtImage historyWrite,
                       RtImage out, SgsrParamsData params) {
        convert.setImages(new RtImage[] {color, depth, velocity, alpha},
                new RtImage[] {yCoCg, motionDepthAlpha});
        activate.setImages(new RtImage[] {lumaRead, motionDepthAlpha, yCoCg},
                new RtImage[] {motionDepthClip, lumaWrite});
        upscale.setImages(new RtImage[] {historyRead, motionDepthClip, yCoCg},
                new RtImage[] {historyWrite, out});
        record(cmd, convert, params, params.renderWidth(), params.renderHeight());
        record(cmd, activate, params, params.renderWidth(), params.renderHeight());
        record(cmd, upscale, params, params.displayWidth(), params.displayHeight());
    }

    private void record(VkCommandBuffer cmd, Pass pass, SgsrParamsData params, int width, int height) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, pass.label)) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pass.layout, 0,
                    stack.longs(pass.set), null);
            ByteBuffer push = stack.malloc(SgsrParamsData.BYTE_SIZE);
            params.write(push);
            VK10.vkCmdPushConstants(cmd, pass.layout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + THREADS - 1) / THREADS, (height + THREADS - 1) / THREADS, 1);
            VulkanCommandEncoder.memoryBarrier(cmd, stack);
        }
    }

    public void destroy() {
        VkDevice vk = ctx.vk();
        for (Pass pass : new Pass[] {convert, activate, upscale}) {
            VK10.vkDestroyPipeline(vk, pass.pipeline, null);
            VK10.vkDestroyPipelineLayout(vk, pass.layout, null);
            VK10.vkDestroyDescriptorPool(vk, pass.pool, null);
            VK10.vkDestroyDescriptorSetLayout(vk, pass.setLayout, null);
        }
        VK10.vkDestroySampler(vk, linearSampler, null);
        VK10.vkDestroySampler(vk, nearestSampler, null);
    }

    private static long createSampler(RtContext ctx, String label, boolean linear) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            int filter = linear ? VK10.VK_FILTER_LINEAR : VK10.VK_FILTER_NEAREST;
            VkSamplerCreateInfo info = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(filter).minFilter(filter)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f).maxLod(0.0f);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateSampler(vk, info, null, handle), "vkCreateSampler(" + label + ")");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, handle.get(0), label);
            return handle.get(0);
        }
    }

    /** One pass: its device objects, its binding numbers, and which sampler each sampled binding uses. */
    private static final class Pass {
        private final RtContext ctx;
        private final long linearSampler;
        private final long nearestSampler;
        private final String label;
        private final int[] sampledBindings;
        private final boolean[] sampledLinear;
        private final int[] storageBindings;
        private final long setLayout;
        private final long pool;
        private final long set;
        private final long layout;
        private final long pipeline;
        private long[] boundViews = new long[0];

        private Pass(RtContext ctx, long linearSampler, long nearestSampler, String shader, String label,
                     int[] sampledBindings, boolean[] sampledLinear, int[] storageBindings) {
            this.ctx = ctx;
            this.linearSampler = linearSampler;
            this.nearestSampler = nearestSampler;
            this.label = label;
            this.sampledBindings = sampledBindings;
            this.sampledLinear = sampledLinear;
            this.storageBindings = storageBindings;
            VkDevice vk = ctx.vk();
            try (MemoryStack stack = MemoryStack.stackPush()) {
                int bindingCount = sampledBindings.length + storageBindings.length;
                VkDescriptorSetLayoutBinding.Buffer bindings =
                        VkDescriptorSetLayoutBinding.calloc(bindingCount, stack);
                for (int binding : sampledBindings) {
                    bindings.get(binding).binding(binding)
                            .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                            .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
                }
                for (int binding : storageBindings) {
                    bindings.get(binding).binding(binding)
                            .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                            .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
                }
                VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                        .sType$Default().pBindings(bindings);
                LongBuffer p = stack.mallocLong(1);
                check(VK10.vkCreateDescriptorSetLayout(vk, dslci, null, p),
                        "vkCreateDescriptorSetLayout(" + label + ")");
                this.setLayout = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, setLayout,
                        label + " descriptor set layout");

                // Pool counts match the binding lists exactly: an undersized pool fails descriptor
                // allocation, which would leave the pass reading stale descriptor memory.
                VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
                poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(storageBindings.length);
                poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(sampledBindings.length);
                VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                        .maxSets(1).pPoolSizes(poolSizes);
                check(VK10.vkCreateDescriptorPool(vk, dpci, null, p), "vkCreateDescriptorPool(" + label + ")");
                this.pool = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, pool, label + " descriptor pool");

                VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                        .descriptorPool(pool).pSetLayouts(stack.longs(setLayout));
                LongBuffer pSet = stack.mallocLong(1);
                check(VK10.vkAllocateDescriptorSets(vk, dsai, pSet), "vkAllocateDescriptorSets(" + label + ")");
                this.set = pSet.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, set, label + " descriptor set");

                VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
                pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0)
                        .size(SgsrParamsData.BYTE_SIZE);
                VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                        .pSetLayouts(stack.longs(setLayout)).pPushConstantRanges(pushRange);
                check(VK10.vkCreatePipelineLayout(vk, plci, null, p),
                        "vkCreatePipelineLayout(" + label + ")");
                this.layout = p.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout,
                        label + " pipeline layout");

                long module = loadModule(vk, stack, shader, label);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " shader module");
                VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                        .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module)
                        .pName(stack.UTF8("main"));
                VkComputePipelineCreateInfo.Buffer cpci = VkComputePipelineCreateInfo.calloc(1, stack);
                cpci.get(0).sType$Default().stage(stage).layout(layout);
                LongBuffer pPipeline = stack.mallocLong(1);
                check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, cpci, null, pPipeline),
                        "vkCreateComputePipelines(" + label + ")");
                this.pipeline = pPipeline.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline, label + " compute pipeline");
                VK10.vkDestroyShaderModule(vk, module, null);
            }
        }

        /** Sampled images in binding order, then storage images in binding order. */
        private void setImages(RtImage[] sampled, RtImage[] storage) {
            long[] views = new long[sampled.length + storage.length];
            for (int i = 0; i < sampled.length; i++) {
                views[i] = sampled[i].view;
            }
            for (int i = 0; i < storage.length; i++) {
                views[sampled.length + i] = storage[i].view;
            }
            if (Arrays.equals(views, boundViews)) {
                return;
            }
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkDescriptorImageInfo.Buffer infos = VkDescriptorImageInfo.calloc(views.length, stack);
                VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(views.length, stack);
                int index = 0;
                for (int i = 0; i < sampled.length; i++) {
                    infos.get(index).imageView(sampled[i].view)
                            .sampler(sampledLinear[i] ? linearSampler : nearestSampler)
                            .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                    writes.get(index).sType$Default().dstSet(set).dstBinding(sampledBindings[i])
                            .descriptorCount(1)
                            .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                            .pImageInfo(VkDescriptorImageInfo.create(infos.address(index), 1));
                    index++;
                }
                for (int i = 0; i < storage.length; i++) {
                    infos.get(index).imageView(storage[i].view).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                    writes.get(index).sType$Default().dstSet(set).dstBinding(storageBindings[i])
                            .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                            .pImageInfo(VkDescriptorImageInfo.create(infos.address(index), 1));
                    index++;
                }
                VK10.vkUpdateDescriptorSets(ctx.vk(), VkWriteDescriptorSet.create(writes.address(), index), null);
            }
            boundViews = views;
        }
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name, String label) {
        byte[] bytes;
        try (InputStream in = RtSgsrPipelines.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo smci = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer pModule = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, smci, null, pModule), "vkCreateShaderModule(" + label + ")");
            return pModule.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}

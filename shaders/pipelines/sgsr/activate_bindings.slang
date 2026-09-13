//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
//
// activate.comp.slang's descriptors. The parameter block is a push constant (sgsr_common.slang), so no binding occupies
// slot 0 the way upstream's Params uniform buffer did, and the textures start at binding 0. Every pass
// declares its own bindings because Slang rejects two parameters that share a (set, binding) pair inside one
// module (E39001) and the passes reuse the same slots for different resources; the slot numbers are in the
// same relative order as upstream's.
//
// The integer textures are read through nearest samplers: R32_UINT cannot be filtered, and upstream's
// textureGather calls read whole texels, never filtered results.
[[vk::binding(0, 0)]] public Sampler2D<uint> prevLumaHistory;                                   // render res, packed luma history
[[vk::binding(1, 0)]] public Sampler2D motionDepthAlpha;                                        // render res, convert's output
[[vk::binding(2, 0)]] public Sampler2D<uint> yCoCgColor;                                        // render res, convert's output
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> outMotionDepthClipAlpha;   // render res, clipped motion + depthclip + alpha
[[vk::binding(4, 0)]] [format("r32ui")] public RWTexture2D<uint> outLumaHistory;                // render res, packed luma history for the next frame

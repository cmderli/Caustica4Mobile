//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
//
// convert.comp.slang's descriptors. The parameter block is a push constant (sgsr_common.slang), so no binding occupies
// slot 0 the way upstream's Params uniform buffer did, and the textures start at binding 0. Every pass
// declares its own bindings because Slang rejects two parameters that share a (set, binding) pair inside one
// module (E39001) and the passes reuse the same slots for different resources; the slot numbers are in the
// same relative order as upstream's.
//
// The integer texture is read through a nearest sampler: R32_UINT cannot be filtered, and upstream's
// textureGather reads whole texels, never filtered results.
[[vk::binding(0, 0)]] public Sampler2D inputColor;                                              // render res, scene-linear radiance (pre-exposed)
[[vk::binding(1, 0)]] public Sampler2D inputDepth;                                              // render res, reversed-Z depth (sky = 0)
[[vk::binding(2, 0)]] public Sampler2D inputVelocity;                                           // render res, pixel motion, previous - current
[[vk::binding(3, 0)]] public Sampler2D inputAlpha;                                              // render res, gAlbedo; .a is the emissive flag
[[vk::binding(4, 0)]] [format("r32ui")] public RWTexture2D<uint> outYCoCg;                      // render res, packed YCoCg
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture2D<float4> outMotionDepthAlpha;       // render res, motion + depth + reactive mask

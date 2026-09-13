//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
//
// SGSR2 convert pass, ported from snapdragon-gsr/sgsr/v2/include/glsl_3_pass_cs/sgsr2_convert.comp
// (see references/SRCompare/upscalers/sgsr/sgsr2_convert.comp).
//
// It picks the nearest surface depth over the pixel's neighbourhood and packs the frame into the two buffers
// the later passes consume: the 11/11/10 YCoCg colour and the (motion, depth, reactive mask) tuple.
import recon_common;
import convert_bindings;
import sgsr_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int2 pix = int2(id.xy);
    if (pix.x >= pc.renderWidth || pix.y >= pc.renderHeight) {
        return;
    }

    // The pack offset is a tone-map constant, not the exposure the tracer applied: it has to be comparable to
    // the stored colour for the 11-bit luma channel to leave a usable headroom to recover the magnitude from.
    float h0 = pc.toneMapOffset;
    float2 viewportSizeInverse = sgsrRenderSizeRcp();

    float2 gatherCoord = float2(pix) * viewportSizeInverse;
    float2 viewportUv = gatherCoord + float2(0.5) * viewportSizeInverse;

    // FindNearestDepth, as in ffx_fsr2_reconstruct_dilated_velocity_and_previous_depth: the nearest surface
    // in the 3x3 neighbourhood, so a pixel on a silhouette takes the depth of the surface it belongs to
    // rather than a background sample. The bottom-right texel is fetched directly and the other eight come
    // out of three gathers. Standard depth keeps this a plain min(): the closest surface has the smallest
    // value, and a pixel with no surface (sky) reports the far plane.
    int2 inputPosBtmRight = int2(1) + pix;
    int2 depthLoadMax = int2(pc.renderWidth - 1, pc.renderHeight - 1);
    float nearestZ =
            sgsrStandardDepth(inputDepth.Load(int3(clamp(inputPosBtmRight, int2(0), depthLoadMax), 0)).x);
    float4 topleft = inputDepth.GatherRed(gatherCoord);

    nearestZ = min(sgsrStandardDepth(topleft.x), nearestZ);
    nearestZ = min(sgsrStandardDepth(topleft.y), nearestZ);
    nearestZ = min(sgsrStandardDepth(topleft.z), nearestZ);
    nearestZ = min(sgsrStandardDepth(topleft.w), nearestZ);

    float2 v11 = float2(viewportSizeInverse.x, 0.0);
    float2 topRight = inputDepth.GatherRed(gatherCoord + v11).yz;

    nearestZ = min(sgsrStandardDepth(topRight.x), nearestZ);
    nearestZ = min(sgsrStandardDepth(topRight.y), nearestZ);

    float2 v13 = float2(0.0, viewportSizeInverse.y);
    float2 bottomLeft = inputDepth.GatherRed(gatherCoord + v13).xy;

    nearestZ = min(sgsrStandardDepth(bottomLeft.x), nearestZ);
    nearestZ = min(sgsrStandardDepth(bottomLeft.y), nearestZ);

    // Motion for the reprojection in activate/upscale. Upstream either decodes an encoded-velocity texture or
    // reprojects the nearest depth through a clipToPrevClip matrix; neither applies here, because this
    // renderer always writes a camera motion guide and has no encoded-velocity texture. The guide's
    // pixel-space previous - current motion is converted to the NDC current - previous the later passes
    // expect, with the nearest depth above already providing the dilation upstream would get from its own
    // depth-aware motion fetch.
    float2 motion = sgsrNdcMotion(inputVelocity.Load(int3(pix, 0)).xy);

    ////////////compute luma
    float3 colorRgb = inputColor.Load(int3(pix, 0)).xyz;

    ///simple tonemap
    colorRgb /= float3(max(max(colorRgb.x, colorRgb.y), colorRgb.z) + h0);

    float3 colorYCoCg;
    colorYCoCg.x = 0.25 * (colorRgb.x + 2.0 * colorRgb.y + colorRgb.z);
    colorYCoCg.y = clamp(0.5 * colorRgb.x + 0.5 - 0.5 * colorRgb.z, 0.0, 1.0);
    colorYCoCg.z = clamp(colorYCoCg.x + colorYCoCg.y - colorRgb.x, 0.0, 1.0);

    //now color YCoCg all in the range of [0,1]
    uint x11 = uint(colorYCoCg.x * 2047.5);
    uint y11 = uint(colorYCoCg.y * 2047.5);
    uint z10 = uint(colorYCoCg.z * 1023.5);

    // Reactive mask. Upstream takes the largest channel difference between the opaque-only colour buffer and
    // the full colour buffer, which flags content the opaque pass could not see — transparent surfaces,
    // particles, anything that moves independently of the surface behind it — because that is the content
    // whose history reprojects onto the wrong surface. This renderer has no opaque-only buffer, so the mask
    // is the emissive flag the gAlbedo guide already carries in .a: it covers glowing geometry, which is the
    // strongest ghosting source, but not translucent content that emits nothing. Upstream's 0.35 * 1000
    // scaling only exists to lift a colour difference into [0,1]; a 0/1 flag needs no scaling.
    float alphaMask = (inputAlpha.Load(int3(pix, 0)).a > 0.5) ? 1.0 : 0.0;

    outYCoCg[pix] = ((x11 << 21u) | (y11 << 10u)) | z10;

    outMotionDepthAlpha[pix] = float4(motion, nearestZ, alphaMask);
}

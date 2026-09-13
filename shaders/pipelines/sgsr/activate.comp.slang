//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
//
// SGSR2 activate pass, ported from snapdragon-gsr/sgsr/v2/include/glsl_3_pass_cs/sgsr2_activate.comp
// (see references/SRCompare/upscalers/sgsr/sgsr2_activate.comp).
//
// It compares the reprojected history depth with the current one to decide how much the history may
// contribute, carries a clamped luma difference forward, and folds that into the reactive mask the upscale
// pass reads.
import recon_common;
import activate_bindings;
import sgsr_common;

static const float EPSILON = 1.19e-07f;

// One iteration of upstream's depth-comparison loop, with the gather offsets passed in so they stay
// compile-time constants: the upstream form used textureGatherOffset with non-constant offsets, which needs
// the shaderImageGatherExtended device feature. Each call weights the four texels of two neighbouring
// 2x2 blocks by how close their nearest depth is to the pixel's own.
void accumulateDepthPair(float2 prevUv, int2 offset0, int2 offset1, float weight0, float weight1, float depth,
                         float ksepKfovDiagonal, inout float wDepth) {
    float4 prevDepth = motionDepthAlpha.GatherBlue(prevUv, offset0);
    float tdepth1 = min(prevDepth.x, prevDepth.y);
    float tdepth2 = min(prevDepth.z, prevDepth.w);
    float fPrevdepth = min(tdepth1, tdepth2);
    float depthsep = ksepKfovDiagonal * (1.0 - min(fPrevdepth, depth));
    wDepth += clamp(depthsep / (abs(fPrevdepth - depth) + EPSILON), 0.0, 1.0) * (weight0);
    float2 prevDepth2 = motionDepthAlpha.GatherBlue(prevUv, offset1).zw;
    fPrevdepth = min(min(prevDepth2.x, prevDepth2.y), tdepth2);
    depthsep = ksepKfovDiagonal * (1.0 - min(fPrevdepth, depth));
    wDepth += clamp(depthsep / (abs(fPrevdepth - depth) + EPSILON), 0.0, 1.0) * (weight1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int2 pix = int2(id.xy);
    if (pix.x >= pc.renderWidth || pix.y >= pc.renderHeight) {
        return;
    }

    float2 viewportUv = (float2(id.xy) + float2(0.5)) * sgsrRenderSizeRcp();
    float2 gatherCoord = viewportUv + 0.5f * sgsrRenderSizeRcp();
    uint lumaReference32 = yCoCgColor.GatherRed(gatherCoord).w;
    float lumaReference = DecodeColorY(lumaReference32);

    float4 mda = motionDepthAlpha.Load(int3(pix, 0)); //motion depth alpha
    float depth = mda.z;
    float alphamask = mda.w;
    float2 motion = mda.xy;

    float2 prevUv = float2(-0.5f * motion.x + viewportUv.x, -0.5f * motion.y + viewportUv.y);
    float depthclip = 0.0;

    // A pixel that reports the far plane has no history depth to compare against, so its history is used
    // as-is (depthclip 0).
    if (depth < 1.0 - 1.0e-05f) {
        float2 prevfSample = prevUv * sgsrRenderSize() - 0.5f;
        float2 prevfrac = prevfSample - floor(prevfSample);
        float oneMinusPrevfacx = 1.0 - prevfrac.x;

        float bilinWeights[4] = {
            oneMinusPrevfacx - oneMinusPrevfacx * prevfrac.y,
            prevfrac.x - prevfrac.x * prevfrac.y,
            oneMinusPrevfacx * prevfrac.y,
            prevfrac.x * prevfrac.y
        };

        float diagonalLength = length(sgsrRenderSize());
        float wDepth = 0.0;
        float ksep = 1.37e-05f;
        float kfov = pc.cameraFovHor;
        float ksepKfovDiagonal = ksep * kfov * diagonalLength;

        accumulateDepthPair(prevUv, int2(-1, -1), int2(-1, 0), bilinWeights[0], bilinWeights[1], depth,
                ksepKfovDiagonal, wDepth);
        accumulateDepthPair(prevUv, int2(0, -1), int2(0, 0), bilinWeights[2], bilinWeights[3], depth,
                ksepKfovDiagonal, wDepth);
        depthclip = clamp(1.0f - wDepth, 0.0, 1.0);
    }

    // The luma difference is clamped to the sign of the previous frame's difference, so a surface that keeps
    // changing in one direction cannot accumulate an unbounded difference across frames.
    float2 currentLumaDiff = float2(0.0, 0.0);
    uint prevLumaDiffPack = prevLumaHistory.GatherRed(prevUv).w;
    float2 prevLumaDiff;
    prevLumaDiff.x = f16tof32(prevLumaDiffPack >> 16u);
    prevLumaDiff.y = f16tof32((prevLumaDiffPack & uint(0xFFFF)));
    bool enable = false;
    if (depthclip + float(pc.reset) < 0.1) {
        enable = (all(prevUv >= float2(0.0)) && all(prevUv <= float2(1.0)));
    }
    float lumaDiff = lumaReference - prevLumaDiff.x;

    if (!enable) {
        currentLumaDiff.x = 0.0;
        currentLumaDiff.y = 0.0;
    } else {
        currentLumaDiff.x = lumaReference;
        currentLumaDiff.y = (prevLumaDiff.y != 0.0f)
                ? ((sign(lumaDiff) == sign(prevLumaDiff.y))
                        ? (sign(lumaDiff) * min(abs(prevLumaDiff.y), abs(lumaDiff)))
                        : prevLumaDiff.y)
                : lumaDiff;
    }

    alphamask = floor(alphamask)
            + 0.5f * float((currentLumaDiff.x != 0.0f) && (abs(currentLumaDiff.y) != abs(lumaDiff)));
    uint pack = (packHalf2x16(float2(currentLumaDiff.x, 0.0)) << 16u)
            | packHalf2x16(float2(currentLumaDiff.y, 0.0));

    outLumaHistory[pix] = pack;

    outMotionDepthClipAlpha[pix] = float4(motion, depthclip, alphamask);
}

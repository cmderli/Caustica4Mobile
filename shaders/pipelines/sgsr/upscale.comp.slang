//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
//
// SGSR2 upscale pass, ported from snapdragon-gsr/sgsr/v2/include/glsl_3_pass_cs/sgsr2_upscale.comp
// (see references/SRCompare/upscalers/sgsr/sgsr2_upscale.comp).
//
// It reconstructs a display-resolution colour from the render-resolution YCoCg samples with a Lanczos-like
// kernel, clamps it against a local colour box, blends it with the reprojected history by how trustworthy
// that history is, and writes both the history for the next frame and the scene colour to present.
import recon_common;
import upscale_bindings;
import sgsr_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int2 pix = int2(id.xy);
    if (pix.x >= pc.displayWidth || pix.y >= pc.displayHeight) {
        return;
    }

    float2 inputInfoViewportSize = sgsrRenderSize();
    float2 historyInfoViewportSize = sgsrDisplaySize();
    float2 historyInfoViewportSizeInverse = sgsrDisplaySizeRcp();
    float2 inputJitter = float2(pc.jitterX, pc.jitterY);

    float biasmaxViewportXScale =
            min(historyInfoViewportSize.x / inputInfoViewportSize.x, 1.99); //Biasmax_viewportXScale
    float scalefactor =
            min(20.0, pow((historyInfoViewportSize.x / inputInfoViewportSize.x)
                                  * (historyInfoViewportSize.y / inputInfoViewportSize.y),
                    3.0));

    float2 hruv = (float2(id.xy) + float2(0.5)) * historyInfoViewportSizeInverse;
    float2 jitteruv;
    jitteruv.x = clamp(hruv.x + (inputJitter.x * sgsrRenderSizeRcp().x), 0.0, 1.0);
    jitteruv.y = clamp(hruv.y + (inputJitter.y * sgsrRenderSizeRcp().y), 0.0, 1.0);

    int2 inputPos = int2(jitteruv * inputInfoViewportSize);

    // jitteruv is clamped to [0,1], so inputPos can land one texel past the render target; clamping the
    // fetch keeps the read defined instead of relying on out-of-range texel behaviour.
    float alphab = motionDepthClipAlpha.Load(int3(clamp(inputPos, int2(0), int2(pc.renderWidth - 1,
            pc.renderHeight - 1)), 0)).w;
    float3 mda = motionDepthClipAlpha.SampleLevel(jitteruv, 0.0).xyz;
    float2 motion = mda.xy;

    ///ScreenPosToViewportScale&Bias
    float2 prevUv;
    prevUv.x = clamp(-0.5 * motion.x + hruv.x, 0.0, 1.0);
    prevUv.y = clamp(-0.5 * motion.y + hruv.y, 0.0, 1.0);

    float depthfactor = mda.z;
    float historyValue = fract(alphab); // clamp(alpha, 0.0f, 1.0f);
    float alphamask = (alphab - historyValue) * 0.001f;
    historyValue *= 2.0;

    float4 history = prevHistory.SampleLevel(prevUv, 0.0);
    // The history texture is this chain's only cross-frame state, so one non-finite sample would survive in
    // every later frame and blacken the image for good: the reset path below zeroes its weight but the blend
    // at the end still evaluates `history * 0`, which is not zero for a NaN. A frame that has never been
    // written (first frame after creation or resize) is undefined memory, and a tracer firefly can put a
    // non-finite value into a written one, so an unusable sample is dropped instead of being propagated.
    history = all(isfinite(history)) ? history : float4(0.0);
    float3 historyColor = history.xyz;
    // On reset the previous-history texture has never been written (its contents are undefined, potentially
    // NaN). Force the feedback weight to 0 so a garbage value cannot latch wfactor to 1 forever.
    float historyW = (pc.reset != 0) ? 0.0 : history.w;
    float wfactor = max(clamp(abs(historyW), 0.0, 1.0), alphamask);

    /////upsample and compute box
    float4 upsampledCw = float4(0.0);
    float kernelfactor = clamp(wfactor + float(pc.reset), 0.0, 1.0);
    float biasmax = biasmaxViewportXScale - biasmaxViewportXScale * kernelfactor;
    float biasmin = max(1.0f, 0.3 + 0.3 * biasmax);
    float biasfactor = max(0.25f * depthfactor, kernelfactor);
    float kernelbias = lerp(biasmax, biasmin, biasfactor);
    float motionViewportLength = length(motion * historyInfoViewportSize);
    float curvebias = lerp(-2.0, -3.0, clamp(motionViewportLength * 0.02, 0.0, 1.0));

    float3 rectboxcenter = float3(0.0);
    float3 rectboxvar = float3(0.0);
    float rectboxweight = 0.0;
    float2 srcpos = float2(inputPos) + float2(0.5) - inputJitter;
    float2 srcOutputPos = hruv * inputInfoViewportSize;

    kernelbias *= 0.5f;
    float kernelbias2 = kernelbias * kernelbias;
    float2 srcposSrcOutputPos = srcpos - srcOutputPos;

    int2 inputPosBtmRight = int2(1) + inputPos;
    float2 gatherCoord = float2(inputPos) * sgsrRenderSizeRcp();
    uint btmRight = yCoCgColor.Load(int3(clamp(inputPosBtmRight, int2(0), int2(pc.renderWidth - 1,
            pc.renderHeight - 1)), 0)).x;
    uint4 topleft = yCoCgColor.GatherRed(gatherCoord);
    uint2 topRight = yCoCgColor.GatherRed(gatherCoord + float2(sgsrRenderSizeRcp().x, 0.0)).yz;
    uint2 bottomLeft = yCoCgColor.GatherRed(gatherCoord + float2(0.0, sgsrRenderSizeRcp().y)).xy;

    float3 rectboxmin;
    float3 rectboxmax;
    {
        rectboxmin = DecodeColor(btmRight);
        float2 baseoffset = srcposSrcOutputPos + float2(1.0, 1.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0, 1.0);
        float weight = FastLanczos(base);
        upsampledCw += float4(rectboxmin * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmax = rectboxmin;
        float3 wsample = rectboxmin * boxweight;
        rectboxcenter = wsample;
        rectboxvar = rectboxmin * wsample;
        rectboxweight = boxweight;
    }
    {
        float3 samplecolor = DecodeColor(bottomLeft.x);
        float2 baseoffset = srcposSrcOutputPos + float2(-1.0, 1.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }
    {
        float3 samplecolor = DecodeColor(bottomLeft.y);
        float2 baseoffset = srcposSrcOutputPos + float2(0.0, 1.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }
    {
        float3 samplecolor = DecodeColor(topRight.x);
        float2 baseoffset = srcposSrcOutputPos + float2(1.0, 0.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }
    {
        float3 samplecolor = DecodeColor(topRight.y);
        float2 baseoffset = srcposSrcOutputPos + float2(1.0, -1.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }
    {
        float3 samplecolor = DecodeColor(topleft.x);
        float2 baseoffset = srcposSrcOutputPos + float2(-1.0, 0.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }
    {
        float3 samplecolor = DecodeColor(topleft.y);
        float2 baseoffset = srcposSrcOutputPos;
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }
    {
        float3 samplecolor = DecodeColor(topleft.z);
        float2 baseoffset = srcposSrcOutputPos + float2(0.0, -1.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }
    {
        float3 samplecolor = DecodeColor(topleft.w);
        float2 baseoffset = srcposSrcOutputPos + float2(-1.0, -1.0);
        float baseoffsetDot = dot(baseoffset, baseoffset);
        float base = clamp(baseoffsetDot * kernelbias2, 0.0f, 1.0f);
        float weight = FastLanczos(base);
        upsampledCw += float4(samplecolor * weight, weight);
        float boxweight = exp(baseoffsetDot * curvebias);
        rectboxmin = min(rectboxmin, samplecolor);
        rectboxmax = max(rectboxmax, samplecolor);
        float3 wsample = samplecolor * boxweight;
        rectboxcenter += wsample;
        rectboxvar += (samplecolor * wsample);
        rectboxweight += boxweight;
    }

    rectboxweight = 1.0 / rectboxweight;
    rectboxcenter *= rectboxweight;
    rectboxvar *= rectboxweight;
    rectboxvar = sqrt(abs(rectboxvar - rectboxcenter * rectboxcenter));

    upsampledCw.xyz = clamp(upsampledCw.xyz / upsampledCw.w, rectboxmin - float3(0.05),
            rectboxmax + float3(0.05));
    upsampledCw.w = upsampledCw.w * (1.0f / 3.0f);

    float tcontribute = historyValue * clamp(rectboxvar.x * 10.0f, 0.0, 1.0);
    float oneMinusWfactor = 1.0f - wfactor;
    tcontribute = tcontribute * oneMinusWfactor;

    float baseupdate = oneMinusWfactor - oneMinusWfactor * depthfactor;
    baseupdate = min(baseupdate, lerp(baseupdate, upsampledCw.w * 10.0f, clamp(10.0f * motionViewportLength, 0.0, 1.0)));
    baseupdate = min(baseupdate, lerp(baseupdate, upsampledCw.w, clamp(motionViewportLength * 0.05f, 0.0, 1.0)));
    float basealpha = baseupdate;

    const float EPSILON = 1.192e-07f;
    float boxscale = max(depthfactor, clamp(motionViewportLength * 0.05f, 0.0, 1.0));
    float boxsize = lerp(scalefactor, 1.0f, boxscale);
    float3 sboxvar = rectboxvar * boxsize;
    float3 boxmin = rectboxcenter - sboxvar;
    float3 boxmax = rectboxcenter + sboxvar;
    rectboxmax = min(rectboxmax, boxmax);
    rectboxmin = max(rectboxmin, boxmin);

    float3 clampedcolor = clamp(historyColor, rectboxmin, rectboxmax);
    float lerpcontribution = (any(rectboxmin > historyColor) || any(historyColor > rectboxmax))
            ? tcontribute
            : 1.0f;
    lerpcontribution = lerpcontribution - lerpcontribution * sqrt(alphamask);
    historyColor = lerp(clampedcolor, historyColor, clamp(lerpcontribution, 0.0, 1.0));
    float basemin = min(basealpha, 0.1f);
    basealpha = lerp(basemin, basealpha, clamp(lerpcontribution, 0.0, 1.0));

    ////blend color
    float alphasum = max(EPSILON, basealpha + upsampledCw.w);
    float alpha = clamp(upsampledCw.w / alphasum + float(pc.reset), 0.0, 1.0);
    upsampledCw.xyz = lerp(historyColor, upsampledCw.xyz, alpha);

    outHistory[pix] = float4(upsampledCw.xyz, wfactor);

    ////ycocg to grb
    float x_z = upsampledCw.x - upsampledCw.z;
    upsampledCw.xyz = float3(
        x_z + upsampledCw.y,
        upsampledCw.x + upsampledCw.z,
        x_z - upsampledCw.y);

    float compMax = max(upsampledCw.x, upsampledCw.y);
    compMax = max(compMax, upsampledCw.z);
    // (1.0f + 1.0f / 65504.0f) = 1.000015e+00: keeps the brightest sample inside fp16 after the exposure
    // scaling below.
    // The offset convert packed with, which is what turns the decoded headroom back into the colour's scale.
    float scale = pc.toneMapOffset / ((1.0f + 1.0f / 65504.0f) - compMax);

    upsampledCw.xyz = upsampledCw.xyz * scale;
    outSceneColor[pix] = upsampledCw;
}

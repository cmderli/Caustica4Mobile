//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
//
// Helpers shared by the three SGSR2 passes, ported from
// snapdragon-gsr/sgsr/v2/include/glsl_3_pass_cs/ (see references/SRCompare/upscalers/sgsr/).
//
// SGSR2 works on standard [0,1] depth (near = 0, far and sky = 1, so a min() over a neighbourhood picks the
// closest surface) and on NDC motion. This renderer supplies reversed-Z depth and pixel-space motion, so both
// conversions live here and every pass reads the converted values rather than re-deriving them.
import recon_common;

// Host-filled per-frame parameters. One block serves all three passes, exactly as upstream's single
// `Params` uniform does.
public [[vk::push_constant]] SgsrParams pc;

// Render and display sizes as floats, and their reciprocals: upstream's Params carries the reciprocals
// precomputed, while SgsrParams keeps only the integer sizes.
public float2 sgsrRenderSize() {
    return float2(float(pc.renderWidth), float(pc.renderHeight));
}

public float2 sgsrDisplaySize() {
    return float2(float(pc.displayWidth), float(pc.displayHeight));
}

public float2 sgsrRenderSizeRcp() {
    return 1.0 / sgsrRenderSize();
}

public float2 sgsrDisplaySizeRcp() {
    return 1.0 / sgsrDisplaySize();
}

// Reversed-Z guide depth -> SGSR2's standard [0,1] depth.
//
// The guide stores ndc z/w of a reversed-Z projection: 0 for a ray that hit nothing and 1 at the near plane,
// with the far plane at whatever guide depth the frame's projection puts there (0 for an infinite far plane).
// Standard depth is the reciprocal-distance mapping SGSR2 expects, so a pixel with no surface reports the far
// plane and every comparison in the passes (nearest-neighbour min, "depth < 1 - 1e-5" as "has history depth")
// reads the way upstream intends.
public float sgsrStandardDepth(float reversedZ) {
    if (!(reversedZ > 0.0)) {
        return 1.0; // sky, or a guide that has not been written yet
    }
    float denom = reversedZ + pc.projM22;
    float distance = abs(pc.projM32 / (abs(denom) < 1.0e-6 ? -1.0e-6 : denom));
    if (isnan(distance)) {
        return 1.0;
    }
    // The host guarantees cameraNear > 0 and cameraFar > cameraNear; the clamps only keep a malformed
    // projection from producing a non-finite depth. Clamping the distance into [near, far] is also what keeps
    // the result inside [0,1]: the mapping is monotonic, near is 0 and far is 1, which is the direction the
    // passes below rely on — convert takes a min() expecting the closest surface to have the smallest value,
    // and activate reads a value below 1 as "this pixel has a surface depth to compare history against".
    float near = max(pc.cameraNear, 1.0e-6);
    float far = max(pc.cameraFar, near * (1.0 + 1.0e-6));
    float distanceReciprocal = 1.0 / clamp(distance, near, far);
    return clamp((1.0 / near - distanceReciprocal) / (1.0 / near - 1.0 / far), 0.0, 1.0);
}

// This renderer's motion guide (render pixels, previous minus current) -> SGSR2's NDC motion
// (current minus previous), which is what the activate and upscale passes reproject with.
public float2 sgsrNdcMotion(float2 motionPixels) {
    return -motionPixels * 2.0 / sgsrRenderSize();
}

// Upstream upscale.comp: the Lanczos-like reconstruction weight for a squared texel distance.
public float FastLanczos(float base) {
    float y = base - 1.0f;
    float y2 = y * y;
    float y_temp = 0.75f * y + y2;
    return y_temp * y2;
}

// Upstream upscale.comp: unpack the 11/11/10 YCoCg bits convert.comp wrote.
public float3 DecodeColor(uint sample32) {
    uint x11 = sample32 >> 21u;
    uint y11 = sample32 & (2047u << 10u);
    uint z10 = sample32 & 1023u;
    float3 samplecolor;
    samplecolor.x = (float(x11) * (1.0 / 2047.5));
    samplecolor.y = (float(y11) * (4.76953602e-7)) - 0.5;
    samplecolor.z = (float(z10) * (1.0 / 1023.5)) - 0.5;
    return samplecolor;
}

// Upstream activate.comp: unpack only the luma that the same bit layout carries.
public float DecodeColorY(uint sample32) {
    uint x11 = sample32 >> 21u;
    return float(x11) * (1.0 / 2047.5);
}

// NRD front end: packs the renderer's guide buffers into the exact input layout NRD expects. Which
// denoiser consumes them is the shim's choice — REBLUR where the device can run it, RELAX where it cannot
// — and the one place the two disagree is how a radiance lane's hit distance is encoded.
//
// Every helper below is a port of the same-named function in NRD v4.17.3's Shaders/NRD.hlsli, evaluated
// for the encodings the shim builds the library with: NRD_NORMAL_ENCODING = RGBA8_UNORM (0) and
// NRD_ROUGHNESS_ENCODING = LINEAR (1). The host and the library must agree on those two numbers; they are
// build-time constants that never reach the API, so a mismatch is silent. LINEAR is the case in which the
// packer stores the roughness it is given unchanged — hence no extra sqrt here.
//
// Conventions taken from the official Vulkan sample, not guessed:
//   - viewZ is the hit position's camera-space Z, so it is NEGATIVE in front of the camera.
//   - the roughness NRD takes is "linear roughness" = sqrt(alpha), where the renderer's guide stores alpha.
//   - the hit-distance normalization takes roughness = 1.0 on the diffuse lane and linear roughness on the
//     specular lane, which is what the sample passes.
//   - radiance is denoised with the material factor divided out; see nrdDemodFactor in recon_common.
import recon_common;
import bindings;

[[vk::push_constant]] NrdFrontendPush pc;

// NRD.hlsli constants.
static const float NRD_FP16_MAX = 65504.0;
static const float NRD_EPS = 1.0e-6;
// Reported for pixels with no surface. NRD compares the magnitude of view-Z against
// "CommonSettings::denoisingRange" and skips whatever falls outside it, so this is derived from the range
// the host passed instead of being fixed: a sentinel inside the range would be denoised as if it were a
// surface, and a range below the tracer's reach would make NRD skip real terrain and never write its output
// there. The factor keeps the value well outside the range while staying finite for downstream divides.
float noHitViewZ() {
    return -pc.denoisingRange * 2.0;
}

float2 texelUv(int2 pix) {
    return (float2(pix) + 0.5) / float2(float(pc.width), float(pc.height));
}

// Camera-space Z of a reversed-Z depth. Reversed-Z puts the near plane at 1 and the far plane at 0, so
// this is negative in front of the camera and tends to -far as the depth tends to 0.
float cameraViewZ(float z) {
    float denom = z + pc.projM22;
    return -abs(pc.projM32 / (abs(denom) < 1.0e-6 ? -1.0e-6 : denom));
}

bool isFinite3(float3 v) {
    return !any(isnan(v)) && !any(isinf(v));
}

float safeRoughness(float alpha) {
    return sqrt(saturate(alpha));
}

// NRD.hlsli:548-562 — the specular magic curve, then the A + B*|viewZ| scale blended toward C by it.
float reblurHitDistanceNormalization(float viewZ, float3 hitDistParams, float roughness) {
    float f = 1.0 - exp2(-200.0 * roughness * roughness);
    f *= pow(saturate(roughness), 0.5);
    return (hitDistParams.x + abs(viewZ) * hitDistParams.y) * lerp(hitDistParams.z, 1.0, f);
}

// NRD.hlsli:884-894 — RELAX's front end packs the distance as it is, sanitized only.
float relaxHitDist(float hitDist) {
    return !isnan(hitDist) && !isinf(hitDist) ? clamp(hitDist, 0.0, NRD_FP16_MAX) : 0.0;
}

// NRD.hlsli:774-782 — "0" means "lobe skipped", which is not the case when this is called.
float reblurNormHitDist(float hitDist, float viewZ, float3 hitDistParams, float roughness) {
    float f = reblurHitDistanceNormalization(viewZ, hitDistParams, roughness);
    return max(saturate(hitDist / f), NRD_EPS);
}

// NRD.hlsli:398-405.
float3 linearToYCoCg(float3 color) {
    return float3(dot(color, float3(0.25, 0.5, 0.25)),
                  dot(color, float3(0.5, 0.0, -0.5)),
                  dot(color, float3(-0.25, 0.5, -0.25)));
}

// NRD.hlsli:787-799 with sanitization on.
float4 packRadianceAndNormHitDist(float3 radiance, float normHitDist) {
    radiance = isFinite3(radiance) ? clamp(radiance, 0.0, NRD_FP16_MAX) : float3(0.0, 0.0, 0.0);
    normHitDist = isfinite(normHitDist) ? saturate(normHitDist) : 0.0;
    return float4(linearToYCoCg(radiance), normHitDist);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int2 pix = int2(id.xy);
    if (pix.x >= pc.width || pix.y >= pc.height) {
        return;
    }
    float2 uv = texelUv(pix);

    float4 normalRough = gNormal.SampleLevel(uv, 0.0);
    float depth = gDepth.SampleLevel(uv, 0.0).x;
    float4 diff = inDiffRadianceHitDist.SampleLevel(uv, 0.0);
    float4 spec = inSpecRadianceHitDist.SampleLevel(uv, 0.0);

    // Sky: view-Z is pushed beyond the denoising range, which is how NRD is told the pixel has no surface.
    // The denoiser then skips it entirely — REBLUR's passes early-out and never write their outputs there —
    // so its radiance is deliberately not packed here: the back end takes the sky from the tracer's own
    // lanes instead. Zeroing the normal-roughness guide keeps the denoiser from blending sky into nearby
    // geometry if it ever does visit the pixel.
    if (dot(normalRough.xyz, normalRough.xyz) < 1.0e-8 || depth <= 0.0) {
        outNormalRoughness[pix] = float4(0.0, 0.0, 0.0, 0.0);
        outViewZ[pix] = noHitViewZ();
        outDiffRadianceHitDist[pix] = float4(0.0, 0.0, 0.0, 0.0);
        outSpecRadianceHitDist[pix] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float viewZ = cameraViewZ(depth);
    float alpha = saturate(normalRough.w);
    float linearRoughness = safeRoughness(alpha);
    float3 hitDistParams = float3(pc.hitDistA, pc.hitDistB, pc.hitDistC);

    // RGBA8_UNORM + LINEAR: the normal remapped to [0,1] and the linear roughness stored as-is. The alpha
    // channel carries roughness; materialID is not packed in this encoding and NRD does not use it.
    float3 n = normalRough.xyz;
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1.0e-20);
    outNormalRoughness[pix] = float4(n * 0.5 + 0.5, linearRoughness);
    outViewZ[pix] = viewZ;

    // Hit distances are averaged world-space distances. REBLUR rescales them through its own parameters,
    // with roughness 1.0 on the diffuse lane and the linear roughness on the specular lane; RELAX takes
    // them raw. The flag follows the denoiser the shim built.
    bool normalize = pc.normalizeHitDistances != 0u;
    float diffPackedHitDist = normalize
            ? reblurNormHitDist(diff.a, viewZ, hitDistParams, 1.0)
            : relaxHitDist(diff.a);
    float specPackedHitDist = normalize
            ? reblurNormHitDist(spec.a, viewZ, hitDistParams, linearRoughness)
            : relaxHitDist(spec.a);
    // The lanes are radiance, so the material has to come out before the denoiser sees them; the back end
    // puts it back with the same factors. The pre-exposure stays in: dividing it out here would push
    // physical-unit radiance past fp16 as soon as a texel is dark.
    float3 diffDemod = nrdDemodFactor(gAlbedo.SampleLevel(uv, 0.0).rgb);
    float3 specDemod = nrdDemodFactor(gSpecAlbedo.SampleLevel(uv, 0.0).rgb);
    outDiffRadianceHitDist[pix] = packRadianceAndNormHitDist(diff.rgb / diffDemod, diffPackedHitDist);
    outSpecRadianceHitDist[pix] = packRadianceAndNormHitDist(spec.rgb / specDemod, specPackedHitDist);
}

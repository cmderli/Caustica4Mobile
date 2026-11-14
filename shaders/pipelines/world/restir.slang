// ReSTIR PT: path-sample generation, temporal reuse, spatial reuse, and reservoir shading.
//
// The primary pass publishes the visible primary surface and leaves one camera continuation per pixel in
// the path queue. This module generates an indirect path from that recorded surface and stores the first
// indirect vertex as the reconnection vertex x2 together with the event that leaves it. A shifted sample
// keeps x2 fixed and only changes its connection to the
// target pixel's visible surface x1: the BSDF at x1, the BSDF at x2, and the shift Jacobian are
// re-evaluated, while the suffix beyond x2 (the BSDF sample's suffix radiance, or the light connection's
// visibility/radiance) is reused unchanged. That is the reconnection shift of generalized resampling,
// with x2 chosen at the first indirect vertex so the prefix is the target pixel's own primary surface and
// needs no replay.
//
// The two entry points are temporal (path generation + previous-frame reuse) and spatial (neighbor reuse
// + final shading). Both backends run the same body: the ray-tracing backend dispatches it as raygen
// stages through the shared hit/miss table, and the ray-query backend dispatches it as compute threads
// that inline-traverse the same TLAS.
//
// The primary vertex must be an opaque surface reached directly by the camera ray. Dielectric,
// particle, delta-mirror, and submerged primary paths fall back to indirect_pass.slang, which traces the
// full per-pixel path for that pixel; the spatial pass sees the fallback's invalid reservoir and leaves
// the image it wrote untouched.
import world_common;
import world_core;
import stage;
import bindings;
import math;
import medium;
import segment;
import water;
import trace;
import lighting;
import sky;
import indirect_pass;
import restir_common;

// A sampled surface, in the form both the generation path and shift mapping need.
public struct RestirDomain {
    public float3 pos;       // rebased world position
    public float3 camRel;    // camera-relative position (what reservoirs store)
    public float3 n;         // shading normal, oriented toward v
    public float3 v;         // direction toward the previous path vertex
    public float3 albedo;
    public float3 f0;
    public float rough;
    public float metal;
    public float sss;
    public float emission;
    public bool emitterInList;
    public bool delta;
};

public RestirDomain restirDomainFromPayload(float3 hitPos, float3 rd) {
    RestirDomain d = {};
    d.pos = hitPos;
    d.camRel = hitPos - worldPush.camOffset;
    d.n = payloadNormal();
    d.v = -rd;
    d.albedo = payloadAlbedo();
    d.f0 = payload.f0;
    float rough = clamp(payloadRoughness(), 0.0, 1.0);
    d.delta = isDeltaAlpha(rough);
    d.rough = d.delta ? 0.0 : rough;
    d.metal = clamp(payloadMetalness(), 0.0, 1.0);
    d.sss = payloadSss();
    d.emission = payloadEmission();
    d.emitterInList = payloadEmitterInList();
    return d;
}

public RestirDomain restirDomainFromPrimary(RestirPrimary p) {
    RestirDomain d = {};
    d.pos = p.pos;
    d.camRel = p.pos - worldPush.camOffset;
    d.n = octDecode(unpackUnorm16x2(p.packedNormal));
    d.v = octDecode(unpackUnorm16x2(p.packedView));
    d.albedo = restirUnpackHalf3(p.packedAlbedo0, p.packedAlbedo1);
    d.f0 = restirUnpackHalf3(p.packedF00, p.packedF01);
    float2 roughMetal = unpackHalf2(p.packedRoughMetal);
    d.rough = roughMetal.x;
    d.metal = roughMetal.y;
    float2 sssEmission = unpackHalf2(p.packedSssEmission);
    d.sss = sssEmission.x;
    d.emission = sssEmission.y;
    d.emitterInList = (p.flags & RESTIR_PRIMARY_EMITTER) != 0u;
    d.delta = isDeltaAlpha(d.rough);
    return d;
}

public float3 restirPrimaryMotion(RestirPrimary p) {
    return restirUnpackHalf3(p.packedMotion0, p.packedMotion1);
}

// The canonical sample carries the primary BRDF parameters of the pixel it belongs to; a shift replaces
// them with the target's. Albedo is deliberately not compared: it is texture detail, so requiring it to
// match would reject almost every neighboring pixel on a stone or grass surface. Roughness/metalness
// distinguish the material response that the shift actually re-evaluates.
public void restirStorePrimaryMaterial(inout RestirSample sample, RestirDomain primary) {
    restirSetPrimaryRoughMetal(sample, float2(primary.rough, primary.metal));
}

public bool restirPrimaryMaterialMatches(RestirSample sample, RestirDomain primary) {
    float2 roughMetal = restirPrimaryRoughMetal(sample);
    return abs(roughMetal.x - primary.rough) <= 0.15
            && abs(roughMetal.y - primary.metal) <= 0.2;
}

// Luminance-preserving clamp: a per-channel clamp of a saturated sample turns it white, which is exactly
// the smeared highlight this pass must not create.
public float3 restirClampLuminance(float3 value, float maxLuminance) {
    float lum = luminance(value);
    if (isnan(lum) || maxLuminance <= 0.0) {
        return float3(0.0);
    }
    if (isinf(lum)) {
        return float3(maxLuminance);
    }
    if (lum > maxLuminance) {
        value *= maxLuminance / lum;
    }
    return max(value, float3(0.0));
}

// ---- Packed sample accessors.
public float3 restirRcPos(RestirSample s) { return float3(s.rcPosX, s.rcPosY, s.rcPosZ); }
public void restirSetRcPos(inout RestirSample s, float3 v) {
    s.rcPosX = v.x; s.rcPosY = v.y; s.rcPosZ = v.z;
}
public float3 restirPrimaryPos(RestirSample s) {
    return float3(s.primaryPosX, s.primaryPosY, s.primaryPosZ);
}
public void restirSetPrimaryPos(inout RestirSample s, float3 v) {
    s.primaryPosX = v.x; s.primaryPosY = v.y; s.primaryPosZ = v.z;
}
public float3 restirRcNormal(RestirSample s) { return restirUnpackDir(s.packedRcNormal); }
public void restirSetRcNormal(inout RestirSample s, float3 v) { s.packedRcNormal = restirPackDir(v); }
public float3 restirPrimaryNormal(RestirSample s) { return restirUnpackDir(s.packedPrimaryNormal); }
public void restirSetPrimaryNormal(inout RestirSample s, float3 v) {
    s.packedPrimaryNormal = restirPackDir(v);
}
public float3 restirRcL(RestirSample s) { return float3(s.rcLX, s.rcLY, s.rcLZ); }
public void restirSetRcL(inout RestirSample s, float3 v) {
    s.rcLX = v.x; s.rcLY = v.y; s.rcLZ = v.z;
}
public float3 restirPHat(RestirSample s) { return restirUnpackHalf3(s.packedPHat0, s.packedPHat1); }
public void restirSetPHat(inout RestirSample s, float3 v) {
    v = restirClampLuminance(v, HALF_MAX);
    s.packedPHat0 = restirPackHalf3A(v);
    s.packedPHat1 = restirPackHalf3B(v);
}
public float2 restirPrimaryRoughMetal(RestirSample s) {
    return unpackHalf2(s.packedPrimaryRoughMetal);
}
public void restirSetPrimaryRoughMetal(inout RestirSample s, float2 v) {
    s.packedPrimaryRoughMetal = packHalf2(v);
}

public float3 restirDiffuseAlbedo(RestirDomain d) {
    return d.albedo * (1.0 - d.metal);
}

// Probability of choosing the specular lobe for a continuation, matching tracePath. A delta mirror is
// always the specular branch.
public float restirSpecularLobeProbability(RestirDomain d) {
    if (d.delta) return 1.0;
    return clamp(luminance(d.f0) / (luminance(d.f0) + luminance(restirDiffuseAlbedo(d)) + 1.0e-4),
            0.1, 0.9);
}

// Diffuse-lobe BRDF * cosine for direction l. The caller has already oriented n toward v.
public float3 restirDiffuseFcos(RestirDomain d, float3 n, float3 l) {
    float ndl = max(0.0, dot(n, l));
    return restirDiffuseAlbedo(d) * (INV_PI * ndl);
}

public float restirDiffusePdf(float3 n, float3 l) {
    return max(0.0, dot(n, l)) * INV_PI;
}

// GGX specular-lobe BRDF * cosine for direction l. This is the same Cook-Torrance term the NEE path
// uses, so a shifted sample and a canonical sample estimate the same integrand.
public float3 restirSpecularFcos(RestirDomain d, float3 n, float3 v, float3 l) {
    float ndl = dot(n, l);
    if (ndl <= 0.0) return float3(0.0, 0.0, 0.0);
    float ndv = dot(n, v);
    if (ndv <= 0.0) return float3(0.0, 0.0, 0.0);
    float3 hv = v + l;
    if (dot(hv, hv) < 1.0e-12) return float3(0.0, 0.0, 0.0);
    float3 h = normalize(hv);
    float ndh = max(0.0, dot(n, h));
    float vdh = max(0.0, dot(v, h));
    float D = ggxD(ndh, d.rough);
    float G = ggxG1(ndv, d.rough) * ggxG1(ndl, d.rough);
    float3 F = fresnelSchlick(vdh, d.f0);
    return (D * G) * F / (4.0 * ndv);
}

// VNDF solid-angle pdf of the reflected direction, matching the sampling in tracePath.
public float restirSpecularPdf(RestirDomain d, float3 n, float3 v, float3 l) {
    float ndl = dot(n, l);
    float ndv = dot(n, v);
    if (ndl <= 0.0 || ndv <= 0.0) return 0.0;
    float3 hv = v + l;
    if (dot(hv, hv) < 1.0e-12) return 0.0;
    float3 h = normalize(hv);
    float ndh = max(0.0, dot(n, h));
    return ggxD(ndh, d.rough) * ggxG1(ndv, d.rough) / (4.0 * ndv);
}

// Full BRDF * cosine, used where the event is a light connection rather than a sampled lobe.
public float3 restirFullFcos(RestirDomain d, float3 n, float3 v, float3 l) {
    float ndl = dot(n, l);
    if (ndl <= 0.0) return float3(0.0, 0.0, 0.0);
    float3 result = restirDiffuseFcos(d, n, l);
    if (!d.delta) {
        result += restirSpecularFcos(d, n, v, l);
    }
    return result;
}

// ---- Reservoir packing helpers. The reservoir ABI is restir_common.RestirReservoir (128 B/entry).

public uint restirPackHalf2(float2 v) {
    return packHalf2(v);
}

public float2 restirUnpackHalf2(uint p) {
    return unpackHalf2(p);
}

public uint restirPackDir(float3 dir) {
    return packUnorm16x2(octEncode(dir));
}

public float3 restirUnpackDir(uint packed) {
    return octDecode(unpackUnorm16x2(packed));
}

public uint restirPackHalf3A(float3 v) {
    return packHalf2(v.xy);
}

public uint restirPackHalf3B(float3 v) {
    return packHalf2(float2(v.z, 0.0));
}

public float3 restirUnpackHalf3(uint a, uint b) {
    float2 lo = unpackHalf2(a);
    return float3(lo.x, lo.y, unpackHalf2(b).x);
}

public RestirReservoir restirEmptyReservoir() {
    RestirReservoir r = {};
    r.sample.flags = 0u;
    r.wSum = 0.0;
    r.M = 0.0;
    return r;
}

public bool restirSampleIsValid(RestirSample s) {
    float lum = luminance(restirPHat(s));
    return (s.flags & RESTIR_FLAG_VALID) != 0u && lum > 0.0 && !isnan(lum) && !isinf(lum);
}

public bool restirReservoirIsValid(RestirReservoir r) {
    return r.M > 0.0 && restirSampleIsValid(r.sample) && r.wSum > 0.0
            && !isnan(r.wSum) && !isinf(r.wSum);
}

public void restirReservoirAddSample(inout RestirReservoir r, RestirSample s, float weight,
        inout uint seed) {
    r.wSum += weight;
    if (rndf(seed) * r.wSum < weight) {
        r.sample = s;
    }
    r.M += 1.0;
}

public void restirReservoirMerge(inout RestirReservoir r, RestirReservoir source, float weight,
        inout uint seed) {
    if (source.M <= 0.0 || weight <= 0.0) {
        return;
    }
    r.wSum += weight;
    if (rndf(seed) * r.wSum < weight) {
        r.sample = source.sample;
    }
    r.M += source.M;
}

public void restirCapConfidence(inout RestirReservoir r, uint cap) {
    if (cap == 0u || r.M <= float(cap)) {
        return;
    }
    float scale = float(cap) / r.M;
    r.wSum *= scale;
    r.M = float(cap);
}

// Unbiased contribution weight of a reservoir. pHat is the selected sample's target function, so
// wSum / (M * luminance(pHat)) is the weight the selected sample is emitted with.
public float restirUcw(RestirReservoir r) {
    float lum = luminance(restirPHat(r.sample));
    if (r.M <= 0.0 || lum <= 0.0) {
        return 0.0;
    }
    return r.wSum / (r.M * lum);
}

// ---- Reconnection shift. Fills outSample with src reconnected to dst's primary surface; returns false
// when the shifted path has zero target function (blocked, backfacing, or a delta BSDF that cannot be
// sampled at the new incidence).
public bool restirShiftSample(RestirSample src, RestirDomain dst, out RestirSample outSample,
        out float jacobian) {
    jacobian = 1.0;
    outSample = src;
    uint eventType = src.flags & RESTIR_EVENT_MASK;
    bool specularLobe = (src.flags & RESTIR_FLAG_LANE_SPECULAR) != 0u;
    float3 n1 = dst.n;
    float3 v1 = dst.v;

    if (eventType == RESTIR_EVENT_SKY) {
        float3 dir = restirUnpackDir(src.packedDir);
        float3 f1cos = float3(0.0);
        if (specularLobe) {
            if (dst.delta) return false;
            f1cos = restirSpecularFcos(dst, n1, v1, dir);
        } else {
            f1cos = restirDiffuseFcos(dst, n1, dir);
        }
        if (luminance(f1cos) <= 0.0) return false;
        // A sky direction that was clear from the source primary may be blocked by geometry from the
        // target primary. The transmittance also tints the sky seen through translucent blocks.
        float3 vis = visibility(offsetSurfaceOrigin(dst.pos, n1, dir, SURF_BIAS),
                dir, 10000.0).transmittance;
        if (max(vis.r, max(vis.g, vis.b)) <= 0.0) return false;
        restirSetPrimaryPos(outSample, dst.camRel);
        restirSetPrimaryNormal(outSample, n1);
        restirSetPHat(outSample, f1cos * restirRcL(src) * vis);
        restirStorePrimaryMaterial(outSample, dst);
        outSample.flags |= RESTIR_FLAG_VALID;
        jacobian = 1.0;
        return luminance(restirPHat(outSample)) > 0.0;
    }

    float3 x2 = restirRcPos(src) + worldPush.camOffset;
    float3 toTarget = dst.pos - x2;
    float dist2 = dot(toTarget, toTarget);
    if (dist2 < 1.0e-6) return false;
    float dist = sqrt(dist2);
    float3 dirToX2 = -toTarget / dist; // from x1 to x2
    float3 v2 = toTarget / dist;       // from x2 toward the new previous vertex

    float3 n2 = restirRcNormal(src);
    float cos2 = dot(n2, v2);
    if (cos2 < 0.0) {
        n2 = -n2;
        cos2 = -cos2;
    }
    if (cos2 <= 0.0) return false;

    // The source prefix's geometric term. The stored normal was oriented toward the source primary, so
    // this cosine is positive by construction.
    float3 srcPrimary = restirPrimaryPos(src) + worldPush.camOffset;
    float3 srcToX2 = x2 - srcPrimary;
    float srcDist2 = dot(srcToX2, srcToX2);
    if (srcDist2 < 1.0e-6) return false;
    float srcCos = dot(restirRcNormal(src), normalize(srcPrimary - x2));
    if (srcCos <= 0.0) return false;
    jacobian = (cos2 / dist2) / (srcCos / srcDist2);
    if (!(jacobian > 0.0) || isnan(jacobian) || isinf(jacobian)) return false;
    // Reconnection to a grazing/distant vertex can make J enormous. Clamp it before it can turn one
    // neighbor's sample into a persistent white firefly that the denoiser then smears.
    jacobian = min(jacobian, 32.0);

    // BSDF at the target's primary vertex for the source sample's lobe.
    float3 f1cos = float3(0.0);
    if (specularLobe) {
        if (dst.delta) return false;
        f1cos = restirSpecularFcos(dst, n1, v1, dirToX2);
    } else {
        f1cos = restirDiffuseFcos(dst, n1, dirToX2);
    }
    if (luminance(f1cos) <= 0.0) return false;

    float3 origin = offsetSurfaceOrigin(dst.pos, n1, dirToX2, SURF_BIAS);
    float originDist = length(x2 - origin);
    if (originDist <= 1.0e-4) return false;
    float3 vis = visibility(origin, dirToX2, originDist * 0.999).transmittance;
    if (max(vis.r, max(vis.g, vis.b)) <= 0.0) return false;

    // Reconnection-vertex material and BSDF under the new incoming direction.
    RestirDomain d2 = {};
    d2.pos = x2;
    d2.camRel = restirRcPos(src);
    d2.n = n2;
    d2.v = v2;
    d2.albedo = restirUnpackHalf3(src.packedAlbedo0, src.packedAlbedo1);
    d2.f0 = restirUnpackHalf3(src.packedF00, src.packedF01);
    float2 roughMetal = restirUnpackHalf2(src.packedRoughMetal);
    d2.rough = roughMetal.x;
    d2.metal = roughMetal.y;
    float2 sssEmission = restirUnpackHalf2(src.packedSssEmission);
    d2.sss = sssEmission.x;
    d2.emission = sssEmission.y;
    d2.delta = (src.flags & RESTIR_FLAG_X2_DELTA) != 0u;

    float3 wo = restirUnpackDir(src.packedDir);
    float3 f2cos = float3(0.0);
    if (eventType == RESTIR_EVENT_BSDF) {
        // A delta lobe only accepts the one reflected direction; changing the incidence invalidates the
        // suffix sample. The canonical sample still shades this pixel; only reuse is dropped.
        if (d2.delta) return false;
        if ((src.flags & RESTIR_FLAG_X2_LIGHT_LOBE) != 0u) {
            f2cos = restirSpecularFcos(d2, n2, v2, wo);
        } else {
            f2cos = restirDiffuseFcos(d2, n2, wo);
        }
    } else {
        f2cos = restirFullFcos(d2, n2, v2, wo);
    }

    float3 emission = (src.flags & RESTIR_FLAG_EMISSION) != 0u
            ? float3(d2.emission, d2.emission, d2.emission)
            : float3(0.0);
    float3 pHat = (f2cos * restirRcL(src) + emission) * (f1cos * vis);
    float lum = luminance(pHat);
    if (!(lum > 0.0) || isnan(lum) || isinf(lum)) return false;

    restirSetPrimaryPos(outSample, dst.camRel);
    restirSetPrimaryNormal(outSample, n1);
    restirSetPHat(outSample, pHat);
    restirStorePrimaryMaterial(outSample, dst);
    outSample.flags |= RESTIR_FLAG_VALID;
    return true;
}

// ---- Path-sample generation at the target primary vertex.

// Choose and evaluate the reconnection vertex's outgoing event. NEE is a single power-weighted light or
// sun sample whose visibility is fixed at generation, so a shift needs no extra shadow ray at x2.
public bool restirSampleEvent(RestirDomain x2, float3 win, bool hasEmitters, bool hasSun,
        uint2 pix, uint sampleIndex, float x2Distance, float coneWidth, float coneSpread,
        inout uint seed, inout uint proposalSeed,
        out float3 wo, out float3 rcL, out float eventPdf, out uint eventFlags) {
    wo = float3(0.0, 0.0, 1.0);
    rcL = float3(0.0);
    eventPdf = 0.0;
    eventFlags = 0u;

    float ps = restirSpecularLobeProbability(x2);
    bool canNee = !x2.delta && (hasSun || hasEmitters);
    float pNee = canNee ? 0.5 : 0.0;

    if (canNee && rndf(seed) < pNee) {
        bool useEmitter = hasEmitters && (!hasSun || rndf(seed) < 0.5);
        float sourceProbability = (hasEmitters && hasSun) ? 0.5 : 1.0;
        if (useEmitter) {
            LightGridCell cell;
            int3 cellCoord;
            bool hasCell = findLightGridCell(x2.pos, cell, cellCoord);
            // The mixture density must describe both branches, so the local probability is fixed by the
            // cell's availability rather than by the branch this sample happened to take.
            float localProbability = hasCell ? 0.75 : 0.0;
            bool useLocal = rndf(seed) < localProbability;
            Light lg = selectLightGridLight(cell, useLocal, seed);
            float s = rndf(seed) * 2.0 - 1.0;
            float t = rndf(seed) * 2.0 - 1.0;
            float3 sp = lg.pos + worldPush.lightRebase.xyz
                    + s * lightHalfU(lg) + t * lightHalfV(lg);
            float3 le = lightRadiance(lg);
            float3 ln = lightGeometricNormal(lg);
            float area = lightArea(lg);
            float3 toL = sp - x2.pos;
            float dist2 = dot(toL, toL);
            if (dist2 < 1.0e-6) return false;
            float dist = sqrt(dist2);
            float3 wi = toL / dist;
            float cosL = max(0.0, dot(ln, -wi));
            if (cosL <= 0.0) return false;
            float3 origin = offsetSurfaceOrigin(x2.pos, x2.n, wi, SURF_BIAS);
            float originDist = length(sp - origin);
            if (originDist <= 1.0e-4) return false;
            float3 vis = visibility(origin, wi, originDist * 0.999).transmittance;
            wo = wi;
            // The light's area-to-solid-angle factor is folded into rcL; eventPdf stays the discrete
            // (light-choice) probability so the estimator is f * area * cosL / dist^2 / pdf.
            rcL = le * (area * cosL / dist2) * vis;
            eventPdf = pNee * sourceProbability
                    * proposalPdf(lg, le, cell, cellCoord, localProbability);
            eventFlags = RESTIR_EVENT_LIGHT;
        } else {
            float3 ld = celestialLight.dir;
            if (celestialLight.halfAngle > 0.0) {
                ld = sampleSquare(ld, celestialLight.halfAngle, seed);
            }
            float3 vis = visibility(offsetSurfaceOrigin(x2.pos, x2.n, ld, SURF_BIAS),
                    ld, 10000.0).transmittance;
            wo = ld;
            // Matches the sun NEE estimator, which treats illuminance as the irradiance a surface would
            // receive and uses no directional pdf division.
            rcL = celestialLight.illuminance * vis;
            eventPdf = pNee * sourceProbability;
            eventFlags = RESTIR_EVENT_LIGHT;
        }
        return eventPdf > 0.0 && !isnan(eventPdf) && !isinf(eventPdf);
    }

    // BSDF event: sample a continuation and trace the suffix as an ordinary path. Its first vertex is
    // the reconnection vertex, so the suffix's own radiance is independent of the incoming direction at
    // x2 and can be reused by every shift.
    bool specular = rndf(seed) < ps;
    float pdfDir = 0.0;
    float pLobe = 0.0;
    float f2cos = 0.0;
    if (x2.delta) {
        specular = true;
        wo = reflect(-win, x2.n);
        if (dot(x2.n, wo) <= 0.0) return false;
        f2cos = luminance(fresnelSchlick(max(0.0, dot(x2.n, win)), x2.f0));
        pdfDir = 1.0;
        pLobe = 1.0;
    } else if (specular) {
        float3 h = sampleGGXVNDF(x2.n, win, x2.rough, seed);
        wo = reflect(-win, h);
        if (dot(x2.n, wo) <= 0.0) return false;
        f2cos = luminance(restirSpecularFcos(x2, x2.n, win, wo));
        pdfDir = restirSpecularPdf(x2, x2.n, win, wo);
        pLobe = ps;
    } else {
        wo = cosineDir(x2.n, seed);
        f2cos = luminance(restirDiffuseFcos(x2, x2.n, wo));
        pdfDir = restirDiffusePdf(x2.n, wo);
        pLobe = 1.0 - ps;
    }
    if (pdfDir <= 0.0 || pLobe <= 0.0 || !(f2cos > 0.0)) return false;

    float coneWidth2 = coneWidth + coneSpread * max(x2Distance, 0.0);
    float coneSpread2 = max(coneSpread,
            specular ? x2.rough * RAY_CONE_GLOSSY_SPREAD_SCALE : RAY_CONE_DIFFUSE_SPREAD);
    PathSegment suffix = makePathSegment(
            offsetSurfaceOrigin(x2.pos, x2.n, wo, SURF_BIAS), wo,
            float3(1.0, 1.0, 1.0), makeMediumStack(airMedium()),
            max(coneWidth2, RAY_CONE_MIN_WIDTH), coneSpread2,
            seed, 2, specular);
    PathRadiance tail = tracePath(suffix, pix, sampleIndex);
    rcL = tail.diffuse + tail.specular;
    eventPdf = (1.0 - pNee) * pLobe * pdfDir;
    eventFlags = RESTIR_EVENT_BSDF
            | (specular ? RESTIR_FLAG_X2_LIGHT_LOBE : 0u)
            | (x2.delta ? RESTIR_FLAG_X2_DELTA : 0u);
    return (eventPdf > 0.0) && !isnan(eventPdf) && !isinf(eventPdf);
}

// Generate one full path sample: sample the first indirect direction at x1, trace to the reconnection
// vertex, choose and evaluate its outgoing event, and return the sample plus its generation pdf.
public bool restirGenerateSample(RestirDomain x1, float3 cameraDir,
        bool risOn, bool hasEmitters, bool hasSun,
        float coneWidth, float coneSpread, uint2 pix, uint sampleIndex,
        inout uint seed, inout uint proposalSeed, out RestirSample sample, out float ucw,
        out bool fallback) {
    sample = {};
    ucw = 0.0;
    fallback = false;
    float3 f0 = x1.f0;
    float ps = restirSpecularLobeProbability(x1);

    bool specular = rndf(seed) < ps;
    if (x1.delta) specular = true;
    float3 w1;
    float3 f1cos = float3(0.0);
    float pdfDir = 0.0;
    float pLobe = 0.0;
    if (specular) {
        if (x1.delta) {
            w1 = reflect(cameraDir, x1.n);
            f1cos = fresnelSchlick(max(0.0, dot(x1.n, x1.v)), f0);
            pdfDir = 1.0;
            pLobe = 1.0;
        } else {
            float3 h = sampleGGXVNDF(x1.n, x1.v, x1.rough, seed);
            w1 = reflect(cameraDir, h);
            if (dot(x1.n, w1) <= 0.0) return false;
            f1cos = restirSpecularFcos(x1, x1.n, x1.v, w1);
            pdfDir = restirSpecularPdf(x1, x1.n, x1.v, w1);
            pLobe = ps;
        }
    } else {
        w1 = cosineDir(x1.n, seed);
        f1cos = restirDiffuseFcos(x1, x1.n, w1);
        pdfDir = restirDiffusePdf(x1.n, w1);
        pLobe = 1.0 - ps;
    }
    float pdf1 = pLobe * pdfDir;
    if (!(pdf1 > 0.0) || !(luminance(f1cos) > 0.0) || isnan(pdf1) || isinf(pdf1)) {
        fallback = true;
        return false;
    }

    // The sampled lobe widens the cone before it reaches x2, exactly as the ordinary bounce loop widens
    // it between vertices; the closest hit at x2 uses that footprint for its texture LOD.
    float coneSpread1 = max(coneSpread,
            specular ? x1.rough * RAY_CONE_GLOSSY_SPREAD_SCALE : RAY_CONE_DIFFUSE_SPREAD);
    float3 ro = offsetSurfaceOrigin(x1.pos, x1.n, w1, SURF_BIAS);
    traceRadiance(CULL_SECONDARY, ro, 0.0, w1, 10000.0, specular,
            max(coneWidth, RAY_CONE_MIN_WIDTH), coneSpread1);

    restirSetPrimaryPos(sample, x1.camRel);
    restirSetPrimaryNormal(sample, x1.n);
    sample.packedDir = restirPackDir(w1);
    sample.flags = (specular ? RESTIR_FLAG_LANE_SPECULAR : 0u) | RESTIR_FLAG_VALID;

    if (payload.hitT < 0.0) {
        float3 sky = payloadSky();
        if (isnan(luminance(sky)) || isinf(luminance(sky))) return false;
        restirSetRcPos(sample, x1.camRel);
        restirSetRcNormal(sample, x1.n);
        restirSetRcL(sample, sky);
        restirSetPHat(sample, f1cos * sky);
        sample.flags = (sample.flags & ~RESTIR_EVENT_MASK) | RESTIR_EVENT_SKY;
        ucw = 1.0 / pdf1;
        return luminance(restirPHat(sample)) > 0.0;
    }

    if (payloadMaterial() != MATERIAL_OPAQUE) {
        // The first indirect hit is a dielectric or a particle. Neither can be represented by the
        // opaque reconnection sample, so let the ordinary path tracer own this pixel.
        fallback = true;
        return false;
    }
    float3 x2pos = ro + w1 * payload.hitT;
    RestirDomain x2 = restirDomainFromPayload(x2pos, w1);
    bool gateEmitter = risOn && x2.emitterInList;
    bool emissionAllowed = x2.emission > 0.0 && (!gateEmitter || specular);

    float3 wo;
    float3 rcL;
    float eventPdf;
    uint eventFlags;
    if (!restirSampleEvent(x2, -w1, hasEmitters, hasSun, pix, sampleIndex,
            max(payload.hitT, 0.0), coneWidth, coneSpread1, seed, proposalSeed,
            wo, rcL, eventPdf, eventFlags)) {
        return false;
    }

    float3 f2cos = float3(0.0);
    if ((eventFlags & RESTIR_FLAG_X2_DELTA) != 0u) {
        // A delta lobe returns its integrated Fresnel reflectance; evaluating a zero-width GGX lobe
        // through the finite-roughness formula would yield zero and discard the whole mirror path.
        f2cos = fresnelSchlick(max(0.0, dot(x2.n, -w1)), x2.f0);
    } else if ((eventFlags & RESTIR_EVENT_MASK) == RESTIR_EVENT_LIGHT) {
        f2cos = restirFullFcos(x2, x2.n, -w1, wo);
    } else if ((eventFlags & RESTIR_FLAG_X2_LIGHT_LOBE) != 0u) {
        f2cos = restirSpecularFcos(x2, x2.n, -w1, wo);
    } else {
        f2cos = restirDiffuseFcos(x2, x2.n, wo);
    }

    float3 emission = emissionAllowed
            ? float3(x2.emission, x2.emission, x2.emission)
            : float3(0.0);
    float3 pHat = f1cos * (f2cos * rcL + emission);
    float lum = luminance(pHat);
    if (!(lum > 0.0) || isnan(lum) || isinf(lum)) {
        return false;
    }

    restirSetRcPos(sample, x2.camRel);
    restirSetRcNormal(sample, x2.n);
    restirSetRcL(sample, rcL);
    sample.packedRoughMetal = restirPackHalf2(float2(x2.rough, x2.metal));
    sample.packedSssEmission = restirPackHalf2(float2(x2.sss, x2.emission));
    sample.packedAlbedo0 = restirPackHalf3A(x2.albedo);
    sample.packedAlbedo1 = restirPackHalf3B(x2.albedo);
    sample.packedF00 = restirPackHalf3A(x2.f0);
    sample.packedF01 = restirPackHalf3B(x2.f0);
    sample.packedDir = restirPackDir(wo);
    sample.flags = (sample.flags & (RESTIR_FLAG_LANE_SPECULAR | RESTIR_FLAG_VALID))
            | eventFlags
            | (emissionAllowed ? RESTIR_FLAG_EMISSION : 0u);
    restirSetPHat(sample, pHat);
    ucw = 1.0 / (pdf1 * eventPdf);
    return ucw > 0.0 && !isnan(ucw) && !isinf(ucw) && luminance(restirPHat(sample)) > 0.0;
}

// ---- Pixel entry points.

// Generate this pixel's canonical path samples and its direct-lighting base from the primary pass's
// record. `eligible` is false when the record is invalid or a sampled path cannot be resampled; the
// ordinary indirect pass then owns the pixel.
public RestirReservoir restirGenerate(uint2 pix, uint2 dimensions, PackedPathSegment packed,
        RestirPrimary primary, inout float3 baseDiffuse, inout float3 baseSpecular,
        out RestirDomain x1, out float3 primaryMotion, out bool eligible) {
    RestirReservoir result = restirEmptyReservoir();
    x1 = {};
    primaryMotion = float3(0.0);
    eligible = false;
    if ((primary.flags & RESTIR_PRIMARY_VALID) == 0u) {
        return result;
    }
    PathSegment seg = unpackPathSegment(packed);
    float coneSpread = max(seg.rayConeSpread, RAY_CONE_MIN_SPREAD);
    float coneWidth = max(seg.rayConeWidth, RAY_CONE_MIN_WIDTH);
    x1 = restirDomainFromPrimary(primary);
    if (x1.delta) {
        return result;
    }
    primaryMotion = restirPrimaryMotion(primary);
    float hitDistance = max(distance(seg.ro, x1.pos), 0.0);
    coneWidth = max(coneWidth + coneSpread * hitDistance, RAY_CONE_MIN_WIDTH);

    // x1's own emission is part of the primary surface, not of the indirect sample, so the direct helper
    // accounts for it along with sun NEE, RIS NEE, and SSS.
    bool risOn = worldPush.risCandidates > 0u && worldPush.lightCount > 0u && pc.lightBufAddr != 0;
    bool hasEmitters = risOn && pc.lightBufAddr != 0 && worldPush.lightCount > 0u;
    bool hasSun = celestialLight.illuminance.r + celestialLight.illuminance.g
            + celestialLight.illuminance.b > 0.0;
    bool waterWaves = (worldPush.flags & 16u) != 0u;
    uint spp = max(worldPush.spp, 1u);
    uint pixelIndex = pix.y * dimensions.x + pix.x;

    for (uint s = 0u; s < spp; ++s) {
        uint directSeed = seg.seed ^ (s * 2246822519u) ^ 0x9e3779b9u;
        directSeed = pcg(directSeed);
        uint directProposal = (pixelIndex * 2654435761u) ^ (s * 2246822519u)
                ^ (worldPush.frameIndex * 747796405u);
        directProposal = pcg(directProposal);
        shadeDirectLighting(x1.pos, x1.n, x1.v, -x1.v, float3(1.0, 1.0, 1.0),
                x1.albedo, restirDiffuseAlbedo(x1), x1.f0, x1.rough, x1.sss, x1.emission,
                x1.emitterInList, -1, 0, risOn, true, seg.medium, waterWaves, 0xFFFFFFFFu, false,
                directSeed, directProposal, baseDiffuse, baseSpecular);

        uint pathSeed = directSeed ^ 0x85ebca6bu;
        pathSeed = pcg(pathSeed);
        uint pathProposal = directProposal ^ 0xc2b2ae35u;
        pathProposal = pcg(pathProposal);
        RestirSample sample;
        float ucw;
        bool fallback = false;
        if (!restirGenerateSample(x1, -x1.v, risOn, hasEmitters, hasSun,
                coneWidth, coneSpread, pix, s,
                pathSeed, pathProposal, sample, ucw, fallback)) {
            if (fallback) {
                return result;
            }
            // A zero-contribution draw still consumes one sampling attempt; counting it keeps the
            // reservoir average over all spp attempts instead of renormalising over the successes.
            result.M += 1.0;
            continue;
        }
        restirStorePrimaryMaterial(sample, x1);
        float weight = luminance(restirPHat(sample)) * ucw;
        if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
            result.M += 1.0;
            continue;
        }
        restirReservoirAddSample(result, sample, weight, pathSeed);
    }
    eligible = restirSampleIsValid(result.sample);
    return result;
}

// Merge the previous frame's reservoir for the reprojected primary surface. The previous sample's
// positions are moved into this frame's camera-relative space, then shifted to this pixel's x1.
public void restirReuseTemporal(inout RestirReservoir result, RestirDomain x1,
        float3 primaryMotion, uint2 dimensions, inout uint seed) {
    if ((worldPush.restirFlags & 1u) == 0u || (worldPush.restirFlags & 2u) == 0u) {
        return;
    }
    float3 prevCamRel = x1.camRel - primaryMotion + worldPush.camDelta;
    float4 clip = mul(worldPush.prevViewProj, float4(prevCamRel, 1.0));
    if (clip.w <= 0.0) {
        return;
    }
    float2 prevUv = clip.xy / clip.w * 0.5 + 0.5;
    if (any(prevUv < float2(0.0, 0.0)) || any(prevUv > float2(1.0, 1.0))) {
        return;
    }
    int2 prevPixel = int2(prevUv * float2(dimensions));
    prevPixel = clamp(prevPixel, int2(0, 0), int2(dimensions) - int2(1, 1));
    DevicePtr<RestirReservoir> history = DevicePtr<RestirReservoir>(worldPush.restirHistoryAddr);
    RestirReservoir previous = history[uint(prevPixel.y) * dimensions.x + uint(prevPixel.x)];
    if (!restirReservoirIsValid(previous)) {
        return;
    }

    // The previous reservoir belongs to the previous pixel's visible surface. Only reuse it if that
    // surface is the one this pixel sees now, which rejects disocclusions and moving objects. The
    // comparison is in the previous frame's camera-relative space, so it includes the camera delta.
    if (distance(restirPrimaryPos(previous.sample), prevCamRel) > 0.1) {
        return;
    }
    float3 previousNormal = restirPrimaryNormal(previous.sample);
    if (dot(previousNormal, previousNormal) < 0.25) {
        return;
    }
    if (dot(normalize(previousNormal), x1.n) < 0.95) {
        return;
    }
    if (!restirPrimaryMaterialMatches(previous.sample, x1)) {
        return;
    }

    RestirSample source = previous.sample;
    restirSetRcPos(source, restirRcPos(source) + worldPush.camDelta);
    restirSetPrimaryPos(source, restirPrimaryPos(source) + worldPush.camDelta);
    restirSetPrimaryNormal(source, x1.n);
    RestirSample shifted;
    float jacobian;
    if (!restirShiftSample(source, x1, shifted, jacobian)) {
        return;
    }
    float ucw = restirUcw(previous);
    float weight = luminance(restirPHat(shifted)) * ucw * jacobian * previous.M;
    if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
        return;
    }
    RestirReservoir incoming = previous;
    incoming.sample = shifted;
    restirReservoirMerge(result, incoming, weight, seed);
}

// Add the reservoir's indirect radiance to the base lanes written by the temporal pass and publish the
// result as the next frame's history. Shared by spatial reuse and the temporal-only fast path.
public void restirShadeAndStore(uint2 dispatchIndex, uint pixelIndex, RestirReservoir result,
        DevicePtr<RestirReservoir> history) {
    float3 phat = restirPHat(result.sample);
    float lum = luminance(phat);
    float3 indirect = phat * (result.wSum / (result.M * lum));
    float4 image = outImage[dispatchIndex];
    // A leaked firefly is far brighter than the direct signal on the same surface. Clamping the
    // pre-exposed indirect luminance here keeps its hue and stops the denoiser from smearing a white
    // blotch across the neighborhood.
    float cap = max(worldPush.restirFireflyClamp, luminance(image.rgb) * 2.0);
    indirect = restirClampLuminance(indirect * worldPush.preExposure, cap);
    bool specular = (result.sample.flags & RESTIR_FLAG_LANE_SPECULAR) != 0u;
    float hitDistance = (result.sample.flags & RESTIR_EVENT_MASK) == RESTIR_EVENT_SKY
            ? SKY_HIT_DISTANCE
            : length(restirRcPos(result.sample) - restirPrimaryPos(result.sample));
    float4 diffLane = gDiffRadianceHitDist[dispatchIndex];
    float4 specLane = gSpecRadianceHitDist[dispatchIndex];
    if (specular) {
        specLane.rgb += indirect;
    } else {
        diffLane.rgb += indirect;
    }
    diffLane.a = specular ? 0.0 : hitDistance;
    specLane.a = specular ? hitDistance : 0.0;
    outImage[dispatchIndex] = float4(clamp(image.rgb + indirect, float3(0.0), float3(HALF_MAX)), 1.0);
    gDiffRadianceHitDist[dispatchIndex] = float4(clamp(diffLane.rgb, float3(0.0), float3(HALF_MAX)),
            diffLane.a);
    gSpecRadianceHitDist[dispatchIndex] = float4(clamp(specLane.rgb, float3(0.0), float3(HALF_MAX)),
            specLane.a);
    history[pixelIndex] = result;
}

public void restirTemporalPass(uint2 dispatchIndex, uint2 dimensions) {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    if (any(dispatchIndex >= dimensions)) {
        return;
    }
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    bool temporalOnly = worldPush.restirSpatialSamples == 0u;
    ConstPtr<PackedPathSegment> queue = ConstPtr<PackedPathSegment>(pc.pathQueueAddr);
    DevicePtr<RestirReservoir> history = DevicePtr<RestirReservoir>(worldPush.restirHistoryAddr);
    DevicePtr<RestirReservoir> scratch = DevicePtr<RestirReservoir>(worldPush.restirScratchAddr);
    DevicePtr<uint> mask = DevicePtr<uint>(pc.restirMaskAddr);
    if (pc.restirMaskAddr != 0) {
        mask[pixelIndex] = 0u;
    }
    PackedPathSegment packed = queue[pixelIndex];
    if (pc.restirPrimaryAddr == 0) {
        if (temporalOnly) {
            history[pixelIndex] = restirEmptyReservoir();
        } else if (worldPush.restirScratchAddr != 0) {
            scratch[pixelIndex] = restirEmptyReservoir();
        }
        return;
    }
    RestirPrimary primary = ConstPtr<RestirPrimary>(pc.restirPrimaryAddr)[pixelIndex];

    uint seed = (pixelIndex * 2654435761u) ^ (worldPush.frameIndex * 2246822519u) ^ 0x9e3779b9u;
    seed = pcg(seed);
    float3 baseDiffuse = float3(0.0);
    float3 baseSpecular = float3(0.0);
    RestirDomain x1;
    float3 primaryMotion;
    bool eligible = false;
    RestirReservoir result = restirGenerate(dispatchIndex, dimensions, packed, primary,
            baseDiffuse, baseSpecular, x1, primaryMotion, eligible);

    if (!eligible) {
        // Dielectric, particle, delta-mirror, submerged, or a failed path sample: leave the pixel to the
        // ordinary indirect pass that follows this dispatch. The zero mask tells it to take over.
        if (temporalOnly) {
            history[pixelIndex] = restirEmptyReservoir();
        } else if (worldPush.restirScratchAddr != 0) {
            scratch[pixelIndex] = restirEmptyReservoir();
        }
        return;
    }

    restirReuseTemporal(result, x1, primaryMotion, dimensions, seed);
    restirCapConfidence(result, worldPush.restirConfidenceCap);

    // Direct lighting at the primary vertex is not resampled; the spatial pass adds the indirect
    // reservoir on top of these base lanes.
    uint spp = max(worldPush.spp, 1u);
    float scale = worldPush.preExposure / float(spp);
    float3 storedDiffuse = clamp(baseDiffuse * scale, float3(0.0), float3(HALF_MAX));
    float3 storedSpecular = clamp(baseSpecular * scale, float3(0.0), float3(HALF_MAX));
    outImage[dispatchIndex] = float4(clamp(storedDiffuse + storedSpecular, float3(0.0),
            float3(HALF_MAX)), 1.0);
    gDiffRadianceHitDist[dispatchIndex] = float4(storedDiffuse, 0.0);
    gSpecRadianceHitDist[dispatchIndex] = float4(storedSpecular, 0.0);

    if (!restirReservoirIsValid(result)) {
        if (temporalOnly) {
            history[pixelIndex] = restirEmptyReservoir();
        } else if (worldPush.restirScratchAddr != 0) {
            scratch[pixelIndex] = restirEmptyReservoir();
        }
        return;
    }
    if (pc.restirMaskAddr != 0) {
        mask[pixelIndex] = 1u;
    }
    if (temporalOnly) {
        restirShadeAndStore(dispatchIndex, pixelIndex, result, history);
        return;
    }
    if (worldPush.restirScratchAddr != 0) {
        scratch[pixelIndex] = result;
    }
}

public void restirSpatialPass(uint2 dispatchIndex, uint2 dimensions) {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    if (any(dispatchIndex >= dimensions)) {
        return;
    }
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    DevicePtr<RestirReservoir> scratch = DevicePtr<RestirReservoir>(worldPush.restirScratchAddr);
    DevicePtr<RestirReservoir> history = DevicePtr<RestirReservoir>(worldPush.restirHistoryAddr);
    RestirReservoir result = scratch[pixelIndex];
    if (!restirReservoirIsValid(result)) {
        history[pixelIndex] = restirEmptyReservoir();
        return;
    }

    if (pc.restirPrimaryAddr == 0) {
        history[pixelIndex] = restirEmptyReservoir();
        return;
    }
    RestirPrimary primary = ConstPtr<RestirPrimary>(pc.restirPrimaryAddr)[pixelIndex];
    if ((primary.flags & RESTIR_PRIMARY_VALID) == 0u) {
        history[pixelIndex] = restirEmptyReservoir();
        return;
    }
    RestirDomain x1 = restirDomainFromPrimary(primary);

    uint seed = (pixelIndex * 747796405u) ^ (worldPush.frameIndex * 2654435761u) ^ 0xc2b2ae35u;
    seed = pcg(seed);
    uint spatialSamples = worldPush.restirSpatialSamples;
    float radius = max(worldPush.restirSpatialRadius, 1.0);
    int2 size = int2(dimensions);
    for (uint i = 0u; i < spatialSamples; ++i) {
        float2 uv = float2(rndf(seed), rndf(seed));
        float r = sqrt(uv.x) * radius;
        float angle = uv.y * 6.2831853;
        int2 offset = int2(round(float2(cos(angle), sin(angle)) * r));
        int2 neighbor = clamp(int2(dispatchIndex) + offset, int2(0, 0), size - int2(1, 1));
        if (all(neighbor == int2(dispatchIndex))) {
            continue;
        }
        RestirReservoir adjacent = scratch[uint(neighbor.y) * dimensions.x + uint(neighbor.x)];
        if (!restirReservoirIsValid(adjacent)) {
            continue;
        }
        // Neighbor selection is in screen space, so validate that the neighbor's primary surface is the
        // same local surface and has the same BRDF response. The tighter distance and normal limits keep
        // one bright reservoir from smearing across a block edge or a material seam.
        if (distance(restirPrimaryPos(adjacent.sample), x1.camRel) > 0.2) {
            continue;
        }
        float3 adjacentNormal = restirPrimaryNormal(adjacent.sample);
        if (dot(adjacentNormal, adjacentNormal) < 0.25) {
            continue;
        }
        if (dot(normalize(adjacentNormal), x1.n) < 0.95) {
            continue;
        }
        if (!restirPrimaryMaterialMatches(adjacent.sample, x1)) {
            continue;
        }
        // Specular paths are directionally sharp: reusing one across neighboring pixels spreads a
        // single sun/sky highlight into a broad white patch. Spatial reuse is therefore limited to
        // diffuse first lobes and diffuse/rough reconnection events; specular samples still contribute
        // through the canonical and temporal paths at their own pixel.
        if ((adjacent.sample.flags & RESTIR_FLAG_LANE_SPECULAR) != 0u) {
            continue;
        }
        if ((adjacent.sample.flags & RESTIR_FLAG_X2_LIGHT_LOBE) != 0u) {
            continue;
        }
        uint adjacentEvent = adjacent.sample.flags & RESTIR_EVENT_MASK;
        float2 adjacentRoughMetal = unpackHalf2(adjacent.sample.packedRoughMetal);
        if (adjacentEvent == RESTIR_EVENT_LIGHT && adjacentRoughMetal.x < 0.3) {
            continue;
        }
        RestirSample shifted;
        float jacobian;
        if (!restirShiftSample(adjacent.sample, x1, shifted, jacobian)) {
            continue;
        }
        float ucw = restirUcw(adjacent);
        float weight = luminance(restirPHat(shifted)) * ucw * jacobian * adjacent.M;
        if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
            continue;
        }
        RestirReservoir incoming = adjacent;
        incoming.sample = shifted;
        restirReservoirMerge(result, incoming, weight, seed);
    }
    restirCapConfidence(result, worldPush.restirConfidenceCap);

    if (!restirReservoirIsValid(result)) {
        history[pixelIndex] = restirEmptyReservoir();
        return;
    }
    restirShadeAndStore(dispatchIndex, pixelIndex, result, history);
}

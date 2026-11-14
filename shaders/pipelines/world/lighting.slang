// Direct lighting: the emitter light grid, RIS reservoirs, and reservoir shading.
// Used by the indirect pass's bounce loop and by ReSTIR's path-sample generation; the primary pass
// shades nothing. Depends on trace and math.

// contribution weight, M the effective sample count. wSum/phat are streaming scratch (not persisted).

import world_common;
import world_core;
import bindings;
import math;
import medium;
import water;
import trace;

// Upper bound on the candidate loop. The config setting and the UI enforce the same cap, so this only
// guards a hand-edited config from making one shading vertex walk an unbounded number of proposals.
public static const uint RIS_MAX_CANDIDATES = 8u;

public struct Reservoir {
    public float3 pos;  // light sample point (rebased world space)
    public float3 lnrm; // emitter outward normal
    public float3 le;   // emitter radiance
    public float area;  // emitter rectangle area (the area-light pdf factor)
    public float M;
    public float W;
    public float wSum;
    public float phat;
    public float pdf;   // light-choice proposal pdf that produced the current survivor
};

public Reservoir resEmpty() {
    Reservoir r;
    r.pos = float3(0.0, 0.0, 0.0);
    r.lnrm = float3(0.0, 0.0, 0.0);
    r.le = float3(0.0, 0.0, 0.0);
    r.area = 0.0;
    r.M = 0.0;
    r.W = 0.0;
    r.wSum = 0.0;
    r.phat = 0.0;
    return r;
}

// Unshadowed contribution of a fixed light sample point at surface (hitPos,n,v); out p-hat = its
// luminance (the resampling target). Evaluated from the UNBIASED hitPos — SURF_BIAS is applied only to
// the shadow-ray origin in shadeReservoir, on whichever side the survivor sample lands. Three receiver
// modes share one target function because they are mutually exclusive per candidate:
//   - front hemisphere (ndl>0): the same Lambert(+GGX) BRDF split the sun NEE uses;
//   - twoSided (particles): billboards have no back face, ndl=abs(ndl) folds both sides into the front;
//   - back hemisphere with sss>0 (LabPBR leaves/grass): HG transmission phase — cosT = dot(wi,rd) peaks
//     when the ray points from the light through the slab toward the camera (same as the sun SSS term).
public float3 evalSampleContrib(float3 sp, float3 lnrm, float3 le, float area, float3 hitPos, float3 n,
                         float3 v, float3 rd, float3 diffAlb, float3 F0, float rough,
                         bool twoSided, float sss, out float phat, out float3 specular) {
    phat = 0.0;
    // The specular half of `contrib`, in the same units. Callers that split the path into the lobes a
    // denoiser reconstructs separately need it; the rest ignore it.
    specular = float3(0.0);
    float3 toL = sp - hitPos;
    float dist2 = dot(toL, toL);
    if (dist2 < 1.0e-6) {
        return float3(0.0, 0.0, 0.0);
    }
    float dist = sqrt(dist2);
    float3 wi = toL / dist;
    float cosL = max(0.0, dot(lnrm, -wi));
    if (cosL <= 0.0) {
        return float3(0.0, 0.0, 0.0); // receiver behind the emitter face
    }
    float ndl = dot(n, wi);
    if (twoSided) {
        ndl = abs(ndl);
    }
    float3 contrib = float3(0.0, 0.0, 0.0);
    if (ndl > 0.0) {
        float G = ndl * cosL / dist2; // area-light geometry term (x area)
        float3 brdf = diffAlb * INV_PI;
        // twoSided callers are particle billboards, which pass rough=1/F0=0 and want diffuse-only
        // shading: fresnelSchlick still returns up to full white at grazing angles even with F0=0 (that
        // IS the Fresnel effect), so skipping the term here — not just zeroing F0 — is what keeps
        // billboards from picking up a rim highlight they were never meant to have.
        if (!twoSided) {
            float3 h = normalize(wi + v);
            float ndh = max(0.0, dot(n, h));
            float ndv = max(1.0e-4, dot(n, v));
            float vdh = max(0.0, dot(v, h));
            float D = ggxD(ndh, rough);
            float Gs = ggxG1(ndv, rough) * ggxG1(ndl, rough);
            float3 Fs = fresnelSchlick(vdh, F0);
            specular = (D * Gs) * Fs / (4.0 * ndv * ndl);
            brdf += specular;
        }
        // Scale both halves by the same transfer factors so the split stays exact: contrib == diffuse + specular.
        float transfer = (G * area);
        contrib = brdf * le * transfer;
        specular = specular * le * transfer;
    } else if (sss > 0.0) {
        float backNdl = -ndl;
        float G = backNdl * cosL / dist2;
        float cosT = dot(wi, rd);
        contrib = diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G) * le * (G * area);
    }
    phat = luminance(contrib);
    return contrib;
}

// Record-returning light proposals. Each helper returns the selected Light record instead of an index and
// prefetches the column's own record before reading the alias column: both addresses are known as soon as
// the column is picked, so the two loads issue together and only the alias outcome (roughly half of the
// samples) needs a second, dependent record fetch. The RIS candidate loop is bound by exactly this chain,
// so one fewer serialized random read per candidate is worth the record the losing column prefetched.

// Select one light from the emitted-power distribution in O(1).
public Light selectGlobalLight(inout uint proposalSeed) {
    float aliasSample = rndf(proposalSeed) * float(worldPush.lightCount);
    uint column = min(uint(aliasSample), worldPush.lightCount - 1u);
    ConstPtr<Light> lights = ConstPtr<Light>(pc.lightBufAddr);
    if (pc.lightAliasAddr == 0) {
        return lights[column];
    }
    Light own = lights[column];
    LightAlias alias = ConstPtr<LightAlias>(pc.lightAliasAddr)[column];
    return aliasSample - float(column) < alias.accept ? own : lights[alias.aliasIndex];
}

public bool findLightGridCell(float3 p, out LightGridCell cell, out int3 cellCoord) {
    cell.spanOffset = 0u;
    cell.spanCount = 0u;
    cell.invWeightSum = 0.0;
    cellCoord = int3(0, 0, 0);
    if (pc.lightGridCellAddr == 0 || pc.lightGridSpanAddr == 0) return false;
    int3 coord = int3(floor((p - worldPush.lightGridOrigin.xyz) / worldPush.lightGridOrigin.w));
    if (coord.x < 0 || coord.y < 0 || coord.z < 0
            || coord.x >= worldPush.lightGridDims.x
            || coord.y >= worldPush.lightGridDims.y
            || coord.z >= worldPush.lightGridDims.z) return false;
    uint linear = (uint(coord.z) * uint(worldPush.lightGridDims.y) + uint(coord.y))
            * uint(worldPush.lightGridDims.x) + uint(coord.x);
    cell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linear];
    cellCoord = coord;
    return cell.spanCount > 0u;
}

public Light selectSectionLight(uint firstLight, uint lightCount, inout uint proposalSeed) {
    float aliasSample = rndf(proposalSeed) * float(lightCount);
    uint column = min(uint(aliasSample), lightCount - 1u);
    uint index = firstLight + column;
    ConstPtr<Light> lights = ConstPtr<Light>(pc.lightBufAddr);
    Light own = lights[index];
    LightAlias alias = ConstPtr<LightAlias>(pc.lightLocalAliasAddr)[index];
    return aliasSample - float(column) < alias.accept ? own : lights[firstLight + alias.aliasIndex];
}

public float unpackUnsignedFloat(uint bits, uint mantissaBits) {
    uint mantissaMask = (1u << mantissaBits) - 1u;
    uint mantissa = bits & mantissaMask;
    uint exponent = (bits >> mantissaBits) & 31u;
    if (exponent == 0u) {
        return float(mantissa) * exp2(float(1 - 15 - int(mantissaBits)));
    }
    return (1.0 + float(mantissa) / float(1u << mantissaBits))
            * exp2(float(int(exponent) - 15));
}

public float3 lightRadiance(Light light) {
    uint packed = light.le;
    return float3(unpackUnsignedFloat(packed & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u));
}

public int3 lightSectionCoord(Light light) {
    uint packed = light.section;
    return int3(int(packed & 0x3ffu), int((packed >> 10u) & 0x3ffu),
            int((packed >> 20u) & 0x3ffu));
}

// The half axes share three lanes two-at-a-time; see the Light record in world_common.
public float3 lightHalfU(Light light) {
    return float3(unpackHalf2(light.halfUxy), unpackHalf2(light.halfUzVx).x);
}

public float3 lightHalfV(Light light) {
    return float3(unpackHalf2(light.halfUzVx).y, unpackHalf2(light.halfVyz));
}

// U x V drives both the emitter normal and the rectangle area, so it is computed once and shared.
public float3 lightCrossUV(Light light) {
    return cross(lightHalfU(light), lightHalfV(light));
}

// The rect spans 2U x 2V, so its area is 4|U x V|.
public float lightArea(Light light) {
    return 4.0 * length(lightCrossUV(light));
}

public float3 lightGeometricNormal(Light light) {
    float3 normal = normalize(lightCrossUV(light));
    return (light.section & 0x40000000u) != 0u ? -normal : normal;
}

public float proposalPdf(Light light, float3 le, LightGridCell cell, int3 cellCoord,
                  float localProbability) {
    float power = max(0.0, lightArea(light) * luminance(le));
    float globalPdf = pc.lightAliasAddr != 0
            ? power * worldPush.lightRebase.w : 1.0 / float(worldPush.lightCount);
    float localPdf = 0.0;
    if (localProbability > 0.0 && power > 0.0) {
        int3 delta = lightSectionCoord(light) - cellCoord;
        if (all(abs(delta) <= int3(2, 2, 2))) {
            float3 deltaF = float3(delta);
            localPdf = power * cell.invWeightSum / max(1.0, dot(deltaF, deltaF));
        }
    }
    return localProbability * localPdf + (1.0 - localProbability) * globalPdf;
}

public Light selectLightGridSpanLight(LightGridCell cell, inout uint proposalSeed) {
    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(pc.lightGridSpanAddr);
    float aliasSample = rndf(proposalSeed) * float(cell.spanCount);
    uint column = min(uint(aliasSample), cell.spanCount - 1u);
    LightGridSpan span = spans[cell.spanOffset + column];
    bool self = aliasSample - float(column) < span.accept;
    uint firstLight = self ? span.firstLight : span.aliasFirstLight;
    uint lightCount = self ? (span.packedLightCounts & 0xffffu)
            : (span.packedLightCounts >> 16u);
    return selectSectionLight(firstLight, lightCount, proposalSeed);
}

// Sample the exact mixture q = alpha*qCell + (1-alpha)*qGlobal. qCell first selects a nearby section
// by emitted power and section distance, then its section-local power alias. Every light in
// the neighborhood is represented without a fixed cap; qGlobal gives distant lights full support.
// Alias spans make weighted local selection O(1). The mixture PDF is reconstructed after the single
// Light48 load: section power cancels with the section-local light PDF.
public Light selectLightGridLight(LightGridCell cell, bool useLocal, inout uint proposalSeed) {
    return useLocal ? selectLightGridSpanLight(cell, proposalSeed) : selectGlobalLight(proposalSeed);
}

// ---- RIS cost profile. Temporary probes pinned the candidate walk out of the shader to bound what
// memory-side RIS work could ever be worth. Against an
// 18.0ms frame at M=8, removing every dependent load (span -> alias -> record) and all lane divergence
// saved 5.9ms — a third of the frame. Split by vertex that was ~4.3ms at secondary vertices against
// ~1.1ms at the primary hit, and forcing coherent selection recovered at most 1.3ms of it. So the cost
// was chasing depth, not divergence, and it lived at secondary vertices. A presampled candidate pool
// would attack the same 5.9ms structurally, but see the note on
// proposalSeed in tracePath for why it should share the pool rather than the seed.

// Initial RIS over M power-weighted candidates -> one resampled sample, no shadow ray yet. The chosen
// sample's W = wSum / (M * p-hat); M counts every candidate, including zero-weight ones.
public Reservoir risInitial(float3 hitPos, float3 n, float3 v, float3 rd, float3 diffAlb, float3 F0,
                     float rough, bool twoSided, float sss, uint candidateLimit,
                     inout uint seed, inout uint proposalSeed) {
    Reservoir r = resEmpty();
    LightGridCell gridCell;
    int3 gridCellCoord;
    float3 gridLookup = hitPos;
    if (pc.lightGridCellAddr != 0) {
        // Stochastically blend across hard section boundaries. Every cell proposal retains full global
        // support, so conditioning on this independently jittered lookup remains unbiased.
        gridLookup += (float3(rndf(proposalSeed), rndf(proposalSeed), rndf(proposalSeed)) - 0.5)
                * worldPush.lightGridOrigin.w;
    }
    bool hasGridCell = findLightGridCell(gridLookup, gridCell, gridCellCoord);
    // Every lane takes the same count and the same local/global split, so the loop body only skips work,
    // it never diverges a warp.
    uint candidateCount = min(min(worldPush.risCandidates, candidateLimit), RIS_MAX_CANDIDATES);
    r.M = float(candidateCount);
    if (candidateCount == 0u) {
        return r;
    }
    // Deterministically stratify the proposal mixture. At the default M=8 this schedules exactly six
    // local and two global candidates, with every lane taking the same branch for a given candidate.
    // Counts that are not divisible by four use the nearest practical split with at least one global
    // candidate; M=1 is global-only so the estimator retains full support.
    uint globalCandidateCount = hasGridCell
            ? max(1u, (candidateCount + 2u) / 4u) : candidateCount;
    uint localCandidateCount = candidateCount - globalCandidateCount;
    float localProbability = float(localCandidateCount) / float(candidateCount);

    for (uint c = 0u; c < candidateCount; c++) {
        uint globalsBefore = (c * globalCandidateCount) / candidateCount;
        uint globalsAfter = ((c + 1u) * globalCandidateCount) / candidateCount;
        bool useLocal = hasGridCell && globalsAfter == globalsBefore;
        Light lg = selectLightGridLight(gridCell, useLocal, proposalSeed);
        // Soft shadows: uniform point on the emitter rectangle ((s,t) in [-1,1]^2, pdf 1/area).
        float s = rndf(seed) * 2.0 - 1.0;
        float t = rndf(seed) * 2.0 - 1.0;
        float3 sp = lg.pos + worldPush.lightRebase.xyz
                + s * lightHalfU(lg) + t * lightHalfV(lg);
        float3 le = lightRadiance(lg);
        float3 lightNormal = lightGeometricNormal(lg);
        float area = lightArea(lg);
        float phat;
        float3 unusedSpecular;
        evalSampleContrib(sp, lightNormal, le, area, hitPos, n, v, rd,
                diffAlb, F0, rough, twoSided, sss, phat, unusedSpecular);
        if (phat <= 0.0) {
            continue;
        }
        float sourcePdf = proposalPdf(lg, le, gridCell, gridCellCoord, localProbability);
        float w = phat / max(sourcePdf, 1.0e-20);
        r.wSum += w;
        if (rndf(seed) * r.wSum < w) { // weighted reservoir update
            r.pos = sp;
            r.lnrm = lightNormal;
            r.le = le;
            r.area = area;
            r.phat = phat;
            r.pdf = sourcePdf;
        }
    }
    r.W = r.phat > 0.0 ? (r.wSum / (r.M * r.phat)) : 0.0;
    return r;
}

// Merge the previous frame's emitter sample for this pixel into the reservoir: re-evaluate the stored sample
// at this frame's shading point and add it as one more candidate whose proposal density is the stored
// light-choice pdf. This is the biased GRIS merge the indirect ReSTIR path also uses (no pairwise MIS), and
// it is what lets the canonical candidate count drop to one while emitter illumination keeps accumulating
// over frames instead of being re-drawn from scratch every frame.
public bool mergeEmitterHistory(uint historyIndex, float3 hitPos, float3 n, float3 v, float3 rd,
                                float3 diffAlb, float3 F0, float rough, float sss,
                                inout uint seed, inout Reservoir r) {
    EmitterHistory h = ConstPtr<EmitterHistory>(worldPush.emitterHistoryAddr)[historyIndex];
    if (h.valid == 0u) {
        return false;
    }
    // Camera-relative history -> this frame's rebased world space.
    float3 sp = float3(h.spX, h.spY, h.spZ) + worldPush.camOffset;
    float3 lnrm = float3(h.nX, h.nY, h.nZ);
    float3 le = float3(h.leX, h.leY, h.leZ);
    float phat;
    float3 unusedSpecular;
    evalSampleContrib(sp, lnrm, le, h.area, hitPos, n, v, rd,
            diffAlb, F0, rough, false, sss, phat, unusedSpecular);
    if (phat <= 0.0) {
        return false;
    }
    float w = phat / max(h.pdf, 1.0e-20);
    r.wSum += w;
    r.M += 1.0;
    if (rndf(seed) * r.wSum < w) {
        r.pos = sp;
        r.lnrm = lnrm;
        r.le = le;
        r.area = h.area;
        r.phat = phat;
        r.pdf = h.pdf;
    }
    return true;
}

// Persist the reservoir's survivor for the next frame. Nothing else writes this pixel's slot, so a single
// array is enough: the read above always happens before this store in the same thread.
public void storeEmitterHistory(uint historyIndex, Reservoir r) {
    EmitterHistory h;
    h.valid = (r.phat > 0.0 && r.W > 0.0) ? 1u : 0u;
    float3 spCam = r.pos - worldPush.camOffset;
    h.spX = spCam.x;
    h.spY = spCam.y;
    h.spZ = spCam.z;
    h.nX = r.lnrm.x;
    h.nY = r.lnrm.y;
    h.nZ = r.lnrm.z;
    h.leX = r.le.x;
    h.leY = r.le.y;
    h.leZ = r.le.z;
    h.area = r.area;
    h.pdf = r.pdf;
    h.reserved0 = 0.0;
    h.reserved1 = 0.0;
    h.reserved2 = 0.0;
    h.reserved3 = 0.0;
    DevicePtr<EmitterHistory>(worldPush.emitterHistoryAddr)[historyIndex] = h;
}

// Shade a finalized reservoir: recompute the survivor's contribution at (hitPos,n,v), cast ONE shadow
// ray, and return throughput-free radiance contrib*vis*W. The one ray serves whichever term fired for
// the survivor (front BRDF, twoSided billboard, or SSS backscatter) — the origin is biased toward the
// sample's side of the surface.
public float3 shadeReservoir(Reservoir s, float3 hitPos, float3 n, float3 v, float3 rd, float3 diffAlb,
                      float3 F0, float rough, bool twoSided, float sss, out float3 specular) {
    specular = float3(0.0);
    if (s.W <= 0.0 || s.phat <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float phat;
    float3 rawSpecular;
    float3 contrib = evalSampleContrib(s.pos, s.lnrm, s.le, s.area, hitPos, n, v, rd,
            diffAlb, F0, rough, twoSided, sss, phat, rawSpecular);
    if (phat <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    // Pick the shadow-ray origin on the sample's side of the surface FIRST, then measure the ray from
    // that origin — measuring dist from hitPos while tracing from the biased origin shifts the ray tip
    // up to SURF_BIAS*ndl toward the emitter plane, beating the 0.001*dist stop-short margin for lights
    // closer than ~SURF_BIAS/0.001 blocks -> shadow ray hits the emitter's own face -> per-texel black.
    float side = dot(n, s.pos - hitPos) >= 0.0 ? 1.0 : -1.0;
    float3 origin = hitPos + side * n * SURF_BIAS;
    float3 toL = s.pos - origin;
    float dist = length(toL);
    // Stop just short of the sample point so the ray doesn't self-occlude on the emitter's own face.
    VisibilityResult shadow = visibility(origin, toL / dist, dist * 0.999);
    float3 vis = shadow.transmittance;
    specular = rawSpecular * vis * s.W;
    return contrib * vis * s.W;
}

// Direct lighting at one surface vertex: its own emission, sun/moon NEE, RIS emitter NEE, and thin-surface
// SSS transmission. The caller supplies the path throughput this vertex is reached with, so the same
// function serves the ordinary bounce loop and the ReSTIR primary vertex (throughput 1).
//
// lane = -1 means no continuation lobe has been chosen yet, so every term is split by BRDF component;
// lane 0/1 means the path already committed to the diffuse/specular lane and the whole term belongs to it.
public void shadeDirectLighting(
        float3 hitPos, float3 n, float3 v, float3 rd, float3 throughput,
        float3 albedo, float3 diffAlb, float3 F0, float rough, float sss, float emission,
        bool emitterInList, int lane, int hitDepth, bool risOn, bool showCelestial,
        MediumStack medium, bool waterWaves, uint emitterHistoryIndex,
        bool storeEmitterHistorySlot,
        inout uint seed, inout uint proposalSeed,
        inout float3 Ldiff, inout float3 Lspec) {
    // One gate drives both the RIS call below and the direct-hit emission suppression. They must agree:
    // gateEmitter exists only because RIS already accounted for this emitter, so a bounce where RIS is
    // skipped must also gather emission directly or that light is lost outright. Specular continuations
    // (showCelestial) never had emitter NEE at the previous vertex, so they keep the direct emission.
    bool gateEmitter = risOn && emitterInList;
    if (emission > 0.0 && (!gateEmitter || showCelestial)) {
        if (lane == 1) {
            Lspec += throughput * albedo * emission;
        } else {
            Ldiff += throughput * albedo * emission;
        }
    }

    float3 p = hitPos + n * SURF_BIAS;
    // NEE: direct light from the dominant celestial body (sun by day, moon by night) — Lambert diffuse +
    // GGX specular, one shadow ray. Jitter the light direction within its square angular extent so the
    // shadow ray samples the light's quad; the same sampled direction drives ndl and the half-vector.
    float3 lightDir = celestialLight.dir;
    float lightHalfAngle = celestialLight.halfAngle;
    if (lightHalfAngle > 0.0) {
        lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
    }
    float ndl = max(0.0, dot(n, lightDir));
    if (ndl > 0.0) {
        VisibilityResult shadow = visibility(p, lightDir, 10000.0);
        float3 vis = shadow.transmittance;
        // Underwater receiver whose shadow ray crossed a water surface: scale the direct light by the
        // wave-refraction caustic at the exit point. Focusing scales incident irradiance, so it applies
        // to the whole NEE term (diffuse + specular).
        if (medium.current.water && waterWaves && shadow.waterHitT > 0.0) {
            vis *= waterCaustic(p + lightDir * shadow.waterHitT, lightDir, shadow.waterHitT);
        }
        if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
            float3 diffBrdf = diffAlb * INV_PI;
            float3 h = normalize(lightDir + v);
            float ndh = max(0.0, dot(n, h));
            float ndv = max(1.0e-4, dot(n, v));
            float vdh = max(0.0, dot(v, h));
            float D = ggxD(ndh, rough);
            float G = ggxG1(ndv, rough) * ggxG1(ndl, rough);
            float3 F = fresnelSchlick(vdh, F0);
            float3 specBrdf = (D * G) * F / (4.0 * ndv * ndl);
            float3 direct = throughput * celestialLight.illuminance * ndl * vis;
            if (lane < 0) {
                Ldiff += direct * diffBrdf;
                Lspec += direct * specBrdf;
            } else if (lane == 1) {
                Lspec += direct * (diffBrdf + specBrdf);
            } else {
                Ldiff += direct * (diffBrdf + specBrdf);
            }
        }
    }

    // RIS direct lighting from block emitters. Front lobe + SSS backlighting are resampled together and
    // share ONE shadow ray — a light can only be in front of n or behind it, never both, so combining them
    // into one target function is exact. SSS is gated off past MAX_SSS_INDIRECT_DEPTH (front-only RIS).
    if (risOn) {
        float activeSss = hitDepth <= MAX_SSS_INDIRECT_DEPTH ? sss : 0.0;
        // Every vertex of the path reuses the pixel's stored emitter sample: the sample is a world-space
        // point on an emitter, so re-evaluating it at a deeper shading point is exactly as valid as at the
        // primary one, and it keeps the candidate count at one canonical draw per vertex instead of the
        // configured count. Only the primary vertex writes the slot back, so the next frame's sample stays
        // the one the visible surface chose.
        bool temporal = emitterHistoryIndex != 0xFFFFFFFFu;
        uint candidateLimit = temporal ? 1u : worldPush.risCandidates;
        Reservoir r = risInitial(hitPos, n, v, rd, diffAlb, F0, rough, false, activeSss,
                                 candidateLimit, seed, proposalSeed);
        if (temporal) {
            mergeEmitterHistory(emitterHistoryIndex, hitPos, n, v, rd, diffAlb, F0, rough,
                    activeSss, seed, r);
            // The survivor does not change while it is shaded, so the slot for the next frame is written
            // before the shadow ray to keep the store off the frame's critical path.
            if (storeEmitterHistorySlot) {
                storeEmitterHistory(emitterHistoryIndex, r);
            }
        }
        float3 risSpecular;
        float3 risDiffuse = shadeReservoir(r, hitPos, n, v, rd, diffAlb, F0, rough, false,
                                           activeSss, risSpecular);
        float3 direct = throughput;
        if (lane < 0) {
            Ldiff += direct * risDiffuse;
            Lspec += direct * risSpecular;
        } else if (lane == 1) {
            Lspec += direct * (risDiffuse + risSpecular);
        } else {
            Ldiff += direct * (risDiffuse + risSpecular);
        }
    }

    // Thin-surface SSS transmission. Light entering from the back face scatters through toward the viewer
    // via a forward-biased HG phase. The shadow ray fires from the back face to avoid self-occlusion;
    // cosT = dot(lightDir, rd) peaks when the ray points from the light through the slab toward the viewer.
    if (sss > 0.0 && hitDepth <= MAX_SSS_INDIRECT_DEPTH) {
        float backNdl = max(0.0, dot(-n, lightDir));
        if (backNdl > 0.0) {
            VisibilityResult shadowBack = visibility(hitPos - n * SURF_BIAS, lightDir, 10000.0);
            float3 visB = shadowBack.transmittance;
            // Same caustic as the front-face NEE, so transmitted foliage flickers with the light bands.
            if (medium.current.water && waterWaves && shadowBack.waterHitT > 0.0) {
                visB *= waterCaustic(hitPos + lightDir * shadowBack.waterHitT,
                                     lightDir, shadowBack.waterHitT);
            }
            if (max(visB.r, max(visB.g, visB.b)) > 0.0) {
                float cosT = dot(lightDir, rd);
                float3 transmission = throughput * diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G)
                        * backNdl * celestialLight.illuminance * visB;
                if (lane == 1) {
                    Lspec += transmission;
                } else {
                    Ldiff += transmission;
                }
            }
        }
    }
}

// TLAS instance-mask bits (set on the Java side, ANDed against these per-ray cull masks):
//   bit 0 (0x01) — secondary rays: shadows, GI bounces, reflections.
//   bit 1 (0x02) — the primary camera ray.
// Terrain / entities / block entities keep 0xFF (both bits). Particles use 0x02: primary-visible

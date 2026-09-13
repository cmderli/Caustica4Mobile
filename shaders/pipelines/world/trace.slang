// Ray dispatch: record/bucket constants, payload construction, radiance/guide traversal and shadow
// visibility for both world-pass backends. Depends on core.

// receiver billboards only. The first-person camera owner uses 0x01 (secondary only).

import world_common;
import world_core;
import bindings;

#ifdef CAUSTICA_RAY_QUERY
import stage;
import any_hit;
import closest_hit;
import sky_miss;
import guide_miss;
#endif

public static const uint CULL_SECONDARY = 0x01u;
public static const uint CULL_PRIMARY = 0x02u;
public static const uint TERRAIN_BUCKETS = 4u;
public static const uint SBT_RADIANCE = 0u;
public static const uint SBT_SHADOW = TERRAIN_BUCKETS;
public static const uint SBT_STRIDE_BUCKET = 1u;
public static const uint MISS_RADIANCE = 0u;
public static const uint MISS_GUIDE = 1u;

public RayDesc makeRay(float3 origin, float tmin, float3 dir, float tmax) {
    RayDesc r;
    r.Origin = origin;
    r.TMin = tmin;
    r.Direction = dir;
    r.TMax = tmax;
    return r;
}

public float3 offsetSurfaceOrigin(float3 position, float3 surfaceNormal,
                           float3 outgoingDirection, float bias) {
    float3 sideNormal = dot(surfaceNormal, outgoingDirection) >= 0.0
            ? surfaceNormal : -surfaceNormal;
    return position + sideNormal * bias;
}

// A radiance payload carrying only the two words the invoked hit/miss code READS: `flags`
// (show-celestial gate, consumed by the sky miss) and the packed ray cone (consumed by closest-hit for
// texture LOD). Every other member is an OUTPUT that the hit/miss code writes, so it carries no
// information into a trace and is initialized to a neutral value here.
public Payload makeRadiancePayload(uint flags, uint rayCone) {
    Payload p;
    // Zeroes the packed albedo/normal words directly: both the hit and the miss view read zero out of
    // them (half 0 and float 0 are the same bit pattern), so this is neutral for either case.
    p.surface0 = 0u;
    p.surface1 = 0u;
    p.surface2 = 0u;
    p.hitT = -1.0; // miss sentinel: the sky miss writes only the surface words, so a miss leaves this negative
    p.motionPrev = half3(0.0h, 0.0h, 0.0h);
    p.f0 = half3(0.0h, 0.0h, 0.0h);
    p.flags = flags;
    p.roughMetal = 0u;
    p.emissionSss = 0u;
    p.iorTransmission = 0u;
    p.rayCone = rayCone;
    return p;
}

public Payload makeShadowPayload() {
    Payload shadowPayload;
    shadowPayload.surface0 = 0u;
    shadowPayload.surface1 = 0u;
    shadowPayload.surface2 = 0u;
    // The albedo view is this ray's accumulated transmittance; the any-hit path multiplies into it.
    packAlbedo(shadowPayload, float3(1.0, 1.0, 1.0));
    shadowPayload.hitT = -1.0; // water-crossing sentinel, filled by the water any-hit
    shadowPayload.motionPrev = half3(0.0h, 0.0h, 0.0h);
    shadowPayload.f0 = half3(0.0h, 0.0h, 0.0h);
    // The guide miss clears this sentinel. An accepted opaque hit skips the miss entirely, while
    // rejected translucent/water hits continue until either a later opaque hit or a miss.
    shadowPayload.flags = 1u;
    shadowPayload.roughMetal = 0u;
    shadowPayload.emissionSss = 0u;
    shadowPayload.iorTransmission = 0u;
    shadowPayload.rayCone = 0u;
    return shadowPayload;
}

#ifdef CAUSTICA_RAY_QUERY

// ---- Ray-query backend ----
//
// A compute thread owns the traversal, so a "hit record" is the alpha/tint predicate deciding whether a
// candidate becomes a hit, followed by the closest-hit body for the one that won. The predicate is called
// only for the records that would have carried an any-hit shader: terrain buckets and the two entity
// buckets share one numbering, making bucket 1 the alpha-tested radiance record (terrain cutout and the
// whole entity any-hit bucket) and every bucket but 0 the alpha-tested shadow record (cutout,
// translucent and water terrain, plus the entity any-hit bucket) — the same split RtAccel's SBT layout
// and RtPipeline.hitGroupUsesAnyHit define for the ray-tracing backend.
static bool radianceRecordAlphaTests(uint geometryIndex) {
    return geometryIndex == BUCKET_CUTOUT;
}

static bool shadowRecordAlphaTests(uint geometryIndex) {
    return geometryIndex != 0u;
}

// Radiance-shaped traversal shared by radiance rays and guide probes: commit every candidate the
// predicate accepts and keep going, because traversal only ends once no closer hit can replace the
// committed one. `missIsSky` selects the miss record — the sky for a radiance ray, the minimal clear for
// a guide probe.
static void traverseRadianceRay(bool missIsSky, uint cullMask, float3 ro, float tmin, float3 rd, float tmax) {
    stageSetRay(ro, rd, false);
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, cullMask, makeRay(ro, tmin, rd, tmax));
    while (query.Proceed()) {
        uint geometryIndex = query.CandidateRayGeometryIndex();
        if (radianceRecordAlphaTests(geometryIndex)) {
            float2 barycentrics = query.CandidateTriangleBarycentrics();
            stageSetCandidate(uint(query.CandidateRayInstanceCustomIndex()), geometryIndex,
                    query.CandidateRayPrimitiveIndex(), query.CandidateTriangleRayT(),
                    query.CandidateObjectToWorld3x4());
            if (!anyHitAccepts(payload, barycentrics)) {
                continue; // rejected candidate: no commit, so traversal resumes
            }
        }
        query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float2 barycentrics = query.CommittedTriangleBarycentrics();
        // Vertex positions are only readable on a committed triangle, and they are in object space, so
        // the hit code applies the instance transform exactly as it does for the builtin in the
        // ray-tracing backend (VK_KHR_ray_tracing_position_fetch).
        float3 positions[3] = query.CommittedGetIntersectionTriangleVertexPositions();
        stageSetCommitted(uint(query.CommittedRayInstanceCustomIndex()),
                query.CommittedRayGeometryIndex(), query.CommittedRayPrimitiveIndex(),
                query.CommittedRayT(), query.CommittedObjectToWorld3x4(),
                positions[0], positions[1], positions[2]);
        closestHitMain(payload, barycentrics);
    } else if (missIsSky) {
        skyMiss(payload);
    } else {
        guideMiss(payload);
    }
}

// Radiance trace for latency-sensitive, coherent work such as the primary pass. The ray-query traversal
// has no reorder capability and no separate hit stages, so there is exactly one form.
public void traceRadiance(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                    bool showCelestial, float rayConeWidth, float rayConeSpread) {
    payload = makeRadiancePayload(showCelestial ? PAYLOAD_SHOW_CELESTIAL : 0u,
            packHalf2(float2(rayConeWidth, rayConeSpread)));
    traverseRadianceRay(true, cullMask, ro, tmin, rd, tmax);
}

// Auxiliary RR probes: one-off, incoherent rays whose result is consumed immediately. The dedicated miss
// record also avoids the atmosphere march: guides need hit-vs-miss, not sky radiance.
public void traceGuide(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                float rayConeWidth, float rayConeSpread) {
    payload = makeRadiancePayload(0u, packHalf2(float2(rayConeWidth, rayConeSpread)));
    traverseRadianceRay(false, cullMask, ro, tmin, rd, tmax);
}

public VisibilityResult visibility(float3 origin, float3 dir, float tmax) {
    // The shadow path uses only the albedo view as accumulated transmittance and hitT as the nearest-water
    // crossing; it owns its payload exactly as the ray-tracing backend's separate shadow trace does.
    Payload shadowPayload = makeShadowPayload();
    stageSetRay(origin, dir, true);
    // Shadow records run the alpha/tint predicate for cutout/translucent/water terrain and the entity
    // any-hit bucket; cutout alpha-tests, translucent and water tint that transmittance and pass through,
    // and everything else terminates traversal. There is no closest-hit work worth executing, so the first
    // accepted hit simply ends the ray.
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, CULL_SECONDARY,
            makeRay(origin, RAY_TMIN, dir, tmax));
    while (query.Proceed()) {
        uint geometryIndex = query.CandidateRayGeometryIndex();
        if (shadowRecordAlphaTests(geometryIndex)) {
            float2 barycentrics = query.CandidateTriangleBarycentrics();
            stageSetCandidate(uint(query.CandidateRayInstanceCustomIndex()), geometryIndex,
                    query.CandidateRayPrimitiveIndex(), query.CandidateTriangleRayT(),
                    query.CandidateObjectToWorld3x4());
            if (!anyHitAccepts(shadowPayload, barycentrics)) {
                continue; // passed through: traversal resumes behind the tinted surface
            }
        }
        query.CommitNonOpaqueTriangleHit();
        break;
    }
    if (query.CommittedStatus() == COMMITTED_NOTHING) {
        // Nothing blocked the ray, so the miss record clears the blocked sentinel and leaves the
        // transmittance the rejected hits accumulated.
        guideMiss(shadowPayload);
    }
    VisibilityResult result;
    result.transmittance = shadowPayload.flags == 0u
            ? unpackAlbedo(shadowPayload) : float3(0.0, 0.0, 0.0);
    result.waterHitT = shadowPayload.hitT;
    return result;
}

#else

// ---- Ray-tracing-pipeline backend ----

// Ordinary radiance trace for latency-sensitive, coherent work such as Pass A. It invokes hit/miss
// shaders directly and therefore requires no invocation-reorder capability.
public void traceRadiance(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                    bool showCelestial, float rayConeWidth, float rayConeSpread) {
    payload = makeRadiancePayload(showCelestial ? PAYLOAD_SHOW_CELESTIAL : 0u,
            packHalf2(float2(rayConeWidth, rayConeSpread)));
    TraceRay(topLevelAS, RAY_FLAG_NONE, cullMask,
            SBT_RADIANCE, SBT_STRIDE_BUCKET, MISS_RADIANCE,
            makeRay(ro, tmin, rd, tmax), payload);
}

// Auxiliary RR probes deliberately use ordinary TraceRay. They are one-off, incoherent rays whose
// result is consumed immediately, so paying a reorder barrier and carrying guide state across it only
// adds latency. The dedicated miss record also avoids the atmosphere march: guides need hit-vs-miss,
// not sky radiance.
public void traceGuide(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                float rayConeWidth, float rayConeSpread) {
    payload = makeRadiancePayload(0u, packHalf2(float2(rayConeWidth, rayConeSpread)));
    TraceRay(topLevelAS, RAY_FLAG_NONE, cullMask, SBT_RADIANCE, SBT_STRIDE_BUCKET,
            MISS_GUIDE, makeRay(ro, tmin, rd, tmax), payload);
}

public VisibilityResult visibility(float3 origin, float3 dir, float tmax) {
    // Vulkan requires an identical payload structure for every stage reachable by this trace. The shadow
    // path uses only the albedo view as accumulated transmittance and hitT as the nearest-water crossing.
    Payload shadowPayload = makeShadowPayload();
    // Shadow SBT records run any-hit only for cutout/translucent/water. Cutout alpha-tests; translucent
    // and water tint that transmittance and pass through. Solid blocks terminate traversal. There is no
    // closest shader worth executing. The lightweight guide miss clears the flags sentinel so this path
    // needs only ordinary TraceRay and no invocation-reorder capability.
    TraceRay(topLevelAS,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            CULL_SECONDARY, SBT_SHADOW, SBT_STRIDE_BUCKET, MISS_GUIDE,
            makeRay(origin, RAY_TMIN, dir, tmax), shadowPayload);
    VisibilityResult result;
    result.transmittance = shadowPayload.flags == 0u
            ? unpackAlbedo(shadowPayload) : float3(0.0, 0.0, 0.0);
    result.waterHitT = shadowPayload.hitT;
    return result;
}

#endif

// `valid` is false when the point projects behind the previous camera, where the perspective divide
// would otherwise manufacture an arbitrarily large NDC. Mirror images reach that case easily: a virtual
// image sits as far behind the reflector as the source is in front, so a reflection seen at a grazing

// Per-invocation stage state, shared by the world pass's two backends.
//
// The ray-tracing backend reads the intersection a hit stage was invoked for from the ray-tracing
// builtins. The ray-query backend has no hit stages: its compute entry owns the traversal and calls the
// same hit code as a plain function, so the intersection it just read from the RayQuery has to be handed
// over explicitly. Both backends reach the hit code through the accessors below, which is what lets one
// copy of the material/shading code compile into either a hit shader or an inline traversal loop.
//
// Only a compute build (CAUSTICA_RAY_QUERY) carries the mutable state; in a ray-tracing build every
// accessor forwards to a builtin and nothing here is writable.
module stage;

import world_common;

// Workgroup edge of the ray-query backend. One thread owns one pixel, and RtPipeline.RAY_QUERY_GROUP_SIZE
// sizes the dispatch from the same value, so the two must stay equal.
public static const uint RQ_GROUP_SIZE = 8u;

#ifdef CAUSTICA_RAY_QUERY

// One intersection as the traversal reports it. A candidate (the any-hit test) has no vertex positions
// yet — only a committed hit can fetch them — so those three fields stay zero until stageSetCommitted.
public struct StageIntersection {
    public uint instanceID;
    public uint geometryIndex;
    public uint primitiveIndex;
    public float t;
    public float3x4 objectToWorld;
    public float3 p0;
    public float3 p1;
    public float3 p2;
};

static StageIntersection stageHit = {};
static float3 stageOrigin = float3(0.0, 0.0, 0.0);
static float3 stageDirection = float3(0.0, 0.0, 1.0);
static bool stageShadow = false;
static uint2 stageIndex = uint2(0u, 0u);

/** The pixel a thread belongs to. Set once by the compute entry; the alpha dither keys off it. */
public void stageSetDispatch(uint2 index) {
    stageIndex = index;
}

/** The ray being traced. Set before every traversal, so hit code sees the ray that produced the hit. */
public void stageSetRay(float3 origin, float3 direction, bool shadow) {
    stageOrigin = origin;
    stageDirection = direction;
    stageShadow = shadow;
}

/** The candidate intersection the traversal is asking about, before it is committed or ignored. */
public void stageSetCandidate(uint instanceID, uint geometryIndex, uint primitiveIndex,
        float t, float3x4 objectToWorld) {
    stageHit.instanceID = instanceID;
    stageHit.geometryIndex = geometryIndex;
    stageHit.primitiveIndex = primitiveIndex;
    stageHit.t = t;
    stageHit.objectToWorld = objectToWorld;
}

/** The committed hit, with the vertex positions only a committed triangle can fetch. */
public void stageSetCommitted(uint instanceID, uint geometryIndex, uint primitiveIndex,
        float t, float3x4 objectToWorld, float3 p0, float3 p1, float3 p2) {
    stageSetCandidate(instanceID, geometryIndex, primitiveIndex, t, objectToWorld);
    stageHit.p0 = p0;
    stageHit.p1 = p1;
    stageHit.p2 = p2;
}

public uint2 stageDispatchIndex() { return stageIndex; }
public float3 stageRayOrigin() { return stageOrigin; }
public float3 stageRayDirection() { return stageDirection; }
public bool stageShadowRay() { return stageShadow; }
public uint stageInstanceID() { return stageHit.instanceID; }
public uint stageGeometryIndex() { return stageHit.geometryIndex; }
public uint stagePrimitiveIndex() { return stageHit.primitiveIndex; }
public float stageRayT() { return stageHit.t; }
public float3x4 stageObjectToWorld() { return stageHit.objectToWorld; }
public float3 stageTriangleVertexPosition(uint index) {
    if (index == 0u) return stageHit.p0;
    if (index == 1u) return stageHit.p1;
    return stageHit.p2;
}

#else

// ---- Ray-tracing-pipeline backend: every accessor is a builtin of the stage being compiled.

public uint2 stageDispatchIndex() { return DispatchRaysIndex().xy; }
public uint2 stageDispatchDimensions() { return DispatchRaysDimensions().xy; }
public float3 stageRayOrigin() { return WorldRayOrigin(); }
public float3 stageRayDirection() { return WorldRayDirection(); }
// The shadow path is the only trace that skips the closest-hit shader, and its any-hit records are the
// ones that need to know which side of that split they are on.
public bool stageShadowRay() { return (RayFlags() & RAY_FLAG_SKIP_CLOSEST_HIT_SHADER) != 0u; }
public uint stageInstanceID() { return InstanceID(); }
public uint stageGeometryIndex() { return GeometryIndex(); }
public uint stagePrimitiveIndex() { return PrimitiveIndex(); }
public float stageRayT() { return RayTCurrent(); }
public float3x4 stageObjectToWorld() { return ObjectToWorld3x4(); }
public float3 stageTriangleVertexPosition(uint index) { return HitTriangleVertexPosition(index); }

#endif

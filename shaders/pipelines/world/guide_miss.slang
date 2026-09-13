// Minimal miss record for reflection/refraction guide probes and shadow visibility. Guide rays consume
// only hit-vs-miss state; fixed sky guide values are supplied by the caller. Shadow rays seed flags=1 and
// use this clear as their miss marker, avoiding any dependence on the sky miss.
//
// Shared by both backends: guide.rmiss.slang is the ray-tracing entry point, and the ray-query traversal
// calls guideMiss() directly.
import world_common;

public void guideMiss(inout Payload payload) {
    // Preserve hitT: guide payloads already seed the -1 miss sentinel, while shadow any-hit may have
    // recorded the nearest water crossing here for underwater caustics. Preserve the surface words too:
    // this record serves shadow rays, whose accumulated transmittance lives in their albedo view.
    packNormal(payload, float3(0.0, 0.0, 0.0));
    payload.flags = 0u;
}

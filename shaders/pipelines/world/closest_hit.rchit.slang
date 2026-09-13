// Ray-tracing backend entry point for closest-hit shading. The body lives in closest_hit.slang, which the
// ray-query compute entry calls directly.
import world_common;
import closest_hit;

[shader("closesthit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    closestHitMain(payload, attr.barycentrics);
}

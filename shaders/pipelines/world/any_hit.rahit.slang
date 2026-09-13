// Ray-tracing backend entry point for alpha/tint candidate handling. The predicate lives in any_hit.slang,
// which the ray-query traversal calls directly.
import world_common;
import any_hit;

[shader("anyhit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    if (!anyHitAccepts(payload, attr.barycentrics)) {
        IgnoreHit();
    }
}

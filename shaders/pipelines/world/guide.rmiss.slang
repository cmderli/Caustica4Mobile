// Ray-tracing backend entry point for the minimal guide/shadow miss. The body lives in guide_miss.slang,
// which the ray-query traversal calls directly.
import world_common;
import guide_miss;

[shader("miss")]
void main(inout Payload payload) {
    guideMiss(payload);
}

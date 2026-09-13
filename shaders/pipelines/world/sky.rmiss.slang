// Ray-tracing backend entry point for a radiance miss. The body lives in sky_miss.slang, which the
// ray-query compute entry calls directly.
import world_common;
import sky_miss;

[shader("miss")]
void main(inout Payload payload) {
    skyMiss(payload);
}

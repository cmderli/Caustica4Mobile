// Ray-query backend entry point for ReSTIR PT path generation + temporal reuse.
//
// The inline traversal lives inside restir.slang's traceRadiance calls, so this dispatch runs the same
// body as the ray-tracing raygen stage. Threads a partial workgroup leaves outside the image return
// before the pass touches any output.
import world_common;
import bindings;
import stage;
import restir;

[shader("compute")]
[numthreads(RQ_GROUP_SIZE, RQ_GROUP_SIZE, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {
    uint width, height;
    outImage.GetDimensions(width, height);
    uint2 dimensions = uint2(width, height);
    uint2 pixel = threadID.xy;
    if (any(pixel >= dimensions)) return;
    stageSetDispatch(pixel);
    restirTemporalPass(pixel, dimensions);
}

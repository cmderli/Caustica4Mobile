// Ray-query backend entry point for the indirect/bounce pass.
//
// The inline traversal lives inside the bounce loop (see trace.slang), so this dispatch resumes the
// queued primary continuations and runs the whole path integral in compute threads: one thread per pixel
// of the render target, with the render target's own dimensions supplying the dispatch size the
// ray-tracing backend reads from DispatchRaysDimensions. Threads a partial workgroup leaves outside the
// image return before the pass touches any output.
import world_common;
import bindings;
import stage;
import indirect_pass;

[shader("compute")]
[numthreads(RQ_GROUP_SIZE, RQ_GROUP_SIZE, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {
    uint width, height;
    outImage.GetDimensions(width, height);
    uint2 dimensions = uint2(width, height);
    uint2 pixel = threadID.xy;
    if (any(pixel >= dimensions)) return;
    stageSetDispatch(pixel);
    indirectPass(pixel, dimensions);
}

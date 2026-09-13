// Ray-query backend entry point for the primary/guide pass.
//
// The inline traversal lives inside the pass body (see trace.slang), so this dispatch is the whole world
// pass: one thread per pixel of the render target, with the render target's own dimensions supplying the
// dispatch size the ray-tracing backend reads from DispatchRaysDimensions. Threads a partial workgroup
// leaves outside the image return before the pass touches any output.
import world_common;
import bindings;
import stage;
import primary_pass;

[shader("compute")]
[numthreads(RQ_GROUP_SIZE, RQ_GROUP_SIZE, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {
    uint width, height;
    outImage.GetDimensions(width, height);
    uint2 dimensions = uint2(width, height);
    uint2 pixel = threadID.xy;
    if (any(pixel >= dimensions)) return;
    // The alpha dither and the guide writes index by dispatch thread, exactly as the ray-tracing backend
    // indexes by DispatchRaysIndex.
    stageSetDispatch(pixel);
    primaryPass(pixel, dimensions);
}

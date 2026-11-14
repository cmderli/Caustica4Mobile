// Ray-tracing backend entry point for ReSTIR PT spatial reuse and final shading. The body lives in
// restir.slang, which the ray-query compute entry dispatches as well.
import stage;
import restir;

[shader("raygeneration")]
void main() {
    restirSpatialPass(stageDispatchIndex(), stageDispatchDimensions());
}

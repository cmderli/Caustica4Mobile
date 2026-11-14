// Ray-tracing backend entry point for ReSTIR PT path generation + temporal reuse. The body lives in
// restir.slang, which the ray-query compute entry dispatches as well.
import stage;
import restir;

[shader("raygeneration")]
void main() {
    restirTemporalPass(stageDispatchIndex(), stageDispatchDimensions());
}

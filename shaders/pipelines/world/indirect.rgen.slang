// Ray-tracing backend entry point for the indirect/bounce pass. The body lives in indirect_pass.slang,
// which the ray-query compute entry dispatches as well.
import stage;
import indirect_pass;

[shader("raygeneration")]
void main() {
    indirectPass(stageDispatchIndex(), stageDispatchDimensions());
}

// Ray-tracing backend entry point for the primary/guide pass. The body lives in primary_pass.slang, which
// the ray-query compute entry dispatches as well.
import stage;
import primary_pass;

[shader("raygeneration")]
void main() {
    primaryPass(stageDispatchIndex(), stageDispatchDimensions());
}

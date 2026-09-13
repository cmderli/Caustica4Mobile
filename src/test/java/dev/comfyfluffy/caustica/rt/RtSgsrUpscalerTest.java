package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.pipeline.RtSgsrUpscaler;
import org.joml.Matrix4f;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The depth conversion SGSR2 needs: the guide stores this renderer's reversed-Z depth, while SGSR2 wants
 * standard [0,1] depth, so the near and far distances have to be recovered from the projection. Getting the
 * order wrong there would silently turn every depth comparison inside out.
 */
final class RtSgsrUpscalerTest {
    // Matrix4f.set takes (column, row), and this renderer's depth mapping reads the same two elements the
    // shaders read: m22 is the ndc-z scale and m32 the ndc-z constant term of a projection of the form
    // ndcZ = m32 / distance - m22.

    @Test
    void derivesBothPlanesFromAReversedZProjection() {
        // Guide depth 1 at 0.05 and 0 at 1000: m22 = near / (far - near), m32 = far * near / (far - near).
        Matrix4f projection = new Matrix4f().zero().set(0, 0, 1.0f).set(1, 1, 1.0f)
                .set(2, 2, 0.0000500025f).set(3, 2, 0.0500025f).set(2, 3, -1.0f);

        float[] bounds = RtSgsrUpscaler.depthBounds(projection);

        assertEquals(0.05f, bounds[0], 1.0e-4f);
        assertEquals(1000.0f, bounds[1], 0.1f);
    }

    @Test
    void fallsBackWhenTheFarPlaneIsInfinite() {
        // An infinite far plane drops the ndc-z constant term, so guide depth 0 names no plane at all and the
        // tracer's own reach has to stand in for the far distance.
        Matrix4f projection = new Matrix4f().zero().set(0, 0, 1.0f).set(1, 1, 1.0f)
                .set(2, 2, 0.0f).set(3, 2, 0.05f).set(2, 3, -1.0f);

        float[] bounds = RtSgsrUpscaler.depthBounds(projection);

        assertEquals(0.05f, bounds[0], 1.0e-4f);
        assertEquals(10000.0f, bounds[1], 1.0f);
    }

    @Test
    void fallsBackWhenTheProjectionIsDegenerate() {
        // A projection with no depth mapping at all must not hand the shader a bounds pair it would divide by.
        Matrix4f projection = new Matrix4f().zero();

        float[] bounds = RtSgsrUpscaler.depthBounds(projection);

        assertTrue(bounds[0] > 0.0f && bounds[1] > bounds[0]);
        assertEquals(0.05f, bounds[0], 1.0e-4f);
        assertEquals(10000.0f, bounds[1], 1.0f);
    }
}

package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.pipeline.RtPipeline;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The ray-query world pass is a Java/shader contract that no test can execute without the device it
 * exists for, so the parts that can be checked on any machine are pinned here: the packaged entry points
 * the backend names, and the workgroup edge its dispatch grid is computed from.
 */
final class RtWorldBackendTest {
    private static final String SHADER_DIR = "/caustica/shaders/pipelines/world/";
    private static final int SPIRV_MAGIC = 0x07230203;
    private static final int OP_EXECUTION_MODE = 16;
    private static final int EXECUTION_MODE_LOCAL_SIZE = 17;

    @Test
    void rayQueryEntryPointsArePackaged() throws IOException {
        for (String shader : rayQueryShaders()) {
            assertNotNull(read(shader), "missing packaged ray-query shader " + SHADER_DIR + shader);
        }
    }

    /**
     * The ray-tracing pipeline loads these raygen stages from the same resource directory. A missing
     * stage only fails when a device with VK_KHR_ray_tracing_pipeline selects that backend, so the
     * packaged set is pinned here.
     */
    @Test
    void rayTracingEntryPointsArePackaged() throws IOException {
        for (String shader : new String[]{"primary.rgen.spv", "indirect.rgen.spv",
                "restir_temporal.rgen.spv", "restir_spatial.rgen.spv"}) {
            assertNotNull(read(shader), "missing packaged ray-tracing shader " + SHADER_DIR + shader);
        }
    }

    /**
     * Every thread of the ray-query dispatch owns exactly the pixel at its dispatch id, so the grid is
     * ceil(size / {@link RtPipeline#RAY_QUERY_GROUP_SIZE}) and each entry point bounds-checks against the
     * render target. A group edge larger than the dispatch assumes would leave the bottom-right of the
     * image untouched, which is why the two are pinned together instead of being written down twice.
     */
    @Test
    void rayQueryWorkgroupMatchesTheDispatchShape() throws IOException {
        for (String shader : rayQueryShaders()) {
            int[] localSize = localSizeOf(read(shader));
            assertEquals(RtPipeline.RAY_QUERY_GROUP_SIZE, localSize[0],
                    shader + " workgroup X must match the dispatch grid");
            assertEquals(RtPipeline.RAY_QUERY_GROUP_SIZE, localSize[1],
                    shader + " workgroup Y must match the dispatch grid");
            assertEquals(1, localSize[2], shader + " is a 2D dispatch");
        }
    }

    private static String[] rayQueryShaders() {
        RtDeviceBringup.WorldBackend backend = RtDeviceBringup.WorldBackend.RAY_QUERY;
        return new String[]{backend.rayQueryPrimaryShader, backend.rayQueryMainShader,
                RtDeviceBringup.RAY_QUERY_RESTIR_TEMPORAL_SHADER,
                RtDeviceBringup.RAY_QUERY_RESTIR_SPATIAL_SHADER};
    }

    private static byte[] read(String shader) throws IOException {
        try (InputStream in = RtWorldBackendTest.class.getResourceAsStream(SHADER_DIR + shader)) {
            return in == null ? null : in.readAllBytes();
        }
    }

    /** Local size of the module's compute execution mode, as {x, y, z}. */
    private static int[] localSizeOf(byte[] spirv) {
        assertNotNull(spirv, "shader binary was not packaged");
        ByteBuffer words = ByteBuffer.wrap(spirv).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(SPIRV_MAGIC, words.getInt(0), "not a SPIR-V module");
        int bound = words.getInt(3);
        int offset = 5 * Integer.BYTES;
        while (offset + Integer.BYTES <= spirv.length) {
            int instruction = words.getInt(offset);
            int wordCount = instruction >>> 16;
            int opcode = instruction & 0xFFFF;
            assertTrue(wordCount > 0, "malformed SPIR-V instruction");
            if (opcode == OP_EXECUTION_MODE && wordCount >= 6
                    && words.getInt(offset + 2 * Integer.BYTES) == EXECUTION_MODE_LOCAL_SIZE) {
                return new int[]{
                        words.getInt(offset + 3 * Integer.BYTES),
                        words.getInt(offset + 4 * Integer.BYTES),
                        words.getInt(offset + 5 * Integer.BYTES)};
            }
            offset += wordCount * Integer.BYTES;
        }
        throw new AssertionError("no LocalSize execution mode in a compute module (bound " + bound + ")");
    }
}

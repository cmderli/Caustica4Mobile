package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.gen.RestirPrimaryData;
import dev.comfyfluffy.caustica.rt.gen.RestirReservoirData;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Layout and tuning invariants for the ReSTIR PT reservoirs: the Slang reflection supplies the byte
 * stride the host allocates, and the interactive settings must stay inside the range the shader's
 * neighbor loop and confidence cap are written for.
 */
final class RtRestirLayoutTest {
    @Test
    void reservoirStrideSupportsDeviceAddressIndexing() {
        assertEquals(96, RestirReservoirData.BYTE_SIZE,
                "the packed sample (88 B) + wSum/M (8 B) is the mobile reservoir budget");
        assertEquals(0, RestirReservoirData.BYTE_SIZE % 16,
                "the reservoir array stride must stay 16-byte aligned");
        assertTrue(RestirPrimaryData.BYTE_SIZE >= 48,
                "primary record must hold the surface, material, and motion state");
        assertEquals(0, RestirPrimaryData.BYTE_SIZE % 16,
                "the primary record is indexed through the same Std430 layout");
    }

    @Test
    void reuseSettingsAreClampedToTheShaderRanges() {
        CausticaConfig.IntSetting spatialSamples = CausticaConfig.Rt.Restir.SPATIAL_SAMPLES;
        CausticaConfig.IntSetting confidenceCap = CausticaConfig.Rt.Restir.CONFIDENCE_CAP;
        CausticaConfig.IntSetting spatialRadius = CausticaConfig.Rt.Restir.SPATIAL_RADIUS;
        int previousSamples = spatialSamples.value();
        int previousCap = confidenceCap.value();
        int previousRadius = spatialRadius.value();
        try {
            spatialSamples.set(99);
            assertEquals(8, spatialSamples.value());
            spatialSamples.set(-1);
            assertEquals(0, spatialSamples.value());

            confidenceCap.set(0);
            assertEquals(1, confidenceCap.value());
            confidenceCap.set(10_000);
            assertEquals(64, confidenceCap.value());

            spatialRadius.set(0);
            assertEquals(1, spatialRadius.value());
            spatialRadius.set(10_000);
            assertEquals(64, spatialRadius.value());
        } finally {
            spatialSamples.set(previousSamples);
            confidenceCap.set(previousCap);
            spatialRadius.set(previousRadius);
        }
    }
}

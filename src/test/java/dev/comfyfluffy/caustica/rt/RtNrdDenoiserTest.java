package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * Pins the NRD denoiser id the shim is built with. NRD's ABI takes the id, and the two denoisers read the
 * radiance lanes' hit-distance channel with different conventions, so this mapping is shared between the
 * native create call and the front-end packer: REBLUR where the device has
 * {@code VK_KHR_compute_shader_derivatives}, RELAX (which issues no quad operations) where it does not.
 */
final class RtNrdDenoiserTest {
    @Test
    void derivativesCapableDevicesGetReblur() {
        assertEquals(0, RtNrd.denoiserId(true), "0 is caustica_nrd's REBLUR_DIFFUSE_SPECULAR");
    }

    @Test
    void devicesWithoutDerivativesGetRelax() {
        assertEquals(1, RtNrd.denoiserId(false), "1 is caustica_nrd's RELAX_DIFFUSE_SPECULAR");
    }

    @Test
    void theTwoDenoisersUseDifferentHitDistanceConventions() {
        assertNotEquals(RtNrd.denoiserId(true), RtNrd.denoiserId(false),
                "the front end picks its hit-distance packing from this id, so the ids must differ");
    }
}

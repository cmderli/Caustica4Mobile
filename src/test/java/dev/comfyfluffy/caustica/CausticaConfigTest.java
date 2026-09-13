package dev.comfyfluffy.caustica;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class CausticaConfigTest {
    /**
     * The shutter angle is the setting the player sees; the shader only ever gets the fraction of the frame
     * interval it means, so the conversion and the angle that disables the streak are pinned here.
     */
    @Test
    void shutterAngleMapsToTheShutterFraction() {
        CausticaConfig.IntSetting angle = CausticaConfig.Rt.MotionBlur.SHUTTER_ANGLE;
        CausticaConfig.BooleanSetting enabled = CausticaConfig.Rt.MotionBlur.ENABLED;
        int previousAngle = angle.value();
        boolean previousEnabled = enabled.value();
        try {
            enabled.set(true);

            angle.set(360);
            assertEquals(1.0f, CausticaConfig.Rt.MotionBlur.shutterFraction(), 1.0e-6f);
            assertTrue(CausticaConfig.Rt.MotionBlur.active());

            angle.set(180);
            assertEquals(0.5f, CausticaConfig.Rt.MotionBlur.shutterFraction(), 1.0e-6f);

            // A closed shutter is not a blur: the pass must not run and change nothing.
            angle.set(0);
            assertEquals(0.0f, CausticaConfig.Rt.MotionBlur.shutterFraction(), 1.0e-6f);
            assertFalse(CausticaConfig.Rt.MotionBlur.active());

            angle.set(90);
            assertTrue(CausticaConfig.Rt.MotionBlur.active());
            enabled.set(false);
            assertFalse(CausticaConfig.Rt.MotionBlur.active(), "the option itself still gates the pass");
        } finally {
            angle.set(previousAngle);
            enabled.set(previousEnabled);
        }
    }

    /** The upscaler setting accepts SGSR2 and the cycle button offers it, since both come from one list. */
    @Test
    void upscalerAcceptsSgsr2AndRejectsUnknownValues() {
        CausticaConfig.StringSetting upscaler = CausticaConfig.Rt.Reconstruction.UPSCALER;
        String previous = upscaler.get();
        try {
            upscaler.set("sgsr");
            assertEquals("sgsr", upscaler.get());
            assertTrue(CausticaConfig.Rt.Reconstruction.UPSCALERS.contains("sgsr"));

            upscaler.set("nonsense");
            assertEquals("auto", upscaler.get());
        } finally {
            upscaler.set(previous);
        }
    }

    @Test
    void invalidPeakNitsFallsBackToDefault() {
        CausticaConfig.IntSetting setting = CausticaConfig.Rt.Hdr.PEAK_NITS;
        int previous = setting.value();
        try {
            setting.set(2000);
            assertEquals(2000, setting.value());

            setting.set(900);
            assertEquals(1000, setting.value());
        } finally {
            setting.set(previous);
        }
    }
}

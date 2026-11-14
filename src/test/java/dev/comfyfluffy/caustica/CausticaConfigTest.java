package dev.comfyfluffy.caustica;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
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

    /**
     * The reset path is only a loop over {@code restoreDefault()}, so every setting type has to answer with its
     * hardcoded default rather than with the value currently loaded from the file.
     */
    @Test
    void restoreDefaultIgnoresTheConfiguredValue() {
        CausticaConfig.IntSetting spp = CausticaConfig.Rt.Composite.SPP;
        CausticaConfig.BooleanSetting frameGeneration = CausticaConfig.Rt.Fg.ENABLED;
        CausticaConfig.FloatSetting gamma = CausticaConfig.Rt.Tonemap.GAMMA;
        CausticaConfig.StringSetting denoiser = CausticaConfig.Rt.Reconstruction.DENOISER;
        CausticaConfig.OptionalStringSetting nrdPath = CausticaConfig.Nrd.PATH;
        int previousSpp = spp.get();
        boolean previousFrameGeneration = frameGeneration.get();
        float previousGamma = gamma.get();
        String previousDenoiser = denoiser.get();
        String previousNrdPath = nrdPath.get();
        try {
            spp.set(8);
            frameGeneration.set(!frameGeneration.defaultValue());
            gamma.set(1.4f);
            denoiser.set("off");
            nrdPath.set("/tmp/does-not-matter.so");

            spp.restoreDefault();
            frameGeneration.restoreDefault();
            gamma.restoreDefault();
            denoiser.restoreDefault();
            nrdPath.restoreDefault();

            assertEquals(spp.defaultValue(), spp.get());
            assertEquals(frameGeneration.defaultValue(), frameGeneration.get());
            assertEquals(gamma.defaultValue(), gamma.get());
            assertEquals(denoiser.defaultValue(), denoiser.get());
            assertNull(nrdPath.get(), "an optional path falls back to unset");
        } finally {
            spp.set(previousSpp);
            frameGeneration.set(previousFrameGeneration);
            gamma.set(previousGamma);
            denoiser.set(previousDenoiser);
            nrdPath.set(previousNrdPath);
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

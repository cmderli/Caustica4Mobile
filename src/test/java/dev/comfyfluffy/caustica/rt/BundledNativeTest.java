package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which bundled build a JVM needs is not readable from {@code os.name}: an Android launcher runs a plain
 * OpenJDK, so the VM reports Linux and the game would be handed the glibc build of a shim — which the
 * platform cannot open at all. Android is identified by {@code os.version} (launchers set it to
 * {@code Android-…}) and by the {@code ANDROID_ROOT} the platform always exports, which is what these
 * tests pin.
 */
final class BundledNativeTest {
    private static final String ANDROID_ROOT = "ANDROID_ROOT";

    @Test
    void androidIsRecognisedFromTheOsVersionLaunchersSet() {
        withProperty("os.version", "Android-17", () -> assertTrue(BundledNative.android()));
    }

    @Test
    void aDesktopJvmIsNotMistakenForAndroid() {
        withProperty("os.version", "6.16.7-arch1-1", () -> assertFalse(BundledNative.android()));
    }

    @Test
    void theLibraryNameFollowsThePlatformConvention() {
        String name = new BundledNative("caustica-nrd", "caustica_nrd").libraryName();
        String os = System.getProperty("os.name", "").toLowerCase();
        assertEquals(os.contains("win") ? "caustica_nrd.dll" : "libcaustica_nrd.so", name);
    }

    private static void withProperty(String key, String value, Runnable assertion) {
        if (System.getenv(ANDROID_ROOT) != null || System.getenv("ANDROID_DATA") != null) {
            // The platform's own markers would decide the answer before os.version is read.
            return;
        }
        String previous = System.getProperty(key);
        try {
            System.setProperty(key, value);
            assertion.run();
        } finally {
            if (previous == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, previous);
            }
        }
    }
}

package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaMod;
import net.fabricmc.loader.api.FabricLoader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A native library that ships inside the mod jar and has to be unpacked before the platform's loader can
 * take it. Two things decide where that copy may live, and neither is a preference:
 *
 * <ul>
 *   <li><b>Which build.</b> Android reports itself as Linux, and launchers run the game on a plain
 *       OpenJDK, so neither {@code os.name} nor the VM name identifies it: the Android build of a shim
 *       links bionic's {@code libc.so}/{@code libm.so}, while the Linux one links {@code libc.so.6} and
 *       cannot be opened at all. {@link #android()} therefore also reads {@code os.version} (the launchers
 *       set {@code -Dos.version=Android-…}) and the {@code ANDROID_ROOT} the platform always exports.</li>
 *   <li><b>Where it may be loaded from.</b> Android's linker only opens files inside the app's own
 *       namespace: the external game directory a launcher hands Minecraft is refused
 *       ({@code not accessible for the namespace}, whose permitted paths are the app's private ones),
 *       while the JVM's temp directory — the app cache — is not. So the bundle is unpacked into several
 *       candidate roots and the caller keeps the first path that actually loads.</li>
 * </ul>
 */
public record BundledNative(String extractedDirName, String baseName) {
    private static final String RESOURCE_ROOT = "/caustica/natives/";

    /** The library file name this platform's loader expects. */
    public String libraryName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("win") ? baseName + ".dll" : "lib" + baseName + ".so";
    }

    /**
     * Which bundled build this JVM needs. Android is checked first because it identifies as Linux; the
     * arm64 targets share one directory layout and differ only in the libc they were linked against.
     */
    public String platformDir() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        boolean x64 = arch.equals("x86_64") || arch.equals("amd64");
        boolean arm64 = arch.equals("aarch64") || arch.equals("arm64");
        if (android() && arm64) {
            return "android-arm64";
        }
        if (os.contains("win") && x64) {
            return "windows-x64";
        }
        if (os.contains("linux") && x64) {
            return "linux-x64";
        }
        if (os.contains("linux") && arm64) {
            return "linux-arm64";
        }
        return os + "/" + arch;
    }

    static boolean android() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String version = System.getProperty("os.version", "").toLowerCase();
        String vm = System.getProperty("java.vm.name", "").toLowerCase();
        String runtime = System.getProperty("java.runtime.name", "").toLowerCase();
        // No single signal is reliable: launchers run a desktop OpenJDK (so no "dalvik"/"android" in the VM
        // name) and few of them override os.version, while the platform itself always exports ANDROID_ROOT
        // and its siblings to every process it starts.
        return os.contains("android") || version.contains("android") || vm.contains("dalvik")
                || runtime.contains("android")
                || System.getenv("ANDROID_ROOT") != null
                || System.getenv("ANDROID_DATA") != null
                || System.getenv("ANDROID_ASSETS") != null;
    }

    /**
     * Every path the library may be at, in the order they should be tried: an explicit override is
     * authoritative and stands alone, then this bundle unpacked into each writable root, then copies the
     * player put next to the game. A caller must try them in order and stop at the first that loads —
     * being present is not the same as being loadable, which is exactly what the Android namespace rule
     * makes of the game directory.
     */
    public List<Path> candidates(String override) {
        if (override != null && !override.isBlank()) {
            Path path = Path.of(override);
            return List.of(Files.isDirectory(path) ? path.resolve(libraryName()) : path);
        }
        List<Path> candidates = new ArrayList<>();
        for (Path root : extractionRoots()) {
            Path extracted = extract(root.resolve(extractedDirName).resolve("natives").resolve(platformDir()));
            if (extracted != null) {
                candidates.add(extracted);
            }
        }
        Path gameDir = FabricLoader.getInstance().getGameDir();
        candidates.add(gameDir.resolve("natives").resolve(libraryName()));
        candidates.add(gameDir.resolve(libraryName()));
        return candidates;
    }

    /**
     * Roots to unpack into. On Android that is the app cache alone — the game directory is external storage
     * its linker will not open a library from, so copying the build there would only waste space; the JVM's
     * temp directory is what {@code java.io.tmpdir} points at, inside the app's own namespace. On a desktop
     * the game directory stays first so the unpacked copy sits with the player's files, with the temp
     * directory as a fallback for a read-only game directory.
     */
    private List<Path> extractionRoots() {
        Path gameDir = FabricLoader.getInstance().getGameDir();
        Path temp = tempDir();
        List<Path> roots = new ArrayList<>();
        if (android()) {
            if (temp != null) {
                return List.of(temp);
            }
        } else {
            roots.add(gameDir);
            if (temp != null) {
                roots.add(temp);
            }
            return roots;
        }
        roots.add(gameDir);
        return roots;
    }

    private static Path tempDir() {
        String tmp = System.getProperty("java.io.tmpdir");
        return tmp == null || tmp.isBlank() ? null : Path.of(tmp);
    }

    /** Unpacks this bundle into {@code dir}, or null when the platform has no bundled build at all. */
    private Path extract(Path dir) {
        String resource = RESOURCE_ROOT + platformDir() + "/" + libraryName();
        byte[] bytes;
        try (InputStream in = BundledNative.class.getResourceAsStream(resource)) {
            if (in == null) {
                return null;
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("Could not read the bundled native {} from the jar", resource, e);
            return null;
        }
        Path library = dir.resolve(libraryName());
        try {
            Files.createDirectories(dir);
            if (!Files.isRegularFile(library) || !sameBytes(library, bytes)) {
                Files.write(library, bytes);
                // Android's loader maps the file executable; an app-private file created with the process
                // umask can end up unreadable to it on some devices. Best effort: a filesystem without
                // POSIX permissions simply keeps its defaults.
                try {
                    Files.setPosixFilePermissions(library, PosixFilePermissions.fromString("rwxr-xr-x"));
                } catch (UnsupportedOperationException | IOException ignored) {
                    // Not a POSIX filesystem, or the mode could not be set; the loader decides from here.
                }
            }
            return library;
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("Could not extract the bundled native {} to {}", libraryName(), dir, e);
            return null;
        }
    }

    private static boolean sameBytes(Path path, byte[] bytes) {
        try {
            return Files.size(path) == bytes.length && Arrays.equals(Files.readAllBytes(path), bytes);
        } catch (IOException e) {
            return false;
        }
    }
}

package dev.comfyfluffy.caustica.rt;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Objects;

/**
 * Minimal uncompressed scanline OpenEXR writer for RGBA half-float screenshots.
 *
 * <p>Keeping this in Java makes F2 capture self-contained: neither Python nor {@code uv} is needed while
 * Minecraft is running. The UV environment remains the reproducible workstation for inspecting and
 * processing the resulting files.
 */
final class RtOpenExrWriter {
    private static final int EXR_MAGIC = 20_000_630;
    private static final int EXR_VERSION = 2;
    private static final int HALF = 1;
    private static final int NO_COMPRESSION = 0;
    private static final DateTimeFormatter CAPTURE_DATE = DateTimeFormatter.ofPattern("yyyy:MM:dd HH:mm:ss");
    private static final int[] CHANNEL_COMPONENT = {3, 2, 1, 0}; // A, B, G, R (lexicographic channel order)
    private static final String[] CHANNEL_NAMES = {"A", "B", "G", "R"};

    private RtOpenExrWriter() {
    }

    record Metadata(
            float preExposure,
            float residualExposure,
            float absoluteExposure,
            String exposureMode,
            float evScene,
            float evTarget,
            float evApplied,
            String look,
            long frame
    ) {
        Metadata {
            Objects.requireNonNull(exposureMode, "exposureMode");
            Objects.requireNonNull(look, "look");
        }
    }

    /**
     * Writes RGBA half values whose rows are in Vulkan image order (row zero is the bottom row).
     * EXR scanline zero is the top row, so scanlines are reversed while writing.
     */
    static void write(Path output, int width, int height, short[] rgba, Metadata metadata) throws IOException {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(rgba, "rgba");
        Objects.requireNonNull(metadata, "metadata");
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("EXR dimensions must be positive: " + width + "x" + height);
        }
        int pixelCount = Math.multiplyExact(width, height);
        if (rgba.length != Math.multiplyExact(pixelCount, 4)) {
            throw new IllegalArgumentException("Expected " + (pixelCount * 4) + " RGBA samples, got " + rgba.length);
        }

        byte[] header = header(width, height, metadata);
        long rowDataBytes = Math.multiplyExact((long) width, 8L);
        long scanlineBlockBytes = Math.addExact(8L, rowDataBytes);
        long firstScanlineOffset = Math.addExact(header.length, Math.multiplyExact((long) height, 8L));

        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (OutputStream raw = Files.newOutputStream(output, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
             BufferedOutputStream stream = new BufferedOutputStream(raw, 1 << 20)) {
            stream.write(header);
            for (int y = 0; y < height; y++) {
                writeLongLe(stream, Math.addExact(firstScanlineOffset, Math.multiplyExact((long) y, scanlineBlockBytes)));
            }

            byte[] row = new byte[Math.toIntExact(rowDataBytes)];
            for (int y = 0; y < height; y++) {
                writeIntLe(stream, y);
                writeIntLe(stream, row.length);
                int sourceRow = height - 1 - y;
                int cursor = 0;
                for (int component : CHANNEL_COMPONENT) {
                    int source = (sourceRow * width * 4) + component;
                    for (int x = 0; x < width; x++, source += 4) {
                        short bits = rgba[source];
                        row[cursor++] = (byte) bits;
                        row[cursor++] = (byte) (bits >>> 8);
                    }
                }
                stream.write(row);
            }
        }
    }

    private static byte[] header(int width, int height, Metadata metadata) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream(1024);
        writeIntLe(bytes, EXR_MAGIC);
        writeIntLe(bytes, EXR_VERSION);

        ByteArrayOutputStream channels = new ByteArrayOutputStream();
        for (String name : CHANNEL_NAMES) {
            writeCString(channels, name);
            writeIntLe(channels, HALF);
            channels.write(0); // pLinear
            channels.write(0);
            channels.write(0);
            channels.write(0);
            writeIntLe(channels, 1); // xSampling
            writeIntLe(channels, 1); // ySampling
        }
        channels.write(0);
        attribute(bytes, "channels", "chlist", channels.toByteArray());
        attribute(bytes, "compression", "compression", new byte[]{NO_COMPRESSION});
        attribute(bytes, "dataWindow", "box2i", box2i(width, height));
        attribute(bytes, "displayWindow", "box2i", box2i(width, height));
        attribute(bytes, "lineOrder", "lineOrder", new byte[]{0});
        attribute(bytes, "pixelAspectRatio", "float", floats(1.0f));
        attribute(bytes, "screenWindowCenter", "v2f", floats(0.0f, 0.0f));
        attribute(bytes, "screenWindowWidth", "float", floats(1.0f));

        // ACEScg/AP1 primaries and ACES white (D60). This is the standard EXR chromaticities attribute,
        // so color-managed applications do not have to infer the working space from the filename.
        attribute(bytes, "chromaticities", "chromaticities", floats(
                0.713f, 0.293f,
                0.165f, 0.830f,
                0.128f, 0.044f,
                0.32168f, 0.33767f));
        attribute(bytes, "adoptedNeutral", "v2f", floats(0.32168f, 0.33767f));

        OffsetDateTime now = OffsetDateTime.now();
        stringAttribute(bytes, "capDate", CAPTURE_DATE.format(now));
        floatAttribute(bytes, "utcOffset", now.getOffset().getTotalSeconds());
        stringAttribute(bytes, "software", "Caustica for Mobile");
        stringAttribute(bytes, "comments",
                "Residual-exposed scene-linear ACEScg; before Look/LMT, ACES output transform, and UI");
        stringAttribute(bytes, "causticaColorSpace", "ACEScg (AP1/D60), scene-linear");
        stringAttribute(bytes, "causticaEncoding",
                "RGB = sceneLinear * preExposure * residualExposure");
        stringAttribute(bytes, "causticaRecovery", "sceneLinear = RGB / causticaAbsoluteExposure");
        floatAttribute(bytes, "causticaPreExposure", metadata.preExposure());
        floatAttribute(bytes, "causticaResidualExposure", metadata.residualExposure());
        floatAttribute(bytes, "causticaAbsoluteExposure", metadata.absoluteExposure());
        stringAttribute(bytes, "causticaExposureMode", metadata.exposureMode());
        finiteFloatAttribute(bytes, "causticaEvScene", metadata.evScene());
        finiteFloatAttribute(bytes, "causticaEvTarget", metadata.evTarget());
        finiteFloatAttribute(bytes, "causticaEvApplied", metadata.evApplied());
        stringAttribute(bytes, "causticaLookIntent", metadata.look());
        stringAttribute(bytes, "causticaFrame", Long.toUnsignedString(metadata.frame()));
        bytes.write(0); // end of header attributes
        return bytes.toByteArray();
    }

    private static byte[] box2i(int width, int height) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream(16);
        writeIntLe(bytes, 0);
        writeIntLe(bytes, 0);
        writeIntLe(bytes, width - 1);
        writeIntLe(bytes, height - 1);
        return bytes.toByteArray();
    }

    private static byte[] floats(float... values) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream(values.length * Float.BYTES);
        for (float value : values) {
            writeIntLe(bytes, Float.floatToRawIntBits(value));
        }
        return bytes.toByteArray();
    }

    private static void floatAttribute(OutputStream output, String name, float value) throws IOException {
        attribute(output, name, "float", floats(value));
    }

    private static void finiteFloatAttribute(OutputStream output, String name, float value) throws IOException {
        if (Float.isFinite(value)) {
            floatAttribute(output, name, value);
        }
    }

    private static void stringAttribute(OutputStream output, String name, String value) throws IOException {
        attribute(output, name, "string", value.getBytes(StandardCharsets.UTF_8));
    }

    private static void attribute(OutputStream output, String name, String type, byte[] value) throws IOException {
        writeCString(output, name);
        writeCString(output, type);
        writeIntLe(output, value.length);
        output.write(value);
    }

    private static void writeCString(OutputStream output, String value) throws IOException {
        output.write(value.getBytes(StandardCharsets.US_ASCII));
        output.write(0);
    }

    private static void writeIntLe(OutputStream output, int value) throws IOException {
        output.write(value);
        output.write(value >>> 8);
        output.write(value >>> 16);
        output.write(value >>> 24);
    }

    private static void writeLongLe(OutputStream output, long value) throws IOException {
        writeIntLe(output, (int) value);
        writeIntLe(output, (int) (value >>> 32));
    }
}

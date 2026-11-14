package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import net.fabricmc.loader.api.FabricLoader;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.KHRSynchronization2;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkQueryPoolCreateInfo;
import org.lwjgl.vulkan.VkQueueFamilyProperties;

/**
 * On-device GPU time per frame stage, measured with Vulkan timestamp queries.
 *
 * <p>{@link RtFrameStats} times how long the CPU spends recording a stage, which says nothing about a pass
 * that is GPU-bound — a single path-tracing dispatch or a denoiser records in microseconds and runs for
 * tens of milliseconds. This class measures the executed cost instead: one timestamp pair per stage around
 * the recorded work, read back once that frame's graphics timeline token completes.
 *
 * <p>Queries live in a fixed ring of frame slots. A slot is only rewritten after the frame that used it has
 * completed, so no readback ever waits on the GPU. A frame whose ring slot is still in flight is simply not
 * profiled, which keeps the measurement free of stalls on slow devices.
 *
 * <p>Nested stages are reported as recorded: {@code recon} covers the whole reconstruction chain, and the
 * NRD and SGSR entries inside it break that envelope down. Their times are not additive with it.
 */
public final class RtGpuProfile {
    /**
     * Measured stages. Each owns two query indices and is reported only in frames where both were written,
     * so a stage that did not run reads as absent rather than as zero.
     */
    public enum Stage {
        FRAME("frame"),
        TLAS("tlas"),
        SKY_LUT("sky-lut"),
        PRIMARY("trace.primary"),
        INDIRECT("trace.indirect"),
        RESTIR_TEMPORAL("restir.temporal"),
        RESTIR_FALLBACK("restir.fallback"),
        RESTIR_SPATIAL("restir.spatial"),
        RECON("recon"),
        NRD_FRONTEND("nrd.front"),
        NRD_CORE("nrd.core"),
        NRD_BACKEND("nrd.back"),
        SGSR_CONVERT("sgsr.convert"),
        SGSR_ACTIVATE("sgsr.activate"),
        SGSR_UPSCALE("sgsr.upscale"),
        MOTION_BLUR("motion-blur"),
        EXPOSURE("exposure"),
        BLOOM("bloom"),
        DISPLAY_MAP("display"),
        COPY_OUTPUT("copy");

        private final String label;

        Stage(String label) {
            this.label = label;
        }

        /** Log/CSV column name for this stage, e.g. {@code trace.indirect}. */
        String label() {
            return label;
        }

        int startMarker() {
            return ordinal() * 2;
        }

        int endMarker() {
            return ordinal() * 2 + 1;
        }

        long markerBits() {
            return (1L << startMarker()) | (1L << endMarker());
        }
    }

    private static final Stage[] STAGES = Stage.values();
    /** Query indices per frame, two per stage. */
    static final int MARKERS = STAGES.length * 2;
    /** Frame slots in the ring. Four frames of history covers even a 1 fps device without a stall. */
    private static final int SLOTS = 4;
    private static final long LOG_INTERVAL_NANOS = 2_000_000_000L;
    /** A slot older than this never completed; dropping it keeps profiling alive after a cancelled frame. */
    private static final long STALE_NANOS = 5_000_000_000L;

    // Declared after the static tables above: the constructor sizes its per-stage arrays from STAGES and
    // MARKERS, which are not initialized yet while this line runs if it appears before them.
    public static final RtGpuProfile INSTANCE = new RtGpuProfile();

    private RtContext ctx;
    private long queryPool;
    private double timestampPeriodNanos = 1.0;
    private int timestampValidBits = 64;
    private boolean unavailable;
    private boolean reportedInitialization;

    private int nextSlot;
    private int currentSlot = -1;
    private long currentMarkers;
    private final long[] slotGraphicsValue = new long[SLOTS];
    private final long[] slotMarkers = new long[SLOTS];
    private final long[] slotRecordedNanos = new long[SLOTS];
    private final boolean[] slotRecorded = new boolean[SLOTS];
    private final long[] values = new long[MARKERS];

    private long lastStaleWarnNanos;
    private long lastNoSampleWarnNanos;
    private int lastQueryResult;

    private final double[] windowSumNanos = new double[STAGES.length];
    private final int[] windowCount = new int[STAGES.length];
    private long windowFrames;
    private long windowStartNanos;

    private PrintWriter csv;
    private boolean csvOpenAttempted;
    private long csvFrame;

    private RtGpuProfile() {
    }

    public static boolean enabled() {
        return CausticaConfig.Rt.FrameStats.GPU.value();
    }

    /** Start this frame's timestamp block. A frame whose ring slot is still in flight is skipped. */
    public void beginFrame(VkCommandBuffer cmd, RtGpuExecutor.GraphicsUse graphicsUse) {
        currentSlot = -1;
        currentMarkers = 0L;
        if (!enabled() || !ensureCreated()) {
            return;
        }
        int slot = nextSlot;
        nextSlot = (nextSlot + 1) % SLOTS;
        if (slotRecorded[slot]) {
            return;
        }
        slotRecorded[slot] = true;
        slotGraphicsValue[slot] = graphicsUse.value();
        slotMarkers[slot] = 0L;
        slotRecordedNanos[slot] = System.nanoTime();
        // Host reset (not vkCmdResetQueryPool): the slot's previous frame has completed, and this is the form
        // Minecraft's own GPU timer uses on this backend.
        VK12.vkResetQueryPool(ctx.vk(), queryPool, slot * MARKERS, MARKERS);
        currentSlot = slot;
    }

    public void markStart(VkCommandBuffer cmd, Stage stage) {
        mark(cmd, stage, false);
    }

    public void markEnd(VkCommandBuffer cmd, Stage stage) {
        mark(cmd, stage, true);
    }

    private void mark(VkCommandBuffer cmd, Stage stage, boolean end) {
        if (currentSlot < 0) {
            return;
        }
        int marker = end ? stage.endMarker() : stage.startMarker();
        long markerBit = 1L << marker;
        if ((currentMarkers & markerBit) != 0L) {
            // A query is reset once per frame; writing the same index again is undefined behaviour.
            return;
        }
        // The synchronization2 form, matching what Minecraft's own writeTimestamp records into this backend.
        KHRSynchronization2.vkCmdWriteTimestamp2KHR(cmd,
                KHRSynchronization2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR, queryPool,
                currentSlot * MARKERS + marker);
        currentMarkers |= 1L << marker;
        slotMarkers[currentSlot] = currentMarkers;
    }

    /**
     * Read back every completed slot, then log and write the window once it has been open long enough.
     * Runs even when the option has been switched off mid-session so a slot recorded before that is drained
     * instead of blocking its ring position forever.
     */
    public void collect(RtContext context) {
        if (queryPool == 0L) {
            return;
        }
        long completed = context.gpuExecutor().completedGraphicsValue();
        long now = System.nanoTime();
        for (int slot = 0; slot < SLOTS; slot++) {
            if (!slotRecorded[slot]) {
                continue;
            }
            if (slotGraphicsValue[slot] > completed) {
                if (now - slotRecordedNanos[slot] < STALE_NANOS) {
                    continue;
                }
                // The frame was recorded but its token never completed (cancelled frame, lost device), or the
                // driver never made its queries available. Dropping the slot keeps the ring alive, and the
                // warning is what distinguishes "no profiling output" from "profiling ran and found frames".
                if (now - lastStaleWarnNanos >= STALE_NANOS) {
                    lastStaleWarnNanos = now;
                    CausticaMod.LOGGER.warn(
                            "RT GPU profiling: frame token {} did not complete after {} ms (graphics={}); "
                                    + "dropping its slot",
                            slotGraphicsValue[slot], STALE_NANOS / 1_000_000L, completed);
                }
                slotRecorded[slot] = false;
                continue;
            }
            // A token can complete before the driver exposes the query results; keep the slot pending in that
            // case and retry on the next frame instead of losing the sample.
            if (readSlot(context, slot)) {
                slotRecorded[slot] = false;
            }
        }
        if (windowFrames > 0 && now - windowStartNanos >= LOG_INTERVAL_NANOS) {
            logWindow(now);
        } else if (windowFrames == 0 && now - lastNoSampleWarnNanos >= STALE_NANOS) {
            // A profiling run with no samples is a bug report on its own: say why instead of writing nothing.
            lastNoSampleWarnNanos = now;
            StringBuilder pending = new StringBuilder();
            int pendingSlots = 0;
            for (int slot = 0; slot < SLOTS; slot++) {
                if (slotRecorded[slot]) {
                    pendingSlots++;
                    pending.append(pendingSlots == 1 ? "" : ",").append(slotGraphicsValue[slot]);
                }
            }
            CausticaMod.LOGGER.warn(
                    "RT GPU profiling: no samples yet after {} s (pending slots=[{}], graphics completed={}, "
                            + "last vkGetQueryPoolResults={})",
                    STALE_NANOS / 1_000_000_000L, pending, completed, lastQueryResult);
        }
    }

    /**
     * Returns false while the frame's executed markers are still unavailable, so the caller retries the same
     * slot. Only markers that were actually written are read: a stage that did not run leaves its queries
     * reset but never executed, and a range read that includes one of those reports VK_NOT_READY for the whole
     * range on the drivers tested, so each written marker is fetched on its own.
     */
    private boolean readSlot(RtContext context, int slot) {
        long mask = slotMarkers[slot];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            var single = stack.mallocLong(1);
            for (int marker = 0; marker < MARKERS; marker++) {
                if ((mask & (1L << marker)) == 0L) {
                    continue; // this marker was never written in that frame
                }
                int rc = VK10.vkGetQueryPoolResults(context.vk(), queryPool, slot * MARKERS + marker, 1,
                        single, Long.BYTES, VK10.VK_QUERY_RESULT_64_BIT);
                lastQueryResult = rc;
                if (rc == VK10.VK_NOT_READY) {
                    return false; // still executing; retry on a later frame
                }
                if (rc != VK10.VK_SUCCESS) {
                    unavailable = true;
                    CausticaMod.LOGGER.warn("RT GPU profiling disabled: vkGetQueryPoolResults failed ({})", rc);
                    return true;
                }
                values[marker] = single.get(0);
            }
        } catch (Throwable t) {
            unavailable = true;
            CausticaMod.LOGGER.warn("RT GPU profiling disabled after a readback failure", t);
            return true;
        }
        double[] frameNanos = new double[STAGES.length];
        for (Stage stage : STAGES) {
            if ((mask & stage.markerBits()) != stage.markerBits()) {
                continue;
            }
            long startTime = maskTimestamp(values[stage.startMarker()]);
            long endTime = maskTimestamp(values[stage.endMarker()]);
            long ticks = endTime - startTime;
            if (ticks < 0L) {
                ticks = 0L;
            }
            double nanos = ticks * timestampPeriodNanos;
            frameNanos[stage.ordinal()] = nanos;
            windowSumNanos[stage.ordinal()] += nanos;
            windowCount[stage.ordinal()]++;
        }
        windowFrames++;
        if (windowStartNanos == 0L) {
            windowStartNanos = System.nanoTime();
        }
        writeCsvRow(frameNanos);
        return true;
    }

    /**
     * Timestamps wider than the queue family's valid bits wrap in the low mask on devices that report a
     * narrower counter, so every raw value is folded to its valid range before the delta is taken.
     */
    private long maskTimestamp(long value) {
        if (timestampValidBits >= 64) {
            return value;
        }
        return value & ((1L << timestampValidBits) - 1L);
    }

    private void logWindow(long now) {
        double seconds = Math.max((now - windowStartNanos) / 1.0e9, 1.0e-9);
        double fps = windowFrames / seconds;
        double frame = windowSumNanos[Stage.FRAME.ordinal()] / Math.max(windowCount[Stage.FRAME.ordinal()], 1)
                / 1.0e6;
        StringBuilder line = new StringBuilder("RT GPU (").append(windowFrames).append(" frames, ")
                .append(String.format(Locale.ROOT, "%.1f", fps)).append(" fps, frame ")
                .append(String.format(Locale.ROOT, "%.1f", frame)).append(" ms):");
        for (Stage stage : STAGES) {
            if (stage == Stage.FRAME || windowCount[stage.ordinal()] == 0) {
                continue;
            }
            double average = windowSumNanos[stage.ordinal()] / windowCount[stage.ordinal()] / 1.0e6;
            line.append(' ').append(stage.label).append('=')
                    .append(String.format(Locale.ROOT, "%.2f", average)).append("ms");
        }
        CausticaMod.LOGGER.info(line.toString());
        java.util.Arrays.fill(windowSumNanos, 0.0);
        java.util.Arrays.fill(windowCount, 0);
        windowFrames = 0;
        windowStartNanos = now;
    }

    private void writeCsvRow(double[] frameNanos) {
        ensureCsv();
        if (csv == null) {
            return;
        }
        StringBuilder row = new StringBuilder();
        row.append(csvFrame++);
        for (Stage stage : STAGES) {
            row.append(',').append(String.format(Locale.ROOT, "%.3f", frameNanos[stage.ordinal()] / 1.0e6));
        }
        csv.println(row);
        csv.flush();
    }

    private void ensureCsv() {
        if (csvOpenAttempted) {
            return;
        }
        csvOpenAttempted = true;
        Path dir = FabricLoader.getInstance().getGameDir().resolve("rt-frame-stats");
        Path file = dir.resolve("gpu.csv");
        try {
            Files.createDirectories(dir);
            csv = new PrintWriter(new BufferedWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8)));
            StringBuilder header = new StringBuilder("frame");
            for (Stage stage : STAGES) {
                header.append(',').append(stage.label).append("Ms");
            }
            csv.println(header);
            csv.flush();
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("RtGpuProfile: failed to open CSV {}: {}", file, e.toString());
            csv = null;
        }
    }

    private boolean ensureCreated() {
        if (queryPool != 0L) {
            return true;
        }
        if (unavailable) {
            return false;
        }
        RtContext context = RtContext.currentOrNull();
        if (context == null) {
            return false;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.calloc(stack);
            VK10.vkGetPhysicalDeviceProperties(context.vk().getPhysicalDevice(), properties);
            timestampPeriodNanos = properties.limits().timestampPeriod();
            int graphicsFamily = context.device().graphicsQueue().queueFamilyIndex();
            var count = stack.callocInt(1);
            VK10.vkGetPhysicalDeviceQueueFamilyProperties(context.vk().getPhysicalDevice(), count, null);
            VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.calloc(count.get(0), stack);
            VK10.vkGetPhysicalDeviceQueueFamilyProperties(context.vk().getPhysicalDevice(), count, families);
            timestampValidBits = families.get(graphicsFamily).timestampValidBits();
            if (timestampValidBits == 0 || !(timestampPeriodNanos > 0.0)) {
                unavailable = true;
                CausticaMod.LOGGER.info("RT GPU profiling unavailable: the graphics queue reports no timestamps");
                return false;
            }
            VkQueryPoolCreateInfo create = VkQueryPoolCreateInfo.calloc(stack).sType$Default()
                    .queryType(VK10.VK_QUERY_TYPE_TIMESTAMP).queryCount(SLOTS * MARKERS);
            var pool = stack.mallocLong(1);
            int rc = VK10.vkCreateQueryPool(context.vk(), create, null, pool);
            if (rc != VK10.VK_SUCCESS) {
                unavailable = true;
                CausticaMod.LOGGER.warn("RT GPU profiling unavailable: vkCreateQueryPool failed ({})", rc);
                return false;
            }
            queryPool = pool.get(0);
            RtDebugLabels.name(context, VK10.VK_OBJECT_TYPE_QUERY_POOL, queryPool, "RT GPU stage timestamps");
            ctx = context;
            if (!reportedInitialization) {
                reportedInitialization = true;
                CausticaMod.LOGGER.info(
                        "RT GPU profiling active: {} stages, {} frame slots, timestampPeriod={} ns, validBits={}",
                        STAGES.length, SLOTS, timestampPeriodNanos, timestampValidBits);
            }
            return true;
        } catch (Throwable t) {
            unavailable = true;
            CausticaMod.LOGGER.warn("RT GPU profiling unavailable", t);
            return false;
        }
    }

    /** Release the query pool. The caller has already made the device idle. */
    public void destroy() {
        if (queryPool != 0L && ctx != null) {
            VK10.vkDestroyQueryPool(ctx.vk(), queryPool, null);
        }
        queryPool = 0L;
        ctx = null;
        currentSlot = -1;
        nextSlot = 0;
        java.util.Arrays.fill(slotRecorded, false);
        java.util.Arrays.fill(windowSumNanos, 0.0);
        java.util.Arrays.fill(windowCount, 0);
        windowFrames = 0;
        windowStartNanos = 0L;
        if (csv != null) {
            csv.close();
            csv = null;
        }
        csvOpenAttempted = false;
    }
}

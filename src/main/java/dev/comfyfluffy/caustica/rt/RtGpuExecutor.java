package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaMod;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import com.mojang.blaze3d.vulkan.VulkanQueue;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkCommandBufferSubmitInfo;
import org.lwjgl.vulkan.VkSemaphoreCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreSubmitInfo;
import org.lwjgl.vulkan.VkSemaphoreTypeCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreWaitInfo;
import org.lwjgl.vulkan.VkSubmitInfo2;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

import static org.lwjgl.vulkan.KHRSynchronization2.VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
import static org.lwjgl.vulkan.KHRSynchronization2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
import static org.lwjgl.vulkan.KHRSynchronization2.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
import static org.lwjgl.vulkan.KHRSynchronization2.VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

/**
 * Single-owner asynchronous GPU submission lane on a queue reserved by Caustica at device creation.
 * Minecraft never fetches or submits to this queue index, so the executor can satisfy Vulkan's external
 * queue-synchronization rule without coordinating a host mutex with Blaze3D.
 */
public final class RtGpuExecutor {
    private static final int MAX_BUILD_BATCH = 32;
    private static final Job STOP = new Job(null, null, null, null, null);
    private static final Job WAKE = new Job(null, null, null, null, null);
    // Stages that consume what an asynchronous terrain build publishes: the acceleration-structure builds
    // that chain onto it, and the world pass itself — a ray-tracing dispatch, or the ray-query compute
    // dispatch that replaces it on devices without a ray-tracing pipeline.
    private static final long TERRAIN_READ_STAGES =
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                    | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
                    | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;

    private final RtContext ctx;
    private final VulkanQueue computeQueue;
    private final long buildTimeline;
    /**
     * Frame-path bound for a timeline wait. Long enough that ordinary load never reaches it (a frame's work
     * is milliseconds), short enough that a wedged queue degrades the renderer instead of freezing the client.
     */
    private static final long FRAME_WAIT_TIMEOUT_MILLIS = 10_000L;

    /** Bound for a build this executor submitted itself; also what stops its thread from wedging shutdown. */
    private static final long BUILD_WAIT_TIMEOUT_MILLIS = 10_000L;

    private final long graphicsTimeline;
    private boolean destroysAbandoned;
    private final LinkedBlockingQueue<Job> jobs = new LinkedBlockingQueue<>();
    private final AtomicLong nextBuildValue = new AtomicLong();
    private final AtomicLong pendingPublishWaitValue = new AtomicLong();
    private final AtomicLong nextGraphicsValue = new AtomicLong();
    private final AtomicLong latestGraphicsUseValue = new AtomicLong();
    private final ArrayList<DestroyJob> destroyJobs = new ArrayList<>();
    private final Object submissionLock = new Object();
    private long submittedBuildValue;
    private final Thread thread;
    private long commandPool;
    private volatile boolean closed;
    private volatile Throwable executorFailure;

    RtGpuExecutor(RtContext ctx) {
        this.ctx = ctx;
        this.computeQueue = ctx.computeQueue();
        this.buildTimeline = createTimeline("RT terrain build timeline");
        this.graphicsTimeline = createTimeline("RT graphics-use timeline");
        createCommandPool();
        this.thread = new Thread(this::run, "Caustica GPU executor");
        this.thread.setDaemon(true);
        this.thread.start();
    }

    /**
     * Enqueue GPU recording. On success, {@code afterSuccess} runs after timeline completion; then
     * {@code finished} receives exactly one terminal success/failure notification on this thread.
     */
    public Build submit(Consumer<VkCommandBuffer> record, Runnable afterSuccess,
                        BiConsumer<Build, Throwable> finished) {
        return submit(() -> false, record, afterSuccess, finished);
    }

    /**
     * Enqueue cancellable GPU recording. Cancellation is sampled immediately before a queued job enters
     * a command-buffer batch; already-recording or submitted work still completes normally.
     */
    public synchronized Build submit(BooleanSupplier cancelled, Consumer<VkCommandBuffer> record,
                                     Runnable afterSuccess, BiConsumer<Build, Throwable> finished) {
        checkExecutorFailure();
        if (closed) {
            throw new IllegalStateException("RT GPU executor is closed");
        }
        long value = nextBuildValue.incrementAndGet();
        Build build = new Build(value);
        jobs.add(new Job(cancelled, record, afterSuccess, finished, build));
        return build;
    }

    /** Mark a build visible to publication; the next graphics frame use waits on it. */
    public void markPublished(Build build) {
        assertRenderThread();
        pendingPublishWaitValue.accumulateAndGet(build.value, Math::max);
    }

    /** Attach published-build waits and reserve the completion token shared by this frame's RT resources. */
    public GraphicsUse beginGraphicsUse(VulkanCommandEncoder encoder) {
        assertRenderThread();
        checkExecutorFailure();
        long waitValue = pendingPublishWaitValue.get();
        if (waitValue != 0L) {
            // vkQueuePresentKHR requires every transitive signal dependency of its binary wait to have
            // already been submitted. A Build is assigned its timeline value when queued on this Java
            // executor, so do not expose that future value to the graphics/present chain prematurely.
            awaitBuildSubmission(waitValue);
            encoder.waitSemaphore(buildTimeline, waitValue, TERRAIN_READ_STAGES);
        }
        return new GraphicsUse(nextGraphicsValue.incrementAndGet());
    }

    /** Signal the frame token after its final terrain, TLAS, entity, and overlay consumer. */
    public void endGraphicsUse(VulkanCommandEncoder encoder, GraphicsUse graphicsUse) {
        assertRenderThread();
        encoder.signalSemaphore(graphicsTimeline, graphicsUse.value, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR);
        latestGraphicsUseValue.accumulateAndGet(graphicsUse.value, Math::max);
        if (hasPendingDestroys()) {
            jobs.offer(WAKE);
        }
    }

    /** Create a waiter that shares one completed-value snapshot across several resource reuse checks. */
    public GraphicsUseWaiter graphicsUseWaiter() {
        assertRenderThread();
        checkExecutorFailure();
        return new GraphicsUseWaiter(queryTimeline(graphicsTimeline));
    }

    /** Latest recorded frame token that can reference currently published RT state. */
    public GraphicsUse latestGraphicsUse() {
        assertRenderThread();
        return new GraphicsUse(latestGraphicsUseValue.get());
    }

    /** Highest frame token the graphics queue has completed; a non-blocking snapshot of the timeline. */
    public long completedGraphicsValue() {
        return queryTimeline(graphicsTimeline);
    }

    /** Rethrow a latched executor failure on the calling thread. */
    public void throwIfFailed() {
        checkExecutorFailure();
    }

    /** Destroy an owner once the specified frame token has completed. */
    public void retireAfterGraphics(GraphicsUse lastUse, Runnable destroy) {
        enqueueDestroyAfterGraphicsValue(lastUse.value, destroy);
    }

    /** Destroy a tracked owner once its exact last frame use has completed. */
    public void retireAfterGraphics(TrackedGraphicsUse trackedUse, Runnable destroy) {
        assertRenderThread();
        enqueueDestroyAfterGraphicsValue(trackedUse.value, destroy);
    }

    private void enqueueDestroyAfterGraphicsValue(long lastUseValue, Runnable destroy) {
        checkExecutorFailure();
        synchronized (destroyJobs) {
            destroyJobs.add(new DestroyJob(lastUseValue, destroy));
        }
        jobs.offer(WAKE);
    }

    /** Queue destruction of a completed build result that was never visible to graphics. */
    public void retireUnpublished(Runnable destroy) {
        enqueueDestroyAfterGraphicsValue(0L, destroy);
    }

    public boolean hasPendingDestroys() {
        synchronized (destroyJobs) {
            return !destroyJobs.isEmpty();
        }
    }

    /**
     * Bounded equivalent of the caller's device-idle wait, used by teardown.
     *
     * <p>It waits on this module's own two timelines rather than the whole device: a device-wide wait cannot
     * be interrupted, while these two cover every command the RT path has submitted (frame work signals the
     * graphics timeline at the end of Minecraft's own submission, async builds signal their own). Values are
     * re-read after each wait so work submitted while waiting is included too.
     */
    public boolean awaitIdleBounded(long timeoutMillis) {
        long deadline = System.nanoTime() + timeoutMillis * 1_000_000L;
        for (int attempt = 0; attempt < 3; attempt++) {
            long graphicsValue = queryTimeline(graphicsTimeline);
            long buildValue = queryTimeline(buildTimeline);
            if (!waitTimeline(graphicsTimeline, graphicsValue, remainingMillis(deadline))
                    || !waitTimeline(buildTimeline, buildValue, remainingMillis(deadline))) {
                return false;
            }
            if (queryTimeline(graphicsTimeline) == graphicsValue && queryTimeline(buildTimeline) == buildValue) {
                return true;
            }
        }
        return true;
    }

    private static long remainingMillis(long deadlineNanos) {
        return Math.max((deadlineNanos - System.nanoTime()) / 1_000_000L, 1L);
    }

    /**
     * Drops queued destruction after a timed-out idle wait. Their resources are still referenced by GPU work
     * that never completed, so freeing them would be use-after-free; leaking them until process exit is the
     * only safe option left.
     */
    public void abandonPendingDestroys() {
        int abandoned;
        synchronized (destroyJobs) {
            abandoned = destroyJobs.size();
            destroyJobs.clear();
        }
        destroysAbandoned = true;
        if (abandoned > 0) {
            CausticaMod.LOGGER.warn("RT shutdown: abandoned {} queued GPU resource destruction", abandoned);
        }
    }

    /** Caller has made the device idle; all queued destruction is now unconditionally safe. */
    public void flushDestroysAfterDeviceIdle() {
        if (destroysAbandoned) {
            return;
        }
        Throwable failure = executorFailure;
        synchronized (destroyJobs) {
            for (DestroyJob job : destroyJobs) {
                try {
                    job.destroy.run();
                } catch (Throwable t) {
                    if (failure == null) {
                        failure = t;
                    } else {
                        failure.addSuppressed(t);
                    }
                }
            }
            destroyJobs.clear();
        }
        if (failure != null) {
            throw new IllegalStateException("RT GPU executor failed", failure);
        }
    }

    /** True when shutdown gave up on this executor, so its command pool and timelines must not be freed. */
    public synchronized boolean abandoned() {
        return thread.isAlive() || destroysAbandoned;
    }

    public synchronized void shutdown() {
        if (closed) {
            return;
        }
        closed = true;
        jobs.add(STOP);
        // Stop and join first, but never without a bound: a thread parked on a timeline wait cannot be
        // interrupted out of it, and a client that never finishes stopping is killed by Minecraft's own
        // shutdown watchdog instead.
        try {
            thread.join(RtShutdown.remainingMillis());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        if (thread.isAlive()) {
            thread.interrupt();
            RtShutdown.timeout("RT GPU executor stop", "its command pool and timelines are left alive");
            return;
        }
        // The idle wait makes graphics-side timeline semaphore use complete before those semaphores are
        // destroyed below; on expiry the same resources are left to process exit instead.
        if (!awaitIdleBounded(RtShutdown.remainingMillis())) {
            RtShutdown.timeout("RT GPU idle wait", "queued frees, the command pool and the timelines are left alive");
            abandonPendingDestroys();
            return;
        }
        Throwable failure = null;
        try {
            flushDestroysAfterDeviceIdle();
        } catch (Throwable t) {
            failure = t;
        }
        VK10.vkDestroyCommandPool(ctx.vk(), commandPool, null);
        commandPool = 0L;
        VK10.vkDestroySemaphore(ctx.vk(), graphicsTimeline, null);
        VK10.vkDestroySemaphore(ctx.vk(), buildTimeline, null);
        if (failure != null) {
            throw new IllegalStateException("RT GPU executor shutdown failed", failure);
        }
    }

    private void run() {
        while (true) {
            Job first;
            try {
                first = jobs.take();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                failQueuedJobs(e);
                return;
            }
            if (first == STOP) {
                return;
            }
            if (first == WAKE) {
                if (!processDestroyJobsSafely()) {
                    failQueuedJobs(executorFailure);
                    return;
                }
                continue;
            }
            ArrayList<Job> batch = new ArrayList<>(MAX_BUILD_BATCH);
            batch.add(first);
            boolean stopAfterBatch = false;
            while (batch.size() < MAX_BUILD_BATCH) {
                Job next = jobs.poll();
                if (next == null) {
                    break;
                }
                if (next == STOP) {
                    stopAfterBatch = true;
                    break;
                }
                if (next == WAKE) {
                    continue;
                }
                batch.add(next);
            }

            ArrayList<Job> executable = new ArrayList<>(batch.size());
            for (Job job : batch) {
                if (job.cancelled.getAsBoolean()) {
                    finishJob(job, new CancellationException("RT GPU job epoch is stale"));
                } else {
                    executable.add(job);
                }
            }
            if (!executable.isEmpty()) {
                try {
                    execute(executable);
                    for (Job job : executable) {
                        finishJob(job, null);
                    }
                } catch (Throwable t) {
                    for (Job job : executable) {
                        finishJob(job, t);
                    }
                }
            }
            if (!processDestroyJobsSafely()) {
                failQueuedJobs(executorFailure);
                return;
            }
            if (executorFailure != null) {
                failQueuedJobs(executorFailure);
                return;
            }
            if (stopAfterBatch) {
                return;
            }
        }
    }

    private void finishJob(Job job, Throwable failure) {
        if (failure == null) {
            try {
                job.afterSuccess.run();
            } catch (Throwable t) {
                failure = t;
            }
        }
        try {
            job.finished.accept(job.build, failure);
        } catch (Throwable t) {
            if (failure != null) {
                failure.addSuppressed(t);
            }
            latchFailure(t);
        }
    }

    /** Fail every accepted build before the executor thread exits so task ownership always unwinds. */
    private synchronized void failQueuedJobs(Throwable failure) {
        Throwable terminal = failure != null ? failure : new IllegalStateException("RT GPU executor stopped");
        latchFailure(terminal);
        Job job;
        while ((job = jobs.poll()) != null) {
            if (job != STOP && job != WAKE) {
                finishJob(job, terminal);
            }
        }
    }

    private synchronized void latchFailure(Throwable failure) {
        if (executorFailure == null) {
            executorFailure = failure;
        } else if (executorFailure != failure) {
            executorFailure.addSuppressed(failure);
        }
        synchronized (submissionLock) {
            submissionLock.notifyAll();
        }
    }

    private boolean processDestroyJobsSafely() {
        try {
            processDestroyJobs();
            return true;
        } catch (Throwable t) {
            latchFailure(t);
            return false;
        }
    }

    private void checkExecutorFailure() {
        Throwable failure = executorFailure;
        if (failure != null) {
            throw new IllegalStateException("RT GPU executor failed", failure);
        }
    }

    private static void assertRenderThread() {
        RenderSystem.assertOnRenderThread();
    }

    private void processDestroyJobs() {
        if (!hasPendingDestroys()) {
            return;
        }
        long completed = queryTimeline(graphicsTimeline);
        synchronized (destroyJobs) {
            Iterator<DestroyJob> it = destroyJobs.iterator();
            while (it.hasNext()) {
                DestroyJob job = it.next();
                if (job.lastUseValue <= completed) {
                    it.remove();
                    job.destroy.run();
                }
            }
        }
    }

    private void execute(List<Job> batch) {
        VkCommandBuffer cmd = null;
        boolean submitted = false;
        boolean completed = false;
        long signalValue = batch.get(batch.size() - 1).build.value;
        long firstValue = batch.get(0).build.value;
        VulkanDiagnostics.setInFlight("async-compute",
                "recording builds=" + firstValue + ".." + signalValue + " batch=" + batch.size()
                        + " queued=" + jobs.size());
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandBufferAllocateInfo ai = VkCommandBufferAllocateInfo.calloc(stack).sType$Default()
                    .commandPool(commandPool).level(VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
            PointerBuffer pCmd = stack.mallocPointer(1);
            RtContext.check(VK10.vkAllocateCommandBuffers(ctx.vk(), ai, pCmd), "vkAllocateCommandBuffers(RT GPU executor)");
            cmd = new VkCommandBuffer(pCmd.get(0), ctx.vk());
            VkCommandBufferBeginInfo bi = VkCommandBufferBeginInfo.calloc(stack).sType$Default()
                    .flags(VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            RtContext.check(VK10.vkBeginCommandBuffer(cmd, bi), "vkBeginCommandBuffer(RT GPU executor)");
            for (Job job : batch) {
                job.record.accept(cmd);
            }
            RtContext.check(VK10.vkEndCommandBuffer(cmd), "vkEndCommandBuffer(RT GPU executor)");
            VkCommandBufferSubmitInfo.Buffer command = VkCommandBufferSubmitInfo.calloc(1, stack)
                    .sType$Default().commandBuffer(cmd);
            VkSemaphoreSubmitInfo.Buffer signal = VkSemaphoreSubmitInfo.calloc(1, stack)
                    .sType$Default().semaphore(buildTimeline).value(signalValue)
                    // Jobs also contain pure transfer uploads (for example the device-local light
                    // proposal tables). Signal only after every command in the batch, not merely the
                    // AS-build stage, so a graphics wait cannot overtake such a copy.
                    .stageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR);
            VkSubmitInfo2.Buffer submit = VkSubmitInfo2.calloc(1, stack).sType$Default()
                    .pCommandBufferInfos(command).pSignalSemaphoreInfos(signal);
            VulkanDiagnostics.noteQueueSubmission(computeQueue.vkQueue(), "Caustica compute queue");
            synchronized (ctx.deviceQueueHostLock()) {
                RtContext.check(org.lwjgl.vulkan.KHRSynchronization2.vkQueueSubmit2KHR(
                        computeQueue.vkQueue(), submit, 0L), "vkQueueSubmit2KHR(RT GPU executor)");
            }
            submitted = true;
            synchronized (submissionLock) {
                submittedBuildValue = Math.max(submittedBuildValue, signalValue);
                submissionLock.notifyAll();
            }
            VulkanDiagnostics.setInFlight("async-compute",
                    "submitted builds=" + firstValue + ".." + signalValue + " batch=" + batch.size());
            if (!waitTimeline(buildTimeline, signalValue, BUILD_WAIT_TIMEOUT_MILLIS)) {
                throw new IllegalStateException("RT GPU build " + signalValue + " did not complete within "
                        + BUILD_WAIT_TIMEOUT_MILLIS + " ms");
            }
            completed = true;
            VulkanDiagnostics.breadcrumb("async-compute completed buildTimeline=" + signalValue);
        } finally {
            // Never retry a failed host wait while unwinding: propagate its original error. A command
            // buffer is safe to release here only if submission never happened or completion was observed.
            // Otherwise the command pool owns it until shutdown waits the device idle and destroys the pool.
            if (cmd != null && (!submitted || completed)) {
                try (MemoryStack stack = MemoryStack.stackPush()) {
                    VK10.vkFreeCommandBuffers(ctx.vk(), commandPool, stack.pointers(cmd));
                }
            }
            if (!submitted || completed) {
                VulkanDiagnostics.setInFlight("async-compute", null);
            }
        }
    }

    private long createTimeline(String label) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreTypeCreateInfo type = VkSemaphoreTypeCreateInfo.calloc(stack).sType$Default()
                    .semaphoreType(VK12.VK_SEMAPHORE_TYPE_TIMELINE).initialValue(0L);
            VkSemaphoreCreateInfo ci = VkSemaphoreCreateInfo.calloc(stack).sType$Default().pNext(type);
            LongBuffer out = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateSemaphore(ctx.vk(), ci, null, out), "vkCreateSemaphore(" + label + ")");
            long semaphore = out.get(0);
            RtDebugLabels.name(this.ctx, VK10.VK_OBJECT_TYPE_SEMAPHORE, semaphore, label);
            return semaphore;
        }
    }

    private void createCommandPool() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandPoolCreateInfo ci = VkCommandPoolCreateInfo.calloc(stack).sType$Default()
                    .flags(VK10.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT)
                    .queueFamilyIndex(computeQueue.queueFamilyIndex());
            LongBuffer out = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateCommandPool(ctx.vk(), ci, null, out), "vkCreateCommandPool(RT GPU executor)");
            commandPool = out.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_COMMAND_POOL, commandPool, "RT GPU executor command pool");
        }
    }

    private long queryTimeline(long semaphore) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer out = stack.mallocLong(1);
            RtContext.check(VK12.vkGetSemaphoreCounterValue(ctx.vk(), semaphore, out), "vkGetSemaphoreCounterValue");
            return out.get(0);
        }
    }

    /**
     * Waits for the timeline to reach {@code value}, bounded by {@code timeoutMillis}.
     *
     * <p>Infinite waits are what turn a wedged queue into a frozen client, and during shutdown they can wait
     * on a signal that will never come at all: frames stop being submitted the moment shutdown starts, so a
     * graphics value reserved by the last frame may never be signalled. Callers get a false return instead
     * and decide what to abandon.
     */
    private boolean waitTimeline(long semaphore, long value, long timeoutMillis) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreWaitInfo wi = VkSemaphoreWaitInfo.calloc(stack).sType$Default()
                    .semaphoreCount(1)
                    .pSemaphores(stack.longs(semaphore))
                    .pValues(stack.longs(value));
            int result = VK12.vkWaitSemaphores(ctx.vk(), wi, timeoutMillis * 1_000_000L);
            if (result == VK10.VK_TIMEOUT) {
                return false;
            }
            RtContext.check(result, "vkWaitSemaphores(RT GPU executor)");
            return true;
        }
    }

    private void awaitBuildSubmission(long value) {
        synchronized (submissionLock) {
            while (submittedBuildValue < value && executorFailure == null) {
                try {
                    submissionLock.wait();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted waiting for RT GPU build submission " + value, e);
                }
            }
        }
        checkExecutorFailure();
    }

    public static final class Build {
        private final long value;

        private Build(long value) {
            this.value = value;
        }

        public long value() {
            return value;
        }
    }

    /** Immutable reservation for one graphics frame's completion on the shared RT graphics timeline. */
    public static final class GraphicsUse {
        private final long value;

        private GraphicsUse(long value) {
            this.value = value;
        }

        public long value() {
            return value;
        }
    }

    /** Mutable last-use owner embedded in reusable or asynchronously retired GPU resource slots. */
    public static final class TrackedGraphicsUse {
        private long value;

        public void mark(GraphicsUse graphicsUse) {
            assertRenderThread();
            value = Math.max(value, graphicsUse.value);
        }
    }

    /**
     * Reuses one timeline query while awaiting several tracked owners. Timeline values are monotonic, so
     * completing a newer value also proves every older value complete.
     */
    public final class GraphicsUseWaiter {
        private long completedValue;

        private GraphicsUseWaiter(long completedValue) {
            this.completedValue = completedValue;
        }

        /** Return true only when this call had to issue a host wait. */
        public boolean await(TrackedGraphicsUse trackedUse) {
            assertRenderThread();
            long requiredValue = trackedUse.value;
            if (requiredValue <= completedValue) {
                return false;
            }
            checkExecutorFailure();
            if (!waitTimeline(graphicsTimeline, requiredValue, FRAME_WAIT_TIMEOUT_MILLIS)) {
                // The frame that reserved this value was never submitted, or the queue stopped completing
                // work. Either way the resource this call guards cannot be reused safely, and the two
                // alternatives are both worse than degrading: waiting forever freezes the client, and
                // throwing takes it down. The RT path is switched off and this frame is drawn any way.
                CausticaMod.LOGGER.error("RT graphics timeline did not reach {} within {} ms; "
                        + "disabling the RT path for this session", requiredValue, FRAME_WAIT_TIMEOUT_MILLIS);
                RtComposite.INSTANCE.latchOff("graphics timeline wait timed out at " + requiredValue);
                completedValue = requiredValue;
                return true;
            }
            completedValue = requiredValue;
            return true;
        }
    }

    private record Job(BooleanSupplier cancelled, Consumer<VkCommandBuffer> record, Runnable afterSuccess,
                       BiConsumer<Build, Throwable> finished, Build build) {
    }

    private record DestroyJob(long lastUseValue, Runnable destroy) {
    }
}

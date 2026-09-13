package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaMod;

/**
 * Deadline bookkeeping for the RT teardown path.
 *
 * <p>Client shutdown runs under a watchdog: if the render thread has not finished within ~15 s the client is
 * killed with a "Client shutdown from post-main" crash report. Teardown therefore must never wait
 * indefinitely on GPU work, and in particular must not wait on work that can no longer be signalled — the
 * client stops submitting frames the moment shutdown begins, so a completion signal still outstanding at that
 * point never arrives and an unbounded wait there hangs until the watchdog fires.
 *
 * <p>Every wait in this path is bounded by the deadline below. A wait that expires is logged with the step it
 * belonged to, and the resources it guarded are then handled best-effort: the process is exiting anyway, and
 * leaking is preferable to a crash report.
 */
public final class RtShutdown {
    /**
     * Teardown budget. Minecraft's client shutdown watchdog allows ~15 s, and the steps after this one
     * (window, device and launcher teardown) need the remainder, so the RT path stops waiting well before it.
     */
    private static final long BUDGET_MILLIS = 8_000L;

    private static long deadlineNanos;
    private static boolean active;
    private static boolean warned;

    private RtShutdown() {
    }

    /** Starts (or restarts) the teardown budget. Idempotent per shutdown; resets on the next client start. */
    public static void begin() {
        active = true;
        warned = false;
        deadlineNanos = System.nanoTime() + BUDGET_MILLIS * 1_000_000L;
    }

    public static boolean active() {
        return active;
    }

    /** Timestamp for the next {@link #step} call. */
    public static long mark() {
        return System.nanoTime();
    }

    /** Logs one completed teardown step with its cost, which is what makes a stuck step identifiable. */
    public static void step(String label, long mark) {
        CausticaMod.LOGGER.info("RT shutdown: {} ({} ms, {} ms budget left)", label,
                (System.nanoTime() - mark) / 1_000_000L, remainingMillis());
    }

    /** Milliseconds left before the deadline, never below 1 so a timed wait still makes progress. */
    public static long remainingMillis() {
        if (!active) {
            return BUDGET_MILLIS;
        }
        long remaining = (deadlineNanos - System.nanoTime()) / 1_000_000L;
        return Math.max(remaining, 1L);
    }

    public static boolean expired() {
        return active && deadlineNanos - System.nanoTime() <= 0L;
    }

    /**
     * Reports a wait that hit the deadline. Logged once per shutdown: every later step that times out is a
     * consequence of the first one, and a single clear warning is more useful than a stream of them.
     */
    public static void timeout(String what, String consequence) {
        if (warned) {
            CausticaMod.LOGGER.warn("RT shutdown: {} also timed out ({})", what, consequence);
            return;
        }
        warned = true;
        CausticaMod.LOGGER.warn("RT shutdown: {} did not complete within the shutdown budget; {}. "
                + "GPU resources are left to process exit", what, consequence);
    }
}

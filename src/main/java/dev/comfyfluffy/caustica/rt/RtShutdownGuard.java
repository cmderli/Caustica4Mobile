package dev.comfyfluffy.caustica.rt;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;

import dev.comfyfluffy.caustica.CausticaMod;

/**
 * Last resort for a client that has finished stopping but cannot exit.
 *
 * <p>JVM exit waits for every non-daemon thread. Android launchers attach their own native threads to the
 * game's JVM — the EGL bridge and native input, for example — and one of those surviving the window close
 * keeps the JVM in that final wait forever: the log shows the launcher's {@code EGLBridge: Terminating}, then
 * nothing for about fifteen seconds, until Minecraft's shutdown watchdog reports "Client shutdown from
 * post-main". That report reads as a renderer crash even though every client and renderer thread has already
 * exited.
 *
 * <p>The guard acts only in the state that has no other explanation: the client's own threads are gone, no
 * thread of this mod is alive, and the process is still running. It writes the surviving threads to the log so
 * the foreign one is identifiable, then halts with status 0. If any thread of this mod is still alive it
 * returns without doing anything, because then the stall is ours and the watchdog's crash report is the
 * evidence worth keeping.
 */
public final class RtShutdownGuard {
    /** Client threads whose exit means the client itself has stopped; anything still running after them is foreign. */
    private static final String[] CLIENT_THREADS = {"Render thread", "Server thread"};
    /** How long to keep watching after those threads exit before concluding that the exit is stuck. */
    private static final long GRACE_MILLIS = 3_000L;
    /** Upper bound on the watch, inside the watchdog's ~15 s so the guard wins that race. */
    private static final long WATCH_MILLIS = 11_000L;
    private static final long POLL_MILLIS = 500L;
    /** Threads this mod creates; if one is still alive the stall is ours and must stay visible. */
    private static final String[] OWN_PREFIXES = {"caustica", "rt-worker-"};

    private static boolean started;

    private RtShutdownGuard() {
    }

    /**
     * Starts the watch, once per session. Called from the client-stopping hook after the teardown ran; the
     * guard thread itself is a daemon, so it cannot keep the process alive by existing.
     */
    public static void start() {
        if (started) {
            return;
        }
        started = true;
        Thread thread = new Thread(RtShutdownGuard::watch, "caustica shutdown guard");
        thread.setDaemon(true);
        thread.start();
    }

    private static void watch() {
        long deadline = System.nanoTime() + WATCH_MILLIS * 1_000_000L;
        long clientGoneAt = 0L;
        while (System.nanoTime() < deadline) {
            if (clientThreadsRunning()) {
                clientGoneAt = 0L;
            } else if (clientGoneAt == 0L) {
                clientGoneAt = System.nanoTime();
            } else if ((System.nanoTime() - clientGoneAt) / 1_000_000L >= GRACE_MILLIS) {
                haltWhenOnlyForeignThreadsRemain();
                return;
            }
            try {
                Thread.sleep(POLL_MILLIS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private static boolean clientThreadsRunning() {
        for (Thread thread : Thread.getAllStackTraces().keySet()) {
            for (String name : CLIENT_THREADS) {
                if (thread.isAlive() && name.equals(thread.getName())) {
                    return true;
                }
            }
        }
        return false;
    }

    private static void haltWhenOnlyForeignThreadsRemain() {
        Verdict verdict = inspect(Thread.getAllStackTraces().keySet());
        if (verdict.survivors().isEmpty()) {
            return; // the process is already on its way out
        }
        if (!verdict.ownThreads().isEmpty()) {
            CausticaMod.LOGGER.warn("Client shutdown is stalled with this mod's own threads still alive ({}); "
                    + "leaving the shutdown watchdog to report it", verdict.ownThreads());
            return;
        }
        if (shutdownAlreadyRunning()) {
            // The JVM is past the wait this guard exists for and is running shutdown hooks: they finish and
            // the process exits on its own, so halting here would cut them short for no reason.
            return;
        }
        CausticaMod.LOGGER.warn("Client shutdown is stalled by a non-daemon thread this mod does not own: {}. "
                        + "Every client and renderer thread has exited, so the JVM is waiting only for {}. "
                        + "Exiting with status 0 rather than letting the shutdown watchdog report a crash",
                describe(verdict.survivors()), verdict.survivors().size() == 1 ? "it" : "them");
        System.out.flush();
        System.err.flush();
        Runtime.getRuntime().halt(0);
    }

    /** The live threads that are holding the JVM open, and which of them belong to this mod. */
    record Verdict(List<Thread> survivors, List<String> ownThreads) {
    }

    /**
     * Whether the JVM has already begun its shutdown sequence. Removing a hook that was never registered
     * returns false normally, and throws exactly while shutdown is in progress, which is the state where the
     * process is about to exit without this guard's help.
     */
    private static boolean shutdownAlreadyRunning() {
        try {
            return Runtime.getRuntime().removeShutdownHook(new Thread());
        } catch (IllegalStateException e) {
            return true;
        }
    }

    /**
     * Classification, kept apart from the halting so it can be tested with threads a test owns. A thread that
     * has already finished or is a daemon cannot hold the JVM open, and the main thread under the name it
     * carries after {@code main} returned is the one doing the waiting.
     */
    static Verdict inspect(Collection<Thread> threads) {
        List<Thread> survivors = new ArrayList<>();
        List<String> own = new ArrayList<>();
        for (Thread thread : threads) {
            if (!thread.isAlive() || thread.isDaemon()) {
                continue;
            }
            String name = thread.getName();
            if ("DestroyJavaVM".equals(name)) {
                continue;
            }
            survivors.add(thread);
            if (isOwnThread(name)) {
                own.add(name);
            }
        }
        return new Verdict(survivors, own);
    }

    /**
     * Name, state and whether the thread has any Java frames at all. A thread attached to the JVM by native
     * code has none, which is what separates a launcher thread from one the game or this mod created.
     */
    private static String describe(List<Thread> threads) {
        StringBuilder out = new StringBuilder();
        for (Thread thread : threads) {
            if (out.length() > 0) {
                out.append(", ");
            }
            boolean nativeOnly = thread.getStackTrace().length == 0;
            out.append(thread.getName()).append('(').append(thread.getState())
                    .append(nativeOnly ? ", no Java frames" : "").append(')');
        }
        return out.toString();
    }

    private static boolean isOwnThread(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        for (String prefix : OWN_PREFIXES) {
            if (lower.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }
}

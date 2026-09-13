package dev.comfyfluffy.caustica.rt;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The shutdown guard decides whether a JVM that outlives its client is stuck on a foreign thread or on one of
 * ours. Getting that wrong in the permissive direction would hide a real hang of this mod behind a forced exit,
 * so the classification is tested with threads the test owns rather than only through the live JVM.
 */
final class RtShutdownGuardTest {
    @Test
    void ignoresFinishedAndDaemonThreads() throws Exception {
        Thread daemon = new Thread(RtShutdownGuardTest::sleepBriefly, "Thread-1");
        daemon.setDaemon(true);
        daemon.start();
        try {
            Thread neverStarted = new Thread(RtShutdownGuardTest::sleepBriefly, "DestroyJavaVM");
            RtShutdownGuard.Verdict verdict = RtShutdownGuard.inspect(List.of(daemon, neverStarted));
            assertTrue(verdict.survivors().isEmpty(), "a daemon or unstarted thread cannot hold the JVM open");
        } finally {
            daemon.join();
        }
    }

    @Test
    void flagsAForeignNonDaemonThread() throws Exception {
        Thread foreign = new Thread(RtShutdownGuardTest::sleepBriefly, "Thread-1");
        foreign.start();
        try {
            RtShutdownGuard.Verdict verdict = RtShutdownGuard.inspect(List.of(foreign));
            assertEquals(List.of(foreign), verdict.survivors());
            assertTrue(verdict.ownThreads().isEmpty(), "a launcher thread is not this mod's");
        } finally {
            foreign.join();
        }
    }

    @Test
    void recognisesThisModsThreads() throws Exception {
        Thread own = new Thread(RtShutdownGuardTest::sleepBriefly, "Caustica GPU executor");
        Thread worker = new Thread(RtShutdownGuardTest::sleepBriefly, "rt-worker-1");
        own.start();
        worker.start();
        try {
            RtShutdownGuard.Verdict verdict = RtShutdownGuard.inspect(List.of(own, worker));
            assertEquals(2, verdict.survivors().size());
            assertEquals(List.of("Caustica GPU executor", "rt-worker-1"), verdict.ownThreads());
            assertFalse(verdict.ownThreads().isEmpty(),
                    "the guard must refuse to force an exit while this mod still has threads alive");
        } finally {
            own.join();
            worker.join();
        }
    }

    private static void sleepBriefly() {
        try {
            Thread.sleep(150L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * The guard's contract, run in a JVM of its own because the accepting half ends it. Without the guard the
     * first probe would wait forever: that is exactly the state a launcher thread left behind produces, and the
     * shutdown watchdog turns it into a crash report about fifteen seconds later.
     */
    @Test
    void exitsWithSuccessWhenOnlyAForeignThreadSurvives() throws Exception {
        Process process = forkProbe("foreign");
        try {
            assertTrue(process.waitFor(8, TimeUnit.SECONDS),
                    "the guard must exit the process instead of leaving the JVM waiting");
            assertEquals(0, process.exitValue(), "the forced exit must report success, not a crash");
            System.out.print(read(process));
        } finally {
            process.destroyForcibly();
        }
    }

    @Test
    void leavesAStallCausedByThisModAlone() throws Exception {
        Process process = forkProbe("own-thread");
        boolean exited;
        try {
            exited = process.waitFor(5, TimeUnit.SECONDS);
        } finally {
            process.destroyForcibly();
        }
        assertFalse(exited, "a stall this mod is part of must keep its watchdog crash report");
    }

    @Test
    void doesNotCutShutdownHooksShort() throws Exception {
        Process process = forkProbe("hooks");
        try {
            assertTrue(process.waitFor(10, TimeUnit.SECONDS), "the hook finishes and the JVM exits on its own");
            assertEquals(0, process.exitValue());
            String output = read(process);
            System.out.print(output);
            assertTrue(output.contains("PROBE hook end"),
                    "the guard must not halt while the JVM is already running its shutdown hooks");
        } finally {
            process.destroyForcibly();
        }
    }

    private static Process forkProbe(String mode) throws IOException {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        return new ProcessBuilder(java, "-cp", System.getProperty("java.class.path"),
                ShutdownGuardProbe.class.getName(), mode)
                .redirectErrorStream(true)
                .start();
    }

    /** What the probe wrote, which for a finished process includes the guard's account of the stall. */
    private static String read(Process process) throws IOException {
        return new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    }
}

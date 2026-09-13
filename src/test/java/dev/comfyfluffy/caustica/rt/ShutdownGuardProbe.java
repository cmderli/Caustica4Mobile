package dev.comfyfluffy.caustica.rt;

/**
 * Forked by {@link RtShutdownGuardTest} so the guard's halting path can run in a JVM the test is allowed to
 * terminate. The argument selects the stall: {@code foreign} starts the thread a launcher would leave behind,
 * {@code own-thread} starts a thread this mod owns, and {@code hooks} exits through {@code System.exit} while a
 * shutdown hook is still running, which the guard must not cut short.
 */
public final class ShutdownGuardProbe {
    public static void main(String[] args) {
        String mode = args.length > 0 ? args[0] : "foreign";
        startNonDaemon("own-thread".equals(mode) ? "Caustica GPU executor" : "Thread-1");
        // Nothing named "Render thread" or "Server thread" exists here, which is the state the guard waits for:
        // the client has stopped, and what is left is whatever kept the JVM alive.
        RtShutdownGuard.start();
        if ("hooks".equals(mode)) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    Thread.sleep(4_000L);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                System.out.println("PROBE hook end");
            }, "probe-shutdown-hook"));
            System.exit(0);
        }
    }

    private static void startNonDaemon(String name) {
        Thread thread = new Thread(() -> {
            try {
                Thread.sleep(Long.MAX_VALUE);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }, name);
        thread.start();
    }
}

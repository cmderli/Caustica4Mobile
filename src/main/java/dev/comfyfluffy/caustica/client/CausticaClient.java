package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtShutdown;
import dev.comfyfluffy.caustica.rt.RtShutdownGuard;
import dev.comfyfluffy.caustica.rt.RtDeviceBringup;
import dev.comfyfluffy.caustica.rt.RtComposite;
import dev.comfyfluffy.caustica.rt.RtFrameStats;
import dev.comfyfluffy.caustica.rt.RtUiOverlay;
import dev.comfyfluffy.caustica.rt.entity.RtEntities;
import dev.comfyfluffy.caustica.rt.entity.RtEntityTextures;
import dev.comfyfluffy.caustica.rt.material.RtBlockMaterials;
import dev.comfyfluffy.caustica.rt.terrain.RtTerrain;
import dev.comfyfluffy.caustica.rt.terrain.RtWorkerPool;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.InvalidateRenderStateCallback;

public final class CausticaClient implements ClientModInitializer {
	private static boolean rtInitDone = false;
	private static boolean loggedRtNeverStarted = false;

	@Override
	public void onInitializeClient() {
		CausticaMod.LOGGER.info("Caustica client initialized");

		// Class-init runs DebugScreenEntries.register(...) via its ID field; touching the class here
		// makes the entry discoverable in F3's entry list. Off by default -- the player opts in the
		// same way as any other optional vanilla entry (e.g. GPU utilization).
		@SuppressWarnings("unused")
		Object registerExposureDebugEntry = RtExposureDebugEntry.ID;

		// The GpuDevice exists well before the first tick, so a one-shot at tick start
		// runs on the render thread with the device idle between frames.
		ClientTickEvents.START_CLIENT_TICK.register(client -> {
			if (!VanillaRenderController.rtRuntimeWorkRequested()) {
				if (rtInitDone) {
					shutdownRt();
				}
				return;
			}

			// Bring up the RT device/context once; terrain residency + the composite follow below.
			if (!rtInitDone && RtDeviceBringup.rtRequested()) {
				RtContext ctx = RtContext.get();
				if (ctx != null) {
					rtInitDone = true;
				}
			} else if (!rtInitDone && !loggedRtNeverStarted) {
				// The Vulkan backend latches rtRequested while it creates the device, which happens before
				// the first tick. Reaching a tick without it means this session never brought up an
				// RT-capable Vulkan device at all: either Minecraft is running on its OpenGL backend — a
				// launcher's "Vulkan"/Zink renderer is OpenGL over Vulkan, not the Vulkan backend — or the
				// device was rejected earlier, which the bring-up logged.
				loggedRtNeverStarted = true;
				CausticaMod.LOGGER.info("Caustica ray tracing is inactive: Minecraft did not come up on an "
						+ "RT-capable Vulkan backend (OpenGL/Zink sessions never start it; see any earlier "
						+ "'Ray tracing unavailable' line for a device that was rejected)");
			}

			// Once RT is up, keep section residency synced to vanilla's loaded chunks around
			// the player — builds newly-in-range sections, frees out-of-range ones, per tick.
			if (rtInitDone) {
				RtContext ctx = RtContext.currentOrNull();
				if (ctx != null) {
					RtFrameStats.FRAME.beginIfInactive();
					// Bring the world pipeline + LabPBR atlases up before terrain tessellates, so per-prim
					// material flags resolve from the first section (PBR on join, no re-extract). No-op
					// until we're in a world with the block atlas loaded, or once already created.
					RtComposite.INSTANCE.ensureResourcesReady(ctx);
					RtTerrain.update(ctx);
					// Log DLSS-FG availability once when frame generation is enabled (capability query only;
					// the present-loop integration that consumes it is built separately).
					if (dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.enabled()) {
						dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.INSTANCE.probeAvailabilityOnce();
					}
				}
			}
		});

		// Vanilla's full render-state invalidation (LevelExtractor.allChanged(): dimension change via
		// setLevel, render-distance change, F3+A) — drop RT terrain residency so it rebuilds for the new
		// world. Fixes stale geometry persisting across an End→Overworld switch (coords alone aren't
		// world-unique). Resource reloads do NOT fire this; that path is handled separately.
		InvalidateRenderStateCallback.EVENT.register(() -> {
			RtTerrain.requestFullClear();
			RtComposite.INSTANCE.resetExposureHistory();
			RtComposite.INSTANCE.resetFailureLatch(); // F3+A doubles as manual RT recovery after a latched failure
		});

		ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
			shutdownRt();
			// Minecraft's exit waits for every non-daemon JVM thread, and Android launchers attach native
			// threads of their own to this JVM. The guard makes the difference between a launcher thread
			// that outlives the window — which the shutdown watchdog would report as a crash — and a clean
			// exit, and it names that thread in the log.
			RtShutdownGuard.start();
		});
	}

	/** Runs one teardown step, logging its cost and swallowing failures so later steps still run. */
	private static void rtShutdownStep(String label, long mark, Runnable step) {
		try {
			step.run();
			RtShutdown.step(label, mark);
		} catch (Throwable t) {
			dev.comfyfluffy.caustica.CausticaMod.LOGGER.warn("RT shutdown: {} failed; continuing teardown",
					label, t);
		}
	}

	private static void shutdownRt() {
		// Everything below runs on the render thread while Minecraft's client shutdown watchdog counts down.
		// Each step is logged with its cost and is individually failure-tolerant, so a step that throws or
		// times out leaves a record in the log instead of silently skipping the rest of the teardown.
		RtShutdown.begin();
		long teardownStart = RtShutdown.mark();
		long mark = RtShutdown.mark();
		rtShutdownStep("window scaler", mark, WorldRenderScaler.INSTANCE::destroy);
		mark = RtShutdown.mark();
		// GUI redirect is not gated by rtInitDone; always release its TextureTarget.
		rtShutdownStep("UI overlay", mark, RtUiOverlay::destroy);
		if (!rtInitDone) {
			mark = RtShutdown.mark();
			rtShutdownStep("worker pool", mark, RtWorkerPool.INSTANCE::shutdown);
			RtShutdown.step("complete (RT never initialized)", teardownStart);
			return;
		}

		RtContext ctx = RtContext.currentOrNull();
		if (ctx != null) {
			// Let the terrain epoch cancel queued work and release every active-task token before
			// shutdownNow(): discarded worker Runnables cannot deliver their terminal callbacks.
			mark = RtShutdown.mark();
			rtShutdownStep("terrain", mark, () -> RtTerrain.shutdown(ctx));
		}
		mark = RtShutdown.mark();
		rtShutdownStep("worker pool", mark, RtWorkerPool.INSTANCE::shutdown);
		if (ctx != null) {
			mark = RtShutdown.mark();
			rtShutdownStep("entities", mark, RtEntities.INSTANCE::shutdown);
		}
		mark = RtShutdown.mark();
		rtShutdownStep("composite", mark, RtComposite.INSTANCE::destroy);
		mark = RtShutdown.mark();
		rtShutdownStep("entity textures", mark, RtEntityTextures.INSTANCE::reset);
		rtShutdownStep("block materials", mark, RtBlockMaterials.INSTANCE::destroy);
		rtShutdownStep("frame generation", mark, dev.comfyfluffy.caustica.rt.pipeline.RtDlssFg.INSTANCE::destroy);
		if (ctx != null) {
			rtShutdownStep("frame presenter", mark,
					() -> dev.comfyfluffy.caustica.rt.RtFramePresenter.INSTANCE.destroy(ctx.device()));
			rtShutdownStep("reflex", mark,
					() -> dev.comfyfluffy.caustica.rt.RtReflex.INSTANCE.destroy(ctx.device().vkDevice()));
		}
		mark = RtShutdown.mark();
		// Shut NGX down once, after every feature (RR + FG) has been released above.
		rtShutdownStep("NGX", mark, dev.comfyfluffy.caustica.ngx.NgxRuntime.INSTANCE::shutdown);
		if (ctx != null) {
			rtShutdownStep("device objects", mark, ctx::destroy);
		}
		RtShutdown.step("complete", teardownStart);
		rtInitDone = false;
	}
}

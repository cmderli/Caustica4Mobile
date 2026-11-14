package dev.comfyfluffy.caustica.client;

import com.mojang.serialization.Codec;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaConfig.BooleanSetting;
import dev.comfyfluffy.caustica.CausticaConfig.FloatSetting;
import dev.comfyfluffy.caustica.CausticaConfig.IntSetting;
import dev.comfyfluffy.caustica.CausticaConfig.StringSetting;
import dev.comfyfluffy.caustica.rt.RtDeviceBringup;
import dev.comfyfluffy.caustica.rt.RtNrd;
import dev.comfyfluffy.caustica.rt.RtRecon;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import net.minecraft.client.Minecraft;
import net.minecraft.client.OptionInstance;
import net.minecraft.client.Options;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.Tooltip;
import net.minecraft.network.chat.Component;

/**
 * Builds the {@link OptionInstance} widgets shown in the RT section of the vanilla Video Settings screen
 * (injected by {@code VideoSettingsScreenMixin}). Each option is bound straight to a {@link CausticaConfig}
 * runtime setting: the initial value is read from the current config, and the value-update listener writes
 * back through {@code set(...)} so changes take effect on the next frame.
 *
 * <p>Only settings the renderer re-reads per-frame are exposed here — toggles that would require a device or
 * buffer-pool rebuild (worker threads, OMM, max-entity capacities, PBR material flags) are intentionally
 * left to the {@code -Dcaustica.*} startup surface. DLSS-RR quality is the exception: the render resolution
 * is queried from NGX for the chosen quality mode on every resize (see
 * {@code RtDlssRr.queryOptimalRenderSize}), and the RR feature itself is recreated live whenever
 * {@code quality} changes (see {@code RtDlssRr.ensureFeature}), so it is safe to expose here.
 */
public final class RtVideoOptions {
    private RtVideoOptions() {
    }

    /**
     * Runtime-tunable RT options, in display order. Paired two-per-row by {@code OptionsList.addSmall}.
     * The HDR entries are omitted entirely (not just disabled) when this session's swapchain isn't
     * PQ-capable ({@code CausticaConfig.Rt.Hdr.swapchainPqAvailable()}) — offering a toggle/sliders that
     * can never do anything is worse than not showing them, and unlike most settings here this one is
     * fixed by hardware/OS/compositor at surface-creation time. The current swapchain may still be native
     * SDR; changing the toggle invalidates its configuration and recreates it in the selected format.
     */
    public static OptionInstance<?>[] runtimeOptions() {
        List<OptionInstance<?>> options = new ArrayList<>(List.of(
            exposureMode(),
            manualEv(),
            gamma(),
            spp(),
            maxBounces(),
            restir(),
            entities(),
            particles(),
            waterWaves(),
            denoiser(),
            superResolutionMethod(),
            superResolution(),
            motionBlur(),
            motionBlurShutter(),
            motionBlurSamples()
        ));
        if (CausticaConfig.Rt.Hdr.swapchainPqAvailable()) {
            options.add(hdrEnabled());
            options.add(hdrUiBrightness());
            options.add(hdrPeak());
        }
        options.add(lightCandidates());
        options.add(gpuProfiling());
        options.add(debugView());
        return options.toArray(OptionInstance<?>[]::new);
    }

    /**
     * Restores every Caustica setting and rewrites {@code config/caustica.toml} with the keys, defaults and
     * comments a first launch writes. A change to the HDR toggle invalidates the surface so the swapchain is
     * recreated in the restored format at the next safe frame boundary. {@code refreshScreen} rebuilds the
     * option widgets, which otherwise keep showing the values they were built with.
     */
    public static Button resetDefaultsButton(Runnable refreshScreen) {
        return Button.builder(
                        Component.translatable("caustica.options.rt.resetDefaults"),
                        button -> {
                            boolean hdrBefore = CausticaConfig.Rt.Hdr.enabled();
                            CausticaConfig.resetToDefaults();
                            // HDR is a swapchain format, not a per-frame read: recreate the swapchain only
                            // when the reset actually changed it, so every other reset stays hiccup-free.
                            if (CausticaConfig.Rt.Hdr.enabled() != hdrBefore) {
                                Minecraft.getInstance().invalidateSurfaceConfiguration();
                            }
                            refreshScreen.run();
                        })
                .tooltip(Tooltip.create(
                        Component.translatable("caustica.options.rt.resetDefaults.tooltip")))
                .build();
    }

    private static OptionInstance<String> exposureMode() {
        StringSetting setting = CausticaConfig.Rt.Exposure.MODE;
        return new OptionInstance<>(
            "caustica.options.rt.exposureMode",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.exposureMode.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.exposureMode." + value),
            new OptionInstance.Enum<>(List.of("auto", "manual"), Codec.STRING),
            setting.get(),
            setting::set);
    }

    private static OptionInstance<Integer> manualEv() {
        FloatSetting setting = CausticaConfig.Rt.Exposure.MANUAL_EV;
        return new OptionInstance<>(
            "caustica.options.rt.manualEv",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.manualEv.tooltip")),
            (caption, tenths) -> {
                float ev = tenths / 10.0f;
                String sign = ev > 0.0f ? "+" : "";
                return Options.genericValueLabel(caption,
                        Component.literal(sign + String.format(Locale.ROOT, "%.1f EV", ev)));
            },
            new OptionInstance.IntRange(-150, 150),
            Math.clamp(Math.round(setting.value() * 10.0f), -150, 150),
            tenths -> setting.set(tenths / 10.0f));
    }

    private static OptionInstance<Integer> gamma() {
        FloatSetting setting = CausticaConfig.Rt.Tonemap.GAMMA;
        return new OptionInstance<>(
            "caustica.options.rt.gamma",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.gamma.tooltip")),
            (caption, hundredths) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(Locale.ROOT, "%.2f", hundredths / 100.0f))),
            new OptionInstance.IntRange(50, 150),
            Math.clamp(Math.round(setting.value() * 100.0f), 50, 150),
            hundredths -> setting.set(hundredths / 100.0f));
    }

    private static OptionInstance<Integer> spp() {
        IntSetting setting = CausticaConfig.Rt.Composite.SPP;
        return new OptionInstance<>(
            "caustica.options.rt.spp",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.spp.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(1, 8),
            Math.clamp(setting.value(), 1, 8),
            setting::set);
    }

    private static OptionInstance<Integer> maxBounces() {
        IntSetting setting = CausticaConfig.Rt.Composite.MAX_BOUNCES;
        return new OptionInstance<>(
            "caustica.options.rt.maxBounces",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.maxBounces.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(1, 8),
            Math.clamp(setting.value(), 1, 8),
            setting::set);
    }

    private static OptionInstance<Boolean> restir() {
        return bool("caustica.options.rt.restir", CausticaConfig.Rt.Restir.ENABLED);
    }

    private static OptionInstance<Boolean> entities() {
        return bool("caustica.options.rt.entities", CausticaConfig.Rt.Entities.ENABLED);
    }

    private static OptionInstance<Boolean> particles() {
        return bool("caustica.options.rt.particles", CausticaConfig.Rt.Entities.PARTICLES_ENABLED);
    }

    private static OptionInstance<Boolean> waterWaves() {
        return bool("caustica.options.rt.waterWaves", CausticaConfig.Rt.Composite.WATER_WAVES);
    }

    /**
     * Which denoiser removes the path tracer's noise. Only the choices this machine can actually run are
     * offered — the same reasoning as the HDR entries below: an option that can never take effect is worse
     * than an absent one. "Auto" is always present and already resolves to the best available denoiser, so
     * hiding an unavailable entry costs nothing.
     */
    private static OptionInstance<String> denoiser() {
        StringSetting setting = CausticaConfig.Rt.Reconstruction.DENOISER;
        List<String> values = new ArrayList<>();
        values.add("auto");
        if (RtDeviceBringup.ngxCapable()) {
            values.add("dlss-rr");
        }
        if (RtNrd.INSTANCE.isAvailable()) {
            values.add("nrd");
        }
        values.add("builtin");
        values.add("off");
        String initial = values.contains(setting.get()) ? setting.get() : "auto";
        return new OptionInstance<>(
            "caustica.options.rt.denoiser",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.denoiser.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.denoiser." + value),
            new OptionInstance.Enum<>(values, Codec.STRING),
            initial,
            setting::set);
    }

    /**
     * How the render-resolution result is scaled to the display: FSR's temporal upscaler, or the renderer's
     * own bilinear resolve. Always offered, because "bilinear" is meaningful everywhere and "auto" adapts;
     * picking FSR without its runtime present falls back to the resolve and says so in the log.
     */
    private static OptionInstance<String> superResolutionMethod() {
        StringSetting setting = CausticaConfig.Rt.Reconstruction.UPSCALER;
        return new OptionInstance<>(
            "caustica.options.rt.superResolutionMethod",
            OptionInstance.cachedConstantTooltip(
                    Component.translatable("caustica.options.rt.superResolutionMethod.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.superResolutionMethod." + value),
            new OptionInstance.Enum<>(CausticaConfig.Rt.Reconstruction.UPSCALERS, Codec.STRING),
            setting.get(),
            setting::set);
    }

    /**
     * NVSDK_NGX_PerfQuality_Value per super-resolution mode, in ladder order: native-aa (DLAA), quality,
     * balanced, performance, ultra-performance. DLSS Ray Reconstruction only accepts Performance(0),
     * Balanced(1), Quality(2), Ultra-Performance(3) and DLAA(5); Ultra Quality(4) is excluded because its
     * optimal-settings query returns a zeroed render size.
     */
    private static final List<Integer> DLSS_BY_MODE = List.of(5, 2, 1, 0, 3);

    /**
     * Super-resolution mode: the render/display ratio the trace runs at. One control drives whichever
     * backend currently owns the upscale — DLSS Ray Reconstruction stores its own quality values, the
     * other backends share the FSR ladder — because the modes mean the same thing to the user either way.
     */
    private static OptionInstance<Integer> superResolution() {
        List<Integer> modes = CausticaConfig.Rt.Reconstruction.QUALITY_STEPS;
        boolean dlss = RtRecon.INSTANCE.superResolutionIsDlss();
        int initialPosition = dlss
                ? Math.max(DLSS_BY_MODE.indexOf(CausticaConfig.Rt.DlssRr.QUALITY.value()), 0)
                : Math.max(modes.indexOf(CausticaConfig.Rt.Reconstruction.QUALITY.value()), 0);
        return new OptionInstance<>(
            "caustica.options.rt.superResolution",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.superResolution.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption,
                    Component.translatable("caustica.options.rt.superResolution." + position)),
            new OptionInstance.IntRange(0, modes.size() - 1),
            initialPosition,
            position -> {
                // Resolved when the value changes, not when the screen opened, so switching the denoiser
                // above and then picking a mode writes to whichever backend is active by then.
                if (RtRecon.INSTANCE.superResolutionIsDlss()) {
                    // DLSS's ladder is shorter than the trace-resolution one; its strongest mode is the
                    // ceiling for the extra step.
                    CausticaConfig.Rt.DlssRr.QUALITY.set(
                            DLSS_BY_MODE.get(Math.min(position, DLSS_BY_MODE.size() - 1)));
                } else {
                    CausticaConfig.Rt.Reconstruction.QUALITY.set(modes.get(position));
                }
            });
    }

    private static OptionInstance<Boolean> motionBlur() {
        BooleanSetting setting = CausticaConfig.Rt.MotionBlur.ENABLED;
        return OptionInstance.createBoolean(
            "caustica.options.rt.motionBlur",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.motionBlur.tooltip")),
            setting.value(),
            setting::set);
    }

    // Film's shutter angle in degrees. 180 (the standard) keeps the virtual shutter open for half the frame
    // interval, 360 for all of it, and 0 disables the streak without turning the option off.
    private static OptionInstance<Integer> motionBlurShutter() {
        IntSetting setting = CausticaConfig.Rt.MotionBlur.SHUTTER_ANGLE;
        return new OptionInstance<>(
            "caustica.options.rt.motionBlurShutter",
            OptionInstance.cachedConstantTooltip(
                Component.translatable("caustica.options.rt.motionBlurShutter.tooltip")),
            (caption, degrees) -> Options.genericValueLabel(caption, Component.literal(degrees + "\u00b0")),
            new OptionInstance.IntRange(0, 360),
            Math.clamp(setting.value(), 0, 360),
            setting::set);
    }

    private static OptionInstance<Integer> motionBlurSamples() {
        IntSetting setting = CausticaConfig.Rt.MotionBlur.SAMPLES;
        List<Integer> steps = CausticaConfig.Rt.MotionBlur.SAMPLE_STEPS;
        int initial = steps.contains(setting.value()) ? steps.indexOf(setting.value()) : steps.indexOf(8);
        return new OptionInstance<>(
            "caustica.options.rt.motionBlurSamples",
            OptionInstance.cachedConstantTooltip(
                Component.translatable("caustica.options.rt.motionBlurSamples.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption, Component.literal(steps.get(position).toString())),
            new OptionInstance.IntRange(0, steps.size() - 1),
            Math.max(initial, 0),
            position -> setting.set(steps.get(position)));
    }

    private static OptionInstance<Boolean> hdrEnabled() {
        BooleanSetting setting = CausticaConfig.Rt.Hdr.ENABLED;
        return OptionInstance.createBoolean(
            "caustica.options.rt.hdr",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdr.tooltip")),
            setting.value(),
            enabled -> {
                if (setting.value() != enabled) {
                    setting.set(enabled);
                    // Reuse the framebuffer-resize path at the next safe frame boundary. GpuSurface
                    // refuses configure() while an image is acquired, so doing it directly here is unsafe.
                    Minecraft.getInstance().invalidateSurfaceConfiguration();
                }
            });
    }

    private static OptionInstance<Integer> hdrUiBrightness() {
        FloatSetting setting = CausticaConfig.Rt.Hdr.UI_NITS;
        return new OptionInstance<>(
            "caustica.options.rt.hdrUiBrightness",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdrUiBrightness.tooltip")),
            (caption, nits) -> Options.genericValueLabel(caption, Component.literal(nits + " nits")),
            new OptionInstance.IntRange(80, 500),
            Math.clamp(Math.round(setting.value()), 80, 500),
            nits -> setting.set(nits.floatValue()));
    }

    // Each step selects a baked ACES HDR mastering target. Changes take effect on the next frame.
    private static OptionInstance<Integer> hdrPeak() {
        IntSetting setting = CausticaConfig.Rt.Hdr.PEAK_NITS;
        List<Integer> steps = CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS;
        int initialPeak = steps.contains(setting.value()) ? setting.value() : 1000;
        int initialPosition = steps.indexOf(initialPeak);
        return new OptionInstance<>(
            "caustica.options.rt.hdrPeak",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdrPeak.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption, Component.literal(steps.get(position) + " nits")),
            new OptionInstance.IntRange(0, steps.size() - 1),
            Math.max(initialPosition, 0),
            position -> setting.set(steps.get(position)));
    }

    /**
     * RIS emitter-NEE candidates per shaded vertex. Each candidate is one shadow ray, so this is the largest
     * single ray-count multiplier in the path pass and the first knob to try when a device is ray-bound.
     * 0 disables emitter NEE entirely (only the direct-hit emission term remains).
     */
    private static OptionInstance<Integer> lightCandidates() {
        IntSetting setting = CausticaConfig.Rt.Lights.RIS_CANDIDATES;
        return new OptionInstance<>(
            "caustica.options.rt.lightCandidates",
            OptionInstance.cachedConstantTooltip(
                Component.translatable("caustica.options.rt.lightCandidates.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(0, 8),
            Math.clamp(setting.value(), 0, 8),
            setting::set);
    }

    /**
     * Turns on the executed-GPU-time report ({@link dev.comfyfluffy.caustica.rt.RtGpuProfile}). It lives in the
     * UI rather than only in the config because a low frame rate cannot be attributed without it: the CPU-side
     * timing in {@code frame-stats.enabled} records a dispatch in microseconds regardless of what it costs on
     * the device. Off by default; the report is one log line every couple of seconds plus a CSV.
     */
    private static OptionInstance<Boolean> gpuProfiling() {
        return bool("caustica.options.rt.gpuProfiling", CausticaConfig.Rt.FrameStats.GPU);
    }

    private static OptionInstance<Integer> debugView() {
        IntSetting setting = CausticaConfig.Rt.Composite.DEBUG_VIEW;
        return new OptionInstance<>(
            "caustica.options.rt.debugView",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.debugView.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.debugView." + value),
            new OptionInstance.Enum<>(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Codec.INT),
            Math.clamp(setting.value(), 0, 9),
            setting::set);
    }

    private static OptionInstance<Boolean> bool(String captionKey, BooleanSetting setting) {
        return OptionInstance.createBoolean(
            captionKey,
            OptionInstance.cachedConstantTooltip(Component.translatable(captionKey + ".tooltip")),
            setting.value(),
            setting::set);
    }
}

package dev.comfyfluffy.caustica.mixin;

import net.minecraft.client.gui.screens.Screen;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

/**
 * Exposes {@link Screen#rebuildWidgets()} so the RT options can rebuild their own screen after a settings
 * reset: {@code OptionInstance} widgets keep the value they were constructed with, so refreshing the values
 * requires rebuilding the list rather than editing it in place.
 */
@Mixin(Screen.class)
public interface ScreenInvoker {
    @Invoker("rebuildWidgets")
    void causticaRebuildWidgets();
}

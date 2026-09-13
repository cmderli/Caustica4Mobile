package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;
import org.lwjgl.vulkan.VkPhysicalDeviceRayQueryFeaturesKHR;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins how the device bring-up reads feature structs whose LWJGL binding class this client may not have.
 * Minecraft ships one LWJGL, but Android launchers substitute their own — the launcher's LWJGL 3.3.6 has
 * no {@code VK_EXT_ray_tracing_invocation_reorder} bindings — and a direct reference to a class that is
 * missing surfaces as {@code NoClassDefFoundError} while the referencing class initializes, before any
 * device exists. Reading the constants by name is what turns that into "this optional extension is
 * unavailable", so the fallback path is the contract worth testing.
 */
final class RtOptionalBindingTest {
    @Test
    void absentBindingFallsBackInsteadOfThrowing() {
        assertEquals(0, RtDeviceBringup.optionalStaticInt(
                "org.lwjgl.vulkan.VkPhysicalDeviceThisBindingDoesNotExist", "SIZEOF", 0));
        assertEquals(-1, RtDeviceBringup.optionalStaticInt(
                "org.lwjgl.vulkan.VkPhysicalDeviceThisBindingDoesNotExist", "ANY_MEMBER_OFFSET", -1));
    }

    @Test
    void presentBindingYieldsItsConstant() {
        int size = RtDeviceBringup.optionalStaticInt(
                "org.lwjgl.vulkan.VkPhysicalDeviceRayQueryFeaturesKHR", "SIZEOF", 0);
        assertEquals(VkPhysicalDeviceRayQueryFeaturesKHR.SIZEOF, size);
        assertTrue(size > 0, "a present binding must report its struct size, not the fallback");
    }
}

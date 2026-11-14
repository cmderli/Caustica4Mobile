package dev.comfyfluffy.caustica.rt;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtGpuProfileTest {
    /**
     * Every stage owns two distinct query indices inside the frame's query range. A collision would make two
     * stages overwrite each other's timestamps, and an out-of-range index would reset or read another slot.
     */
    @Test
    void stageMarkersAreDisjointAndInRange() {
        Set<Integer> used = new HashSet<>();
        Set<String> labels = new HashSet<>();
        for (RtGpuProfile.Stage stage : RtGpuProfile.Stage.values()) {
            assertTrue(stage.startMarker() < stage.endMarker(), stage.name());
            assertTrue(used.add(stage.startMarker()), "duplicate start marker for " + stage.name());
            assertTrue(used.add(stage.endMarker()), "duplicate end marker for " + stage.name());
            assertTrue(stage.endMarker() < RtGpuProfile.MARKERS,
                    stage.name() + " marker " + stage.endMarker() + " outside " + RtGpuProfile.MARKERS);
            assertTrue(labels.add(stage.label()), "duplicate stage label " + stage.label());
        }
        assertEquals(RtGpuProfile.Stage.values().length * 2, RtGpuProfile.MARKERS);
    }
}

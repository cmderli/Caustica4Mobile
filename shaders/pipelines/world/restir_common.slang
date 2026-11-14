// ReSTIR PT reservoir layout and the constants shared by both reservoir passes.
//
// A reservoir stores one selected path sample: the reconnection vertex (x2), the event that leaves it,
// the radiance that event delivers, the target function at the pixel it currently belongs to, and the
// sum/count of the resampling weights that produced it. Positions are camera-relative so terrain
// rebasing cannot invalidate a sample, and every material/radiance value is packed to keep the per-pixel
// buffer at 128 B while leaving room for both a history and a scratch array at render resolution.
//
// This module is imported by the reflection probe, so it must not import any binding or render module.
import world_common;

// Event at the reconnection vertex. BSDF = a sampled continuation; LIGHT = a sampled sun/emitter
// connection; SKY = the first continuation escaped, so the "suffix" is the sky radiance along rcDir.
public static const uint RESTIR_EVENT_MASK = 3u;
public static const uint RESTIR_EVENT_BSDF = 0u;
public static const uint RESTIR_EVENT_LIGHT = 1u;
public static const uint RESTIR_EVENT_SKY = 2u;

// Sample flags. The event occupies the low two bits.
public static const uint RESTIR_FLAG_LANE_SPECULAR = 4u;  // first lobe at the primary vertex was specular
public static const uint RESTIR_FLAG_X2_DELTA = 8u;       // reconnection BSDF is a delta lobe
public static const uint RESTIR_FLAG_EMISSION = 16u;      // reconnection vertex's own emission is part of p-hat
public static const uint RESTIR_FLAG_VALID = 32u;
public static const uint RESTIR_FLAG_X2_LIGHT_LOBE = 64u; // BSDF event sampled the specular lobe at x2

// Primary-record flags. The record is written by the primary pass so the two reservoir passes do not
// have to re-trace the camera ray at the render resolution.
public static const uint RESTIR_PRIMARY_VALID = 1u;
public static const uint RESTIR_PRIMARY_EMITTER = 2u;

// A stored path sample. rc* values describe the reconnection vertex x2; primary* describe the visible
// surface x1 of the pixel the sample currently belongs to. pHat is the full path's target-function value
// evaluated for that pixel, and rcL is the radiance the event delivers at x2 (already including the
// light/area/visibility factors for LIGHT; already the suffix radiance estimate for BSDF).
//
// Positions and rcL are scalar triples (not float3) so std430 does not insert 16-byte alignment padding;
// normals are oct16 and pHat is half3 (the shader clamps pHat to HALF_MAX anyway). The sample is 88 B and
// the reservoir 96 B, which keeps history/scratch traffic and the mobile working set down.
public struct RestirSample {
    public float rcPosX;              // camera-relative x2 position
    public float rcPosY;
    public float rcPosZ;
    public uint  packedRcNormal;      // oct16x2 x2 shading normal, oriented toward the primary vertex
    public float primaryPosX;         // camera-relative x1 position of the pixel owning this sample
    public float primaryPosY;
    public float primaryPosZ;
    public uint  packedPrimaryNormal; // oct16x2 x1 shading normal
    public uint  packedDir;           // oct16x2 outgoing event direction
    public uint  packedRoughMetal;    // half2(roughness, metalness) at x2
    public uint  packedSssEmission;   // half2(sss, emission) at x2
    public uint  packedAlbedo0;       // half2(albedo.r, albedo.g)
    public uint  packedAlbedo1;       // half2(albedo.b, 0)
    public uint  packedF00;           // half2(f0.r, f0.g)
    public uint  packedF01;           // half2(f0.b, 0)
    public float rcLX;                // radiance delivered by the event at x2 (float range)
    public float rcLY;
    public float rcLZ;
    public uint  packedPHat0;         // half2(pHat.r, pHat.g)
    public uint  packedPHat1;         // half2(pHat.b, 0)
    public uint  flags;
    public uint  packedPrimaryRoughMetal; // half2(x1 roughness, x1 metalness) for reuse compatibility
};

// The reservoir itself: the selected sample plus the resampling weight sum and confidence. wSum/M is the
// unbiased contribution weight times p-hat, so the shaded value is pHat * wSum / (M * luminance(pHat)).
public struct RestirReservoir {
    public RestirSample sample; // 88 B
    public float wSum;
    public float M;
}; // 96 B

// The visible primary surface of one render pixel, written once by the primary pass. pos/n/view are in
// rebased world space, normals and directions are oct16, and materials are packed the same way as the
// reservoir's reconnection-vertex material so both sides of the shift use one representation.
public struct RestirPrimary {
    public float3 pos;                // rebased world position of x1
    public uint   packedNormal;       // oct16 shading normal, oriented toward the camera
    public uint   packedView;         // oct16 direction toward the camera
    public uint   flags;              // RESTIR_PRIMARY_*
    public uint   packedAlbedo0;      // half2(albedo.r, albedo.g)
    public uint   packedAlbedo1;      // half2(albedo.b, 0)
    public uint   packedF00;          // half2(f0.r, f0.g)
    public uint   packedF01;          // half2(f0.b, 0)
    public uint   packedRoughMetal;   // half2(roughness, metalness)
    public uint   packedSssEmission;  // half2(sss, emission)
    public uint   packedMotion0;      // half2(previous-frame world displacement .xy)
    public uint   packedMotion1;      // half2(previous-frame world displacement .z, 0)
    public uint   reserved0;
    public uint   reserved1;
};

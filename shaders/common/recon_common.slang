// Push-constant structs for the reconstruction chain: the passes that turn the noisy render-resolution
// trace into the display-resolution image. They live beside the display-chain structs rather than inside a
// pipeline folder because layout/layout_probe.slang reflects push constants per module, and because a
// second reconstruction backend adds its struct here.

public struct DenoisePush {
    public int mode;              // 0 atrous iteration, 1 bilinear resolve into the display-res output
    public int step;              // atrous tap spacing in render texels: 1, 2, 4
    public int renderWidth;
    public int renderHeight;
    public int displayWidth;      // resolve output size; unused by the atrous iterations
    public int displayHeight;
    public float projM22;         // frame projection [2][2]
    public float projM32;         // frame projection [3][2]
    public float depthSigma;      // relative view-distance tolerance between taps
    public float normalPower;     // normal-similarity exponent
    public float albedoSigma;     // diffuse-albedo luminance tolerance
    public float lumaSigma;       // demodulated-luminance tolerance
    public float mirrorRoughness; // roughness below which a tap keeps its own sample
    public float fireflyClamp;    // luminance ceiling as a multiple of the local mean
};

// Diffuse and specular lanes arrive as radiance: the material factor is still in them. NRD denoises with no
// albedo input of its own, so both ends of the chain apply NRD's material demodulation
// ("NRD(diffuseRadiance / albedo) * albedo", see NRD's README) to keep its spatial filters from averaging
// neighbouring albedo into each other — without it, block textures come back smeared. The floor matches the
// built-in denoiser's (shaders/pipelines/denoise/main.comp.slang): a near-black texel would otherwise
// amplify the signal without bound.
public static const float NRD_MIN_DEMOD = 0.01;

public float3 nrdDemodFactor(float3 albedo) {
    return max(albedo, float3(NRD_MIN_DEMOD));
}

// Qualcomm Snapdragon Game Super Resolution 2 (SGSR2) inputs. SGSR2 is a three-pass temporal upscaler
// (convert -> activate -> upscale) whose passes share one parameter block; the host fills it once per frame.
// Only the fields this integration actually uses are present: upstream also carries a clipToPrevClip matrix
// for its "no motion vector available" fallback, which cannot be used here — this renderer always writes a
// motion guide, and its zero is a legitimately static pixel rather than "unknown".
public struct SgsrParams {
    public int renderWidth;
    public int renderHeight;
    public int displayWidth;      // upscale output size
    public int displayHeight;
    public float jitterX;         // signed primary-ray jitter in render pixels, the value DLSS-RR receives
    public float jitterY;
    // Tone-map offset of the YCoCg pack, in the domain the stored colour is already in: the passes encode
    // colour / (max(colour) + offset) and recover the magnitude from the remaining headroom, so the offset has
    // to be comparable to the stored values or the quantisation of the luma channel swamps that headroom.
    public float toneMapOffset;
    // tan(fovY/2) * renderWidth / renderHeight, SGSR2's horizontal camera-fov term.
    public float cameraFovHor;
    // Depth conversion bounds: the guide stores this renderer's reversed-Z depth, SGSR2 wants standard [0,1]
    // depth (near -> 0, far and sky -> 1). Distances come from the projection terms below, so it stays
    // correct for any projection that mapping fits.
    public float cameraNear;
    public float cameraFar;
    public float projM22;         // frame projection [2][2]
    public float projM32;         // frame projection [3][2]
    public int reset;             // 1 = history invalid this frame (first frame, resize, camera cut)
};

// NRD front-end inputs. The path tracer writes pre-exposed radiance and raw hit distances; NRD wants the
// material factor divided out, YCoCg-packed, with hit distance normalized through its own hit-distance
// parameters (which must match the library build). The signal stays pre-exposed: the exposure is what keeps
// physical-unit radiance inside fp16 at all, and demodulation can amplify it by 1/NRD_MIN_DEMOD.
public struct NrdFrontendPush {
    public int width;
    public int height;
    public float projM22;         // frame projection [2][2]
    public float projM32;         // frame projection [3][2]
    public float hitDistA;        // ReblurSettings::hitDistanceParameters (REBLUR only)
    public float hitDistB;
    public float hitDistC;
    public float denoisingRange;  // CommonSettings::denoisingRange, the same value the shim passes to NRD
    // 1 = REBLUR packs a hit distance normalized through the parameters above, 0 = RELAX packs the raw
    // world-space distance. Both denoisers read the same channel with different conventions, so the packer
    // follows whichever one the shim built (see RtNrd.denoiserId).
    public uint normalizeHitDistances;
};

// NRD back-end inputs: turns the denoised YCoCg lanes back into the pre-exposed linear radiance the rest
// of the frame graph expects.
public struct NrdBackendPush {
    public int width;
    public int height;
    public float pad0;
    public float pad1;
};

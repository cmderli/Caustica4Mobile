# caustica_nrd — NVIDIA Real-time Denoisers (NRD) C-ABI shim

`native/nrd_shim` builds `libcaustica_nrd.so`: a small shared library that drives
[NVIDIA NRD](https://github.com/NVIDIA-RTX/NRD) v4.17.3 against the Vulkan
`VkDevice`, `VkPhysicalDevice` and `VkCommandBuffer` the Java side already owns.
It exists because NRD makes **no** graphics-API calls itself — it hands out POD
pipeline/resource descriptions plus embedded SPIR-V bytecode, and something has
to turn those into Vulkan objects and dispatches. That "something" is this shim;
the Java caller only has to publish seven textures and one command buffer per
frame.

The ABI is a flat C surface designed for `java.lang.foreign` (FFM) downcalls: no
callbacks, no exported C++ types, no exceptions across the boundary, and every
Vulkan handle passed as a `uint64_t`.

---

## 1. The ABI (the Java side is written against this verbatim)

```c
#define CAUSTICA_NRD_OK 0

typedef struct CausticaNrdImageView { uint64_t image; uint64_t view; } CausticaNrdImageView;

typedef struct CausticaNrdCreateInfo {
    uint64_t device;            // VkDevice
    uint64_t physicalDevice;    // VkPhysicalDevice
    uint64_t getDeviceProcAddr; // PFN_vkGetDeviceProcAddr
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t denoiser;          // 0 = REBLUR_DIFFUSE_SPECULAR, 1 = RELAX_DIFFUSE_SPECULAR
    uint32_t reserved;
} CausticaNrdCreateInfo;

typedef struct CausticaNrdFrameInfo {
    uint64_t commandBuffer;     // VkCommandBuffer, already recording
    uint32_t frameIndex;        // advances by exactly 1 per denoised frame
    float    jitterX, jitterY;  // render-pixel jitter applied to the primary ray this frame
    float    denoisingRange;
    float    viewToClip[16];
    float    clipToView[16];
    float    worldToView[16];
    float    viewToWorld[16];
    CausticaNrdImageView viewZ;                 // r32f, render res
    CausticaNrdImageView normalRoughness;       // render res
    CausticaNrdImageView motion;                // rg16f, render res
    CausticaNrdImageView diffRadianceHitDist;   // rgba16f, render res
    CausticaNrdImageView specRadianceHitDist;   // rgba16f, render res
    CausticaNrdImageView outDiffRadianceHitDist;
    CausticaNrdImageView outSpecRadianceHitDist;
    float worldPrevToWorld[16]; // previous world -> current world; all-zero = derive
} CausticaNrdFrameInfo;

int32_t caustica_nrd_create(const CausticaNrdCreateInfo* info);
int32_t caustica_nrd_denoise(const CausticaNrdFrameInfo* frame);
void    caustica_nrd_destroy(void);
```

`caustica_nrd_create` / `caustica_nrd_denoise` return `CAUSTICA_NRD_OK` (0) on
success and a negative code otherwise (the Java side only has to test `!= 0`):

| code | meaning |
|---|---|
| `-1` | bad arguments (null pointer, zero image handle, zero `denoisingRange`, all-zero matrix, `renderWidth/Height` outside `1..65535`, unknown `denoiser`) |
| `-2` | a Vulkan object could not be created |
| `-3` | an `nrd::` call failed (see the `[caustica_nrd]` line on stderr) |
| `-4` | internal inconsistency / C++ exception |

Every failure also prints a `[caustica_nrd] ...` line to stderr; the shim never
throws across the ABI boundary.

### Struct layout (for the FFM `MemoryLayout`)

Compiled with the host ABI (`sizeof`/`offsetof`, no padding surprises):

| struct | size | field offsets |
|---|---|---|
| `CausticaNrdImageView` | 16 | image 0, view 8 |
| `CausticaNrdCreateInfo` | 40 | device 0, physicalDevice 8, getDeviceProcAddr 16, renderWidth 24, renderHeight 28, denoiser 32, reserved 36 |
| `CausticaNrdFrameInfo` | 520 | commandBuffer 0, frameIndex 8, jitterX 12, jitterY 16, denoisingRange 20, viewToClip 24, clipToView 88, worldToView 152, viewToWorld 216, viewZ 280, normalRoughness 296, motion 312, diffRadianceHitDist 328, specRadianceHitDist 344, outDiffRadianceHitDist 360, outSpecRadianceHitDist 376, worldPrevToWorld 392, worldToViewPrev 456 |

Fields are appended at the end on purpose: adding one keeps every earlier offset
stable, so a host compiled against an older revision of this table stays valid
until it uses the new field.

### Lifecycle

* `caustica_nrd_create` may be called again while an instance is alive; it
  releases the previous one first. That is the resize path (NRD has no resize —
  resolution changes need a full rebuild of pipelines and pool textures).
* `caustica_nrd_destroy` releases every Vulkan object the shim owns. Vulkan
  objects it does **not** own (the caller's textures, device, command buffer)
  are left untouched.
* The shim is a process-wide singleton and is **not thread-safe**: NRD instances
  are stateful and `GetComputeDispatches` returns instance-owned memory. Call it
  from one thread (or one thread at a time).

### What the shim owns

Shader modules, compute pipelines, both descriptor set layouts, the shared
pipeline layout, `3` per-frame descriptor pools, the sampler objects (bound as
immutable samplers in set 1), the constants uniform buffer, and every NRD
internal pool texture (`InstanceDesc::permanentPool` / `transientPool` — 13
permanent + 8 transient for `REBLUR_DIFFUSE_SPECULAR`).

Everything Vulkan-specific comes from the descriptions NRD returns
(`nrd::GetInstanceDesc`, `nrd::LibraryDesc::spirvBindingOffsets`): set 0 =
resources (`SAMPLED_IMAGE` at `textureOffset`, `STORAGE_IMAGE` at
`storageTextureAndBufferOffset`), set 1 = two immutable `CLAMP_TO_EDGE` samplers
(binding 0 nearest, 1 linear) plus the constants `UNIFORM_BUFFER_DYNAMIC` at
binding `constantBufferOffset + constantBufferRegisterIndex`. The pipeline
layout has `pushConstantRangeCount = 0`; NRD passes its constants through the
uniform buffer and its SPIR-V has no push-constant block. Binding offsets and
pool sizes are read at runtime and never hardcoded.

---

## 2. Build

Requirements: CMake ≥ 3.22, a C++17 compiler, Ninja (or any generator), git and
network access at configure time (NRD, ShaderMake, MathLib and — when no Vulkan
SDK is installed — the Khronos Vulkan-Headers are fetched), plus a **DXC with
SPIR-V support** (NRD compiles and embeds its shaders at build time). No Vulkan
loader is linked: the shim resolves every entry point through the
`PFN_vkGetDeviceProcAddr` the caller supplies.

```sh
cd /home/mizuki/Documents/CausticaForMobile

export DXC_PATH=/tmp/dxctest/linux_dxc_2026_07_29.x86_x64/bin/dxc
export DXC_SPIRV_PATH=/tmp/dxctest/linux_dxc_2026_07_29.x86_x64/bin/dxc

cmake -S native/nrd_shim -B build/native/nrd_shim/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DSHADERMAKE_DXC_VK_PATH="$DXC_SPIRV_PATH"

cmake --build build/native/nrd_shim/build --parallel
```

Notes that cost time to rediscover:

* `-DSHADERMAKE_DXC_VK_PATH=...` is required **when a system `dxc` exists**.
  NRD 4.17 fetches ShaderMake itself; that project sets `SHADERMAKE_PATH` and
  then resolves its own `SHADERMAKE_DXC_VK_PATH` with `find_program(dxc)`. NRD
  only falls back to the `DXC_PATH`/`DXC_SPIRV_PATH` environment variables when
  `SHADERMAKE_PATH` is unset, so the environment variables alone are **not**
  enough once a `dxc` is on `PATH` (it silently used `/usr/bin/dxc` here).
  Pre-seeding the ShaderMake cache variable makes the shader pass use exactly
  the requested compiler; the build log prints it as `--compiler <path>`.
* `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is only needed when a fetched dependency
  declares an ancient `cmake_minimum_required` and CMake ≥ 4 refuses it.
* `CMAKE_BUILD_TYPE` defaults to `Release` when unset (mirrors
  `native/ngx_shim`).

### Cross-compiling for arm64 (Android and Linux)

`build_cross.sh` stages an arm64 `libcaustica_nrd.so` for each target beside the host build:

```sh
# Arm64 Android (NDK) and Arm64 Linux (Zig). DXC is the host shader compiler, as for the host build.
export DXC_PATH=/path/to/dxc

native/nrd_shim/build_cross.sh                     # every target this host can build
native/nrd_shim/build_cross.sh android             # Android only, needs the NDK
ZIG=/path/to/zig native/nrd_shim/build_cross.sh linux
```

| target | artifact | toolchain | runtime dependencies |
|---|---|---|---|
| `android-arm64` | `build/native/nrd_shim/android-arm64/release/libcaustica_nrd.so` | NDK clang, `arm64-v8a`, API 26, `c++_static` | `libm`, `libdl`, `libc` |
| `linux-arm64` | `build/native/nrd_shim/linux-arm64/release/libcaustica_nrd.so` | Zig, `-target aarch64-linux-gnu.2.17` | `libm`, `libpthread`, `libc`, `libdl` |

`gradle bundleNrdNatives` picks up every `build/native/nrd_shim/<platform>/release/` directory it
finds, so both are packaged as `caustica/natives/<platform>/libcaustica_nrd.so` next to the host
artifact, and `RtNrd` selects the directory from the JVM's OS/arch (Android identifies itself as
Linux, so it is detected by `java.vm.name`/`java.runtime.name` first).

What the cross builds rely on:

* **Run the host build once first.** They reuse its fetched `_deps` (`nrd-src`, `mathlib-src`,
  `shadermake-src`, `vulkan-headers-src`), its host `ShaderMake` and a host DXC, and are configured
  with `FETCHCONTENT_FULLY_DISCONNECTED=ON`, so a cross build performs no network I/O. NRD's shaders
  are target-independent and are compiled by host tools; only the C++ library is cross-compiled.
* **`sse2neon`**: MathLib fetches it for every ARM target. The script expects a checkout at
  `build/native/nrd_shim/build/_deps/sse2neon-src` (a shallow clone of
  `https://github.com/DLTcollab/sse2neon.git` is enough).
* **Zig** supplies the arm64 Linux compiler, its libc and its libc++ in one tarball, and `zig c++`
  links libc++ statically — which is why the artifact needs no `libc++.so` on the target. Pinning the
  target to glibc 2.17 (the aarch64 baseline) is what keeps the required symbol floor low.
* Zig's driver rejects `-Wl,--exclude-libs,ALL`, so `CMakeLists.txt` checks for that flag and falls
  back to an export version script, which pins the same three-symbol ABI (GNU ld, LLD and the NDK
  linker all keep using `--exclude-libs`).
* Both artifacts link with `-Wl,--no-undefined`, so every import has to resolve from their own
  `DT_NEEDED` list. That is also what proves the shim never calls a Vulkan entry point directly.
* Debug info is stripped with `llvm-objcopy --strip-debug` (~12 MiB → ~5.4 MiB) unless
  `KEEP_DEBUG_INFO=1` is set. The dynamic symbol table — and therefore the three ABI entry points —
  is untouched.

### Environment variables

| variable | purpose |
|---|---|
| `DXC_PATH`, `DXC_SPIRV_PATH` | DXC used for NRD's shader pass (see the caveat above) |
| `VULKAN_SDK` | optional; `$VULKAN_SDK/include` is used for `vulkan/vulkan.h` when present |
| `NRD_SHIM_VULKAN_INCLUDE_DIR` (CMake `-D`) | explicit override for the Vulkan headers |
| `FETCHCONTENT_SOURCE_DIR_NRD` (CMake `-D`) | offline build against a pre-downloaded NRD checkout |

NRD's configure options are pinned as part of the ABI:
`NRD_STATIC_LIBRARY=ON`, `NRD_NRI=OFF`, `NRD_EMBEDS_SPIRV_SHADERS=ON`
(**mandatory** — without it `PipelineDesc::computeShaderSPIRV.bytecode` is null
and no pipeline can be created), `NRD_EMBEDS_DXIL_SHADERS=OFF`,
`NRD_EMBEDS_DXBC_SHADERS=OFF`, `NRD_SUPPORTS_VIEWPORT_OFFSET=OFF`,
`NRD_SUPPORTS_CHECKERBOARD=ON`, `NRD_SUPPORTS_HISTORY_CONFIDENCE=ON`,
`NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX=ON`, `NRD_SUPPORTS_ANTIFIREFLY=ON`,
`REBLUR_PERFORMANCE_MODE=OFF`, and the two encodings below. No compile options or
definitions are added to the `NRD` target (it builds with `-Wextra -Werror`).

### IN_NORMAL_ROUGHNESS encoding

```cmake
set(NRD_NORMAL_ENCODING 0 CACHE STRING "" FORCE)      # 0 = NormalEncoding::RGBA8_UNORM
set(NRD_ROUGHNESS_ENCODING 1 CACHE STRING "" FORCE)   # 1 = RoughnessEncoding::LINEAR
```

Encoding `0` lets the host write `IN_NORMAL_ROUGHNESS` from Slang as a plain
`[format("rgba8")]` storage image; Slang has no `r10g10b10a2_unorm` format string,
so NRD's default (`2 = R10_G10_B10_A2_UNORM`) would force a format-less
`[format("unknown")]` write with hand-packed bits. Roughness value `1` is
`LINEAR` (NRD's recommended setting) — note that `2` is `SQRT_LINEAR`, `0` is
`SQ_LINEAR`.

The host must therefore write `float4(normal * 0.5 + 0.5, linearRoughness)` and
read nothing back (NRD decodes `p.xyz * 2 - 1`, roughness as-is). Material IDs
are not available with this encoding, so NRD's material-ID settings keep their
`999` defaults.

`caustica_nrd_create` logs what the linked library reports and **fails** if it
does not match the pinned values:

```
[caustica_nrd] NRD v4.17.3: normalEncoding=0 (expected 0, 0 = RGBA8_UNORM), roughnessEncoding=1 (expected 1, 1 = LINEAR)
```

---

## 3. Artifact

```
build/native/nrd_shim/release/libcaustica_nrd.so
```

5,000,368 bytes, `sha256
4a5e7243c812a340352f2151a11005f5c6d9b9436176bff9ddac28afafd3a262`. A from-scratch
configure+build (empty binary directory, separate output root) reproduces that
hash byte for byte.

One self-contained shared library: NRD 4.17.3 (static, with the SPIR-V bytecode
embedded inside) plus this shim. NRD source, its fetched dependencies and its
generated `_Shaders` all live under `build/native/nrd_shim/build` — NRD is
fetched at configure time and is never vendored into this repository.

The arm64 builds land in the same shape under `build/native/nrd_shim/<platform>/release/`
(5.5 MiB for Linux arm64, 5.4 MiB for Android arm64, both with debug info stripped; see the
cross-compiling section above).

Exported symbols — the dynamic symbol table contains nothing else (NRD's
internals are built hidden, this translation unit is built with
`CXX_VISIBILITY_PRESET hidden`, and the static libraries are linked with
`-Wl,--exclude-libs,ALL`, or with a three-symbol export version script where the
compiler driver rejects that flag):

```
$ nm -D --defined-only build/native/nrd_shim/release/libcaustica_nrd.so
0000000000007cc0 T caustica_nrd_create
00000000000032c0 T caustica_nrd_denoise
00000000000053a0 T caustica_nrd_destroy
```

NRD is licensed by NVIDIA (NVIDIA RTX SDKs License) and is **not** relicensed by
this project's LGPL: `libcaustica_nrd.so` must be shipped as a separate object
code artifact with NVIDIA's notice ("This software contains source code provided
by NVIDIA Corporation."), an entry in `THIRD_PARTY_NOTICES.md`, and credit
attribution in end-user documentation.

---

## 4. `caustica_nrd_create`

`denoiser` selects `REBLUR_DIFFUSE_SPECULAR` (0) or `RELAX_DIFFUSE_SPECULAR` (1);
both consume the same seven textures. `renderWidth`/`renderHeight` are the render
resolution NRD allocates its internal pools for (≤ 65535, and NRD has no resize:
destroy + create on a resolution change).

The shim verifies the linked library version (4.17.3), the two encodings and the
NRD set indices at create time, then builds samplers, layouts, pipelines,
descriptor pools, the constants UBO and the pool textures.

Denoiser settings are NRD's defaults, applied once at create:

* `ReblurSettings{}` — `hitDistanceParameters {A=3.0, B=0.1, C=20.0}`,
  antilag `{2.0, 3.0}`, `maxAccumulatedFrameNum=30`,
  `maxFastAccumulatedFrameNum=6`, `maxStabilizedFrameNum=63`,
  `historyFixFrameNum=3`, prepass radii `30/50`,
  `checkerboardMode=OFF`, `hitDistanceReconstructionMode=OFF`,
  `enableAntiFirefly=true`, `returnHistoryLengthInsteadOfOcclusion=false`.
* `RelaxSettings{}` — `atrousIterationNum=5`, diffuse/specular accumulation
  `30/30`, fast history `6`, `enableAntiFirefly=false`, same checkerboard and
  hit-distance-reconstruction choices.

### Hit-distance normalization (the caller must match it)

REBLUR's front-end must normalize hit distances with the same A/B/C the denoiser
uses, or the hit-distance signal is biased:

```
f(viewZ, roughness) = (A + |viewZ| * B) * lerp(C, 1, specMagicCurve(roughness, 0.5))
normHitDist         = saturate(hitT / f)
```

With the defaults this is `REBLUR_FrontEnd_GetNormHitDist(hitT, viewZ,
float3(3.0, 0.1, 20.0), roughness)` from `NRD.hlsli`. `A` is a constant in world
units, `B` is the view-Z-proportional term (10 cm at 1 m, 1 m at 10 m, 10 m at
100 m) and `C > 1` clamps hit distance to a larger value at low roughness. The
defaults assume a scene measured in metres/blocks, which is what this renderer
uses. RELAX uses the same numbers (its front-end is
`RELAX_FrontEnd_PackRadianceAndHitDist(radiance, hitDistance)`).

---

## 5. `caustica_nrd_denoise`

Call it once per denoised frame, inside the recording of `commandBuffer`.
`frameIndex` must advance by exactly 1 per denoised frame (it is NRD's
accumulation counter **and** the shim's ring index); `frameIndex == 0` requests a
full history reset (`CLEAR_AND_RESTART`), which is also what NRD forces on the
first frame after create. The shim never submits and never waits.

Per frame it: resets the frame's descriptor pool → fills `CommonSettings` →
`SetCommonSettings` → `GetComputeDispatches` → for every reported dispatch
allocates a descriptor set, writes exactly the bindings the dispatch names,
records the image barriers it needs, copies that dispatch's constants into its
own uniform-buffer slice with `vkCmdUpdateBuffer`, then binds pipeline + both
descriptor sets (`firstSet = 0`), records the barriers and calls
`vkCmdDispatch(gridWidth, gridHeight, 1)` with NRD's grid — the grid is never
recomputed.

### Caller contract

* **Layouts.** Every bound texture must be in `VK_IMAGE_LAYOUT_GENERAL` when
  `caustica_nrd_denoise` is called, and is left in
  `VK_IMAGE_LAYOUT_GENERAL` when it returns. Guides are normally produced by
  storage-image writes, so this is the natural state.
* **Usage.** All seven textures need `VK_IMAGE_USAGE_SAMPLED_BIT |
  VK_IMAGE_USAGE_STORAGE_BIT`, one mip, one layer, and a single 2D colour view
  (the shim uses the same view for `SAMPLED_IMAGE` and `STORAGE_IMAGE` bindings,
  and NRD writes `IN_MV` internally — REBLUR's temporal-stabilization pass
  patches motion vectors in place, so do not reuse them as "raw" MVs
  afterwards).
* **Synchronization.** The shim records compute→compute barriers for NRD's own
  dispatches only. The caller must have the guide-writing passes synchronized
  with the NRD block (one barrier before `denoise` making the guide writes
  visible to `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`).
* **Frames in flight.** The shim sizes its descriptor pools and constants ring
  for 3 frames in flight and resets pool `frameIndex % 3`. Do not have more than
  3 denoised frames recording/executing at once.
* **Device setup.** The embedded SPIR-V is **version 1.5**, so the device needs
  `apiVersion >= VK_API_VERSION_1_2`, plus these features/extensions (all
  verified against the shader modules with the Khronos validation layer):
  * `VkPhysicalDeviceFeatures::shaderStorageImageWriteWithoutFormat` **and**
    `shaderStorageImageReadWithoutFormat` — NRD's outputs are `[[vk::image_format("unknown")]]`
    storage images and the spatial passes also read them.
  * `VK_KHR_compute_shader_derivatives` with
    `VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR::computeDerivativeGroupQuads`
    — REBLUR's Blur, HistoryFix and PostBlur shaders call `QuadReadAcrossX/Y`
    inside a compute shader. **RELAX does not need this** (its shaders use no
    quad operations), which matters on hardware without the extension.
  * `VK_KHR_compute_shader_derivatives` is *not* satisfied by apiVersion 1.4
    alone on the loader/layer tested here: the extension name and the feature
    both have to be enabled (chain `VkPhysicalDeviceFeatures2` →
    `VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR`).
  * Vulkan 1.1-device errors look like `SPIR-V binary version 1.5 for target
    environment SPIR-V 1.3`, and a missing feature surfaces as
    `VUID-VkShaderModuleCreateInfo-pCode-0873x` at `vkCreateShaderModule`.
* **Queue family.** Record NRD on the same queue family that produced the
  guides, or make the textures `VK_SHARING_MODE_CONCURRENT`; the shim cannot know
  how to emit a queue-family ownership transfer.

### Pixels outside `denoisingRange`

NRD never writes its output for them, which is how sky and other unlit pixels are
excluded, so the host has to composite those pixels itself (from its own
un-denoised signal). The test NRD applies is
`abs(IN_VIEWZ) < CommonSettings::denoisingRange`, i.e. the *magnitude* of view-Z:
`Shaders/Common.hlsli` unpacks as `abs(z * viewZScale)` with `viewZScale = 1.0`,
and every `IsInDenoisingRange` call site runs on the unpacked value. Two
consequences for the host:

* mark sky with a view-Z whose **magnitude** exceeds `denoisingRange` (the sign
  does not matter, NRD takes the absolute value);
* keep `denoisingRange` **above the tracer's reach**. Any surface beyond it is
  classified as sky by NRD and left unwritten, which is indistinguishable from a
  hole in the denoised output.

---

## 6. Matrix conventions

NRD's `CommonSettings` requires **column-major storage** with **column-vector
usage**, i.e. for the `float m[16]` the caller passes:

```
m[column * 4 + row]  =  M[row][column]
clip                 =  M * viewPosition        (M applied to a column vector)
depth                =  z / w                   (no reversed-Z, INF far plane allowed)
```

A matrix from GLM (`glm::value_ptr(mat4)`) is already in that layout; a
row-major array (e.g. a C `float m[4][4]` used as `m[row][col]`) must be
transposed before it is handed over, or the denoiser smears. The matrices must
be **non-jittered**; the jitter is passed separately through `jitterX/jitterY`
(sample offset in render pixels, `[-0.5, 0.5]`).

The ABI carries only the current frame, so the shim tracks the previous frame's
`viewToClip`, `worldToView` and jitter itself and fills
`viewToClipMatrixPrev`/`cameraJitterPrev`. `worldToViewMatrixPrev` is taken from
the `worldToViewPrev` field when that array is non-zero and from the cached
previous `worldToView` otherwise. `clipToView` is accepted for the caller's
convenience but is not forwarded — NRD has no such field and derives its own
inverse.

`worldToViewPrev` is the current world space → **previous** view space transform.
A world-stable host can leave it zero, because the cached previous `worldToView`
is then already correct:

* if `worldToViewPrev` is **all zero**, the shim uses the previous frame's
  `worldToView`;
* otherwise the array is copied straight into
  `CommonSettings::worldToViewMatrixPrev`.

`worldPrevToWorld` is the previous world space → current world space transform
used for virtual-motion (specular) reprojection:

* if `worldPrevToWorld` is **all zero**, the shim derives it as
  `viewToWorld(current) * worldToView(previous)`;
* otherwise the array is copied straight into
  `CommonSettings::worldPrevToWorldMatrix`.

This renderer works in camera-relative world space, where that product is
identity even while the camera translates, so the host passes
`worldPrevToWorld` explicitly (a camera-delta translation) *and*
`worldToViewPrev` as `worldToView(previous) * inverse(worldPrevToWorld)`, which
is what puts a static point's history on the right pixel while the camera moves.
On the first frame after create there is no history: an all-zero
`worldPrevToWorld` falls back to NRD's identity default and the shim leaves
`worldToViewMatrixPrev` untouched.

`viewToClip` must be the standard (non-reversed-Z) projection that matches the
`IN_VIEWZ` values supplied: NRD decomposes it to detect handedness and the
frustum, converts to LH internally and never sees the depth buffer. Render with
reversed-Z if you like, but compute `viewZ` yourself and pass the equivalent
standard projection here.

---

## 7. What was verified

Built with the commands in §2 (NRD's 159 shader compilations ran through the DXC
at `/tmp/dxctest/linux_dxc_2026_07_29.x86_x64/bin/dxc`; the build log prints it as
`--compiler <path>`), then executed against an NVIDIA GeForce RTX 4070 (driver
610.57.04, Vulkan 1.4.341) with a 320×240 synthetic constant scene and 40 frames:

* `REBLUR_DIFFUSE_SPECULAR` and `RELAX_DIFFUSE_SPECULAR`, both including the
  `CLEAR_AND_RESTART` first frame and the ~24 clear dispatches it injects, the
  alternating-jitter path, the `worldPrevToWorld` derivation and the explicit
  `worldPrevToWorld` path, and a second `caustica_nrd_create` on a live instance
  (the resize/recreate path). Both converged to the expected values (diffuse and
  specular outputs within 0.003 of their inputs, correctly separated, i.e. no
  binding swap), and RELAX's output alpha settled at its history-length cap (30).
* The same two runs under `VK_LAYER_KHRONOS_validation` with
  `VALIDATION_CHECK_ENABLE_SYNCHRONIZATION_VALIDATION`, on a device created with
  `VK_KHR_compute_shader_derivatives` + the format-less read/write features:
  **0 errors and 0 warnings** — no layout, barrier, descriptor or pipeline
  complaints from the shim's recording.
* A control run on a Vulkan 1.1 device with no extensions reproduces exactly the
  device-setup errors listed in §5 and nothing else, which is how the required
  feature list above was established.
* The build is reproducible: a from-scratch configure+build into a separate
  output root produced the identical `sha256`.

The arm64 Linux artifact was also loaded and exercised on this machine under
`qemu-aarch64-static` (with an arm64 Ubuntu base rootfs supplying the dynamic loader and libc):
`dlopen` succeeds, all three entry points resolve, and `caustica_nrd_create(NULL)` /
`caustica_nrd_denoise(NULL)` return the ABI's `-1` "bad arguments" code, so the C++ runtime, the
static initializers and the C ABI all work on the target architecture.

Not verified: behaviour on a mobile driver (no Android device here — the Android artifact is
verified statically: architecture, API level, `DT_NEEDED`, undefined-symbol check and exported
symbol set), and image quality on real path-traced input (the test scene is constant by design).

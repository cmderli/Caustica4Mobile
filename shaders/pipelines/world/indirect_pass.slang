// Indirect pass of the wavefront renderer.
//
// Resumes the continuations the primary pass queued and runs the bounce loop: NEE toward the sun, RIS
// emitter lighting, thin-surface SSS, BSDF continuation and Russian roulette. It captures no guides and
// owns no motion vectors — guides and primary are deliberately absent from the include list
// below, so guide state cannot leak into this pass by accident.
//
// Lighting is iterative here rather than recursive from a hit stage, so nothing recurses at all.
// At each surface vertex:
//   * NEE toward the sun — a single analytic directional light, gated by one hit-object visibility
//     query (TerminateOnFirstHit | SkipClosestHit). Lambertian BRDF = albedo/PI.
//   * a cosine-weighted Lambertian continuation ray (throughput *= albedo; the PI and cos cancel).
// Sky / ambient light needs no separate AO pass: continuation rays that escape land in world.rmiss,
// which returns sky radiance — so soft sky fill and contact AO both fall out of the path integral.
//
// Each frame emits a single SPP-averaged (still noisy) estimate; temporal convergence is the denoiser's
// job. With RR disabled the output is the raw noisy path trace — a reference view, not a converged
// image. Output is scene-linear ACEScg HDR (R16G16B16A16_SFLOAT); display mapping happens afterward.
//
// The body below is shared by both backends: indirect.rgen.slang is the ray-tracing entry point (with an
// ordinary TraceRay form and an optional EXT SER variant) and indirect_rq.comp.slang dispatches the same
// code as a compute grid over the render target. restir.slang reuses tracePath for its suffix path; this
// pass is also dispatched after ReSTIR's temporal pass and covers only the pixels ReSTIR hands back.
import world_common;
import world_core;
import stage;
import math;
import medium;
import segment;
import water;
import trace;
// SER reorders a hit object, which only the ray-tracing backend has; the ray-query traversal has no
// equivalent and never imports it.
#if defined(CAUSTICA_ENABLE_EXT_SER) && !defined(CAUSTICA_RAY_QUERY)
import trace_ser;
#endif
import lighting;
import sky;
import bindings;

// Atmospheric transmittance LUT (RtSkyLut), also bound to world.rmiss at the same binding: raygen reads it
// to colour the NEE sun/moonlight, the miss shader reads it to tint the visible discs and stars.

// Reported as a lane's hit distance when the ray escaped to sky. The renderer works in blocks and the
// scene is bounded far below this, so it reads as "no hit" without being a special case downstream.
public static const float SKY_HIT_DISTANCE = 1.0e5;
// Radiance split by the lobe that produced it, plus each lane's first-bounce hit distance in blocks. A
// path belongs to the lane of the FIRST lobe sampled at the primary vertex, because that is the lobe a
// denoiser reconstructs; direct lighting at that vertex is evaluated for both lobes at once (one shadow
// ray), so it is split by BRDF component instead. Adding the two lanes back together reproduces exactly
// the combined radiance the display chain consumes when no denoiser asks for the split.
public struct PathRadiance {
    public float3 diffuse;
    public float3 specular;
    public float diffHitDist;
    public float specHitDist;
};

public PathRadiance tracePath(PathSegment seg, uint2 pix, uint sampleIndex) {
    float3 Ldiff = float3(0.0, 0.0, 0.0);
    float3 Lspec = float3(0.0, 0.0, 0.0);
    // -1 until the primary vertex picks a continuation lobe, then 0 diffuse / 1 specular for the rest of
    // the path. Everything after that choice belongs to the chosen lane, including deeper bounces.
    int lane = -1;
    // Negative until the lane's own ray has been traced, so the first hit after the choice wins.
    float diffHitDist = -1.0;
    float specHitDist = -1.0;
    float3 ro = seg.ro;
    float3 rd = seg.rd;
    float3 throughput = seg.throughput;
    // Pass A is fixed at one sample per pixel. Decorrelate each Pass B resample here so configured SPP
    // produces independent lighting/BSDF paths from the shared terminal continuation.
    uint seed = seg.seed ^ sampleIndex * 2246822519u;
    seed = pcg(seed);
    float rayConeWidth = seg.rayConeWidth;
    float rayConeSpread = max(seg.rayConeSpread, RAY_CONE_MIN_SPREAD);
    int maxBounces = int(worldPush.maxBounces);
    int rrStart = maxBounces <= 3 ? 1 : 2;
    // RIS emitter NEE: direct lighting from block emitters is active when lights are published and the
    // candidate count is non-zero. Otherwise emitters contribute only when a path hits them.
    bool risOn = worldPush.risCandidates > 0u && worldPush.lightCount > 0u && pc.lightBufAddr != 0;
    // Independent light-proposal RNG. Screen-space coherent selection is intentionally deferred until
    // there is a real presampled RIS tile buffer; direct sharing made tile-shaped lighting noise visible.
    // Measured: making a 16x16 tile share this seed recovers at most 1.3ms, so a future pool should share
    // the POOL rather than the seed — the win is in removing dependent loads, not in coherence.
    uint proposalSeed = pix.x * 1973u + pix.y * 9277u + 26699u
            ^ worldPush.frameIndex * 2654435761u ^ sampleIndex * 2246822519u;
    proposalSeed = pcg(proposalSeed);
    // The medium the segment starts in. For the camera segment that is water when the eye is submerged
    // (so the first ray already carries the right relative index and absorption); for a segment split off
    // a dielectric it is whatever the parent was travelling through.
    MediumStack medium = seg.medium;
    bool waterWaves = (worldPush.flags & 16u) != 0u; // animated wave-normal perturbation
    // Sky-disc gate (see world.rmiss): the sun/moon disc is shown to the primary ray and to rays spawned
    // by a specular/dielectric bounce (mirror reflections + refractions of the sun/moon), but hidden from
    // diffuse-sampled continuation rays — those are already covered by the sun NEE, so the tiny disc
    // would be a double-counted firefly. Reset true on every specular/water bounce, false on a diffuse one.
    bool showCelestial = seg.showCelestial;
    // Pass B treats every hit it sees as one indirect-depth step, regardless of material or selected
    // continuation lobe. Deliberately start at zero instead of seg.bounce: the first hit of either
    // queued split branch is still the primary-terminal shading point. Hits 0 and 1 retain SSS; RIS
    // remains active with its full configured candidate count at every hit. Pass A's primary/interface
    // prefix is outside this SSS quality budget.
    int indirectDepth = 0;
    for (int bounce = seg.bounce; bounce <= maxBounces; bounce++) {
        // Radiance SBT records run any-hit only for true alpha cutout. Translucent/water go straight to
        // closest-hit for dielectric handling. Geometry is double-sided; the chit flips the normal.
        // Primary (bounce 0) is the camera ray (CULL_PRIMARY): sees particles but not the first-person
        // player. Bounce rays are secondary (CULL_SECONDARY): exclude particles, include the player.
#if defined(CAUSTICA_ENABLE_EXT_SER) && !defined(CAUSTICA_RAY_QUERY)
        // SER lifetime phase: keep paths that are not roulette-eligible, paths that may terminate via
        // roulette, and paths guaranteed to end at the bounce cap in separate coherence groups.
        uint pathPhaseHint = bounce >= maxBounces ? 2u : (bounce >= rrStart ? 1u : 0u);
        traceRadianceReordered(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0.0, rd, 10000.0,
                               showCelestial, rayConeWidth, rayConeSpread, pathPhaseHint);
#else
        traceRadiance(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0.0, rd, 10000.0,
                      showCelestial, rayConeWidth, rayConeSpread);
#endif

        // First hit along whichever lane the primary vertex chose: this trace is the ray that lobe
        // generated, so its length is that lane's hit distance. A miss reports the sky sentinel, which a
        // denoiser reads as "infinitely far" (maximum blur) rather than "close".
        if (lane == 1 && specHitDist < 0.0) {
            specHitDist = payload.hitT < 0.0 ? SKY_HIT_DISTANCE : payload.hitT;
        } else if (lane == 0 && diffHitDist < 0.0) {
            diffHitDist = payload.hitT < 0.0 ? SKY_HIT_DISTANCE : payload.hitT;
        }

        if (payload.hitT < 0.0) {
            // A ray still in water should leave through a water interface. A miss instead means the
            // streamed/open volume has no known exit; do not reinterpret that unknown region as air
            // and reveal sky.
            if (medium.current.water) {
                break;
            }

            // world.rmiss writes sky radiance (gradient + sun/moon disc + stars) into the payload's
            // surface words as a full float3, so this is absolute scene units with nothing to undo. The
            // packed show-celestial flag decides whether this path segment may see the bright disc.
            float3 sky = payloadSky();
            // Escaped to sky. A camera ray that hit nothing is the background: no lobe chose it, so it
            // belongs to the diffuse lane.
            if (lane == 1) {
                Lspec += throughput * sky;
            } else {
                Ldiff += throughput * sky;
            }
            break;
        }

        // Beer–Lambert: attenuate along the segment just travelled by the medium it lay inside. Applies
        // to every hit reached while inside a volume dielectric (its own exit face, or whatever content
        // lies within it), shifting the transmitted radiance with distance. Air's extinction is zero, so
        // the ordinary case costs one comparison.
        if (any(medium.current.extinction > 0.0)) {
            throughput *= exp(-medium.current.extinction * payload.hitT);
        }

        float3 n = payloadNormal(); // oriented toward the incoming ray by the closest-hit
        float3 hitPos = ro + rd * payload.hitT;
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.0), RAY_CONE_MIN_WIDTH);
        uint material = payloadMaterial();
        int hitDepth = indirectDepth++;

        // ---- Dielectric interface: water, stained glass, ice. ONE handler and one behaviour — every
        // dielectric is a volume. Fresnel from the relative index across the face, then reflect or
        // refract; no diffuse response and no sun NEE, because the surface is specular. Transmission
        // pushes or pops a participating medium, and the segment travelled inside it is attenuated by
        // that medium's extinction rather than by a tint multiply at the interface.
        //
        // Water is the only material that adds anything: the animated wave normal, and a medium marked
        // for the caustic term. Which side of the face we are on comes from the geometry
        // (payloadDielectricEntering), so a stray or missing face cannot desync the medium for the rest
        // of the path.
        if (material == MATERIAL_WATER || material == MATERIAL_DIELECTRIC) {
            // A dielectric has no local shading term. At the path cap its continuation cannot be traced,
            // so avoid computing an unused Fresnel choice, medium update, and roulette step.
            if (bounce >= maxBounces) {
                break;
            }
            bool isWater = material == MATERIAL_WATER;
            bool entering = payloadDielectricEntering();
            float3 tint = payloadAlbedo();
            float transmission = clamp(payloadTransmission(), 0.0, 1.0);

            // Ripple the (near-flat) surface normal before any Fresnel/guide use → glints, broken
            // reflections, wobbling refraction. The wave domain is world-stable: rebased hit XZ plus the
            // pushed anchor (terrain origin mod 4096) survives the per-move terrain rebase.
            float3 geometricNormal = n; // unrippled; ray origins offset along this
            if (isWater && waterWaves) {
                float2 waterDomain = hitPos.xz + worldPush.waterAnchor.xy;
                float waterFootprint = rayConeWidth / max(abs(dot(-rd, geometricNormal)), 0.2);
                n = applyWaterWaves(geometricNormal, waterDomain,
                                    worldPush.waterParams.w, waterFootprint);
            }

            // The medium this face opens into, and the one it returns to on the way out — which is what
            // the stack remembers.
            Medium entered = makeDielectricMedium(isWater, max(payloadIor(), 1.0), tint, transmission);
            float etaI = medium.current.ior;
            float etaT = entering ? entered.ior : medium.outer.ior;

            float cosI = clamp(dot(-rd, n), 0.0, 1.0);
            float F = fresnelDielectric(cosI, etaI, etaT);
            float3 transmittedDir = refract(rd, n, etaI / etaT);
            // The translucent terrain layer is recessed by TRANSLUCENT_INSET (RtTerrainMesher), so a
            // glass/ice face touching a slab or stair sits a hair behind that neighbour's surface. A full
            // SURF_BIAS would restart the transmitted ray past the neighbour and see through it. Water
            // comes from the fluid mesher, which applies no inset, so it takes the ordinary bias.
            float transmitBias = isWater ? SURF_BIAS : INSET_TRANSMIT_BIAS;

            bool chooseReflection = rndf(seed) < F;
            if (chooseReflection) {
                rd = reflect(rd, n);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, SURF_BIAS);
            } else {
                if (dot(transmittedDir, transmittedDir) <= 0.0) break; // TIR with F < 1 cannot happen
                rd = normalize(transmittedDir);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, transmitBias);
                // Crossing into or out of the volume. Absorption is the medium's job from here, so the
                // tint is NOT also multiplied into the throughput — that would double-count it.
                if (entering) {
                    mediumPush(medium, entered);
                } else {
                    mediumPop(medium);
                }
            }
            showCelestial = true; // specular interface: the continuation ray may see the sun/moon disc
            // A dielectric reflects or refracts, never scatters diffusely, so a primary ray that lands on
            // one makes the whole path specular.
            if (lane < 0) {
                lane = 1;
            }
            if (bounce >= rrStart) {
                float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
                if (rndf(seed) > q) {
                    break;
                }
                throughput /= q;
            }
            continue;
        }

        // Particles (material == 2): camera-facing receiver billboard. The instance mask keeps particles
        // off secondary rays, so this only fires at bounce 0; they receive direct light and one diffuse
        // path continuation into the scene, but never appear in shadows/reflections/GI or emit radiance.
        if (material == MATERIAL_PARTICLE) {
            float3 albedo = payloadAlbedo();
            // Billboards are diffuse receivers.
            if (lane < 0) {
                lane = 0;
            }

            float3 lightDir = celestialLight.dir;
            float lightHalfAngle = celestialLight.halfAngle;
            if (lightHalfAngle > 0.0) {
                lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
            }
            // Billboards are effectively two-sided receivers. Bias from the side facing the light; shadow
            // rays exclude particles anyway, but this keeps the origin sane when the light is behind the
            // camera side.
            float signedNdl = dot(n, lightDir);
            float ndl = abs(signedNdl);
            if (ndl > 0.0) {
                float3 shadowOrigin = hitPos + (signedNdl >= 0.0 ? n : -n) * SURF_BIAS;
                float3 vis = visibility(shadowOrigin, lightDir, 10000.0).transmittance;
                if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
                    Ldiff += throughput * albedo * INV_PI * celestialLight.illuminance * ndl * vis;
                }
            }

            // RIS direct lighting from block emitters (lava, glowstone, torches, ...) — same reservoir
            // sampler as terrain/entities, but two-sided: a billboard has no back face, so light
            // striking either side of the quad should still land (matches the sun/moon NEE above).
            if (risOn) {
                float3 v = -rd;
                Reservoir r = risInitial(hitPos, n, v, rd, albedo, float3(0.0, 0.0, 0.0), 1.0,
                                         true, 0.0, worldPush.risCandidates, seed, proposalSeed);
                float3 unusedSpecular;
                Ldiff += throughput * shadeReservoir(r, hitPos, n, v, rd, albedo, float3(0.0, 0.0, 0.0), 1.0,
                                                     true, 0.0, unusedSpecular);
            }

            if (bounce >= maxBounces) {
                break;
            }
            throughput *= albedo;
            ro = hitPos + n * SURF_BIAS;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse receiver: direct sun/moon was handled by NEE above
            continue;
        }

        float3 albedo = payloadAlbedo();
        float sss = payloadSss(); // LabPBR SSS strength (0 when absent)
        float3 p = hitPos + n * SURF_BIAS;
        float3 v = -rd; // view direction (toward the camera / incoming ray)

        // Material split. Metals tint F0 by albedo and have no diffuse lobe; dielectrics use a fixed
        // 0.04 F0, sourced from the chit (LabPBR custom/metal F0, or the dielectric default it applies).
        float rough = clamp(payloadRoughness(), 0.0, 1.0);
        float metal = clamp(payloadMetalness(), 0.0, 1.0);
        // Unresolvably tight lobes become exact delta mirrors; everything else keeps its authored value.
        // Snapping to 0 also propagates into the guide buffers, which is what tells RR to hold the
        // reflection sharp instead of accumulating it as a glossy lobe.
        bool exactSpecular = isDeltaAlpha(rough);
        rough = exactSpecular ? 0.0 : rough;
        float3 diffAlb = albedo * (1.0 - metal);
        float3 F0 = payload.f0;

        // Emission, sun/moon NEE, RIS emitter NEE and thin-surface SSS transmission at this vertex. The
        // helper owns the emitter-emission gate and the lane split so every caller shades one vertex the
        // same way.
        // Every vertex reuses the pixel's emitter sample; only the visible surface (hitDepth 0) owns the
        // slot and writes it back for the next frame.
        uint emitterHistoryIndex = 0xFFFFFFFFu;
        if (worldPush.emitterHistoryAddr != 0
                && pix.x < worldPush.renderWidth && pix.y < worldPush.renderHeight) {
            emitterHistoryIndex = pix.y * worldPush.renderWidth + pix.x;
        }
        shadeDirectLighting(hitPos, n, v, rd, throughput, albedo, diffAlb, F0, rough, sss,
                payloadEmission(), payloadEmitterInList(), lane, hitDepth,
                risOn && (hitDepth == 0 || worldPush.lightBounceRis != 0u), showCelestial,
                medium, waterWaves, emitterHistoryIndex, hitDepth == 0, seed, proposalSeed, Ldiff, Lspec);

        // Preserve all local lighting at the terminal hit, but do not sample a continuation that the
        // bounce loop cannot trace.
        if (bounce >= maxBounces) {
            break;
        }

        // Indirect continuation: importance-sample the diffuse OR the GGX specular lobe, chosen by their
        // relative reflectance.
        float ps = exactSpecular && luminance(diffAlb) <= 1.0e-6
                       ? 1.0
                       : clamp(luminance(F0) / (luminance(F0) + luminance(diffAlb) + 1.0e-4), 0.1, 0.9);
        if (rndf(seed) < ps) {
            float3 l;
            if (exactSpecular) {
                // Authored zero is a delta distribution, not a narrow finite GGX lobe. This avoids the
                // degenerate VNDF at alpha=0 and preserves a genuinely sharp reflection without a floor.
                l = reflect(rd, n);
                throughput *= fresnelSchlick(clamp(dot(n, v), 0.0, 1.0), F0) / ps;
            } else {
                float3 h = sampleGGXVNDF(n, v, rough, seed);
                l = reflect(rd, h); // rd = -v: reflect the incoming ray about the microfacet normal
                float ndl2 = dot(n, l);
                if (ndl2 <= 0.0) {
                    break; // microfacet reflects below the surface: terminate this path
                }
                // VNDF + separable Smith ⇒ weight = F · G2/G1(v) = F · G1(l); divide by the lobe pdf ps.
                float3 F = fresnelSchlick(max(0.0, dot(v, h)), F0);
                throughput *= F * ggxG1(ndl2, rough) / ps;
                rayConeSpread = max(rayConeSpread, rough * RAY_CONE_GLOSSY_SPREAD_SCALE);
            }
            ro = p;
            rd = l;
            showCelestial = true; // specular lobe: this ray is a (glossy) mirror — show the disc
            if (lane < 0) {
                lane = 1;
            }
        } else {
            throughput *= diffAlb / (1.0 - ps);
            ro = p;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse lobe: disc covered by NEE, hide it (anti-firefly)
            if (lane < 0) {
                lane = 0;
            }
        }

        // Russian roulette on deeper bounces: survive with prob = max channel, rescale survivors.
        if (bounce >= rrStart) {
            float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
            if (rndf(seed) > q) {
                break;
            }
            throughput /= q;
        }
    }
    PathRadiance result;
    result.diffuse = Ldiff;
    result.specular = Lspec;
    // Negative when this sample never traced that lane (the other lobe was chosen, or the path ended at
    // the primary vertex). The caller averages only over the samples that did trace it — dividing by the
    // full sample count would bias both distances toward zero.
    result.diffHitDist = diffHitDist;
    result.specHitDist = specHitDist;
    return result;
}

public void indirectPass(uint2 dispatchIndex, uint2 dimensions) {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    // With ReSTIR PT enabled this dispatch follows the temporal pass and only owns the pixels ReSTIR
    // handed back to the ordinary path tracer (dielectrics, particles, mirrors, misses, and failed
    // samples). The temporal pass publishes a one-uint valid mask; a set pixel already has its reservoir
    // shaded (temporal-only) or pending shading (spatial), so the fallback must leave it alone.
    if (pc.restirMaskAddr != 0) {
        uint restirPixel = dispatchIndex.y * dimensions.x + dispatchIndex.x;
        if (DevicePtr<uint>(pc.restirMaskAddr)[restirPixel] != 0u) {
            return;
        }
    }
    // Derive this dispatch's directional light once. The same transmittance LUT drives terrain lighting
    // and the visible sun/moon, keeping atmospheric tint consistent across both consumers.
    celestialLight = dominantCelestialLight(skyState(worldPush), transmittanceLut);
    // The path integral is ACEScg end to end; sky.slang works in linear BT.709, so this is the one seam.
    celestialLight.illuminance = bt709ToAcesCg(celestialLight.illuminance);
    int2 pix = int2(dispatchIndex);
    uint spp = max(worldPush.spp, 1u);
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    ConstPtr<PackedPathSegment> queue = ConstPtr<PackedPathSegment>(pc.pathQueueAddr);
    float3 frameDiffuse = float3(0.0, 0.0, 0.0);
    float3 frameSpecular = float3(0.0, 0.0, 0.0);
    float diffHitDistSum = 0.0;
    float specHitDistSum = 0.0;
    uint diffHitDistCount = 0u;
    uint specHitDistCount = 0u;
    uint recordIndex = pixelIndex;
    for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
        PackedPathSegment packed = queue[recordIndex];
        PathSegment segment = unpackPathSegment(packed);
        for (uint s = 0u; s < spp; ++s) {
            PathRadiance radiance = tracePath(segment, uint2(pix), s + leaf * spp);
            frameDiffuse += radiance.diffuse;
            frameSpecular += radiance.specular;
            // Averaged over the samples that actually traced the lane, not over every sample: a lane is
            // only taken by the paths whose primary vertex sampled that lobe.
            if (radiance.diffHitDist >= 0.0) {
                diffHitDistSum += radiance.diffHitDist;
                diffHitDistCount++;
            }
            if (radiance.specHitDist >= 0.0) {
                specHitDistSum += radiance.specHitDist;
                specHitDistCount++;
            }
        }
        if (packed.nextRecord == PATH_NO_NEXT) {
            break;
        }
        recordIndex = packed.nextRecord;
    }
    // Pre-exposure is applied here and nowhere else: the entire path-traced result above is in absolute
    // scene units, and only the fp16 store is normalized.
    //
    // The three outputs are rgba16f, so anything over 65504 would round to +inf and propagate as NaN
    // through the denoiser. Pre-exposed, it only bites when a pixel would render >65504x mid-grey — the
    // sun mid-transition, a firefly off a tiny emitter — and ACES 2.0 renders anything a few EV over
    // white as white, so it costs nothing visible.
    float scale = worldPush.preExposure / float(spp);
    float3 storedDiffuse = frameDiffuse * scale;
    float3 storedSpecular = frameSpecular * scale;
    outImage[pix] = float4(clamp(storedDiffuse + storedSpecular, float3(0.0), float3(HALF_MAX)), 1.0);
    // The split lanes, for a denoiser that reconstructs the lobes separately. Their alpha carries the
    // lane's mean first-bounce hit distance in blocks; radiance is clamped for the same reason as above.
    // A lane no sample took reports zero distance, which reads as "hit at the primary surface".
    float diffHitDist = diffHitDistCount > 0u ? diffHitDistSum / float(diffHitDistCount) : 0.0;
    float specHitDist = specHitDistCount > 0u ? specHitDistSum / float(specHitDistCount) : 0.0;
    gDiffRadianceHitDist[pix] = float4(clamp(storedDiffuse, float3(0.0), float3(HALF_MAX)), diffHitDist);
    gSpecRadianceHitDist[pix] = float4(clamp(storedSpecular, float3(0.0), float3(HALF_MAX)), specHitDist);
}

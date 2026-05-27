#version 450

// Dedicated SSGI (screen-space one-bounce diffuse) pass — runs between
// the G-buffer pass and the lit pass, writes raw single-frame SSGI
// irradiance to a separate R16G16B16A16_SFLOAT image. lit.frag then
// samples this with a cross-bilateral 9-tap filter to suppress the
// per-pixel sample-set noise that TAA alone couldn't fully resolve.
// See mds/sponza_visual_diagnosis.md N2b for the design rationale.

// ---- Scene UBO (set 0, binding 0) — same std140 layout as lit.frag ----
struct DirectionalLight { vec4 direction; vec4 colorIntensity; };
struct PointLight       { vec4 position;  vec4 colorIntensity; };
struct SpotLight {
    vec4 position; vec4 direction; vec4 colorIntensity; vec4 cutoffAngles;
};
layout(set = 0, binding = 0) uniform SceneUniformBuffer {
    mat4 projection;
    mat4 view;
    mat4 lightSpaceMatrices[4];
    mat4 pointShadowMatrices[6];
    vec4 cameraPosition;
    vec4 cascadeSplits;
    DirectionalLight directionalLight;
    PointLight pointLights[4];
    SpotLight spotLights[2];
    ivec4 lightCounts;
    vec4 shadowParams;
    mat4 invProj;
    mat4 invView;
    vec4 fogParams;
    int  debugMode;
    vec4 qualityToggles;
    vec4 qualityToggles2;
    mat4 prevViewProj;
    vec4 taaParams;
    vec4 viewportSize;
} scene;

// CSM shadow map (set 0 binding 1) — declared so per-SSGI-sample shadow
// tests can suppress samples that geometrically face the sun but are
// actually occluded by other geometry. Without this, walls in cast
// shadow (NoL > 0 but light blocked) contribute false "bright bounce"
// into SSGI, surfacing as the bright square clusters on the shadowed
// floor (the user's "light coming through the wall" symptom).
layout(set = 0, binding = 1) uniform sampler2DArrayShadow shadowMap;

// ---- G-buffer (set 1) — bindings 0-3 match the deferred-pass layout ----
layout(set = 1, binding = 0) uniform sampler2D gBuffer0Sampler;  // albedo + metallic
layout(set = 1, binding = 1) uniform sampler2D gBuffer1Sampler;  // worldN + roughness
layout(set = 1, binding = 3) uniform sampler2D gBufferDepthSampler;

// Set 2: prev-frame SSGI history. Sampled with bilinear + CLAMP_TO_EDGE;
// CPU rotates the binding through a history ring so this is always last-frame
// data, never this frame's in-flight output.
layout(set = 2, binding = 0) uniform sampler2D ssgiHistoryPrev;

layout(location = 0) out vec4 outSSGI;

vec3 reconstructViewPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 v    = scene.invProj * clip;
    return v.xyz / v.w;
}

bool ssgiPathVisible(vec2 uv, vec2 sUV, float viewZ, float sViewZ,
                     vec3 recvN, vec3 sampleN) {
    // SSGI is screen-space, so without a cheap path test a sunlit floor tile
    // on the far side of a curtain/wall can be treated as a valid bounce
    // emitter for a shadowed tile. Those false emitters are the beige square
    // clusters visible on the Sponza floor.
    const float STEPS = 3.0;
    for (int step = 1; step < 3; ++step) {
        float t = float(step) / STEPS;
        vec2 midUV = mix(uv, sUV, t);
        float midDepth = texture(gBufferDepthSampler, midUV).r;
        if (midDepth >= 0.9999)
            return false;

        float expectedZ = mix(viewZ, sViewZ, t);
        float midZ = -reconstructViewPos(midUV, midDepth).z;
        float zTolerance = max(0.04 * expectedZ, 0.035);
        if (abs(midZ - expectedZ) > zTolerance)
            return false;

        vec3 midN = texture(gBuffer1Sampler, midUV).xyz;
        if (dot(midN, midN) < 0.1)
            return false;
        midN = normalize(midN);
        if (dot(midN, recvN) < 0.72 || dot(midN, sampleN) < 0.72)
            return false;
    }
    return true;
}

float interleavedGradientNoise(vec2 px) {
    vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(px, m.xy)));
}

vec2 vogelDisk(int i, int n, float phi) {
    float r  = sqrt((float(i) + 0.5) / float(n));
    float th = float(i) * 2.4 + phi;
    return r * vec2(cos(th), sin(th));
}

// Cheap 1-tap shadow lookup for SSGI samples. NO PCF, NO Vogel disc -
// just project + hardware-compare. Returns visibility in [0,1] where
// 1.0 = fully lit by the sun, 0.0 = fully shadowed. Walks cascades
// (0..3) and stops at the first one the sample falls inside.
// 32 of these per receiver pixel adds noticeable cost; keep simple.
float sampleSunVisibility(vec3 worldPos, float viewZ) {
    int primaryCascade = 3;
    if      (viewZ < scene.cascadeSplits.x) primaryCascade = 0;
    else if (viewZ < scene.cascadeSplits.y) primaryCascade = 1;
    else if (viewZ < scene.cascadeSplits.z) primaryCascade = 2;

    for (int c = primaryCascade; c <= 3; ++c) {
        vec4 lsPos = scene.lightSpaceMatrices[c] * vec4(worldPos, 1.0);
        vec3 proj  = lsPos.xyz / lsPos.w;
        proj.xy    = proj.xy * 0.5 + 0.5;
        if (proj.z >= 0.0 && proj.z <= 1.0 &&
            proj.x >= 0.0 && proj.x <= 1.0 &&
            proj.y >= 0.0 && proj.y <= 1.0) {
            // Small constant bias is fine here - SSGI samples don't need
            // the receiver-plane gradient bias because we only care about
            // "is this pixel in shadow at all", not penumbra accuracy.
            float bias = 0.001 * float(c + 1);
            float tapDepth = proj.z - bias;
            return texture(shadowMap,
                           vec4(proj.xy, float(c), tapDepth));
        }
    }
    // Outside all cascades — conservative: assume shadowed (no SSGI
    // contribution from samples that fall past CSM far).
    return 0.0;
}

// Same body as lit.frag::computeSSGI(). Kept identical so we can A/B
// against the inline path by toggling the lit.frag bilateral sample
// out and inlining computeSSGI again.
vec3 computeSSGI(vec2 uv, vec3 worldPos, vec3 worldN, float viewZ) {
    if (scene.shadowParams.y < 0.001) return vec3(0.0);

    vec2 texelSize = 1.0 / vec2(textureSize(gBuffer0Sampler, 0));
    vec3 sunDir   = normalize(-scene.directionalLight.direction.xyz);
    vec3 sunColor = scene.directionalLight.colorIntensity.rgb *
                    scene.directionalLight.colorIntensity.a;

    // SSGI is there to recover missing bounce in shadowed regions. On pixels
    // that are already directly sunlit, the gather is mostly invisible but
    // still costs the full sample loop and can add high-frequency shimmer.
    float recvNoL = max(dot(worldN, sunDir), 0.0);
    if (recvNoL > 0.0 && sampleSunVisibility(worldPos, viewZ) > 0.75)
        return vec3(0.0);

    const int   SSGI_SAMPLES   = 12;
    const float SSGI_RADIUS_PX = 18.0;
    const float SSGI_DEPTH_TOL = 0.06;

    float temporalSeed = fract(scene.taaParams.x * 9311.0 +
                               scene.taaParams.y * 7919.0);
    float phi          = (interleavedGradientNoise(gl_FragCoord.xy) + temporalSeed)
                         * 6.2831853;

    vec3  bounce = vec3(0.0);
    float totalW = 0.0;
    int   validCount = 0;

    for (int i = 0; i < SSGI_SAMPLES; ++i) {
        vec2 off = vogelDisk(i, SSGI_SAMPLES, phi) * SSGI_RADIUS_PX * texelSize;
        vec2 sUV = uv + off;
        if (sUV.x < 0.0 || sUV.x > 1.0 || sUV.y < 0.0 || sUV.y > 1.0) continue;

        float sDepth = texture(gBufferDepthSampler, sUV).r;
        if (sDepth >= 0.9999) continue;

        vec3 sN = texture(gBuffer1Sampler, sUV).xyz;
        if (dot(sN, sN) < 0.1) continue;
        sN = normalize(sN);

        vec3 sViewPos = reconstructViewPos(sUV, sDepth);
        float sViewZ  = -sViewPos.z;
        if (abs(sViewZ - viewZ) > SSGI_DEPTH_TOL * viewZ) continue;
        if (!ssgiPathVisible(uv, sUV, viewZ, sViewZ, worldN, sN)) continue;

        vec3 sWorldPos = (scene.invView * vec4(sViewPos, 1.0)).xyz;
        vec3 dir       = sWorldPos - worldPos;
        float dist2    = dot(dir, dir);
        if (dist2 < 1e-4) continue;
        dir *= inversesqrt(dist2);

        float cosRecv = max(dot(worldN, dir), 0.0);
        if (cosRecv < 0.08) continue;

        float cosEmit = max(-dot(sN, dir), 0.0);
        if (cosEmit < 0.05) continue;

        float falloff = 1.0 / (dist2 + 0.25);

        vec3  sAlbedo = texture(gBuffer0Sampler, sUV).rgb;
        float sNoL    = max(dot(sN, sunDir), 0.0);
        // Shadow-test THIS sample so walls/floors that geometrically
        // face the sun but are actually occluded don't contribute false
        // bright bounce. This is the fix for the "square lights coming
        // through the wall" symptom - without it, every sample point's
        // sNoL > 0 surface gets counted as fully sunlit. 1-tap lookup
        // (no PCF) - cheap, single-frame Vogel temporal rotation washes
        // the per-pixel binary visibility into a smooth gradient.
        float sVisibility = (sNoL > 0.0) ? sampleSunVisibility(sWorldPos, sViewZ)
                                         : 0.0;
        vec3  sLit    = sAlbedo * sunColor * sNoL * sVisibility;
        sLit = min(sLit, vec3(1.0));

        float w = cosRecv * cosEmit * falloff;
        bounce += sLit * w;
        totalW += w;
        validCount++;
    }

    if (totalW < 1e-4 || validCount < 3) return vec3(0.0);
    // Sparse one/two-hit gathers are the root of the blocky white islands:
    // one random sunlit candidate was being normalized as if the whole local
    // neighborhood agreed. Require several agreeing samples before allowing
    // full-strength bounce, but keep a smooth ramp so real contact bounce
    // near sun patches does not hard-pop.
    float confidence = smoothstep(3.0, 8.0, float(validCount));
    return bounce / totalW * scene.shadowParams.y * confidence;
}

// Reconstruct world position from a depth-buffer sample. Same as
// second.frag's helper — scene.invProj is the JITTERED inverse, but
// SSGI runs OUTSIDE the TAA jitter loop (no jitter applied here), so
// for SSGI we want the un-jittered projection. Since we're reading the
// G-buffer depth (which was rendered with the same jittered projection
// as TAA), the reconstructed pos matches what TAA already uses for
// reprojection — same convention, no off-by-jitter bug.
vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 vp = scene.invProj * clip;
    vp /= vp.w;
    vec4 wp = scene.invView * vp;
    return wp.xyz;
}

float reconstructLinearViewZ(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 vp = scene.invProj * clip;
    return vp.z / vp.w;
}

void main() {
    // The SSGI target is half-resolution. Map each low-res output pixel back
    // to normalized full-screen UVs, then sample the full-res G-buffer there.
    vec2 outSize = vec2(textureSize(ssgiHistoryPrev, 0));
    vec2 uv      = (gl_FragCoord.xy + vec2(0.5)) / outSize;

    float depth = texture(gBufferDepthSampler, uv).r;
    if (depth >= 0.9999) {
        // Sky — no bounce contribution.
        outSSGI = vec4(0.0);
        return;
    }

    vec3 worldN = texture(gBuffer1Sampler, uv).xyz;
    if (dot(worldN, worldN) < 0.1) { outSSGI = vec4(0.0); return; }
    worldN = normalize(worldN);

    vec3 viewPos  = reconstructViewPos(uv, depth);
    vec3 worldPos = (scene.invView * vec4(viewPos, 1.0)).xyz;

    // 1. Current-frame raw bounce. Keep the default sample count modest:
    //    every SSGI candidate also runs a path-visibility test and a sun
    //    shadow lookup, so this full-screen pass gets expensive quickly.
    vec3 ssgiCurr = computeSSGI(uv, worldPos, worldN, -viewPos.z);
    ssgiCurr = min(ssgiCurr, vec3(1.0)); // per-pass firefly clamp

    // 2. Reproject last frame's SSGI to find where this pixel was on
    //    screen one frame ago. Same math as second.frag's TAA.
    bool historyValid = scene.taaParams.w > 0.5;
    vec3 ssgiBlended  = ssgiCurr;

    if (historyValid) {
        vec4 prevClip = scene.prevViewProj * vec4(worldPos, 1.0);
        if (abs(prevClip.w) > 1e-6) {
            vec2 prevNDC = prevClip.xy / prevClip.w;
            vec2 prevUV  = prevNDC * 0.5 + 0.5;

            // Off-screen reject.
            bool onScreen = all(greaterThanEqual(prevUV, vec2(0.0))) &&
                            all(lessThanEqual   (prevUV, vec2(1.0)));
            if (onScreen) {
                // Depth-disocclusion reject (same proxy as second.frag:
                // compare view-Z of current pixel vs current-frame depth
                // sampled at prevUV).
                float depthAtPrev = texture(gBufferDepthSampler, prevUV).r;
                float viewZCur  = abs(reconstructLinearViewZ(uv,     depth));
                float viewZPrev = abs(reconstructLinearViewZ(prevUV, depthAtPrev));
                bool depthOK = (depthAtPrev >= 0.9999) ||
                               (abs(viewZCur - viewZPrev) <=
                                max(0.1 * viewZCur, 0.05));

                if (depthOK) {
                    vec3 ssgiHist = texture(ssgiHistoryPrev, prevUV).rgb;
                    // Sanitize history — NaN/Inf can creep in from
                    // float-edge cases in the bilinear tap.
                    if (any(isnan(ssgiHist)) || any(isinf(ssgiHist))) {
                        ssgiHist = ssgiCurr;
                    }
                    // 0.95 history weight is the doc spec for N6 Option C.
                    // Effective convergence over ~20 frames; visible noise
                    // drops ~4.5x on a static camera.
                    ssgiBlended = mix(ssgiCurr, ssgiHist, 0.95);
                }
            }
        }
    }

    outSSGI = vec4(ssgiBlended, 1.0);
}

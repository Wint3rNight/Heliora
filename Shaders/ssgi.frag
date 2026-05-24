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

// ---- G-buffer (set 1) — bindings 0-3 match the deferred-pass layout ----
layout(set = 1, binding = 0) uniform sampler2D gBuffer0Sampler;  // albedo + metallic
layout(set = 1, binding = 1) uniform sampler2D gBuffer1Sampler;  // worldN + roughness
layout(set = 1, binding = 3) uniform sampler2D gBufferDepthSampler;

layout(location = 0) out vec4 outSSGI;

vec3 reconstructViewPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 v    = scene.invProj * clip;
    return v.xyz / v.w;
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

// Same body as lit.frag::computeSSGI(). Kept identical so we can A/B
// against the inline path by toggling the lit.frag bilateral sample
// out and inlining computeSSGI again.
vec3 computeSSGI(vec2 uv, vec3 worldPos, vec3 worldN, float viewZ) {
    if (scene.shadowParams.y < 0.001) return vec3(0.0);

    vec2 texelSize = 1.0 / vec2(textureSize(gBuffer0Sampler, 0));
    vec3 sunDir   = normalize(-scene.directionalLight.direction.xyz);
    vec3 sunColor = scene.directionalLight.colorIntensity.rgb *
                    scene.directionalLight.colorIntensity.a;

    const int   SSGI_SAMPLES   = 32;
    const float SSGI_RADIUS_PX = 18.0;
    const float SSGI_DEPTH_TOL = 0.12;

    float temporalSeed = fract(scene.taaParams.x * 9311.0 +
                               scene.taaParams.y * 7919.0);
    float phi          = (interleavedGradientNoise(gl_FragCoord.xy) + temporalSeed)
                         * 6.2831853;

    vec3  bounce = vec3(0.0);
    float totalW = 0.0;

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
        vec3  sLit    = sAlbedo * sunColor * sNoL;
        sLit = min(sLit, vec3(2.5));

        float w = cosRecv * cosEmit * falloff;
        bounce += sLit * w;
        totalW += w;
    }

    if (totalW < 1e-4) return vec3(0.0);
    return bounce / totalW * scene.shadowParams.y;
}

void main() {
    vec2 texSize = vec2(textureSize(gBuffer0Sampler, 0));
    vec2 uv      = gl_FragCoord.xy / texSize;

    float depth = texture(gBufferDepthSampler, uv).r;
    if (depth >= 0.9999) {
        // Sky — no bounce contribution; write zero so lit.frag's
        // bilateral filter doesn't pull SSGI onto sky pixels at edges.
        outSSGI = vec4(0.0);
        return;
    }

    vec3 worldN = texture(gBuffer1Sampler, uv).xyz;
    if (dot(worldN, worldN) < 0.1) { outSSGI = vec4(0.0); return; }
    worldN = normalize(worldN);

    vec3 viewPos  = reconstructViewPos(uv, depth);
    vec3 worldPos = (scene.invView * vec4(viewPos, 1.0)).xyz;

    vec3 ssgi = computeSSGI(uv, worldPos, worldN, -viewPos.z);
    // Per-pass firefly clamp on the gathered result — second line of
    // defence on top of the per-sample clamp inside computeSSGI.
    ssgi = min(ssgi, vec3(2.5));
    outSSGI = vec4(ssgi, 1.0);
}

#version 450

layout(location = 0) in vec3 fragCol;
layout(location = 1) in vec2 fragTex;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in mat3 fragTBN;

// Scene UBO — only qualityToggles is consumed here. Full struct must match
// the lit pass so std140 offsets line up.
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
    vec4 qualityToggles;  // x = sRGB albedo decode
    vec4 qualityToggles2; // x = mipmap sampling enable
} scene;

layout(set = 1, binding = 0) uniform sampler2D albedoSampler;
layout(set = 1, binding = 1) uniform sampler2D normalSampler;
layout(set = 1, binding = 2) uniform sampler2D metallicSampler;
layout(set = 1, binding = 3) uniform sampler2D roughnessSampler;
layout(set = 1, binding = 4) uniform sampler2D aoSampler;

layout(location = 0) out vec4 gBuffer0;  // albedo.rgb + metallic
layout(location = 1) out vec4 gBuffer1;  // world-space normal.xyz + roughness
layout(location = 2) out vec4 gBuffer2;  // AO

// Material-texture fetch: implicit LOD (with aniso + trilinear) when mipmaps
// are enabled, force-mip-0 otherwise. Lets the user A/B the no-mip bug state
// against the fixed pipeline. Note: the ternary still evaluates both reads;
// uniform-branching via `if` would let one elide, but the texture cache
// makes the duplicate fetch essentially free here.
vec4 sampleTex(sampler2D s, vec2 uv) {
    return scene.qualityToggles2.x > 0.5
        ? texture(s, uv)
        : textureLod(s, uv, 0.0);
}

void main() {
    vec4 albedoSample = sampleTex(albedoSampler, fragTex);
    if (albedoSample.a < 0.1) discard;

    // sRGB → linear. Albedo textures are stored as UNORM, so the bytes
    // arrive already in sRGB-encoded space; PBR math requires linear.
    // Pow(2.2) is the conventional shading approximation of the sRGB EOTF
    // (RTR4 §5.6.1). Gated by a toggle so the bug behavior can be A/B'd.
    vec3 albedoRgb = albedoSample.rgb;
    if (scene.qualityToggles.x > 0.5)
        albedoRgb = pow(albedoRgb, vec3(2.2));

    vec3 albedo = albedoRgb * fragCol;
    float metallic = sampleTex(metallicSampler, fragTex).b;
    float roughness = clamp(sampleTex(roughnessSampler, fragTex).g, 0.04, 1.0);
    float ao = sampleTex(aoSampler, fragTex).r;

    vec3 normalSample = sampleTex(normalSampler, fragTex).rgb * 2.0 - 1.0;
    vec3 worldNormal = normalize(fragTBN * normalSample);

    gBuffer0 = vec4(albedo, metallic);
    gBuffer1 = vec4(worldNormal, roughness);
    gBuffer2 = vec4(ao, 0.0, 0.0, 1.0);
}

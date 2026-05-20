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
    mat4 prevViewProj;
    vec4 taaParams;
    vec4 viewportSize;
} scene;

layout(set = 1, binding = 0) uniform sampler2D albedoSampler;
layout(set = 1, binding = 1) uniform sampler2D normalSampler;
layout(set = 1, binding = 2) uniform sampler2D metallicSampler;
layout(set = 1, binding = 3) uniform sampler2D roughnessSampler;
layout(set = 1, binding = 4) uniform sampler2D aoSampler;

layout(location = 0) out vec4 gBuffer0;  // albedo.rgb + metallic
layout(location = 1) out vec4 gBuffer1;  // world-space normal.xyz + roughness
layout(location = 2) out vec4 gBuffer2;  // AO

void main() {
    vec4 albedoSample = texture(albedoSampler, fragTex);
    if (albedoSample.a < 0.1) discard;

    // sRGB → linear. Albedo textures are uploaded as UNORM, so the bytes
    // arrive already in sRGB-encoded space; PBR math requires linear.
    // Pow(2.2) is the conventional shading approximation of the sRGB EOTF
    // (RTR4 §5.6.1).
    vec3 albedo = pow(albedoSample.rgb, vec3(2.2)) * fragCol;
    float metallic = texture(metallicSampler, fragTex).b;
    float roughness = clamp(texture(roughnessSampler, fragTex).g, 0.04, 1.0);
    float ao = texture(aoSampler, fragTex).r;

    vec3 normalSample = texture(normalSampler, fragTex).rgb * 2.0 - 1.0;
    vec3 worldNormal = normalize(fragTBN * normalSample);
    // P2 diagnostic: when qualityToggles2.y > 0.5, replace normal-map result
    // with the geometric (interpolated TBN basis Z) normal. Used to isolate
    // whether floor dither is normal-map driven.
    if (scene.qualityToggles2.y > 0.5)
        worldNormal = normalize(fragTBN[2]);

    gBuffer0 = vec4(albedo, metallic);
    gBuffer1 = vec4(worldNormal, roughness);
    gBuffer2 = vec4(ao, 0.0, 0.0, 1.0);
}

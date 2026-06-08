#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 fragCol;
layout(location = 1) in vec2 fragTex;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in mat3 fragTBN;
layout(location = 6) flat in uvec4 fragTexIdx0;
layout(location = 7) flat in uvec4 fragTexIdx1;

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

layout(set = 1, binding = 0) uniform sampler2D textures[];

layout(location = 0) out vec4 gBuffer0;
layout(location = 1) out vec4 gBuffer1;
layout(location = 2) out vec4 gBuffer2;

void main() {
    vec4 albedoSample = texture(textures[nonuniformEXT(fragTexIdx0.x)], fragTex);
    bool alphaMasked = (fragTexIdx1.y & 2u) != 0u;
    float alphaCutoff = float(fragTexIdx1.z) / 255.0;
    if (alphaMasked && albedoSample.a < alphaCutoff) discard;

    vec3 albedo = pow(albedoSample.rgb, vec3(2.2)) * fragCol;
    float metallic = texture(textures[nonuniformEXT(fragTexIdx0.z)], fragTex).b;
    float roughness = clamp(texture(textures[nonuniformEXT(fragTexIdx0.w)], fragTex).g, 0.04, 1.0);
    float nonMetalFloor = scene.qualityToggles2.z * (1.0 - step(0.5, metallic));
    roughness = max(roughness, nonMetalFloor);
    float ao = texture(textures[nonuniformEXT(fragTexIdx1.x)], fragTex).r;

    float normalStrength = clamp(scene.qualityToggles2.y, 0.0, 1.5);
    vec3 normalTex = texture(textures[nonuniformEXT(fragTexIdx0.y)], fragTex).rgb * 2.0 - 1.0;
    vec3 normalSample = normalize(vec3(normalTex.xy * normalStrength,
                                       max(normalTex.z, 0.001)));
    vec3 worldNormal = normalize(fragTBN * normalSample);
    vec3 geomNormal  = normalize(fragTBN[2]);

    float normalVariance =
        dot(dFdx(normalSample), dFdx(normalSample)) +
        dot(dFdy(normalSample), dFdy(normalSample));
    float normalDeviation = 1.0 - clamp(normalSample.z, 0.0, 1.0);
    float normalRoughnessBoost =
        (1.0 - step(0.5, metallic)) *
        clamp(normalVariance * 0.20 + normalDeviation * 0.10, 0.0, 0.35);
    float alphaRoughness = roughness * roughness;
    roughness = sqrt(clamp(alphaRoughness + normalRoughnessBoost,
                           0.04 * 0.04, 1.0));

    float clothBit = float((fragTexIdx1.y & 1u));

    vec2 octN;
    {
        vec3 n = geomNormal;
        n /= (abs(n.x) + abs(n.y) + abs(n.z));
        if (n.z < 0.0) {
            vec2 nxy = n.xy;
            n.x = (1.0 - abs(nxy.y)) * (nxy.x >= 0.0 ? 1.0 : -1.0);
            n.y = (1.0 - abs(nxy.x)) * (nxy.y >= 0.0 ? 1.0 : -1.0);
        }
        octN = n.xy * 0.5 + 0.5;
    }

    gBuffer0 = vec4(albedo, metallic);
    gBuffer1 = vec4(worldNormal, roughness);
    gBuffer2 = vec4(ao, clothBit, octN.x, octN.y);
}

#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 fragCol;
layout(location = 1) in vec2 fragTex;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in mat3 fragTBN;

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

layout(push_constant) uniform PushModel {
    mat4  model;
    mat4  normal;
    uvec4 texIdx0;
    uvec4 texIdx1;
} push;

layout(location = 0) out vec4 outColor;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NoH = max(dot(N, H), 0.0);
    float denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denom * denom, 0.0001);
}

float GeometrySchlickGGX(float NoV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NoV / max(NoV * (1.0 - k) + k, 0.0001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 cookTorrance(vec3 albedo, vec3 N, vec3 V, vec3 L, vec3 radiance,
                  vec3 F0, float metallic, float roughness,
                  float clothFactor) {
    vec3 H = normalize(V + L);
    float NoL = max(dot(N, L), 0.0);
    float NoV = max(dot(N, V), 0.0);
    if (NoL <= 0.0 || NoV <= 0.0)
        return vec3(0.0);

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (D * G * F) / max(4.0 * NoV * NoL, 0.001);
    specular *= mix(1.0, 0.12, clothFactor);
    vec3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo / 3.14159265;
    return (diffuse + specular) * radiance * NoL;
}

void main() {
    vec4 albedoSample = texture(textures[nonuniformEXT(push.texIdx0.x)], fragTex);
    float alpha = clamp(albedoSample.a, 0.0, 1.0);
    if (alpha <= 0.01)
        discard;

    vec3 albedo = pow(albedoSample.rgb, vec3(2.2)) * fragCol;
    float metallic = texture(textures[nonuniformEXT(push.texIdx0.z)], fragTex).b;
    float roughness = clamp(texture(textures[nonuniformEXT(push.texIdx0.w)], fragTex).g, 0.04, 1.0);
    float nonMetalFloor = scene.qualityToggles2.z * (1.0 - step(0.5, metallic));
    roughness = max(roughness, nonMetalFloor);
    float ao = texture(textures[nonuniformEXT(push.texIdx1.x)], fragTex).r;

    float normalStrength = clamp(scene.qualityToggles2.y, 0.0, 1.5);
    vec3 normalTex = texture(textures[nonuniformEXT(push.texIdx0.y)], fragTex).rgb * 2.0 - 1.0;
    vec3 normalSample = normalize(vec3(normalTex.xy * normalStrength,
                                       max(normalTex.z, 0.001)));
    vec3 N = normalize(fragTBN * normalSample);
    vec3 geomN = normalize(fragTBN[2]);

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

    float materialCloth = ((push.texIdx1.y & 1u) != 0u) ? 1.0 : 0.0;
    float saturation = max(max(albedo.r, albedo.g), albedo.b) -
                       min(min(albedo.r, albedo.g), albedo.b);
    float nonMetal = 1.0 - smoothstep(0.10, 0.20, metallic);
    float roughBand = smoothstep(0.30, 0.55, roughness);
    float chromaFactor = smoothstep(0.15, 0.40, saturation);
    float clothFactor = max(materialCloth, nonMetal * roughBand * chromaFactor);
    float clothSuppression = clothFactor * nonMetal;
    roughness = mix(roughness, max(roughness, 0.90), clothSuppression);

    vec3 V = normalize(scene.cameraPosition.xyz - fragWorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    F0 = mix(F0, vec3(0.018), clothSuppression);
    vec3 color = vec3(0.0);

    vec3 sunL = normalize(-scene.directionalLight.direction.xyz);
    vec3 sunRadiance = scene.directionalLight.colorIntensity.rgb *
                       scene.directionalLight.colorIntensity.a;
    color += cookTorrance(albedo, N, V, sunL, sunRadiance, F0, metallic,
                          roughness, clothFactor);

    for (int i = 0; i < min(scene.lightCounts.x, 4); ++i) {
        vec3 toLight = scene.pointLights[i].position.xyz - fragWorldPos;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 0.001);
        float attenuation = 1.0 / max(dist * dist, 1.0);
        vec3 radiance = scene.pointLights[i].colorIntensity.rgb *
                        scene.pointLights[i].colorIntensity.a * attenuation;
        color += cookTorrance(albedo, N, V, L, radiance, F0, metallic,
                              roughness, clothFactor);
    }

    for (int i = 0; i < min(scene.lightCounts.y, 2); ++i) {
        vec3 toLight = scene.spotLights[i].position.xyz - fragWorldPos;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 0.001);
        float theta = dot(L, normalize(-scene.spotLights[i].direction.xyz));
        float inner = scene.spotLights[i].cutoffAngles.x;
        float outer = scene.spotLights[i].cutoffAngles.y;
        float spot = clamp((theta - outer) / max(inner - outer, 0.001), 0.0, 1.0);
        float attenuation = spot / max(dist * dist, 1.0);
        vec3 radiance = scene.spotLights[i].colorIntensity.rgb *
                        scene.spotLights[i].colorIntensity.a * attenuation;
        color += cookTorrance(albedo, N, V, L, radiance, F0, metallic,
                              roughness, clothFactor);
    }

    vec3 ambient = albedo * ao * scene.qualityToggles2.w * 0.08;
    color += ambient;

    float viewDist = length(scene.cameraPosition.xyz - fragWorldPos);
    float fog = 1.0 - exp(-scene.fogParams.x * viewDist);
    fog = clamp(fog * scene.fogParams.y, 0.0, scene.fogParams.z);
    color = mix(color, vec3(0.55, 0.62, 0.66) * scene.qualityToggles2.w, fog);

    outColor = vec4(min(color, vec3(12.0)), alpha);
}

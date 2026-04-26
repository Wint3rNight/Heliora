#version 450

layout(location = 0) in vec3 fragCol;
layout(location = 1) in vec2 fragTex;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in mat3 fragTBN;

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

    vec3 albedo = albedoSample.rgb * fragCol;
    float metallic = texture(metallicSampler, fragTex).b;
    float roughness = clamp(texture(roughnessSampler, fragTex).g, 0.04, 1.0);
    float ao = texture(aoSampler, fragTex).r;

    vec3 normalSample = texture(normalSampler, fragTex).rgb * 2.0 - 1.0;
    vec3 worldNormal = normalize(fragTBN * normalSample);

    gBuffer0 = vec4(albedo, metallic);
    gBuffer1 = vec4(worldNormal, roughness);
    gBuffer2 = vec4(ao, 0.0, 0.0, 1.0);
}

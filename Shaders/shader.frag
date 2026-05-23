#version 450
#extension GL_EXT_nonuniform_qualifier : require

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

// Phase 7.2: Bindless texture array. All PBR textures live in a single
// global descriptor set (set 1, binding 0). Each texture slot is identified
// by an integer index passed via push constants. The nonuniformEXT qualifier
// tells the driver that different invocations in the same subgroup may index
// different textures (non-uniform control flow).
layout(set = 1, binding = 0) uniform sampler2D textures[];

// Push constants carry the model/normal matrices (for the vertex stage)
// and the bindless texture indices (for this fragment stage).
layout(push_constant) uniform PushModel {
    mat4  model;
    mat4  normal;
    uvec4 texIdx0;  // x=albedo, y=normal, z=metallic, w=roughness
    uvec4 texIdx1;  // x=ao
} push;

layout(location = 0) out vec4 gBuffer0;  // albedo.rgb + metallic
layout(location = 1) out vec4 gBuffer1;  // world-space normal.xyz + roughness
layout(location = 2) out vec4 gBuffer2;  // AO

void main() {
    vec4 albedoSample = texture(textures[nonuniformEXT(push.texIdx0.x)], fragTex);
    if (albedoSample.a < 0.1) discard;

    // sRGB → linear. Albedo textures are uploaded as UNORM, so the bytes
    // arrive already in sRGB-encoded space; PBR math requires linear.
    // Pow(2.2) is the conventional shading approximation of the sRGB EOTF
    // (RTR4 §5.6.1).
    vec3 albedo = pow(albedoSample.rgb, vec3(2.2)) * fragCol;
    float metallic = texture(textures[nonuniformEXT(push.texIdx0.z)], fragTex).b;
    float roughness = clamp(texture(textures[nonuniformEXT(push.texIdx0.w)], fragTex).g, 0.04, 1.0);
    // Per-material minimum-roughness floor (qualityToggles2.z). Sponza's
    // authored metalRoughness textures bottom out around 0.1 on the floor
    // and column surfaces, which makes stone act like polished marble and
    // produces (a) clean banner reflections in the floor via SSR / IBL
    // specular and (b) sub-pixel sparkle on normal-mapped pillars under
    // bright HDR sun. Floor stays out of metals' way — `step(0.5, metallic)`
    // zeros the floor on metallic surfaces so real gold/silver trim still
    // reads as polished.
    float nonMetalFloor = scene.qualityToggles2.z * (1.0 - step(0.5, metallic));
    roughness = max(roughness, nonMetalFloor);
    float ao = texture(textures[nonuniformEXT(push.texIdx1.x)], fragTex).r;

    vec3 normalSample = texture(textures[nonuniformEXT(push.texIdx0.y)], fragTex).rgb * 2.0 - 1.0;
    vec3 worldNormal = normalize(fragTBN * normalSample);
    // P2 diagnostic: when qualityToggles2.y > 0.5, replace normal-map result
    // with the geometric (interpolated TBN basis Z) normal. Used to isolate
    // whether floor dither is normal-map driven.
    if (scene.qualityToggles2.y > 0.5)
        worldNormal = normalize(fragTBN[2]);

    // texIdx1.y is a packed material-flag bitfield (bit 0 = isCloth).
    // Bake the cloth bit into gBuffer2.g so the deferred lit pass can drive
    // its Charlie sheen lobe without an extra UBO lookup or shader variant.
    // R8 UNORM → exact 0.0 / 1.0 readback after bilinear at material edges.
    float clothBit = float((push.texIdx1.y & 1u));

    gBuffer0 = vec4(albedo, metallic);
    gBuffer1 = vec4(worldNormal, roughness);
    gBuffer2 = vec4(ao, clothBit, 0.0, 1.0);
}

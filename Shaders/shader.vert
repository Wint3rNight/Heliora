#version 450 // using glsl 4.5

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;
layout(location = 2) in vec2 tex;
layout(location = 3) in vec3 normal;
layout(location = 4) in vec3 tangent;
layout(location = 5) in vec3 bitangent;

struct DirectionalLight {
  vec4 direction;
  vec4 colorIntensity;
};

struct PointLight {
  vec4 position;
  vec4 colorIntensity;
};

struct SpotLight {
  vec4 position;
  vec4 direction;
  vec4 colorIntensity;
  vec4 cutoffAngles;
};

// Full SceneUniformBuffer layout — must match Utilities.h field-for-field.
// This stage only reads projection/view (offsets 0/64), but declaring a
// truncated block puts every later member at wrong offsets and silently
// reads garbage the moment someone touches e.g. scene.cameraPosition here.
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
  int debugMode;
  vec4 qualityToggles;
  vec4 qualityToggles2;
  mat4 prevViewProj;
  vec4 taaParams;
  vec4 viewportSize;
  vec4 visualToggles;
} scene;

// Phase 7.2: push constant block matches ModelPushConstants on the C++ side.
// The vertex stage only uses model and normal; texIdx0/texIdx1 are consumed
// by the fragment stage.
layout(push_constant) uniform PushModel {
  mat4 model;
  mat4 normal;
  uvec4 texIdx0;  // (albedo, normal, metallic, roughness)
  uvec4 texIdx1;  // (ao, _, _, _)
} pushModel;

layout(location = 0) out vec3 fragCol;
layout(location = 1) out vec2 fragTex;
layout(location = 2) out vec3 fragWorldPos;
layout(location = 3) out mat3 fragTBN;

void main() {
  vec4 worldPosition = pushModel.model * vec4(pos, 1.0);
  vec3 worldNormal = normalize(mat3(pushModel.normal) * normal);
  vec3 worldTangent = normalize(mat3(pushModel.model) * tangent);
  worldTangent = normalize(worldTangent - dot(worldTangent, worldNormal) * worldNormal);
  // Degenerate-input test must run BEFORE normalize: normalize(0) is NaN
  // and NaN < eps is false, so a post-normalize guard can never fire.
  vec3 worldBitangent = mat3(pushModel.model) * bitangent;
  if (dot(worldBitangent, worldBitangent) < 1e-8)
    worldBitangent = cross(worldNormal, worldTangent);
  worldBitangent = normalize(worldBitangent);

  gl_Position = scene.projection * scene.view * worldPosition;
  fragCol = col;
  fragTex = tex;
  fragWorldPos = worldPosition.xyz;
  fragTBN = mat3(worldTangent, worldBitangent, worldNormal);
}

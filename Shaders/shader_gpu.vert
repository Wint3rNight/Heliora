#version 450

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

struct TransformRecord {
  mat4 model;
  mat4 normal;
  uvec4 texIdx0;
  uvec4 texIdx1;
};

layout(set = 2, binding = 2, std430) readonly buffer TransformRecords {
  TransformRecord transforms[];
};

layout(location = 0) out vec3 fragCol;
layout(location = 1) out vec2 fragTex;
layout(location = 2) out vec3 fragWorldPos;
layout(location = 3) out mat3 fragTBN;
layout(location = 6) flat out uvec4 fragTexIdx0;
layout(location = 7) flat out uvec4 fragTexIdx1;

void main() {
  TransformRecord tr = transforms[gl_InstanceIndex];
  vec4 worldPosition = tr.model * vec4(pos, 1.0);
  vec3 worldNormal = normalize(mat3(tr.normal) * normal);
  vec3 worldTangent = normalize(mat3(tr.model) * tangent);
  worldTangent = normalize(worldTangent - dot(worldTangent, worldNormal) * worldNormal);
  // Degenerate-input test must run BEFORE normalize: normalize(0) is NaN
  // and NaN < eps is false, so a post-normalize guard can never fire.
  vec3 worldBitangent = mat3(tr.model) * bitangent;
  if (dot(worldBitangent, worldBitangent) < 1e-8)
    worldBitangent = cross(worldNormal, worldTangent);
  worldBitangent = normalize(worldBitangent);

  gl_Position = scene.projection * scene.view * worldPosition;
  fragCol = col;
  fragTex = tex;
  fragWorldPos = worldPosition.xyz;
  fragTBN = mat3(worldTangent, worldBitangent, worldNormal);
  fragTexIdx0 = tr.texIdx0;
  fragTexIdx1 = tr.texIdx1;
}

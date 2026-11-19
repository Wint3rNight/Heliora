#version 450

// Light effect strips (EffectsPass): tracers, muzzle flashes, built on the
// CPU in world space.

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 core;
layout(location = 2) in vec4 edge;
layout(location = 3) in vec2 uv;

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

layout(location = 0) out vec4 outCore;
layout(location = 1) out vec4 outEdge;
layout(location = 2) out vec2 outUv;

void main() {
  outCore = core;
  outEdge = edge;
  outUv = uv;
  gl_Position = scene.projection * scene.view * vec4(pos, 1.0);
}

#version 450

// The powers' glow (step 1.4a): a mesh drawn again as a shell a hair out
// along its normals, so it sits in front of the surface it covers.

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

// The full SceneUniformBuffer, as in shader.vert: a truncated block would put
// every later member at the wrong offset.
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

layout(push_constant) uniform Glow {
  mat4 model;
  mat4 normal;
  vec4 colour; // rgb in HDR, a the strength
  vec4 params; // x time, y how far the shell sits out along the normals, z the floor
} pc;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;

void main() {
  worldNormal = normalize(mat3(pc.normal) * normal);
  vec4 world = pc.model * vec4(pos, 1.0);
  world.xyz += worldNormal * pc.params.y;
  worldPos = world.xyz;
  gl_Position = scene.projection * scene.view * world;
}

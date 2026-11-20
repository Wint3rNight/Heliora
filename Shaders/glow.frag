#version 450

// The powers' glow: see-through, not a recolour. Bright where the surface
// turns away from the camera (a fresnel rim), its brightness moved by slow
// noise drifting up through it. A target, or what's held or torn, also gets
// an even floor of light over its faces (a piece's face turned to the camera
// has no rim), only gently moved by the noise: thresholded noise there shows
// as blotches on a big flat wall. Added on top in HDR, so bloom spreads it.

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormal;

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

// Value noise, 0 to 1, smooth between whole-number points (a quintic ease,
// so no creases show along the grid), and two octaves of it.
float hash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float noise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
                 mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
                 mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
             f.z);
}
float fbm(vec3 x) { return 0.65 * noise(x) + 0.35 * noise(2.03 * x + vec3(17.1, 3.7, 9.2)); }

layout(location = 0) out vec4 outColor;

void main() {
  vec3 toEye = normalize(scene.cameraPosition.xyz - worldPos);
  float facing = clamp(abs(dot(normalize(worldNormal), toEye)), 0.0, 1.0);
  float rim = pow(1.0 - facing, 3.0);
  float t = pc.params.x;
  // Light drifting up through it, brightening and dimming the rim.
  float drift = fbm(worldPos * 1.6 + vec3(0.0, -0.8 * t, 0.25 * t));
  float glow = rim * (0.55 + 0.9 * drift) + pc.params.z * (0.85 + 0.3 * drift);
  outColor = vec4(pc.colour.rgb * glow * pc.colour.a, 0.0);
}

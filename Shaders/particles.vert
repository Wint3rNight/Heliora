#version 450

// The particles' sprites: six corners a slot, a square turned to face the
// camera, fading in quickly and out over its life and shrinking a little as
// it goes. A dead slot's corners all land outside the view.

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

// One particle, as ParticleSpawn lays it out (64 bytes).
struct Particle {
  vec4 positionAge; // xyz position, w seconds lived
  vec4 velocityLife; // xyz m/s, w seconds it lasts
  vec4 colourSize;  // rgb HDR, w m across
  vec4 motion;      // x drag, y swirl, z its share of gravity, w opacity (0: it glows)
};

layout(std430, set = 1, binding = 0) readonly buffer Ring { Particle ring[]; };

layout(location = 0) out vec2 corner;
layout(location = 1) out vec3 colour;
layout(location = 2) out float opacity; // 0 glows; above 0 smoke, faded in and out
layout(location = 3) out float fadeOut;

const vec2 kCorners[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1),
                                vec2(-1, 1));

void main() {
  Particle p = ring[gl_InstanceIndex];
  float age = p.positionAge.w, life = p.velocityLife.w;
  if (life <= 0.0 || age >= life) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // outside the view
    corner = vec2(0.0);
    colour = vec3(0.0);
    opacity = 0.0;
    fadeOut = 0.0;
    return;
  }
  float t = age / life;
  float fade = smoothstep(0.0, 0.08, t) * (1.0 - t);
  vec2 c = kCorners[gl_VertexIndex];
  vec3 right = vec3(scene.view[0][0], scene.view[1][0], scene.view[2][0]);
  vec3 up = vec3(scene.view[0][1], scene.view[1][1], scene.view[2][1]);
  float radius = 0.5 * p.colourSize.w * (1.0 - 0.4 * t);
  vec3 world = p.positionAge.xyz + (right * c.x + up * c.y) * radius;
  gl_Position = scene.projection * scene.view * vec4(world, 1.0);
  corner = c;
  colour = p.colourSize.rgb * fade;
  opacity = p.motion.w;
  fadeOut = fade;
}

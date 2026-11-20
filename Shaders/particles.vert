#version 450

// The particles' sprites: six corners a slot, a square turned to face the
// camera, fading in quickly and out over its life, shrinking or (smoke)
// spreading as it goes. A dead slot's corners all land outside the view.

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

// One particle, as ParticleSpawn lays it out (80 bytes).
struct Particle {
  vec4 positionAge; // xyz position, w seconds lived
  vec4 velocityLife; // xyz m/s, w seconds it lasts
  vec4 colourSize;  // rgb HDR, w m across
  vec4 motion;      // x drag, y swirl, z its share of gravity, w opacity (0: it glows)
  vec4 shape;       // x spread (its size at the end over at the start)
};

layout(set = 0, binding = 3) uniform samplerCube irradianceMap;

layout(std430, set = 1, binding = 0) readonly buffer Ring { Particle ring[]; };

layout(location = 0) out vec2 corner;
layout(location = 1) out vec3 colour;
layout(location = 2) out float opacity; // 0 glows; above 0 smoke, faded in and out
layout(location = 3) out float fadeOut;

const float PI = 3.14159265359;

// The light on a puff of smoke (its colour an albedo): as a white surface
// half facing the sun (its lit side and its shaded side together; no
// shadows), plus the sky's irradiance from above and from the side it's
// seen from (toEye), scaled as lit.frag scales its ambient (dimmed at
// night).
vec3 smokeLight(vec3 toEye) {
  vec3 sunDir = normalize(-scene.directionalLight.direction.xyz);
  float sunUp = smoothstep(-0.05, 0.1, sunDir.y);
  vec3 sun = scene.directionalLight.colorIntensity.rgb * scene.directionalLight.colorIntensity.a;
  vec3 sky = 0.5 * (textureLod(irradianceMap, vec3(0.0, 1.0, 0.0), 0.0).rgb +
                    textureLod(irradianceMap, toEye, 0.0).rgb);
  return sun * (sunUp * 0.5 / PI) + sky * scene.qualityToggles2.w;
}

const float kFewestPixels = 1.5; // a sprite's radius on screen, at least
const float kWidest = 0.5;        // and at most, as a share of the view's height (est.)

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
  // Its size: toward its spread, fast at first (particleSizeAt).
  float spread = p.shape.x;
  float u = 1.0 - t;
  float grown = 1.0 + (spread - 1.0) * (1.0 - u * u);
  opacity = p.motion.w;
  if (opacity > 0.0 && spread > 1.0) {
    // Smoke spreading out keeps its smoke, so it thins as one over its
    // area (smokeCover); it only fades out at the very end.
    opacity = 1.0 - pow(1.0 - min(opacity, 0.999), 1.0 / (grown * grown));
    fade = smoothstep(0.0, 0.03, t) * (1.0 - smoothstep(0.75, 1.0, t));
  }
  float radius = 0.5 * p.colourSize.w * grown;
  // Faded out as the camera gets inside it, so smoke it stands in (a
  // rocket's back blast right behind the gunner) doesn't fill the view.
  float seen = distance(p.positionAge.xyz, scene.cameraPosition.xyz);
  fade *= smoothstep(0.6 * radius, 0.6 * radius + 1.0, seen);
  vec3 middle = (scene.view * vec4(p.positionAge.xyz, 1.0)).xyz; // in the camera's frame
  float depth = -middle.z;
  // Its radius as a share of the view's height.
  float screen = radius * scene.projection[1][1] * 0.5 / max(depth, 1e-3);
  // Never under a pixel and a half across, made fainter to match (the same
  // cover), so a far trail stays a line instead of flickering dots.
  float pixels = screen * scene.viewportSize.y;
  if (pixels > 0.0 && pixels < kFewestPixels) {
    float share = pixels / kFewestPixels;
    radius /= share;
    fade *= share * share;
  }
  // Across most of the view, smoke reads as haze and costs the most to draw
  // (every pixel under it): it fades out from a quarter of the view's height
  // to a half. Nothing left to see, or behind the camera: not drawn at all.
  fade *= 1.0 - smoothstep(kWidest * 0.5, kWidest, screen);
  float left = opacity > 0.0 ? opacity * fade : fade;
  if (depth <= 0.0 || left < 1.0 / 255.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    corner = vec2(0.0);
    colour = vec3(0.0);
    opacity = 0.0;
    fadeOut = 0.0;
    return;
  }
  // A square facing the camera: its corners in the camera's frame.
  vec2 c = kCorners[gl_VertexIndex];
  gl_Position = scene.projection * vec4(middle + vec3(c * radius, 0.0), 1.0);
  corner = c;
  colour = p.colourSize.rgb * fade;
  if (opacity > 0.0)
    colour *= smokeLight(seen > 1e-4 ? (scene.cameraPosition.xyz - p.positionAge.xyz) / seen
                                     : vec3(0.0, 1.0, 0.0));
  fadeOut = fade;
}

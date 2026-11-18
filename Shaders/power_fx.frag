#version 450

// A strip of the powers' light: a bright core down its middle, fading to its
// edge colour and to nothing at its sides, with noise flowing along it.
// Added on top in HDR, so bloom spreads it.

layout(location = 0) in vec4 core; // rgb in HDR, a how far it's faded in
layout(location = 1) in vec4 edge;
layout(location = 2) in vec2 uv;   // x along it in metres, y across it, 0 to 1

layout(push_constant) uniform Fx {
  vec4 params; // x time
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
  float across = 1.0 - abs(uv.y * 2.0 - 1.0); // 1 down the middle, 0 at the sides
  float soft = smoothstep(0.0, 1.0, across);
  float centre = across * across;
  float t = pc.params.x;
  float flow = fbm(vec3(uv.x * 2.5 - 4.0 * t, uv.y * 2.0, 0.5 * t));
  // The core colour down the middle, the edge colour in the fringe round it.
  vec3 colour = core.rgb * centre * (0.7 + 0.5 * flow) +
                edge.rgb * soft * (1.0 - centre) * (0.4 + 0.6 * flow);
  outColor = vec4(colour * core.a, 0.0);
}

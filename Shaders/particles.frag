#version 450

// A particle: a soft round dot, brightest in its middle, added on top in HDR
// so bloom spreads it.

layout(location = 0) in vec2 corner;
layout(location = 1) in vec3 colour;

layout(location = 0) out vec4 outColor;

void main() {
  float r = length(corner);
  float light = 1.0 - smoothstep(0.0, 1.0, r);
  outColor = vec4(colour * light * light, 0.0);
}

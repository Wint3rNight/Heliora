#version 450

// A particle: a soft round dot, densest in its middle. A glowing one is
// added on top in HDR (alpha 0) so bloom spreads it; smoke and dust write
// their opacity, premultiplied, and cover what's behind that much (their
// colour already lit, in particles.vert).

layout(location = 0) in vec2 corner;
layout(location = 1) in vec3 colour;
layout(location = 2) in float opacity;
layout(location = 3) in float fadeOut;

layout(location = 0) out vec4 outColor;

void main() {
  float r = length(corner);
  float light = 1.0 - smoothstep(0.0, 1.0, r);
  float shape = light * light;
  if (opacity <= 0.0) {
    outColor = vec4(colour * shape, 0.0);
  } else {
    float a = clamp(opacity * fadeOut * shape, 0.0, 1.0);
    outColor = vec4(colour * shape * opacity, a); // colour is already faded and lit
  }
}

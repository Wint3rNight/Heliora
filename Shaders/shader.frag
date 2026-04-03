#version 450

layout(location = 0) in vec3 fragCol; // color from vertex shader
layout(location = 1) in vec2 fragTex; // texture coordinate from vertex shader

layout(set = 1, binding = 0) uniform sampler2D textureSampler; // texture sampler

layout(location = 0) out vec4 outColor; // final color output

void main() {
  outColor = texture(textureSampler, fragTex); // modulate color with texture
}

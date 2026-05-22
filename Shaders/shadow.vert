#version 450

layout(location = 0) in vec3 pos;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec2 fragUV;

layout(push_constant) uniform ShadowPush {
  mat4 model;
  mat4 lightSpaceMatrix;
  uint albedoIdx;      // Phase 7.2: bindless index for alpha-test discard
} pushShadow;

void main() {
  gl_Position = pushShadow.lightSpaceMatrix * pushShadow.model * vec4(pos, 1.0);
  fragUV = uv;
}

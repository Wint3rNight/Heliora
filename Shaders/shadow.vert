#version 450

layout(location = 0) in vec3 pos;

layout(push_constant) uniform ShadowPush {
  mat4 model;
  mat4 lightSpaceMatrix;
} pushShadow;

void main() {
  gl_Position = pushShadow.lightSpaceMatrix * pushShadow.model * vec4(pos, 1.0);
}

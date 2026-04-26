#version 450

layout(location = 0) in vec3 texCoord;
layout(set = 1, binding = 0) uniform samplerCube skyboxSampler;
layout(location = 0) out vec4 outColor;

void main() {
  vec3 hdrColor = texture(skyboxSampler, texCoord).rgb;
  outColor = vec4(hdrColor * 0.35, 1.0);
}

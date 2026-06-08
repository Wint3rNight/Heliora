#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Alpha-tested shadow caster. Without this, Sponza's foliage casts solid
// quad-shaped block shadows on the floor (one quad per leaf) producing the
// stippled pattern the player sees on lit floor. Discarding sub-threshold
// alpha makes the shadow follow the leaf silhouette in the albedo texture.

layout(location = 0) in vec2 fragUV;

// Phase 7.2: bindless texture array. The shadow pipeline layout carries this
// single set at set=0. Only the albedo texture is sampled (indexed by
// push constant); the rest of the 4096-slot array is untouched.
layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform ShadowPush {
  mat4 model;
  mat4 lightSpaceMatrix;
  uint albedoIdx;
  uint materialFlags;
  uint alphaCutoff255;
} pushShadow;

float interleavedGradientNoise(vec2 p) {
  vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
  return fract(magic.z * fract(dot(p, magic.xy)));
}

void main() {
  bool alphaMasked = (pushShadow.materialFlags & 2u) != 0u;
  if (alphaMasked) {
    float a = texture(textures[nonuniformEXT(pushShadow.albedoIdx)], fragUV).a;
    float cutoff = float(pushShadow.alphaCutoff255) / 255.0;
    float width = max(fwidth(a) * 1.35, 1.0 / 255.0);
    float threshold =
        cutoff + (interleavedGradientNoise(gl_FragCoord.xy) - 0.5) * width * 0.45;
    if (a < threshold) discard;
  }
}

#version 450

// Alpha-tested shadow caster. Without this, Sponza's foliage casts solid
// quad-shaped block shadows on the floor (one quad per leaf) producing the
// stippled pattern the player sees on lit floor. Discarding sub-threshold
// alpha makes the shadow follow the leaf silhouette in the albedo texture.

layout(location = 0) in vec2 fragUV;

// Material descriptor set layout matches the geometry pass (5 samplers),
// but the shadow pipeline layout only carries this single set, so it lives
// at set=0 here (vs set=1 in the lit pass). Only binding 0 (albedo) is
// sampled; the rest are silent.
layout(set = 0, binding = 0) uniform sampler2D albedoSampler;

void main() {
  float a = texture(albedoSampler, fragUV).a;
  if (a < 0.5) discard;
}

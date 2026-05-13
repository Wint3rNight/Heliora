#pragma once

struct Material {
  int albedoTextureId = 0;
  int normalTextureId = 0;
  int metallicTextureId = 0;
  int roughnessTextureId = 0;
  int aoTextureId = 0;
  int descriptorSetId = 0;
  // glTF authoring flags. Foliage in Sponza is doubleSided + alphaMode=MASK;
  // honoring these stops leaves popping in/out and using the wrong cutoff.
  bool doubleSided = false;
};

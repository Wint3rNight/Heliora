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
  // Permanent per-material cloth flag. Drives the Charlie sheen lobe in
  // lit.frag so banners/curtains stop reading as polished satin. Detected
  // at glTF import time from material name / texture filename keywords;
  // unrecognized models fall back to the shader's chroma heuristic.
  bool isCloth = false;
  // glTF alphaMode=MASK. Opaque materials must not alpha-discard in either
  // the G-buffer or shadow pass, even if their source image has a non-opaque
  // alpha channel for authoring/packing reasons.
  bool alphaMasked = false;
  bool alphaBlended = false;
  float alphaCutoff = 0.5f;
};

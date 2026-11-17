#pragma once

#include "AnimationClip.h"
#include "CharacterAnimator.h"
#include "CharacterFile.h"
#include "PrimitiveMesh.h"
#include "Skeleton.h"
#include "Skinning.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

// Reading a .character's model and clips with Assimp into the animation
// types of heliora_core. The game's side, since heliora_core doesn't use
// Assimp.

struct CharacterAsset {
  Skeleton skeleton; // every node of the model file, parents first
  int hips = -1;     // the bone root motion comes from
  MeshData mesh;     // the skinned meshes merged, in their own space
  std::vector<VertexSkin> skin; // one per mesh vertex
  // The albedo texture embedded in the model file (PNG or JPEG bytes), or
  // empty with none.
  std::vector<unsigned char> albedo;
  std::string albedoName;
  // Around the mesh in the bind pose, in the skeleton's space.
  glm::vec3 boundsMin{0.0f};
  glm::vec3 boundsMax{0.0f};
  AnimationLibrary library;
  const CharacterClipData *clip(ClipSlot slot) const { return library.clip(slot); }
};

// A path from a .character file, as written or one folder up (the game
// usually runs from a build folder); empty if neither exists.
std::string resolveCharacterPath(const std::string &path);

// Reads the model and every clip. Nothing, with the reason in error, when
// the model can't be read or has no skinned mesh; a clip that can't be read
// is left out with a log line.
std::optional<CharacterAsset> importCharacter(const CharacterDescription &description,
                                              std::string &error);

// One clip read again (the Clips section gave its move another file), onto
// the asset's skeleton. Nothing, with a log line, if it can't be read.
std::optional<CharacterClipData> importCharacterClip(const CharacterClip &clip,
                                                     const CharacterAsset &asset);

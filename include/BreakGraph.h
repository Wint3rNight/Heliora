#pragma once

#include "Fracture.h"
#include "Stress.h"

#include <glm/glm.hpp>

#include <vector>

// The breakable pieces of a scene as one graph: the links the cutter makes
// inside each object, and links between pieces of objects that touch, so a
// building of breakable walls and slabs holds together as one. Pieces
// resting on something that never breaks are anchors. Hits break links;
// what's left falls apart into connected groups, and a group with no anchor
// falls. No Vulkan and no physics here, so it's tested like the rest of
// heliora_core.

// Two pieces holding onto each other, a < b, until an impulse (N s) of at
// least strength reaches the link.
struct BreakLink {
  int a = 0;
  int b = 0;
  float strength = 0.0f;
  bool broken = false;
};

// The links of one object's cut, its pieces numbered from firstPiece: each
// as strong as its face's area times strengthPerArea (N s per m^2), and no
// weaker than a quarter of the object's average link, so the slivers where
// cells barely touch don't snap at any tap anywhere.
std::vector<BreakLink> cutLinks(const Fracture &fracture, float strengthPerArea, int firstPiece);
// Each of the cut's links' shared face (in the cut's space), facing from its
// first piece to its second, for the stress solver; in the links' order.
std::vector<LinkFace> cutFaces(const Fracture &fracture);

// Pieces joined by unbroken links, smallest index first, and whether any of
// them is an anchor (if not, the group falls).
struct BreakGroup {
  std::vector<int> pieces;
  bool anchored = false;
};

// The connected groups, in the order of their first piece.
std::vector<BreakGroup> splitGroups(int pieceCount, const std::vector<BreakLink> &links,
                                    const std::vector<bool> &anchors);

// A link that might break, and where it is now (world space).
struct LinkAt {
  int link = 0;
  glm::vec3 position{0.0f};
};

// How far a hit reaches: the impulse that arrives at a link d metres away is
// impulse / (1 + (d / kBreakSpread)^2), so a harder hit breaks a wider patch.
constexpr float kBreakSpread = 0.5f; // m

// The unbroken links among the candidates the hit breaks (indices into links).
std::vector<int> linksBrokenBy(const std::vector<BreakLink> &links,
                               const std::vector<LinkAt> &candidates, glm::vec3 hit,
                               float impulse);

// A piece put in the world: its faces' corners and normals transformed.
struct PlacedPiece {
  int object = 0;
  std::vector<FractureFace> faces;
};
PlacedPiece placePiece(const FracturePiece &piece, const glm::mat4 &where, int object);

// Pieces of different objects whose outside faces lie against each other:
// opposite normals, no more than tolerance apart, overlapping by this area.
struct PieceTouch {
  int a = 0;
  int b = 0;
  float area = 0.0f;
  glm::vec3 centre{0.0f};
  LinkFace face; // the biggest overlap's, facing from a to b
};
std::vector<PieceTouch> touchingPieces(const std::vector<PlacedPiece> &pieces, float tolerance);

// How much two pieces touch through any of their faces (opposite normals,
// within tolerance apart), and the contact's middle; 0 if they don't.
float touchArea(const PlacedPiece &a, const PlacedPiece &b, float tolerance, glm::vec3 &centre);
// The face of their biggest overlap, facing from a to b (area 0 if none).
LinkFace touchFace(const PlacedPiece &a, const PlacedPiece &b, float tolerance);

// Whether an outside face of the piece faces down at this height (within
// tolerance): it rests on a floor there.
bool restsOn(const PlacedPiece &piece, float height, float tolerance);

// Whether an anchored group stays where it is as part of the structure: it
// holds at least a quarter of the volume of an object it's part of. A
// smaller one (a piece lying on the ground, a stub of wall) moves, and
// gravity keeps it there. A group with no anchor always moves.
constexpr float kStructuralShare = 0.25f;
bool staysPut(const BreakGroup &group, const std::vector<int> &objectOf,
              const std::vector<float> &volume, const std::vector<float> &objectVolume);

// A hit's momentum split between the groups it frees, by how far each is
// from the hit (as links feel it), adding up to momentum.
std::vector<float> shareHit(float momentum, const std::vector<float> &distances);

// Every link's strength changed by up to amount either way (0.25 for a
// quarter), from the seed: the same structure breaks differently each Play.
void varyStrengths(std::vector<BreakLink> &links, uint32_t seed, float amount);

// The impulse it takes to split a piece where it's hit: as strong as its
// cross-section, volume^(2/3) times strengthPerArea.
float splitImpulse(float volume, float strengthPerArea);

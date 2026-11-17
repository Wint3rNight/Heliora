#pragma once

#include "Fracture.h"

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
};
std::vector<PieceTouch> touchingPieces(const std::vector<PlacedPiece> &pieces, float tolerance);

// Whether an outside face of the piece faces down at this height (within
// tolerance): it rests on a floor there.
bool restsOn(const PlacedPiece &piece, float height, float tolerance);

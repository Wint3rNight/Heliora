#pragma once

#include "PrimitiveMesh.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

// Pre-cutting a box into pieces for destruction, ahead of time: Voronoi
// cells. Points are scattered inside the box from a seed number, and each
// piece is the part of the box closer to its point than to any other, so
// every piece is convex. Pieces that share a cut face hold onto each other
// (a link), as strongly as the face is big. No Vulkan and no physics here,
// so it's tested like the rest of heliora_core.

// What's across a face that isn't another piece of the same cut.
constexpr int kOutsideFace = -1; // the object's own outside
constexpr int kCutFace = -2;     // an earlier cut, onto a piece not in this set

// How a material breaks when a piece is cut where it's hit.
enum class FracturePattern : uint8_t {
  None,      // it doesn't split (metal, rubber)
  Clustered, // small pieces at the hit, bigger further out (stone)
  Splinters, // the same, cells stretched along the grain (wood)
  Shards,    // rings around the hit, across the face it was hit on (ice)
};

// One flat face of a piece: its corners in order, counter-clockwise seen from
// outside, and the piece across it, or kOutsideFace / kCutFace.
struct FractureFace {
  std::vector<glm::vec3> points;
  glm::vec3 normal{0.0f}; // outward, unit length
  int neighbour = -1;
};

// A convex piece, in the box's own space (centred on its middle).
struct FracturePiece {
  std::vector<FractureFace> faces;
  float volume = 0.0f;
  glm::vec3 centre{0.0f}; // its volume's middle
};

// Two pieces sharing a cut face of this area, a < b.
struct FractureLink {
  int a = 0;
  int b = 0;
  float area = 0.0f;
  glm::vec3 centre{0.0f}; // the shared face's middle
};

struct Fracture {
  std::vector<FracturePiece> pieces;
  std::vector<FractureLink> links;
};

// A box of this size (full extents) cut into this many pieces (1 is the
// whole box). The same size, count and seed always give the same pieces.
Fracture fractureBox(glm::vec3 size, int pieces, uint32_t seed);

float faceArea(const FractureFace &face);

// A piece to draw: every face flat, a fan of triangles with its own
// normal; the box's outside faces tinted outside, the cut faces inside.
// Texture coordinates run 1 per metre across each face.
MeshData makePieceMesh(const FracturePiece &piece, glm::vec3 outside, glm::vec3 inside);

// A convex piece cut again into Voronoi cells around these points (in its
// own space), where it was hit. Its faces keep what they were: the object's
// outside stays outside, an earlier cut becomes kCutFace; the new faces are
// onto the new pieces, and links come from them. With stretch above 1 the
// cells are that much longer along axis (unit): wood's splinters.
Fracture fractureConvex(const FracturePiece &piece, const std::vector<glm::vec3> &points,
                        glm::vec3 axis = glm::vec3(0.0f), float stretch = 1.0f);

// Up to count points inside the piece for cutting it where it was hit: hit
// in its space, push the way the hit went in. Clustered and Splinters crowd
// the hit; Shards lie in rings around it, across the face it was hit on.
// The same seed gives the same points.
std::vector<glm::vec3> pointsAround(const FracturePiece &piece, glm::vec3 hit, glm::vec3 push,
                                    int count, FracturePattern pattern, uint32_t seed);

// Rubble: what's too small to pick up settles into a heap that's part of
// the ground. A heap is a low, rough dome, radius 1 and kRubbleHeight high
// at its middle, base at y = 0; it's scaled to the volume it holds.
constexpr float kRubbleHeight = 0.35f;
MeshData makeRubbleMound(uint32_t seed, glm::vec3 colour);
// The radius of a heap holding this volume (m^3): a paraboloid dome holds
// pi r^2 h / 2.
float rubbleRadius(float volume);

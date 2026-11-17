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

// One flat face of a piece: its corners in order, counter-clockwise seen from
// outside, and the piece across it, or -1 for the box's own outside.
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

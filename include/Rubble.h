#pragma once

#include "PrimitiveMesh.h"

#include <glm/glm.hpp>

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

// Rubble as part of the ground: what's too small to pick up settles into a
// layer of rubble depth over the ground, in tiles of samples. It comes in
// as deposits (a volume spread as a smooth bump around a point) and slumps
// downhill until no slope is steeper than loose rubble rests at, so heaps
// close together run into one pile: a ridge at a broken wall's foot, a cone
// under a tower. No Vulkan and no physics here, so it's tested like the
// rest of heliora_core.

constexpr int kRubbleSamples = 33;                                    // per tile side
constexpr float kRubbleSpacing = 0.125f;                              // m between samples
constexpr float kRubbleTile = (kRubbleSamples - 1) * kRubbleSpacing; // 4 m
constexpr float kRubbleSlope = 0.78128563f;  // tan(38 degrees): loose rubble rests at 35 to 40
constexpr float kRubbleSunk = 0.03f;         // m: where there's none, the mesh hides under the ground
constexpr float kRubbleLumps = 0.04f;        // m: the most a drawn sample is raised or lowered
constexpr float kRubbleThinnest = 0.005f;    // m: thinner than this collides as a hole

// A tile, by where it is: the floor of x / 4 m and of z / 4 m. A tile's
// samples run from its corner, 33 along x and 33 along z, and neighbours
// share their border row.
using RubbleKey = std::pair<int, int>;
RubbleKey rubbleKeyAt(glm::vec2 xz);
glm::vec2 rubbleTileCorner(RubbleKey key);

class RubbleLayer {
public:
  // The ground's height under a point (x, z), or none: rubble there is lost.
  using Ground = std::function<std::optional<float>(glm::vec2)>;

  explicit RubbleLayer(Ground ground);

  // volume m^3 spread around at (x, z), most at the middle and none from
  // radius out, in this colour. Over no ground it's lost.
  void deposit(glm::vec2 at, float volume, float radius, glm::vec3 colour);
  // Up to passes passes of slumping; true while anything is still moving.
  bool settle(int passes);

  float depth(glm::vec2 at) const;     // m of rubble, between samples blended
  float surface(glm::vec2 at) const;   // the ground plus the rubble; NaN over no ground
  glm::vec3 colour(glm::vec2 at) const; // the nearest sample's
  float volume() const;                 // m^3 in all of it

  // Tiles changed since the last call.
  std::vector<RubbleKey> takeChanged();
  std::vector<RubbleKey> tiles() const;

  // A tile to draw, relative to its corner: one vertex per sample (always
  // 33 x 33, so it can be rewritten in place), smooth normals across tile
  // borders, lumps and colour noise fixed to the world; where there's no
  // rubble it sits kRubbleSunk under the ground.
  MeshData tileMesh(RubbleKey key) const;
  // Its heights for a height field, 33 x 33, x fastest: the ground plus the
  // rubble, NaN where there's under kRubbleThinnest (a hole, so the ground
  // under it is what's hit) or no ground.
  std::vector<float> tileHeights(RubbleKey key) const;

  void clear();

private:
  struct Tile {
    std::vector<float> ground; // NaN: no ground
    std::vector<float> depth;
    std::vector<glm::vec3> colour;
    bool unsettled = false;
    bool changed = false;
  };

  Tile &tile(RubbleKey key); // made, with its ground, if it's new
  const Tile *findTile(RubbleKey key) const;
  float groundAt(int gx, int gz);
  float sampleDepth(int gx, int gz) const;
  float sampleGround(int gx, int gz) const;
  void addDepth(int gx, int gz, float amount, glm::vec3 colour);
  float drawnHeight(int gx, int gz, float fallbackGround) const;

  Ground ground;
  std::map<RubbleKey, Tile> tiles_;
};

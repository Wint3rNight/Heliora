#include "Rubble.h"
#include "TestRunner.h"

#include <algorithm>
#include <cmath>

namespace {

const glm::vec3 kGrey(0.6f);

RubbleLayer flatLayer() {
  return RubbleLayer([](glm::vec2) -> std::optional<float> { return 0.0f; });
}

void settleFully(RubbleLayer &layer) {
  for (int i = 0; i < 2000 && layer.settle(50); ++i) {
  }
}

// The steepest slope between neighbouring samples in a square around the
// middle, where the higher one holds rubble (a bare ground step can't
// slump), as rise over run.
float steepestSlope(const RubbleLayer &layer, glm::vec2 middle, float reach) {
  float steepest = 0.0f;
  const int n = static_cast<int>(reach / kRubbleSpacing);
  const int cx = static_cast<int>(std::round(middle.x / kRubbleSpacing));
  const int cz = static_cast<int>(std::round(middle.y / kRubbleSpacing));
  for (int x = cx - n; x <= cx + n; ++x)
    for (int z = cz - n; z <= cz + n; ++z) {
      const glm::vec2 p(x * kRubbleSpacing, z * kRubbleSpacing);
      for (int dx = -1; dx <= 1; ++dx)
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dz == 0)
            continue;
          const glm::vec2 q = p + glm::vec2(dx, dz) * kRubbleSpacing;
          if (layer.depth(p) < 1e-4f)
            continue;
          const float rise = layer.surface(p) - layer.surface(q);
          steepest = std::max(steepest, rise / glm::length(q - p));
        }
    }
  return steepest;
}

} // namespace

TEST(a_tile_key_is_the_floor_of_four_metres) {
  CHECK(rubbleKeyAt(glm::vec2(0.1f, 3.9f)) == RubbleKey(0, 0));
  CHECK(rubbleKeyAt(glm::vec2(4.0f, -0.1f)) == RubbleKey(1, -1));
  CHECK(rubbleKeyAt(glm::vec2(-4.1f, 8.5f)) == RubbleKey(-2, 2));
  CHECK(rubbleTileCorner(RubbleKey(-2, 3)) == glm::vec2(-8.0f, 12.0f));
}

TEST(a_deposit_adds_exactly_its_volume) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.3f, 2.7f), 0.2f, 0.5f, kGrey);
  CHECK_NEAR(layer.volume(), 0.2, 1e-5);
  CHECK(layer.depth(glm::vec2(1.3f, 2.7f)) > 0.0f);
  CHECK(layer.depth(glm::vec2(2.0f, 2.7f)) == 0.0f);
}

TEST(slumping_keeps_the_volume) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 1.0f, 0.2f, kGrey);
  CHECK(layer.settle(1)); // a spike: it moves
  settleFully(layer);
  CHECK(!layer.settle(1));
  CHECK_NEAR(layer.volume(), 1.0, 1e-4);
}

TEST(slumping_leaves_no_slope_steeper_than_38_degrees) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 1.0f, 0.2f, kGrey);
  CHECK(steepestSlope(layer, glm::vec2(1.0f), 1.5f) > 2.0f);
  settleFully(layer);
  CHECK(steepestSlope(layer, glm::vec2(1.0f), 2.5f) < kRubbleSlope + 0.02f);
  // A cone of 1 m^3 at 38 degrees is about 1.07 m across its foot.
  CHECK(layer.depth(glm::vec2(1.8f, 1.0f)) > 0.0f);
  CHECK(layer.surface(glm::vec2(1.0f, 1.0f)) < 1.0f);
}

TEST(deposits_close_together_end_up_one_pile) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.15f, 0.3f, kGrey);
  layer.deposit(glm::vec2(1.8f, 1.0f), 0.15f, 0.3f, kGrey);
  settleFully(layer);
  for (float x = 1.0f; x <= 1.8f; x += kRubbleSpacing)
    CHECK(layer.depth(glm::vec2(x, 1.0f)) > 0.01f);
}

TEST(a_pile_on_a_tile_corner_spreads_into_its_neighbours_seamlessly) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(0.0f, 0.0f), 0.3f, 0.4f, kGrey);
  settleFully(layer);
  CHECK_NEAR(layer.volume(), 0.3, 1e-4);
  const std::vector<RubbleKey> keys = layer.tiles();
  for (RubbleKey key : {RubbleKey(-1, -1), RubbleKey(0, -1), RubbleKey(-1, 0), RubbleKey(0, 0)})
    CHECK(std::find(keys.begin(), keys.end(), key) != keys.end());
  CHECK(layer.depth(glm::vec2(-0.3f, -0.3f)) > 0.0f);
  // The column a tile shares with its left neighbour holds the same heights.
  const std::vector<float> right = layer.tileHeights(RubbleKey(0, 0));
  const std::vector<float> left = layer.tileHeights(RubbleKey(-1, 0));
  CHECK(right.size() == static_cast<size_t>(kRubbleSamples * kRubbleSamples));
  CHECK(left.size() == right.size());
  int compared = 0;
  for (int z = 0; z < kRubbleSamples; ++z) {
    const float a = right[z * kRubbleSamples];
    const float b = left[z * kRubbleSamples + kRubbleSamples - 1];
    CHECK(std::isnan(a) == std::isnan(b));
    if (!std::isnan(a)) {
      CHECK(a == b);
      ++compared;
    }
  }
  CHECK(compared > 0);
}

TEST(a_small_deposit_stays_thin) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.005f, 0.35f, kGrey);
  settleFully(layer);
  CHECK(layer.depth(glm::vec2(1.0f, 1.0f)) > 0.0f);
  CHECK(layer.depth(glm::vec2(1.0f, 1.0f)) < 0.05f);
}

TEST(rubble_on_a_ledge_spills_off_its_edge) {
  // A 1 m ledge for x under 0.9375, between two rows of samples.
  RubbleLayer layer([](glm::vec2 p) -> std::optional<float> { return p.x < 0.9375f ? 1.0f : 0.0f; });
  layer.deposit(glm::vec2(0.75f, 0.5f), 0.3f, 0.3f, kGrey);
  settleFully(layer);
  CHECK_NEAR(layer.volume(), 0.3, 1e-4);
  CHECK(layer.depth(glm::vec2(1.0f, 0.5f)) > 0.01f);
  // The ledge stays sharp away from the pile: ground cast on both sides.
  CHECK_NEAR(layer.surface(glm::vec2(0.875f, 3.5f)), 1.0, 1e-5);
  CHECK_NEAR(layer.surface(glm::vec2(1.0f, 3.5f)), 0.0, 1e-5);
}

TEST(colours_mix_by_volume) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.1f, 0.4f, glm::vec3(1.0f, 0.0f, 0.0f));
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.1f, 0.4f, glm::vec3(0.0f, 0.0f, 1.0f));
  const glm::vec3 c = layer.colour(glm::vec2(1.0f, 1.0f));
  CHECK_NEAR(c.r, 0.5, 1e-4);
  CHECK_NEAR(c.b, 0.5, 1e-4);
}

TEST(the_same_deposits_give_the_same_layer) {
  auto build = [] {
    RubbleLayer layer = flatLayer();
    layer.deposit(glm::vec2(1.0f, 1.0f), 0.3f, 0.3f, kGrey);
    layer.deposit(glm::vec2(1.5f, 0.2f), 0.2f, 0.5f, glm::vec3(0.5f, 0.4f, 0.3f));
    layer.deposit(glm::vec2(-0.5f, 1.1f), 0.4f, 0.2f, kGrey);
    settleFully(layer);
    return layer;
  };
  const RubbleLayer a = build();
  const RubbleLayer b = build();
  CHECK(a.tiles() == b.tiles());
  for (RubbleKey key : a.tiles()) {
    const std::vector<float> ha = a.tileHeights(key), hb = b.tileHeights(key);
    for (size_t i = 0; i < ha.size(); ++i)
      CHECK((std::isnan(ha[i]) && std::isnan(hb[i])) || ha[i] == hb[i]);
  }
}

TEST(a_tile_mesh_keeps_its_vertex_count_and_hides_where_there_is_none) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(0.5f, 0.5f), 0.02f, 0.3f, kGrey);
  const MeshData before = layer.tileMesh(RubbleKey(0, 0));
  CHECK(before.vertices.size() == static_cast<size_t>(kRubbleSamples * kRubbleSamples));
  CHECK(before.indices.size() == static_cast<size_t>((kRubbleSamples - 1) * (kRubbleSamples - 1) * 6));
  layer.deposit(glm::vec2(1.0f, 1.0f), 1.0f, 0.2f, kGrey);
  settleFully(layer);
  const MeshData after = layer.tileMesh(RubbleKey(0, 0));
  CHECK(after.vertices.size() == before.vertices.size());
  CHECK(after.indices == before.indices);
  // Far from the pile: under the ground. On it: above.
  const Vertex &far = after.vertices[30 * kRubbleSamples + 30];
  CHECK_NEAR(far.pos.y, -kRubbleSunk, 1e-5);
  CHECK_NEAR(far.pos.x, 30 * kRubbleSpacing, 1e-5);
  CHECK_NEAR(far.pos.z, 30 * kRubbleSpacing, 1e-5);
  const Vertex &top = after.vertices[8 * kRubbleSamples + 8];
  CHECK(top.pos.y > 0.3f);
  CHECK(top.normal.y > 0.5f);
  // Every triangle faces up, the way its normals do (culled otherwise).
  int badWinding = 0;
  for (size_t i = 0; i < after.indices.size(); i += 3) {
    const Vertex &a = after.vertices[after.indices[i]];
    const Vertex &b = after.vertices[after.indices[i + 1]];
    const Vertex &c = after.vertices[after.indices[i + 2]];
    if (glm::dot(glm::cross(b.pos - a.pos, c.pos - a.pos), a.normal + b.normal + c.normal) <= 0.0f)
      ++badWinding;
  }
  CHECK(badWinding == 0);
}

TEST(rubble_too_thin_is_a_hole_for_physics) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.1f, 0.4f, kGrey);
  const std::vector<float> h = layer.tileHeights(RubbleKey(0, 0));
  CHECK(std::isnan(h[30 * kRubbleSamples + 30]));
  CHECK_NEAR(h[8 * kRubbleSamples + 8], layer.depth(glm::vec2(1.0f, 1.0f)), 1e-6);
}

TEST(a_deposit_over_no_ground_is_lost) {
  RubbleLayer layer([](glm::vec2 p) -> std::optional<float> {
    if (p.x > 2.0f)
      return std::nullopt;
    return 0.0f;
  });
  layer.deposit(glm::vec2(3.0f, 3.0f), 0.1f, 0.3f, kGrey);
  CHECK(layer.volume() == 0.0f);
  CHECK(std::isnan(layer.surface(glm::vec2(3.0f, 3.0f))));
}

TEST(changed_tiles_are_handed_out_once) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.1f, 0.3f, kGrey);
  const std::vector<RubbleKey> changed = layer.takeChanged();
  CHECK(changed.size() == 1);
  CHECK(changed.front() == RubbleKey(0, 0));
  CHECK(layer.takeChanged().empty());
}

TEST(clearing_empties_the_layer) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.1f, 0.3f, kGrey);
  layer.clear();
  CHECK(layer.volume() == 0.0f);
  CHECK(layer.tiles().empty());
}

TEST(a_new_tile_takes_its_shared_border_from_the_tile_already_there) {
  // The ground rays of a later tile start somewhere else and see other
  // ground: the border both tiles hold must still agree.
  float offset = 0.0f;
  RubbleLayer layer([&offset](glm::vec2) -> std::optional<float> { return offset; });
  layer.deposit(glm::vec2(1.0f, 1.0f), 0.1f, 0.3f, kGrey);
  offset = 0.5f;
  layer.deposit(glm::vec2(5.0f, 1.0f), 0.1f, 0.3f, kGrey);
  layer.deposit(glm::vec2(4.0f, 1.0f), 0.1f, 0.3f, kGrey);
  const std::vector<float> left = layer.tileHeights(RubbleKey(0, 0));
  const std::vector<float> right = layer.tileHeights(RubbleKey(1, 0));
  int compared = 0;
  for (int z = 0; z < kRubbleSamples; ++z) {
    const float a = left[z * kRubbleSamples + kRubbleSamples - 1];
    const float b = right[z * kRubbleSamples];
    CHECK(std::isnan(a) == std::isnan(b));
    if (!std::isnan(a)) {
      CHECK(a == b);
      ++compared;
    }
  }
  CHECK(compared > 0);
}

TEST(no_slope_steeper_than_38_degrees_survives_across_a_tile_border) {
  RubbleLayer layer = flatLayer();
  layer.deposit(glm::vec2(4.6f, 1.0f), 0.8f, 0.2f, kGrey);
  settleFully(layer);
  layer.deposit(glm::vec2(3.6f, 1.0f), 0.3f, 0.2f, kGrey);
  settleFully(layer);
  layer.deposit(glm::vec2(4.1f, 1.6f), 0.4f, 0.2f, kGrey);
  settleFully(layer);
  CHECK(steepestSlope(layer, glm::vec2(4.0f, 1.0f), 2.0f) < kRubbleSlope + 0.02f);
}

TEST(no_skirt_hangs_off_a_raised_block_where_there_is_no_rubble) {
  // A 1 m block 3 m high on flat ground, the rubble beside it: the mesh's
  // bare samples mustn't join the block's top to the ground outside it, a
  // slanted sheet that would stick out of the block's sides.
  auto ground = [](glm::vec2 p) {
    return (p.x > 0.97f && p.x < 2.03f && p.y > 0.97f && p.y < 2.03f) ? 3.0f : 0.0f;
  };
  RubbleLayer layer([&](glm::vec2 p) -> std::optional<float> { return ground(p); });
  layer.deposit(glm::vec2(3.0f, 3.0f), 0.05f, 0.3f, kGrey);
  const MeshData mesh = layer.tileMesh(RubbleKey(0, 0));
  int skirts = 0;
  for (size_t i = 0; i < mesh.indices.size(); i += 3) {
    float outsideLowest = 1e9f, highest = -1e9f;
    for (int k = 0; k < 3; ++k) {
      const Vertex &v = mesh.vertices[mesh.indices[i + k]];
      highest = std::max(highest, v.pos.y);
      if (ground(glm::vec2(v.pos.x, v.pos.z)) == 0.0f)
        outsideLowest = std::min(outsideLowest, v.pos.y);
    }
    if (outsideLowest < 1e9f && highest > outsideLowest + 0.5f)
      ++skirts;
  }
  CHECK(skirts == 0);
}

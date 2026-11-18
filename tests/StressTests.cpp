#include "Stress.h"
#include "TestRunner.h"

#include <cmath>

namespace {

constexpr float kWeight = 100.0f * 9.81f; // a 100 kg piece's, N
const glm::vec3 kDown(0.0f, -1.0f, 0.0f);

// A square face of this side around c, facing n (along an axis).
std::vector<glm::vec3> square(glm::vec3 c, glm::vec3 n, float side) {
  glm::vec3 a = std::fabs(n.y) > 0.5f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
  const glm::vec3 b = glm::normalize(glm::cross(n, a));
  a = glm::cross(b, n);
  const float h = 0.5f * side;
  return {c + (-a - b) * h, c + (a - b) * h, c + (a + b) * h, c + (-a + b) * h};
}

StressLink squareLink(int a, int b, glm::vec3 c, glm::vec3 n, float side, bool contact = false) {
  return {a, b, linkFace(square(c, n, side), n), contact};
}

StressPiece cube(glm::vec3 centre) { return {centre, 100.0f}; }

void settle(StressSolver &solver) { solver.solve(20000, 1e-7f); }

// A column of three cubes on the ground.
void buildColumn(StressSolver &solver) {
  solver.build({cube({0, 0.5f, 0}), cube({0, 1.5f, 0}), cube({0, 2.5f, 0})},
               {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true),
                squareLink(0, 1, {0, 1, 0}, {0, 1, 0}, 1.0f),
                squareLink(1, 2, {0, 2, 0}, {0, 1, 0}, 1.0f)});
}

// Five cubes in a row resting on the ground at both ends.
void buildSlab(StressSolver &solver) {
  std::vector<StressPiece> pieces;
  for (int i = 0; i < 5; ++i)
    pieces.push_back(cube({float(i), 0.5f, 0}));
  std::vector<StressLink> links = {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true),
                                   squareLink(4, -1, {4, 0, 0}, kDown, 1.0f, true)};
  for (int i = 0; i < 4; ++i)
    links.push_back(squareLink(i, i + 1, {i + 0.5f, 0.5f, 0}, {1, 0, 0}, 1.0f));
  solver.build(pieces, links);
}

} // namespace

TEST(a_rectangle_face_knows_its_area_moments_and_reach) {
  const LinkFace f = linkFace({{-1, 0, -0.5f}, {1, 0, -0.5f}, {1, 0, 0.5f}, {-1, 0, 0.5f}}, {0, 1, 0});
  CHECK_NEAR(f.area, 2.0, 1e-5);
  CHECK_NEAR(glm::length(f.centre), 0.0, 1e-5);
  CHECK_NEAR(std::fabs(glm::dot(f.axisU, f.normal)), 0.0, 1e-5);
  // About the long side's axis (x) it's thin: b h^3 / 12 = 2 / 12; about z, 8 / 12.
  const bool uAlongX = std::fabs(f.axisU.x) > 0.9f;
  CHECK_NEAR(uAlongX ? f.inertiaU : f.inertiaV, 2.0 / 12.0, 1e-4);
  CHECK_NEAR(uAlongX ? f.inertiaV : f.inertiaU, 8.0 / 12.0, 1e-4);
  CHECK_NEAR(uAlongX ? f.reachU : f.reachV, 1.0, 1e-5);
  CHECK_NEAR(uAlongX ? f.reachV : f.reachU, 0.5, 1e-5);
}

TEST(one_piece_on_the_ground_pushes_with_its_weight) {
  StressSolver solver;
  solver.build({cube({0, 0.5f, 0})}, {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true)});
  settle(solver);
  CHECK(solver.settled());
  const LinkStress s = solver.stress(0);
  CHECK_NEAR(s.normalForce, -kWeight, 1.0);
  CHECK_NEAR(s.bending, 0.0, 1.0);
  CHECK_NEAR(s.push, kWeight, 1.0);
  CHECK(s.pull <= 0.0f);
  CHECK(s.liftedShare == 0.0f);
}

TEST(a_column_carries_three_two_and_one_weights) {
  StressSolver solver;
  buildColumn(solver);
  settle(solver);
  CHECK_NEAR(solver.stress(0).normalForce, -3.0 * kWeight, 3.0);
  CHECK_NEAR(solver.stress(1).normalForce, -2.0 * kWeight, 3.0);
  CHECK_NEAR(solver.stress(2).normalForce, -1.0 * kWeight, 3.0);
}

TEST(a_slab_on_two_supports_puts_half_on_each_and_bends_most_in_the_middle) {
  StressSolver solver;
  buildSlab(solver);
  settle(solver);
  CHECK_NEAR(solver.stress(0).normalForce, -2.5 * kWeight, 0.01 * kWeight);
  CHECK_NEAR(solver.stress(1).normalForce, -2.5 * kWeight, 0.01 * kWeight);
  CHECK(solver.stress(3).bending > 1.5f * solver.stress(2).bending); // 1-2 against 0-1
  CHECK_NEAR(solver.stress(3).bending, solver.stress(4).bending, 0.01 * kWeight);
}

TEST(an_overhang_carries_all_of_it_at_its_root_and_bends_by_half_its_length) {
  // Four cubes out of a wall, the first glued to it.
  StressSolver solver;
  std::vector<StressPiece> pieces;
  for (int i = 0; i < 4; ++i)
    pieces.push_back(cube({float(i), 0.5f, 0}));
  std::vector<StressLink> links = {squareLink(0, -1, {-0.5f, 0.5f, 0}, {-1, 0, 0}, 1.0f)};
  for (int i = 0; i < 3; ++i)
    links.push_back(squareLink(i, i + 1, {i + 0.5f, 0.5f, 0}, {1, 0, 0}, 1.0f));
  solver.build(pieces, links);
  settle(solver);
  const LinkStress root = solver.stress(0);
  CHECK_NEAR(root.shearForce, 4.0 * kWeight, 0.01 * kWeight);
  CHECK_NEAR(root.bending, 8.0 * kWeight, 0.02 * kWeight); // 4 weights, 2 m out
  CHECK(root.pull > 0.0f);                                   // its top edge
  CHECK_NEAR(solver.stress(1).shearForce, 3.0 * kWeight, 0.01 * kWeight);
  CHECK_NEAR(solver.stress(1).bending, 4.5 * kWeight, 0.02 * kWeight);
}

TEST(a_wall_lifts_its_light_edge_but_tips_only_once_its_weight_leaves_its_base) {
  const LinkLimits resting{1e6f, 1e9f, 1e9f, true, 0.6f};
  auto base = [&](float offset) {
    StressSolver solver;
    solver.build({cube({offset, 1.5f, 0})}, {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true)});
    settle(solver);
    return std::pair(solver.stress(0), overload(solver.stress(0), resting));
  };
  const auto [centred, centredOver] = base(0.1f);
  CHECK(centred.liftedShare == 0.0f); // pressed all over
  CHECK(centredOver < 1.0f);
  const auto [leaning, leaningOver] = base(0.35f);
  CHECK(leaning.liftedShare > 0.0f); // its light edge lifts...
  CHECK(leaningOver < 1.0f);         // ...but it stands on the rest
  CHECK(leaning.normalForce < -0.99f * kWeight);
  const auto [tipping, tippingOver] = base(0.65f); // past the edge of its base
  CHECK(tippingOver > 1.0f);
}

TEST(things_side_by_side_on_the_ground_dont_hold_each_other_up) {
  // Two cubes standing next to each other, touching: each stands on its own.
  StressSolver solver;
  solver.build({cube({0, 0.5f, 0}), cube({1, 0.5f, 0})},
               {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true),
                squareLink(1, -1, {1, 0, 0}, kDown, 1.0f, true),
                squareLink(0, 1, {0.5f, 0.5f, 0}, {1, 0, 0}, 1.0f, true)});
  settle(solver);
  CHECK_NEAR(solver.stress(0).normalForce, -kWeight, 0.01 * kWeight);
  CHECK_NEAR(solver.stress(1).normalForce, -kWeight, 0.01 * kWeight);
  CHECK(std::fabs(solver.stress(2).normalForce) < 0.01f * kWeight);
  CHECK(solver.stress(2).shearForce < 0.01f * kWeight);
}

TEST(a_slab_resting_on_a_lopsided_pair_of_supports_lifts_off_the_one_it_pulls) {
  // A slab whose middle is out past one of the two supports it rests on:
  // the far support would have to pull, so it carries nothing, and the slab
  // stands on the near one (its middle still over that support's top).
  StressSolver solver;
  solver.build({{{1.3f, 1.5f, 0}, 100.0f}, cube({0, 0.5f, 0}), cube({1, 0.5f, 0})},
               {squareLink(1, -1, {0, 0, 0}, kDown, 1.0f, true),
                squareLink(2, -1, {1, 0, 0}, kDown, 1.0f, true),
                squareLink(1, 0, {0, 1, 0}, {0, 1, 0}, 1.0f, true),
                squareLink(2, 0, {1, 1, 0}, {0, 1, 0}, 1.0f, true)});
  settle(solver);
  CHECK(std::fabs(solver.stress(2).normalForce) < 0.05f * kWeight);
  CHECK_NEAR(solver.stress(3).normalForce, -kWeight, 0.05 * kWeight);
}

TEST(a_resting_block_slides_only_past_friction) {
  const LinkLimits resting{1e6f, 1e9f, 1e9f, true, 0.5f};
  auto slideOverload = [&](float push) {
    StressSolver solver;
    solver.build({{{0, 0.1f, 0}, 100.0f}}, {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true)});
    solver.addLoad(0, {push * kWeight, 0, 0});
    settle(solver);
    return overload(solver.stress(0), resting);
  };
  CHECK(slideOverload(0.3f) < 1.0f);
  CHECK(slideOverload(0.7f) > 1.0f);
}

TEST(links_side_by_side_share_by_area) {
  StressSolver solver;
  solver.build({cube({0, 0.5f, 0})}, {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true),
                                      squareLink(0, -1, {0, 0, 0}, kDown, std::sqrt(3.0f), true)});
  settle(solver);
  CHECK_NEAR(solver.stress(0).normalForce, -0.25 * kWeight, 0.01 * kWeight);
  CHECK_NEAR(solver.stress(1).normalForce, -0.75 * kWeight, 0.01 * kWeight);
}

TEST(a_pulled_concrete_link_breaks_before_a_pushed_one) {
  const LinkLimits concrete{3e6f, 30e6f, 4e6f, false, 0.6f};
  LinkStress pulled;
  pulled.pull = 1e6f;
  pulled.push = -1e6f;
  LinkStress pushed;
  pushed.pull = -1e6f;
  pushed.push = 1e6f;
  CHECK(overload(pulled, concrete) > 5.0f * overload(pushed, concrete));
  CHECK_NEAR(overload(pulled, concrete), 1.0 / 3.0, 1e-4);
}

TEST(a_warm_start_settles_in_fewer_iterations) {
  StressSolver cold;
  buildSlab(cold);
  settle(cold);
  StressSolver warm;
  buildSlab(warm);
  warm.setStart(cold.shifts(), cold.turns());
  settle(warm);
  CHECK(cold.iterations() > 2);
  CHECK(warm.iterations() < cold.iterations());
  CHECK_NEAR(warm.stress(0).normalForce, -2.5 * kWeight, 0.01 * kWeight);
}

TEST(a_piece_cut_off_from_the_ground_carries_nothing) {
  StressSolver solver;
  solver.build({cube({0, 0.5f, 0}), cube({3, 2.5f, 0}), cube({4, 2.5f, 0})},
               {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true),
                squareLink(1, 2, {3.5f, 2.5f, 0}, {1, 0, 0}, 1.0f)});
  settle(solver);
  CHECK(solver.settled());
  CHECK_NEAR(solver.stress(0).normalForce, -kWeight, 1.0);
  CHECK(solver.stress(1).normalForce == 0.0f);
  CHECK(solver.stress(1).bending == 0.0f);
}

TEST(the_same_structure_gives_the_same_answer) {
  StressSolver a, b;
  buildColumn(a);
  buildColumn(b);
  a.solve(7);
  b.solve(7);
  CHECK(a.stress(0).normalForce < -kWeight); // past the first steps already
  for (int i = 0; i < 3; ++i) {
    CHECK(a.stress(i).normalForce == b.stress(i).normalForce);
    CHECK(a.stress(i).bending == b.stress(i).bending);
  }
}

TEST(an_added_force_changes_the_link_forces_in_proportion) {
  StressSolver plain, loaded;
  plain.build({cube({0, 0.5f, 0})}, {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true)});
  loaded.build({cube({0, 0.5f, 0})}, {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true)});
  loaded.addLoad(0, {0, -kWeight, 0});
  settle(plain);
  settle(loaded);
  CHECK_NEAR(plain.stress(0).normalForce, -kWeight, 1.0);
  CHECK_NEAR(loaded.stress(0).normalForce, 2.0 * plain.stress(0).normalForce, 2.0);
}

TEST(a_thin_wall_tips_across_its_thin_side) {
  // A base 0.3 m thick and 1 m long, the weight 0.1 m past its thin edge:
  // it tips, though it's well inside the base's long reach.
  const LinkLimits resting{1e6f, 1e9f, 1e9f, true, 0.6f};
  const std::vector<glm::vec3> base = {{-0.15f, 0, -0.5f}, {0.15f, 0, -0.5f}, {0.15f, 0, 0.5f},
                                       {-0.15f, 0, 0.5f}};
  StressSolver solver;
  solver.build({cube({0.25f, 1.5f, 0})}, {{0, -1, linkFace(base, kDown), true}});
  solver.solve(2000, 1e-7f);
  CHECK(overload(solver.stress(0), resting) > 1.0f);
}

TEST(only_a_contact_tipping_counts_as_tipping) {
  const LinkLimits resting{1e6f, 1e9f, 1e9f, true, 0.5f};
  LinkStress sliding; // pressed half lifted, pushed sideways past friction, not turning
  sliding.normalForce = -1000.0f;
  sliding.shearForce = 2000.0f;
  sliding.liftedShare = 0.5f;
  sliding.area = 1.0f;
  sliding.reachU = sliding.reachV = 0.5f;
  CHECK(overload(sliding, resting) > 1.0f);
  CHECK(tipping(sliding, resting) == 0.0f);
  LinkStress tipped = sliding; // turning about its u axis past its v reach
  tipped.shearForce = 0.0f;
  tipped.bendU = 500.0f;
  CHECK(tipping(tipped, resting) > 1.0f);
  LinkStress pressed = tipped; // the same turning with all its springs down
  pressed.liftedShare = 0.0f;
  CHECK(tipping(pressed, resting) == 0.0f);
}

TEST(a_piece_held_only_by_a_link_with_no_face_carries_nothing) {
  StressSolver solver;
  solver.build({cube({0, 0.5f, 0}), cube({0, 1.5f, 0})},
               {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true), {0, 1, LinkFace{}, false}});
  CHECK(solver.solve(2000, 1e-6f));
  CHECK_NEAR(solver.stress(0).normalForce, -kWeight, 1.0);
}

TEST(a_load_near_one_end_of_a_slab_moves_its_supports_shares_by_the_lever) {
  // Two more weights on the fourth of five cubes resting at both ends: the
  // near end takes three quarters of them.
  StressSolver solver;
  buildSlab(solver);
  solver.setLiveLoads({{3, {0, -2.0f * kWeight, 0}, {0, 0, 0}}});
  settle(solver);
  CHECK_NEAR(solver.stress(0).normalForce, -3.0 * kWeight, 0.15 * kWeight);
  CHECK_NEAR(solver.stress(1).normalForce, -4.0 * kWeight, 0.15 * kWeight);
}

TEST(no_live_loads_gives_gravitys_answer_back) {
  StressSolver solver;
  buildSlab(solver);
  solver.setLiveLoads({{3, {0, -2.0f * kWeight, 0}, {0, 0, 0}}});
  settle(solver);
  solver.setLiveLoads({});
  settle(solver);
  CHECK_NEAR(solver.stress(0).normalForce, -2.5 * kWeight, 0.01 * kWeight);
  CHECK_NEAR(solver.stress(1).normalForce, -2.5 * kWeight, 0.01 * kWeight);
}

TEST(a_live_load_on_a_piece_outside_the_solve_does_nothing) {
  StressSolver solver;
  solver.build({cube({0, 0.5f, 0}), cube({3, 2.5f, 0})},
               {squareLink(0, -1, {0, 0, 0}, kDown, 1.0f, true)});
  solver.setLiveLoads({{1, {0, -5.0f * kWeight, 0}, {0, 0, 0}}, {7, {0, -kWeight, 0}, {0, 0, 0}}});
  CHECK(solver.solve(2000, 1e-6f));
  CHECK_NEAR(solver.stress(0).normalForce, -kWeight, 1.0);
}

TEST(the_overload_time_adds_up_while_over_and_drops_to_zero_under) {
  CHECK_NEAR(overloadTime(0.0f, 1.5f, 0.05f), 0.05, 1e-6);
  CHECK_NEAR(overloadTime(0.05f, 1.2f, 0.05f), 0.1, 1e-6);
  CHECK(overloadTime(0.1f, 0.9f, 0.05f) == 0.0f);
}

TEST(a_settled_solve_counts_afresh_when_its_live_loads_change) {
  // Otherwise every change adds to the count until the solve looks stuck.
  StressSolver solver;
  buildSlab(solver);
  settle(solver);
  CHECK(solver.iterations() > 0);
  solver.setLiveLoads({{3, {0, -kWeight, 0}, {0, 0, 0}}});
  CHECK(solver.iterations() == 0);
  settle(solver);
  CHECK(solver.settled());
}

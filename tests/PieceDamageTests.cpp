#include "PieceDamage.h"
#include "TestRunner.h"

TEST(a_pieces_toughness_is_its_volume_times_its_materials) {
  CHECK_NEAR(pieceToughness(0.2f, 3.0e5f), 6.0e4, 1.0);
  CHECK_NEAR(pieceToughness(0.0f, 3.0e5f), 0.0, 1e-6);
}

TEST(damage_adds_up_and_breaks_a_piece_at_its_toughness_not_before) {
  PieceDamage d;
  d.toughness = 1000.0f;
  CHECK(!d.add(400.0f));
  CHECK(!d.add(500.0f));
  CHECK(!d.broken());
  CHECK_NEAR(d.energy, 900.0, 1e-4);
  CHECK(d.add(150.0f)); // the hit that broke it
  CHECK(d.broken());
  CHECK(!d.add(100.0f)); // broken already
  CHECK(!d.add(-50.0f)); // nothing taken back
  CHECK_NEAR(d.energy, 1150.0, 1e-3);
}

TEST(a_damaged_pieces_links_hold_less_in_proportion) {
  PieceDamage whole, half, gone;
  whole.toughness = half.toughness = gone.toughness = 100.0f;
  half.energy = 50.0f;
  gone.energy = 120.0f;
  CHECK_NEAR(whole.strengthLeft(), 1.0, 1e-6);
  CHECK_NEAR(half.strengthLeft(), 0.5, 1e-6);
  CHECK_NEAR(gone.strengthLeft(), 0.0, 1e-6);
  // A link is as strong as its weaker end.
  CHECK_NEAR(linkStrengthLeft(whole, half), 0.5, 1e-6);
  // Its pull, push and slide shrink; friction is the surfaces', it stays.
  LinkLimits limits;
  limits.pull = 10.0f;
  limits.push = 20.0f;
  limits.slide = 30.0f;
  limits.friction = 0.6f;
  const LinkLimits weak = weakened(limits, 0.5f);
  CHECK_NEAR(weak.pull, 5.0, 1e-5);
  CHECK_NEAR(weak.push, 10.0, 1e-5);
  CHECK_NEAR(weak.slide, 15.0, 1e-5);
  CHECK_NEAR(weak.friction, 0.6, 1e-6);
  // A piece without toughness (nothing given) never weakens.
  PieceDamage unknown;
  unknown.energy = 10.0f;
  CHECK_NEAR(unknown.strengthLeft(), 1.0, 1e-6);
  CHECK(!unknown.broken());
}

TEST(a_cut_shares_the_damage_out_by_volume) {
  const std::vector<float> shares = shareDamage(900.0f, {0.1f, 0.2f, 0.0f, 0.3f});
  CHECK(shares.size() == 4);
  CHECK_NEAR(shares[0], 150.0, 1e-3);
  CHECK_NEAR(shares[1], 300.0, 1e-3);
  CHECK_NEAR(shares[2], 0.0, 1e-6);
  CHECK_NEAR(shares[3], 450.0, 1e-3);
  // No volume to share by: equal shares.
  const std::vector<float> even = shareDamage(90.0f, {0.0f, 0.0f, 0.0f});
  CHECK_NEAR(even[1], 30.0, 1e-4);
  CHECK(shareDamage(10.0f, {}).empty());
}

#include "Loadout.h"
#include "TestRunner.h"

#include <cmath>

namespace {

void runFor(Loadout &l, float seconds) {
  constexpr float kDt = 1.0f / 100.0f;
  for (int i = 0; i < int(std::lround(seconds / kDt)); ++i)
    l.update(kDt);
}

LoadoutTiming timing() {
  LoadoutTiming t;
  t.pistolDraw = 0.6f;
  t.pistolHolster = 0.7f;
  t.longDraw = 1.2f;
  t.longHolster = 1.3f;
  return t;
}

} // namespace

TEST(drawing_the_pistol_takes_its_draw_time) {
  Loadout l(timing());
  l.give(true, true);
  CHECK(l.inHand() == Holdable::None);
  l.select(Holdable::Pistol);
  runFor(l, 0.3f);
  CHECK(l.inHand() == Holdable::Pistol);
  CHECK_NEAR(l.drawn(), 0.5, 0.02);
  CHECK(!l.ready());
  CHECK(l.busy());
  runFor(l, 0.31f);
  CHECK(l.ready());
  CHECK(!l.busy());
  CHECK_NEAR(l.drawn(), 1.0, 1e-6);
}

TEST(switching_puts_the_first_gun_away_before_drawing_the_next) {
  Loadout l(timing());
  l.give(true, true);
  l.select(Holdable::Pistol);
  runFor(l, 1.0f);
  l.select(Holdable::Long);
  runFor(l, 0.35f);
  CHECK(l.inHand() == Holdable::Pistol); // still going back in
  CHECK_NEAR(l.drawn(), 0.5, 0.02);
  runFor(l, 0.36f + 0.6f); // away, and the rifle half out
  CHECK(l.inHand() == Holdable::Long);
  CHECK_NEAR(l.drawn(), 0.5, 0.03);
  runFor(l, 0.62f);
  CHECK(l.ready());
  CHECK(l.inHand() == Holdable::Long);
}

TEST(holstering_puts_the_gun_away) {
  Loadout l(timing());
  l.give(true, false);
  l.select(Holdable::Pistol);
  runFor(l, 1.0f);
  l.select(Holdable::None);
  runFor(l, 0.71f);
  CHECK(l.inHand() == Holdable::None);
  CHECK_NEAR(l.drawn(), 0.0, 1e-6);
  CHECK(!l.busy());
}

TEST(a_gun_not_carried_cant_be_drawn) {
  Loadout l(timing());
  l.give(false, true);
  l.select(Holdable::Pistol);
  runFor(l, 1.0f);
  CHECK(l.inHand() == Holdable::None);
  CHECK(!l.has(Holdable::Pistol) && l.has(Holdable::Long));
}

TEST(changing_your_mind_mid_draw_puts_it_back_from_where_it_got_to) {
  Loadout l(timing());
  l.give(true, true);
  l.select(Holdable::Pistol);
  runFor(l, 0.3f); // half out
  l.select(Holdable::None);
  runFor(l, 0.36f); // half of the 0.7 s holster
  CHECK(l.inHand() == Holdable::None);
  // Asking for the one already out again changes nothing.
  l.select(Holdable::Pistol);
  runFor(l, 1.0f);
  l.select(Holdable::Pistol);
  runFor(l, 0.1f);
  CHECK(l.ready());
}

TEST(taking_the_guns_away_empties_the_hands_at_once) {
  // Respawning or a scene without a kit: nothing carried, nothing out.
  Loadout l(timing());
  l.give(true, true);
  l.select(Holdable::Long);
  runFor(l, 2.0f);
  l.give(false, false);
  CHECK(l.inHand() == Holdable::None);
  CHECK_NEAR(l.drawn(), 0.0, 1e-6);
}

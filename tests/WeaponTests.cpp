#include "Weapon.h"
#include "TestRunner.h"

#include <cmath>
#include <vector>

namespace {

constexpr float kDt = 1.0f / 60.0f;

// The AKM from its file's numbers (NSD).
WeaponSpec akmSpec() {
  WeaponSpec s;
  s.name = "AKM";
  s.magazine = 30;
  s.rate = 600.0f;
  s.automatic = true;
  s.locksOpen = false;
  s.firstSpread = {0.44f, 0.59f};
  s.followSpread = {1.63f, 1.19f};
  s.reload = 2.5f;
  s.emptyReload = 3.2f;
  s.tracerEvery = 5;
  return s;
}

WeaponSpec p38Spec() {
  WeaponSpec s = akmSpec();
  s.name = "P38";
  s.magazine = 8;
  s.automatic = false;
  s.locksOpen = true;
  s.tracerEvery = 0;
  s.reload = 2.0f;
  s.emptyReload = 2.5f;
  return s;
}

// Holds the trigger (or not) for this long; the shots, and whether it clicked.
WeaponEvents hold(Weapon &w, bool trigger, float seconds, Stance stance = Stance::Supported,
                  float dt = kDt) {
  WeaponEvents all;
  for (int i = 0; i < int(std::lround(seconds / dt)); ++i) {
    WeaponEvents e;
    w.update(dt, trigger, stance, e);
    all.shots.insert(all.shots.end(), e.shots.begin(), e.shots.end());
    all.dry = all.dry || e.dry;
    all.reloaded = all.reloaded || e.reloaded;
  }
  return all;
}

// Spread of a set of numbers around their mean.
float sigma(const std::vector<float> &x) {
  double mean = 0.0, square = 0.0;
  for (float v : x)
    mean += v;
  mean /= static_cast<double>(x.size());
  for (float v : x)
    square += (v - mean) * (v - mean);
  return static_cast<float>(std::sqrt(square / static_cast<double>(x.size() - 1)));
}

} // namespace

TEST(automatic_fire_keeps_its_rate_whatever_the_frame_rate) {
  // 600 a minute: shots at 0, 0.1, ... 0.9 s, the same at 60 or 37 frames.
  for (float dt : {1.0f / 60.0f, 1.0f / 37.0f, 0.25f}) {
    Weapon w(akmSpec());
    w.load(2);
    const WeaponEvents e = hold(w, true, 0.95f, Stance::Supported, dt);
    CHECK(e.shots.size() == 10);
  }
}

TEST(each_shot_says_when_in_its_frame_it_went) {
  // A quarter-second frame holds three shots: at 0, 0.1 and 0.2 s.
  Weapon w(akmSpec());
  w.load(1);
  WeaponEvents e;
  w.update(0.25f, true, Stance::Supported, e);
  CHECK(e.shots.size() == 3);
  if (e.shots.size() == 3) {
    CHECK_NEAR(e.shots[0].time, 0.0, 1e-5);
    CHECK_NEAR(e.shots[1].time, 0.1, 1e-5);
    CHECK_NEAR(e.shots[2].time, 0.2, 1e-5);
  }
}

TEST(semi_fires_once_a_pull) {
  Weapon w(akmSpec());
  w.load(1);
  w.switchMode();
  CHECK(w.mode() == FireMode::Semi);
  CHECK(hold(w, true, 1.0f).shots.size() == 1);
  int shots = 0;
  for (int i = 0; i < 5; ++i) {
    hold(w, false, 0.1f);
    shots += static_cast<int>(hold(w, true, 0.1f).shots.size());
  }
  CHECK(shots == 5);
  w.switchMode();
  CHECK(w.mode() == FireMode::Auto);
  // A pistol has no auto.
  Weapon p(p38Spec());
  CHECK(p.mode() == FireMode::Semi);
  p.switchMode();
  CHECK(p.mode() == FireMode::Semi);
}

TEST(a_loaded_gun_has_a_full_magazine_and_one_more_in_the_chamber) {
  Weapon w(akmSpec());
  w.load(5);
  CHECK(w.chambered());
  CHECK(w.magazineRounds() == 30);
  CHECK(w.pouch().size() == 4);
  CHECK(w.roundsInGun() == 31);
  CHECK(w.rounds() == 151);
  hold(w, true, 0.05f);
  CHECK(w.chambered());
  CHECK(w.magazineRounds() == 29);
}

TEST(the_last_round_leaves_an_empty_chamber_and_the_trigger_clicks) {
  Weapon w(akmSpec());
  w.load(1);
  CHECK(hold(w, true, 4.0f).shots.size() == 31);
  CHECK(!w.chambered());
  CHECK(w.magazineRounds() == 0);
  CHECK(!w.slideLocked()); // the AK's bolt doesn't hold open
  hold(w, false, 0.1f);
  const WeaponEvents click = hold(w, true, 0.1f);
  CHECK(click.shots.empty());
  CHECK(click.dry);
  // The P38's slide stays back after its last round.
  Weapon p(p38Spec());
  p.load(1);
  for (int i = 0; i < 9; ++i) {
    hold(p, true, 0.2f);
    hold(p, false, 0.2f);
  }
  CHECK(p.slideLocked());
  CHECK(p.roundsInGun() == 0);
}

TEST(a_partial_reload_keeps_the_partial_magazine_and_the_chambered_round) {
  Weapon w(akmSpec());
  w.load(3);
  hold(w, true, 0.95f); // 10 rounds: 20 left in the magazine
  hold(w, false, 0.1f);
  CHECK(w.magazineRounds() == 20);
  CHECK(w.reload());
  CHECK(w.reloading());
  CHECK(!w.emptyReload());
  CHECK(hold(w, true, 2.4f).shots.empty()); // nothing while reloading
  const WeaponEvents done = hold(w, false, 0.2f);
  CHECK(done.reloaded);
  CHECK(!w.reloading());
  CHECK(w.chambered());
  CHECK(w.magazineRounds() == 30);
  CHECK(w.pouch().size() == 2);
  int total = 0;
  for (int m : w.pouch())
    total += m;
  CHECK(total == 50); // a full one and the partial 20
}

TEST(the_fullest_magazine_goes_in_next) {
  Weapon w(akmSpec());
  w.load(1);
  w.setPouch({5, 20, 12});
  hold(w, true, 0.35f); // 4 rounds: 26 left
  hold(w, false, 0.1f);
  CHECK(w.reload());
  hold(w, false, 2.6f);
  CHECK(w.magazineRounds() == 20);
}

TEST(an_empty_reload_takes_longer_and_works_the_bolt) {
  // The AK has no hold-open: an empty gun needs the bolt pulled too, so
  // after an empty reload it holds 30, after a partial one 31.
  Weapon w(akmSpec());
  w.load(2);
  hold(w, true, 4.0f);
  hold(w, false, 0.1f);
  CHECK(w.reload());
  CHECK(w.emptyReload());
  CHECK_NEAR(w.reloadLeft(), 3.2, 1e-4);
  hold(w, false, 2.6f);
  CHECK(w.reloading());
  hold(w, false, 0.7f);
  CHECK(!w.reloading());
  CHECK(w.chambered());
  CHECK(w.roundsInGun() == 30);
}

TEST(no_reload_without_a_spare_or_with_a_full_gun) {
  Weapon w(akmSpec());
  w.load(2);
  CHECK(!w.reload()); // full and chambered
  Weapon last(akmSpec());
  last.load(1);
  hold(last, true, 0.35f);
  CHECK(!last.reload()); // nothing to put in
  w.cancelReload();
  CHECK(!w.reloading());
}

TEST(a_tracer_magazine_lights_every_fifth_round) {
  Weapon w(akmSpec());
  w.load(1);
  const WeaponEvents e = hold(w, true, 0.95f);
  int tracers = 0;
  for (size_t i = 0; i < e.shots.size(); ++i)
    if (e.shots[i].tracer) {
      ++tracers;
      CHECK((i + 1) % 5 == 0);
    }
  CHECK(tracers == 2);
}

TEST(single_shots_spread_as_the_manual_says_by_stance) {
  // 10,000 shots a second apart, so the recoil has gone each time: NSD's
  // prone supported sigma, times 1.34 standing and 1.52 moving; from the
  // hip, 15 mrad (est.).
  struct Case {
    Stance stance;
    glm::vec2 expected;
  } cases[] = {{Stance::Supported, {0.44f, 0.59f}},
               {Stance::Standing, {0.44f * 1.34f, 0.59f * 1.34f}},
               {Stance::Moving, {0.44f * 1.52f, 0.59f * 1.52f}},
               {Stance::Hip, {15.0f, 15.0f}}};
  for (const Case &c : cases) {
    WeaponSpec spec = akmSpec();
    spec.magazine = 1000000;
    Weapon w(spec, 7);
    w.load(1);
    w.switchMode();
    std::vector<float> side, up;
    for (int i = 0; i < 10000; ++i) {
      WeaponEvents e;
      w.update(0.5f, true, c.stance, e);
      w.update(0.5f, false, c.stance, e);
      for (const Shot &s : e.shots) {
        side.push_back(s.deviation.x);
        up.push_back(s.deviation.y);
      }
    }
    CHECK(side.size() == 10000);
    CHECK_NEAR(sigma(side), c.expected.x, 0.03 * c.expected.x);
    CHECK_NEAR(sigma(up), c.expected.y, 0.03 * c.expected.y);
  }
}

TEST(recoil_spreads_bursts_as_the_manual_measured) {
  // NSD p.161, prone with a rest, short bursts of 3 to 5: the first round
  // at 0.44 / 0.59 mrad, the ones after at 1.63 / 1.19.
  WeaponSpec spec = akmSpec();
  spec.magazine = 1000000;
  Weapon w(spec, 11);
  w.load(1);
  std::vector<float> firstSide, firstUp, followSide, followUp;
  for (int b = 0; b < 9000; ++b) {
    const int rounds = 3 + b % 3;
    const WeaponEvents e = hold(w, true, 0.1f * static_cast<float>(rounds - 1) + 0.05f);
    hold(w, false, 2.0f);
    for (size_t i = 0; i < e.shots.size(); ++i) {
      (i == 0 ? firstSide : followSide).push_back(e.shots[i].deviation.x);
      (i == 0 ? firstUp : followUp).push_back(e.shots[i].deviation.y);
    }
  }
  CHECK(firstSide.size() == 9000);
  CHECK(followSide.size() == 27000);
  CHECK_NEAR(sigma(firstSide), 0.44, 0.03 * 0.44);
  CHECK_NEAR(sigma(firstUp), 0.59, 0.03 * 0.59);
  CHECK_NEAR(sigma(followSide), 1.63, 0.03 * 1.63);
  CHECK_NEAR(sigma(followUp), 1.19, 0.03 * 1.19);
}

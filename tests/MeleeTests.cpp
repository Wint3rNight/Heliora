#include "Melee.h"
#include "TestRunner.h"

#include <cmath>

namespace {

constexpr float kDt = 1.0f / 120.0f;

MeleeInput held() {
  MeleeInput in;
  in.attack = true;
  return in;
}

// Runs for `seconds` with the same input; whether a strike opened and
// closed on the way.
struct Run {
  bool opened = false, closed = false;
  int started = -1;
};
Run runFor(MeleeController &m, const MeleeInput &in, float seconds, const MeleeTuning &t) {
  Run r;
  for (float time = 0.0f; time < seconds - 1e-6f; time += kDt) {
    const MeleeEvents e = m.update(in, kDt, t);
    r.opened = r.opened || e.strikeOpened;
    r.closed = r.closed || e.strikeClosed;
    if (e.started >= 0)
      r.started = e.started;
  }
  return r;
}

// A tap: down for a few frames, then up.
Run tap(MeleeController &m, const MeleeTuning &t) {
  Run r = runFor(m, held(), 0.05f, t);
  const Run up = runFor(m, MeleeInput{}, kDt, t);
  r.started = up.started >= 0 ? up.started : r.started;
  r.opened = r.opened || up.opened;
  return r;
}

float length(const MeleeAttack &a) { return a.windup + a.active + a.recovery; }

} // namespace

TEST(a_tap_throws_the_first_blow_which_strikes_only_in_its_window) {
  const MeleeTuning t;
  MeleeController m;
  const Run r = tap(m, t);
  CHECK(r.started == 0);
  CHECK(m.state() == MeleeState::Attacking && m.attack() == 0);
  CHECK(!m.striking());
  // Into the active window, out of it, then back to idle.
  Run w = runFor(m, MeleeInput{}, t.light[0].windup, t);
  CHECK(w.opened && m.striking());
  w = runFor(m, MeleeInput{}, t.light[0].active + 0.02f, t);
  CHECK(w.closed && !m.striking());
  runFor(m, MeleeInput{}, t.light[0].recovery, t);
  CHECK(m.state() == MeleeState::Idle);
}

TEST(presses_in_the_chain_window_run_the_combo_one_two_three) {
  const MeleeTuning t;
  MeleeController m;
  tap(m, t);
  runFor(m, MeleeInput{}, t.light[0].chainFrom, t);
  CHECK(tap(m, t).started == 1);
  CHECK(m.attack() == 1);
  runFor(m, MeleeInput{}, t.light[1].chainFrom, t);
  CHECK(tap(m, t).started == 2);
  // After the third the combo is over: another press waits for the end, then
  // starts again from the first.
  runFor(m, MeleeInput{}, t.light[2].chainFrom, t);
  const Run again = tap(m, t);
  CHECK(again.started < 0 && m.attack() == 2);
  const Run after = runFor(m, MeleeInput{}, length(t.light[2]), t);
  CHECK(after.started == 0 || m.state() == MeleeState::Idle);
}

TEST(a_press_during_the_wind_up_is_remembered_and_an_old_one_is_not) {
  const MeleeTuning t;
  MeleeController m;
  tap(m, t);
  tap(m, t); // early, still winding up the first
  CHECK(m.attack() == 0);
  const Run r = runFor(m, MeleeInput{}, t.light[0].chainFrom, t);
  CHECK(r.started == 1 && m.attack() == 1);
  // A press long before the window is forgotten.
  MeleeTuning slow;
  slow.light[0].chainFrom = slow.buffer + 0.3f;
  slow.light[0].recovery = slow.light[0].chainFrom + 0.3f;
  MeleeController n;
  tap(n, slow);
  tap(n, slow);
  runFor(n, MeleeInput{}, slow.light[0].chainFrom + 0.05f, slow);
  CHECK(n.attack() == 0);
}

TEST(holding_the_button_throws_the_heavy_blow_without_waiting_for_the_release) {
  const MeleeTuning t;
  MeleeController m;
  const Run r = runFor(m, held(), t.heavyHold + 2.0f * kDt, t);
  CHECK(r.started == kHeavyAttack);
  CHECK(m.attack() == kHeavyAttack);
  // Letting go doesn't throw a light one after it.
  const Run up = runFor(m, MeleeInput{}, kDt, t);
  CHECK(up.started < 0);
}

TEST(a_dodge_cancels_a_blows_recovery_but_not_its_wind_up_or_strike) {
  const MeleeTuning t;
  MeleeController m;
  tap(m, t);
  MeleeInput dodge;
  dodge.dodge = true;
  m.update(dodge, kDt, t);
  CHECK(m.state() == MeleeState::Attacking); // committed
  runFor(m, MeleeInput{}, t.light[0].windup + t.light[0].active, t);
  const MeleeEvents e = m.update(dodge, kDt, t);
  CHECK(e.cancelled && m.state() == MeleeState::Idle);
}

TEST(holding_block_blocks_counts_its_parry_time_and_stops_attacks) {
  const MeleeTuning t;
  MeleeController m;
  MeleeInput block;
  block.block = true;
  runFor(m, block, 0.1f, t);
  CHECK(m.state() == MeleeState::Blocking);
  CHECK(m.parrying(t));
  runFor(m, block, t.parryWindow, t);
  CHECK(!m.parrying(t) && m.state() == MeleeState::Blocking);
  block.attack = true;
  const Run r = runFor(m, block, 0.5f, t);
  CHECK(r.started < 0);
  runFor(m, MeleeInput{}, kDt, t);
  CHECK(m.state() == MeleeState::Idle);
}

TEST(no_blows_while_the_powers_hold_something_and_none_while_staggered) {
  const MeleeTuning t;
  MeleeController m;
  MeleeInput busy = held();
  busy.busy = true;
  CHECK(runFor(m, busy, 0.5f, t).started < 0);
  runFor(m, MeleeInput{}, kDt, t);
  m.stagger(0.4f);
  CHECK(m.state() == MeleeState::Staggered);
  CHECK(runFor(m, held(), 0.3f, t).started < 0);
  runFor(m, MeleeInput{}, 0.2f, t);
  CHECK(m.state() == MeleeState::Idle);
}

TEST(a_blows_timing_comes_from_its_clips_strike_frames) {
  // A clip trimmed from 0.25 s to 0.85 s that can hit from 0.52 to 0.67 s:
  // wound up 0.27 s, striking 0.15 s, recovering 0.18 s; the next blow can
  // be pressed in from the strike on.
  MeleeTuning t;
  timeBlowFromClip(t.light[0], 0.25f, 0.85f, 0.52f, 0.67f);
  CHECK_NEAR(t.light[0].windup, 0.27, 1e-5);
  CHECK_NEAR(t.light[0].active, 0.15, 1e-5);
  CHECK_NEAR(t.light[0].recovery, 0.18, 1e-5);
  CHECK_NEAR(t.light[0].chainFrom, 0.27, 1e-5);
  // Nonsense (no window, or one outside the trim) leaves it as it was.
  MeleeTuning u;
  const MeleeAttack before = u.light[1];
  timeBlowFromClip(u.light[1], 0.2f, 0.9f, 1.5f, 1.6f);
  CHECK(u.light[1] == before);
}

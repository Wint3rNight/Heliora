#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

// Bare-handed melee (step F3): the blows and the controller that throws
// them. A tap of the attack button throws the next light blow of a three-hit
// combo; holding it past heavyHold throws the heavy one. A press during a
// blow is remembered for a moment and chains into the next blow once its
// window opens; a dodge can cancel a blow's recovery, never its wind-up or
// its strike. Block is held. No physics or animation here, so it's tested
// like the rest of heliora_core.

// What a blow strikes with.
enum class Limb : uint8_t { RightHand, LeftHand, RightFoot, LeftFoot };

// One blow, in seconds from its start: the wind-up, the window it can hit
// in, the recovery after; from chainFrom a press goes on to the next blow.
// What it does when it lands: points of damage, a push (m/s) and the focus
// it earns is the powers' (focusHit). All est., to tune by playing.
struct MeleeAttack {
  Limb limb = Limb::RightHand;
  float windup = 0.12f;
  float active = 0.1f;
  float recovery = 0.3f;
  float chainFrom = 0.18f;
  float damage = 12.0f;
  float push = 1.5f;
  float energy = 100.0f; // J it leaves in a breakable piece it lands on (est.)
  bool operator==(const MeleeAttack &) const = default;
};

struct MeleeTuning {
  // The combo: a jab, a cross, a hook (est.).
  std::array<MeleeAttack, 3> light = {
      MeleeAttack{Limb::LeftHand, 0.12f, 0.10f, 0.30f, 0.18f, 12.0f, 1.5f, 100.0f},
      MeleeAttack{Limb::RightHand, 0.14f, 0.10f, 0.32f, 0.20f, 15.0f, 2.0f, 150.0f},
      MeleeAttack{Limb::LeftHand, 0.18f, 0.12f, 0.45f, 0.30f, 22.0f, 4.0f, 200.0f}};
  // The heavy blow: a kick (est.).
  MeleeAttack heavy{Limb::RightFoot, 0.35f, 0.15f, 0.6f, 1.0f, 40.0f, 7.0f, 500.0f};
  float heavyHold = 0.2f;   // s the button is held for the heavy blow
  float buffer = 0.3f;      // s a press is remembered
  float parryWindow = 0.2f; // s after the block goes up: a blow then is parried
  float blockReduce = 0.8f; // the share of a blow's damage a block takes off
  bool operator==(const MeleeTuning &) const = default;
};

// A blow timed from its clip (seconds into the file): trimmed from start to
// end, its fist or foot able to hit from hitFrom to hitTo. The wind-up is up
// to hitFrom, the strike to hitTo, the recovery to the end; the next blow
// can be pressed in from the strike on. A window that isn't inside the
// trim leaves the blow as it was.
void timeBlowFromClip(MeleeAttack &blow, float start, float end, float hitFrom, float hitTo);

// Which blow: 0 to 2 the combo's, this the heavy one.
inline constexpr int kHeavyAttack = 3;

enum class MeleeState : uint8_t { Idle, Attacking, Blocking, Staggered };
const char *meleeStateName(MeleeState state); // "Idle", "Attacking" and so on

struct MeleeInput {
  bool attack = false; // the attack button, held
  bool block = false;  // held
  bool dodge = false;  // went down this frame
  bool busy = false;   // the powers hold something: the attack button throws it instead
};

struct MeleeEvents {
  int started = -1;          // a blow started: which
  bool strikeOpened = false; // its window to hit opened this frame
  bool strikeClosed = false; // and closed
  bool cancelled = false;    // a dodge cut its recovery short
};

class MeleeController {
public:
  MeleeEvents update(const MeleeInput &input, float dt, const MeleeTuning &t);
  MeleeState state() const { return state_; }
  int attack() const { return attack_; }          // the blow under way, or -1
  float attackTime() const { return time_; }      // s into it
  bool striking() const { return striking_; }     // in its window to hit
  bool parrying(const MeleeTuning &t) const {     // the block just went up
    return state_ == MeleeState::Blocking && time_ < t.parryWindow;
  }
  // Hit hard while not blocking: no moves for this long.
  void stagger(float seconds);
  const MeleeAttack &blow(int which, const MeleeTuning &t) const {
    return which == kHeavyAttack ? t.heavy : t.light[static_cast<std::size_t>(which < 0 ? 0 : which % 3)];
  }

private:
  void start(int which, MeleeEvents &e);

  MeleeState state_ = MeleeState::Idle;
  int attack_ = -1;
  float time_ = 0.0f;
  bool striking_ = false;
  float downFor = -1.0f;   // s the attack button has been held, -1 while up
  bool heavyThrown = false; // this press already threw the heavy blow
  int request = -1;        // a blow asked for, waiting: 0 light, kHeavyAttack heavy
  float requestAge = 0.0f;
  float staggered = 0.0f;
};

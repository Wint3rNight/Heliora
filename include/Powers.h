#pragma once

#include "Ballistics.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

// The powers (step 1.4a): grabbing loose things and pieces of what breaks,
// holding them out along the aim, throwing them, tearing pieces out of what
// they're part of, and a burst that blasts what's in front away. The maths
// and the states, no Vulkan and no physics, so they're tested like the rest
// of heliora_core.

struct PowerTuning {
  // What lifts.
  float liftCap = 2500.0f; // kg; anything heavier only shudders
  // Holding.
  float holdDistance = 4.0f; // m past the character, along the aim
  float holdNearest = 2.5f;  // m, as close as the wheel brings it
  float holdFarthest = 8.0f; // m, as far as the wheel pushes it
  float holdRaise = 0.5f;    // m up
  float liftSeconds = 0.4f;  // from where it was to the hold point
  float holdLag = 0.12f;     // s: it closes its distance in about this
  float holdSpeed = 20.0f;   // m/s at most
  float holdResponse = 30.0f; // 1/s: how quickly its velocity turns to what it wants
  float holdHeavy = 300.0f;  // kg: heavier is slower by (this / mass)^0.5
  float tumble = 0.4f;       // rad/s, its slow turn while held
  float lostDistance = 6.0f; // m from the hold point, for lostSeconds: dropped
  float lostSeconds = 1.0f;
  // Throwing.
  float throwSpeed = 30.0f;  // m/s
  // kg: heavier is slower by (this / mass)^0.5, so a 100 kg stone piece goes
  // half as fast and hits like one heavy ball
  float throwHeavy = 25.0f;
  float throwReach = 60.0f;  // m out, when the crosshair hits nothing
  float throwDelay = 0.1f;   // s from the click to letting go
  float throwRecover = 0.45f; // s from the click until another power
  float throwTumble = 3.0f;  // rad/s at most, given as it goes
  // Tearing.
  float tearPace = 0.1f;     // s per square root of its links' strength (N s)
  float tearShortest = 0.4f; // s
  float tearLongest = 4.0f;  // s
  float tearShake = 0.02f;   // m, how far it shakes once nearly out
  // The burst.
  float burstWidth = 70.0f;  // degrees across its cone
  float burstReach = 8.0f;   // m
  float burstSpeed = 16.0f;  // m/s given at the hand
  float burstHeavy = 200.0f; // kg: heavier gets less by (this / mass)^0.5
  float burstLift = 0.3f;    // how much what it throws tilts up
  float burstImpulse = 5000.0f; // N s on what breaks, at the hand (a heavy ball is about 1,500)
  float burstDelay = 0.1f;   // s from the click to the blast
  float burstRecover = 0.5f; // s from the click until another power
  float burstCooldown = 3.0f; // s between bursts (est.; 0.8 before focus, step F2)
  // Picking.
  float pickAngle = 4.0f;    // degrees off the crosshair
  float pickReach = 30.0f;   // m
  float pickDwell = 0.15f;   // s on one target, right mouse held, to grab it
  // s after a throw or a burst before the crosshair resting on something
  // grabs it (what they sent flying passes the crosshair and lands where it
  // points, broken into new things); a press still does.
  float regrabSeconds = 2.0f;
  // Revealing.
  float revealSpeed = 25.0f; // m/s, the pulse going out
  float revealReach = 30.0f; // m
  float revealSeconds = 6.0f; // the glow fading out after the pulse passes
  float revealCooldown = 4.0f; // s between reveals (est.)
  // Focus (step F2): what the powers run on. Melee hits and kills fill it
  // (F3); each power costs some, holding and tearing drain it; under
  // focusRegenUpTo it comes back by itself, so the powers never lock out
  // for good. All est., to tune by playing.
  float focusMax = 100.0f;
  float focusRegen = 2.0f;        // a second, by itself...
  float focusRegenUpTo = 40.0f;   // ...up to this
  float focusLift = 8.0f;         // to take hold of something
  float focusLiftPerTonne = 12.0f; // and this much more for each 1000 kg of it
  float focusHold = 3.0f;         // a second while it's held out (lifted, held, strained at)
  float focusTear = 8.0f;         // a second while tearing
  float focusBurst = 35.0f;
  float focusReveal = 10.0f;
  float focusHit = 8.0f;          // a melee hit earns this (F3)
  float focusKill = 25.0f;        // a kill this
  bool operator==(const PowerTuning &) const = default;
};

// How much slower something heavy goes: 1 up to `from` kg, then
// (from / mass)^0.5.
float heavySlowdown(float mass, float from);
// Whether the power can lift a body of this mass (0 is a static body).
bool liftable(float mass, const PowerTuning &t);

// Where a held thing is pulled to: from the camera along the aim (unit),
// past the character (characterDistance from the camera) by distance, and
// raised.
glm::vec3 holdPoint(glm::vec3 camera, glm::vec3 aim, float characterDistance, float distance,
                    float raise);
// Where it's pulled to `seconds` after the grab: from where it was to the
// hold point over liftSeconds, eased in and out.
glm::vec3 liftPoint(glm::vec3 from, glm::vec3 hold, float seconds, const PowerTuning &t);
// What a held thing swept from `start` by `move` runs into: how far along it
// gets (0 to 1) and the way the surface faces (unit).
struct HoldContact {
  float fraction = 1.0f;
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
};
using HoldSweep = std::function<std::optional<HoldContact>(glm::vec3 start, glm::vec3 move)>;
// Where a held thing at `from` is pulled to, with what doesn't move in its
// way: as far toward `target` as it goes, then along what it meets for the
// rest, swept again each time (a corner has two walls). Without it a hold
// point behind a wall rams what's held into the wall every step.
glm::vec3 slideHoldPoint(glm::vec3 from, glm::vec3 target, const HoldSweep &sweep);
// The velocity to give a held thing before a physics step of dt: toward
// target, no faster than its cap, heavier things slower, with the step's
// gravity taken back off so it floats. Its spin eased toward a slow tumble
// about axis (unit).
glm::vec3 holdVelocity(glm::vec3 position, glm::vec3 velocity, glm::vec3 target, float mass,
                       glm::vec3 gravity, float dt, const PowerTuning &t);
glm::vec3 holdSpin(glm::vec3 spin, glm::vec3 axis, float dt, const PowerTuning &t);

// How fast something of this mass is thrown.
float throwSpeed(float mass, const PowerTuning &t);
// The launch velocity that lands on `to`: throwVelocity in Ballistics.h.

// How hard the burst from origin along aim (unit) reaches a point, 0 to 1:
// 1 at the hand, 0 at its reach, outside its cone and behind.
float burstFalloff(glm::vec3 origin, glm::vec3 aim, glm::vec3 at, const PowerTuning &t);
// The velocity it gives a body of this mass whose middle is at `at`: away
// from the hand, tilted up.
glm::vec3 burstKick(glm::vec3 origin, glm::vec3 aim, glm::vec3 at, float mass,
                    const PowerTuning &t);

// How long tearing out a piece takes, from its links' strengths added up
// (N s).
float tearSeconds(float strength, const PowerTuning &t);
// When each of a piece's links gives way, in their order: weakest first,
// each once the tear has gone through its share of the whole strength, the
// strongest last at tearSeconds.
std::vector<float> tearTimes(const std::vector<float> &strengths, const PowerTuning &t);

// What the crosshair might be on: a body's middle and how far it reaches.
struct PickCandidate {
  glm::vec3 centre{0.0f};
  float radius = 0.0f;
};
// The candidates within pickAngle of the line from origin along aim (unit)
// and pickReach of origin, nearest the line first (by the angle to their
// edge, so big things are easier to hit); none behind.
std::vector<int> pickOrder(const std::vector<PickCandidate> &candidates, glm::vec3 origin,
                           glm::vec3 aim, const PowerTuning &t);

// How brightly something this far away glows sinceReveal seconds after the
// reveal, 0 to 1: nothing until the pulse reaches it, then fading out over
// revealSeconds.
float revealGlow(float distance, float sinceReveal, const PowerTuning &t);

// The focus meter (step F2): spent all or nothing, drained while
// something's held out or torn at, filled by melee, and coming back by
// itself up to focusRegenUpTo. It starts full.
class Focus {
public:
  explicit Focus(const PowerTuning &t = {}) : amount(t.focusMax) {}
  float value() const { return amount; }
  bool spend(float cost);           // false, and nothing spent, if there isn't enough
  bool drain(float rate, float dt); // false once it's run dry
  void gain(float earned, const PowerTuning &t);
  void update(float dt, const PowerTuning &t); // what comes back by itself

private:
  float amount;
};
// The focus it takes to grab something of this mass (kg).
float grabCost(float mass, const PowerTuning &t);

enum class PowerState : uint8_t { Idle, Lifting, Holding, Tearing, Straining, Throwing, Bursting };
const char *powerStateName(PowerState state); // "Idle", "Lifting" and so on

// What grabbing the target under the crosshair does: lifts it (a loose body
// under the cap), tears it out first (a piece still part of something), or
// strains against it (too heavy).
enum class GrabKind : uint8_t { Lift, Tear, Strain };

// The player and the world, this frame.
struct PowerInput {
  bool aim = false;  // right mouse held
  bool cast = false;  // left mouse went down this frame
  bool burst = false; // the burst's key (E) went down this frame
  bool reveal = false; // the reveal's key (Q) went down this frame
  int target = -1;   // what's under the crosshair and in sight, or -1
  GrabKind targetKind = GrabKind::Lift;
  float targetMass = 0.0f; // kg, what grabbing it costs goes by
  bool torn = false; // the tear is done: the piece is loose and light enough
  bool lost = false; // the held thing is gone, or stuck away from the hold point
};

// What happened this frame, for the world to do.
struct PowerEvents {
  int grabbed = -1;     // took hold of this target
  bool dropped = false; // let go of it where it is
  bool thrown = false;  // let go of it flying
  bool burst = false;   // the blast goes off
  bool revealed = false; // the reveal's pulse goes out
  // A power was asked for and refused: not enough focus, or cooling down.
  bool refused = false;
};

// The powers' state machine: right mouse grabs what's under the crosshair
// (as it goes down, or for something loose once the crosshair rests on it;
// a structure or something too heavy takes a press), lets it go when
// released; left mouse throws what's held; the burst's key blasts, aiming
// or not, letting go of whatever's held. Each costs focus, and what can't
// be paid for (or is cooling down) is refused.
class PowerController {
public:
  PowerEvents update(const PowerInput &input, float dt, const PowerTuning &t);
  PowerState state() const { return state_; }
  float stateTime() const { return stateTime_; } // s in this state
  int held() const { return held_; }             // the grabbed target, or -1
  // Something is held out: lifting, holding, tearing, straining, or a throw
  // that hasn't let go yet.
  bool holding() const { return held_ >= 0; }
  Focus &focus() { return focus_; }
  const Focus &focus() const { return focus_; }
  float burstCooling() const { return cooldown; }   // s until the next burst
  float revealCooling() const { return revealCool; } // s until the next reveal

private:
  void enter(PowerState state);

  PowerState state_ = PowerState::Idle;
  float stateTime_ = 0.0f;
  int held_ = -1;
  bool wasAiming = false;
  int dwellTarget = -1;
  float dwell = 0.0f;
  float cooldown = 0.0f; // s until the next burst
  float revealCool = 0.0f;
  Focus focus_;
  bool fired = false;    // this throw or burst has gone off
  // s since the last throw let go or burst went off: what they sent flying
  // passes the crosshair and lands where it points.
  float sinceCast = 1e9f;
};

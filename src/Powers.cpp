#include "Powers.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

constexpr float kSpinEase = 3.0f;    // 1/s, a held thing's spin toward its tumble
constexpr float kRevealRise = 0.15f; // s for a revealed thing's glow to come up
// Timings compare with this much slack, so a delay of 0.1 s isn't missed by
// six steps of 1/60 s adding up to a hair under it.
constexpr float kSlack = 1e-4f;
// Sweeps for a held thing's way: the first, then along what it meets, then
// along the next (a corner).
constexpr int kSlides = 3;

} // namespace

float heavySlowdown(float mass, float from) {
  return mass <= from ? 1.0f : std::sqrt(from / mass);
}

bool liftable(float mass, const PowerTuning &t) { return mass > 0.0f && mass <= t.liftCap; }

glm::vec3 holdPoint(glm::vec3 camera, glm::vec3 aim, float characterDistance, float distance,
                    float raise) {
  return camera + aim * (characterDistance + distance) + glm::vec3(0.0f, raise, 0.0f);
}

glm::vec3 liftPoint(glm::vec3 from, glm::vec3 hold, float seconds, const PowerTuning &t) {
  const float s = t.liftSeconds > 0.0f ? std::clamp(seconds / t.liftSeconds, 0.0f, 1.0f) : 1.0f;
  return glm::mix(from, hold, s * s * (3.0f - 2.0f * s));
}

glm::vec3 slideHoldPoint(glm::vec3 from, glm::vec3 target, const HoldSweep &sweep) {
  glm::vec3 at = from, move = target - from;
  for (int i = 0; i < kSlides; ++i) {
    if (glm::length(move) < 1e-5f)
      return at;
    const std::optional<HoldContact> hit = sweep(at, move);
    if (!hit)
      return at + move;
    at += move * hit->fraction;
    move *= 1.0f - hit->fraction;
    if (const float into = glm::dot(move, hit->normal); into < 0.0f)
      move -= hit->normal * into;
  }
  return at; // still meeting things: no further
}

// The velocity it wants closes its distance in holdLag, and the velocity it
// has turns toward that at holdResponse: a spring and a damper, nearly
// critically damped (the damping ratio is 0.5 sqrt(response * lag), 0.95 by
// default). A heavy thing gets both slowed by the same factor, so it's
// slower but just as steady.
glm::vec3 holdVelocity(glm::vec3 position, glm::vec3 velocity, glm::vec3 target, float mass,
                       glm::vec3 gravity, float dt, const PowerTuning &t) {
  const float slow = heavySlowdown(mass, t.holdHeavy);
  glm::vec3 want = (target - position) * (slow / t.holdLag);
  const float cap = t.holdSpeed * slow;
  const float speed = glm::length(want);
  if (speed > cap)
    want *= cap / speed;
  const float blend = 1.0f - std::exp(-t.holdResponse * slow * dt);
  // The step adds gravity's pull after this, which this takes back off.
  return velocity + (want - velocity) * blend - gravity * dt;
}

glm::vec3 holdSpin(glm::vec3 spin, glm::vec3 axis, float dt, const PowerTuning &t) {
  return glm::mix(spin, axis * t.tumble, 1.0f - std::exp(-kSpinEase * dt));
}

float throwSpeed(float mass, const PowerTuning &t) {
  return t.throwSpeed * heavySlowdown(mass, t.throwHeavy);
}

// The launch angle above the level for a speed v to reach a point x across
// and y up under gravity g: tan = (v^2 - sqrt(v^4 - g (g x^2 + 2 y v^2))) / (g x),
// the low of the two arcs; with nothing under the root it can't get there.
glm::vec3 throwVelocity(glm::vec3 from, glm::vec3 to, float speed, glm::vec3 gravity) {
  const float g = glm::length(gravity);
  const glm::vec3 up = g > 0.0f ? -gravity / g : glm::vec3(0.0f, 1.0f, 0.0f);
  const glm::vec3 d = to - from;
  const float y = glm::dot(d, up);
  const glm::vec3 across = d - up * y;
  const float x = glm::length(across);
  if (x < 1e-4f || g <= 0.0f) {
    const float length = glm::length(d);
    return (length > 1e-6f ? d / length : up) * speed;
  }
  const float v2 = speed * speed;
  const float root = v2 * v2 - g * (g * x * x + 2.0f * y * v2);
  const float angle = root < 0.0f ? 0.25f * glm::pi<float>()
                                  : std::atan((v2 - std::sqrt(root)) / (g * x));
  return (across / x * std::cos(angle) + up * std::sin(angle)) * speed;
}

float burstFalloff(glm::vec3 origin, glm::vec3 aim, glm::vec3 at, const PowerTuning &t) {
  const glm::vec3 d = at - origin;
  const float distance = glm::length(d);
  if (distance >= t.burstReach)
    return 0.0f;
  if (distance < 1e-4f)
    return 1.0f;
  if (glm::dot(d / distance, aim) < std::cos(glm::radians(0.5f * t.burstWidth)))
    return 0.0f;
  const float r = distance / t.burstReach;
  return 1.0f - r * r;
}

glm::vec3 burstKick(glm::vec3 origin, glm::vec3 aim, glm::vec3 at, float mass,
                    const PowerTuning &t) {
  const float falloff = burstFalloff(origin, aim, at, t);
  if (falloff <= 0.0f)
    return glm::vec3(0.0f);
  const glm::vec3 d = at - origin;
  const float distance = glm::length(d);
  const glm::vec3 away = distance < 1e-4f ? aim : d / distance;
  const glm::vec3 way = glm::normalize(away + glm::vec3(0.0f, t.burstLift, 0.0f));
  return way * (t.burstSpeed * falloff * heavySlowdown(mass, t.burstHeavy));
}

float tearSeconds(float strength, const PowerTuning &t) {
  return std::clamp(t.tearPace * std::sqrt(std::max(strength, 0.0f)), t.tearShortest,
                    t.tearLongest);
}

std::vector<float> tearTimes(const std::vector<float> &strengths, const PowerTuning &t) {
  std::vector<float> times(strengths.size(), 0.0f);
  const float total = std::accumulate(strengths.begin(), strengths.end(), 0.0f);
  if (total <= 0.0f)
    return times;
  std::vector<int> order(strengths.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return strengths[a] < strengths[b]; });
  const float seconds = tearSeconds(total, t);
  float sofar = 0.0f;
  for (int i : order) {
    sofar += std::max(strengths[i], 0.0f);
    times[i] = seconds * sofar / total;
  }
  return times;
}

std::vector<int> pickOrder(const std::vector<PickCandidate> &candidates, glm::vec3 origin,
                           glm::vec3 aim, const PowerTuning &t) {
  struct Ranked {
    float off;      // radians from the line to its edge
    float distance; // m, nearer first among the same angle
    int index;
  };
  std::vector<Ranked> ranked;
  const float widest = glm::radians(t.pickAngle);
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    const PickCandidate &c = candidates[i];
    const glm::vec3 d = c.centre - origin;
    const float distance = glm::length(d);
    if (distance - c.radius > t.pickReach)
      continue;
    float off = 0.0f;
    if (distance > 1e-4f) {
      const float along = glm::dot(d, aim);
      if (along <= 0.0f)
        continue; // behind, even with the origin inside its reach
      const float angle = std::acos(std::clamp(along / distance, -1.0f, 1.0f));
      // From inside its reach there's no edge to aim at: the angle to its middle.
      off = distance > c.radius ? std::max(0.0f, angle - std::asin(c.radius / distance)) : angle;
    }
    if (off <= widest)
      ranked.push_back({off, distance, i});
  }
  std::sort(ranked.begin(), ranked.end(), [](const Ranked &a, const Ranked &b) {
    return a.off != b.off ? a.off < b.off : a.distance < b.distance;
  });
  std::vector<int> order;
  order.reserve(ranked.size());
  for (const Ranked &r : ranked)
    order.push_back(r.index);
  return order;
}

float revealGlow(float distance, float sinceReveal, const PowerTuning &t) {
  if (distance > t.revealReach)
    return 0.0f;
  const float since = sinceReveal - distance / t.revealSpeed;
  if (since <= 0.0f || since >= t.revealSeconds - kSlack)
    return 0.0f;
  return std::min(1.0f, since / kRevealRise) * (1.0f - since / t.revealSeconds);
}

const char *powerStateName(PowerState state) {
  switch (state) {
  case PowerState::Idle: return "Idle";
  case PowerState::Lifting: return "Lifting";
  case PowerState::Holding: return "Holding";
  case PowerState::Tearing: return "Tearing";
  case PowerState::Straining: return "Straining";
  case PowerState::Throwing: return "Throwing";
  case PowerState::Bursting: return "Bursting";
  }
  return "";
}

void PowerController::enter(PowerState state) {
  state_ = state;
  stateTime_ = 0.0f;
  fired = false;
  dwellTarget = -1;
  dwell = 0.0f;
}

PowerEvents PowerController::update(const PowerInput &in, float dt, const PowerTuning &t) {
  PowerEvents e;
  stateTime_ += dt;
  cooldown = std::max(0.0f, cooldown - dt);
  sinceCast += dt;
  const bool pressed = in.aim && !wasAiming;
  wasAiming = in.aim;
  const auto drop = [&] {
    e.dropped = true;
    held_ = -1;
    enter(PowerState::Idle);
  };
  // The burst's key goes off from anything but a throw or a burst under way,
  // aiming or not, letting go of whatever's held first.
  if (in.burst && cooldown <= 0.0f && state_ != PowerState::Throwing &&
      state_ != PowerState::Bursting) {
    if (held_ >= 0) {
      e.dropped = true;
      held_ = -1;
    }
    cooldown = t.burstCooldown;
    enter(PowerState::Bursting);
    return e;
  }
  switch (state_) {
  case PowerState::Idle: {
    if (!in.aim || in.target < 0) {
      dwellTarget = -1;
      dwell = 0.0f;
      break;
    }
    if (in.target != dwellTarget) {
      dwellTarget = in.target;
      dwell = 0.0f;
    }
    dwell += dt;
    // Resting on it grabs only something loose, and not while what the last
    // throw or burst sent flying is still about; tearing into a structure or
    // straining at something too heavy takes a press.
    const bool rests = in.targetKind == GrabKind::Lift && sinceCast + kSlack >= t.regrabSeconds &&
                       dwell + kSlack >= t.pickDwell;
    if (pressed || rests) {
      held_ = in.target;
      e.grabbed = in.target;
      enter(in.targetKind == GrabKind::Lift   ? PowerState::Lifting
            : in.targetKind == GrabKind::Tear ? PowerState::Tearing
                                              : PowerState::Straining);
    }
    break;
  }
  case PowerState::Lifting:
  case PowerState::Holding:
    if (!in.aim || in.lost)
      drop();
    else if (in.cast)
      enter(PowerState::Throwing);
    else if (state_ == PowerState::Lifting && stateTime_ + kSlack >= t.liftSeconds)
      enter(PowerState::Holding);
    break;
  case PowerState::Tearing:
  case PowerState::Straining:
    if (!in.aim || in.lost) {
      drop();
    } else if (state_ == PowerState::Tearing && in.torn) {
      enter(PowerState::Lifting);
    }
    break;
  case PowerState::Throwing:
    // Once it's clicked, it goes, right mouse held or not.
    if (!fired && in.lost) {
      fired = true;
      held_ = -1;
    }
    if (!fired && stateTime_ + kSlack >= t.throwDelay) {
      fired = true;
      e.thrown = true;
      sinceCast = 0.0f;
      held_ = -1;
    }
    if (stateTime_ + kSlack >= t.throwRecover)
      enter(PowerState::Idle);
    break;
  case PowerState::Bursting:
    if (!fired && stateTime_ + kSlack >= t.burstDelay) {
      fired = true;
      e.burst = true;
      sinceCast = 0.0f;
    }
    if (stateTime_ + kSlack >= t.burstRecover)
      enter(PowerState::Idle);
    break;
  }
  return e;
}

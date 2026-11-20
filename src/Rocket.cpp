#include "Rocket.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kAirDensity = 1.225f; // kg/m^3, sea level
constexpr float kStep = 1e-3f;        // s, the most a step goes

// One step of h: the motor on between motorFrom and motorTo, the nose along
// the air the rocket moves through, thrust along it, drag against that air.
void stepOnce(RocketFlight &f, const RocketSpec &spec, const RocketFit &fit, float h, glm::vec3 wind,
              glm::vec3 gravity) {
  if (!f.burntOut && !f.motorOn && f.flown >= spec.motorFrom)
    f.motorOn = true;
  if (f.motorOn && f.flown >= spec.motorTo) {
    f.motorOn = false;
    f.burntOut = true;
  }
  const glm::vec3 air = f.velocity - wind;
  const float speed = glm::length(air);
  if (speed > 1e-3f)
    f.axis = air / speed;
  const float area = 0.25f * 3.14159265f * spec.diameter * spec.diameter;
  const float mass = std::max(spec.mass, 1e-3f);
  const float drag = 0.5f * kAirDensity * fit.dragCoefficient * area * speed / mass;
  const glm::vec3 push = f.motorOn ? f.axis * (fit.thrust / mass) : glm::vec3(0.0f);
  f.velocity += (push - drag * air + gravity) * h;
  const glm::vec3 move = f.velocity * h;
  f.position += move;
  f.flown += glm::length(move);
  f.time += h;
}

// Level, in still air, without gravity: the speed at burnout and how far it
// had gone at its self-destruct.
struct FitFlight {
  float topSpeed = 0.0f, flown = 0.0f;
};
FitFlight flyLevel(const RocketSpec &spec, const RocketFit &fit) {
  RocketFlight f = launchRocket(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), spec);
  FitFlight out;
  while (!rocketSpent(f, spec)) {
    stepRocket(f, spec, fit, 0.01f, glm::vec3(0.0f), glm::vec3(0.0f));
    out.topSpeed = std::max(out.topSpeed, glm::length(f.velocity));
  }
  out.flown = f.flown;
  return out;
}

} // namespace

RocketFit fitRocket(const RocketSpec &spec) {
  // Each round, the thrust scaled toward the top speed and the drag toward
  // the reach (more drag, less reach); both settle in a few dozen rounds.
  RocketFit fit{200.0f, 0.5f};
  for (int round = 0; round < 80; ++round) {
    const FitFlight f = flyLevel(spec, fit);
    if (!(f.topSpeed > 0.0f) || !(f.flown > 0.0f))
      break;
    const float speedOff = spec.topSpeed / f.topSpeed, reachOff = f.flown / std::max(spec.reach, 1.0f);
    fit.thrust *= std::pow(speedOff, 1.5f);
    fit.dragCoefficient *= std::pow(reachOff, 3.0f);
    if (std::abs(speedOff - 1.0f) < 1e-5f && std::abs(reachOff - 1.0f) < 1e-5f)
      break;
  }
  return fit;
}

RocketFlight launchRocket(glm::vec3 from, glm::vec3 direction, const RocketSpec &spec) {
  RocketFlight f;
  const float length = glm::length(direction);
  f.axis = length > 1e-6f ? direction / length : glm::vec3(0.0f, 0.0f, 1.0f);
  f.position = from;
  f.velocity = f.axis * spec.launchSpeed;
  return f;
}

void stepRocket(RocketFlight &flight, const RocketSpec &spec, const RocketFit &fit, float dt,
                glm::vec3 wind, glm::vec3 gravity) {
  const bool toTheEnd = dt >= spec.selfDestruct - flight.time;
  float left = std::min(dt, spec.selfDestruct - flight.time);
  while (left > 1e-7f) {
    const float h = std::min(left, kStep);
    stepOnce(flight, spec, fit, h, wind, gravity);
    left -= h;
  }
  // Thousands of small steps add up a little off; its fuze doesn't.
  if (toTheEnd)
    flight.time = spec.selfDestruct;
}

bool rocketSpent(const RocketFlight &flight, const RocketSpec &spec) {
  return flight.time >= spec.selfDestruct - 1e-6f;
}

float rocketLift(const RocketSpec &spec, const RocketFit &fit, float range, glm::vec3 gravity) {
  // Past about 95% of its reach it can't be lifted onto the point before it
  // goes off: aim for that.
  const float target = std::clamp(range, 1.0f, 0.95f * spec.reach);
  // The height it passes target at, launched at `lift` (or far under if it
  // never gets there).
  auto heightAt = [&](float lift) {
    RocketFlight f = launchRocket(glm::vec3(0.0f), glm::vec3(0.0f, std::sin(lift), std::cos(lift)), spec);
    while (!rocketSpent(f, spec)) {
      const glm::vec3 before = f.position;
      stepRocket(f, spec, fit, 0.005f, glm::vec3(0.0f), gravity);
      if (f.position.z >= target) {
        const float t = (target - before.z) / std::max(f.position.z - before.z, 1e-6f);
        return glm::mix(before.y, f.position.y, t);
      }
    }
    return -1e6f;
  };
  // The height there goes almost in a straight line with the lift, so a
  // secant search lands it in three or four flights (a bisection took 30).
  float a = 0.0f, ha = heightAt(a);
  float b = std::clamp(-ha / target, 1e-4f, 0.3f), hb = heightAt(b);
  for (int i = 0; i < 8 && std::abs(hb) > 0.005f && std::abs(hb - ha) > 1e-6f; ++i) {
    const float next = std::clamp(b - hb * (b - a) / (hb - ha), 0.0f, 0.3f);
    a = b;
    ha = hb;
    b = next;
    hb = heightAt(b);
  }
  return b;
}

float jetDepth(float steelLeft, float density) {
  return std::max(steelLeft, 0.0f) * std::sqrt(kSteelDensity / std::max(density, 1.0f));
}

float jetUsed(float thickness, float density) {
  return std::max(thickness, 0.0f) * std::sqrt(std::max(density, 1.0f) / kSteelDensity);
}

float matterDensity(Matter matter) {
  switch (matter) {
  case Matter::Stone: return 2400.0f;
  case Matter::Wood: return 600.0f;
  case Matter::Metal: return 7800.0f;
  case Matter::Rubber: return 1100.0f;
  case Matter::Ice: return 920.0f;
  case Matter::Dirt: return 1600.0f; // est.: dry soil
  case Matter::Flesh: return 1060.0f;
  case Matter::Count: break;
  }
  return 1000.0f;
}

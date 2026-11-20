#include "Blast.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kAmbient = 101325.0f; // Pa, sea level
constexpr float kAirDensity = 1.225f; // kg/m^3
constexpr float kSteelDensity = 7850.0f;
constexpr float kPi = 3.14159265f;

double sq(double x) { return x * x; }

} // namespace

float tntEquivalent(Explosive explosive, float pressure, bool impulse) {
  switch (explosive) {
  case Explosive::CompB: return 1.32f;
  case Explosive::C4: return impulse ? 1.19f : (pressure < 1.38e6f ? 1.20f : 1.37f);
  case Explosive::Tnt: return 1.0f;
  }
  return 1.0f;
}

float smokeShade(Explosive explosive) {
  switch (explosive) {
  case Explosive::Tnt: return 0.08f;
  case Explosive::CompB: return 0.16f;
  case Explosive::C4: return 0.4f;
  }
  return 0.3f;
}

float scaledDistance(float range, float tntKg) {
  return range / std::cbrt(std::max(tntKg, 1e-9f));
}

float peakOverpressure(float z) {
  const double s = std::max(static_cast<double>(z), 1e-3);
  const double ratio = 808.0 * (1.0 + sq(s / 4.5)) /
                       std::sqrt((1.0 + sq(s / 0.048)) * (1.0 + sq(s / 0.32)) * (1.0 + sq(s / 1.35)));
  return static_cast<float>(kAmbient * ratio);
}

float positiveDuration(float z, float tntKg) {
  const double s = std::max(static_cast<double>(z), 1e-3);
  const double ms = 980.0 * (1.0 + std::pow(s / 0.54, 10.0)) /
                    ((1.0 + std::pow(s / 0.02, 3.0)) * (1.0 + std::pow(s / 0.74, 6.0)) *
                     std::sqrt(1.0 + sq(s / 6.9)));
  return static_cast<float>(ms * 1e-3 * std::cbrt(std::max(tntKg, 1e-9f)));
}

float positiveImpulse(float z, float tntKg) {
  const double s = std::max(static_cast<double>(z), 1e-3);
  const double barMs = 0.067 * std::sqrt(1.0 + std::pow(s / 0.23, 4.0)) /
                       (sq(s) * std::cbrt(1.0 + std::pow(s / 1.55, 3.0)));
  // bar ms is 100 Pa s; per kg^(1/3).
  return static_cast<float>(barMs * 100.0 * std::cbrt(std::max(tntKg, 1e-9f)));
}

float reflectionFactor(float overpressure) {
  // Pr / Ps = 2 (7 P0 + 4 Ps) / (7 P0 + Ps).
  const double r = overpressure / kAmbient;
  return static_cast<float>(2.0 * (7.0 + 4.0 * r) / (7.0 + r));
}

float blastPush(float dragCoefficient, float area, float impulse, float mass) {
  return mass > 0.0f ? dragCoefficient * area * impulse / mass : 0.0f;
}

float gurneySphere(float charge, float caseMass, float gurney) {
  return gurney / std::sqrt(caseMass / std::max(charge, 1e-6f) + 0.6f);
}

float fragmentSpeed(float v0, float mass, float range) {
  if (!(mass > 0.0f))
    return 0.0f;
  const float area = 1.5f * std::pow(mass / kSteelDensity, 2.0f / 3.0f);
  const float k = kAirDensity * 1.0f * area / (2.0f * mass); // per m
  return v0 * std::exp(-k * std::max(range, 0.0f));
}

FragmentSpread fitFragments(const CasedCharge &charge) {
  // Hit at least once with chance 1 - e^-n, n = count A / (4 pi R^2): half
  // the time at the casualty radius means n = ln 2 there.
  const float r = charge.casualtyRadius;
  FragmentSpread spread;
  spread.count = std::max(1, static_cast<int>(std::lround(std::log(2.0f) * 4.0f * kPi * r * r / kPersonArea)));
  spread.mass = charge.caseMass / static_cast<float>(spread.count);
  spread.speed = gurneySphere(charge.charge, charge.caseMass, kGurneyCompB);
  return spread;
}

float casualtyChance(const FragmentSpread &spread, const CasedCharge &, float range) {
  const float r = std::max(range, 0.1f);
  const float hits = static_cast<float>(spread.count) * kPersonArea / (4.0f * kPi * r * r);
  return 1.0f - std::exp(-hits);
}

float throwRange(float speed, float angle, float height) {
  constexpr float mass = 0.397f, radius = 0.032f, drag = 0.47f;
  const float k = 0.5f * kAirDensity * drag * kPi * radius * radius / mass; // per m
  glm::vec2 p(0.0f, height), v(speed * std::cos(angle), speed * std::sin(angle));
  constexpr float dt = 1.0f / 1000.0f;
  for (int i = 0; i < 20000; ++i) {
    const glm::vec2 last = p;
    const glm::vec2 a = glm::vec2(0.0f, -9.81f) - k * glm::length(v) * v;
    v += a * dt;
    p += v * dt;
    if (p.y <= 0.0f) {
      const float t = last.y / std::max(last.y - p.y, 1e-6f);
      return last.x + (p.x - last.x) * t;
    }
  }
  return p.x;
}

float fuseTime(float dice) { return 4.0f + std::clamp(dice, 0.0f, 1.0f); }

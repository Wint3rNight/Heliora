#pragma once

#include <glm/glm.hpp>

// What flies: arcs for what's thrown or launched (grenades in step 1.7,
// kept from the powers' throw when they went), and bullets (step 1.6):
// a point mass under gravity and air drag from the standard drag tables,
// the way external ballistics programs do it (McCoy, Modern Exterior
// Ballistics), checked against the AKM manual's own table.

// The launch velocity, at this speed, that lands on `to` under gravity: the
// low arc; out of reach, 45 degrees up toward it.
glm::vec3 throwVelocity(glm::vec3 from, glm::vec3 to, float speed, glm::vec3 gravity);

// The two standard projectiles bullets are measured against: G1 (a flat-
// based bullet, what pistol rounds are rated by) and G7 (a long boat-tail,
// the better fit for rifle bullets).
enum class DragModel { G1, G7 };

// The air: the ICAO standard atmosphere at sea level, and the wind.
struct Air {
  float density = 1.225f;       // kg/m^3
  float speedOfSound = 340.29f; // m/s
  glm::vec3 wind{0.0f};         // m/s
};

// One pound per square inch in kilograms per square metre: published
// ballistic coefficients are in lb/in^2.
constexpr float kBallisticCoefficientUnit = 703.0696f;

// A bullet's drag: its standard projectile and its ballistic coefficient
// against it, lb/in^2 as published (the AKM's ball round, G7 0.15).
struct Projectile {
  DragModel model = DragModel::G7;
  float coefficient = 0.15f;
};

// The standard projectile's drag coefficient at a Mach number: the
// published tables, straight lines between rows, held past either end.
float standardDrag(DragModel model, float mach);

// m/s^2 that drag takes off at this speed through the air:
// pi * rho * v^2 * Cd(M) / (8 * BC), BC in kg/m^2.
float dragDeceleration(const Projectile &projectile, float airspeed, const Air &air);

// A bullet in the air.
struct Flight {
  glm::vec3 position{0.0f};
  glm::vec3 velocity{0.0f};
};

// dt seconds of flight (a midpoint step: at 1/240 s a rifle bullet stays
// within half a m/s of the exact path to 800 m). Drag acts against the
// bullet's speed through the air, so wind pushes it sideways.
void stepFlight(Flight &flight, const Projectile &projectile, const Air &air,
                glm::vec3 gravity, float dt);

// Radians above a level line to aim so a bullet leaving at `speed` comes
// back down to the line `range` m away: zeroing a sight at that range.
// 0 for no range, or one the bullet can't reach.
float zeroAngle(const Projectile &projectile, float speed, float range, const Air &air,
                glm::vec3 gravity);

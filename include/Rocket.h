#pragma once

#include "Bullets.h"

#include <glm/glm.hpp>

// A launcher's rocket (step 1.7): its flight, worked out from a few
// documented facts, and its warhead's shaped-charge jet. No physics here, so
// it's tested on its own; the game casts the rays.
//
// The defaults are the RPG-7's PG-7V, from Wikipedia's RPG-7 article (read
// 2026-11-20): 115 m/s off the launcher, the motor lit after 10 m and
// "sustains flight out to 500 m at a maximum velocity of 295 m/s"; the fuze
// sets the usual maximum range, 920 m, with a 4.5 s fuze; the 85 mm warhead
// goes through 260 mm of steel armour.

struct RocketSpec {
  float mass = 2.2f;          // kg (est.: not in the article)
  float diameter = 0.085f;    // m, its widest: the warhead
  float launchSpeed = 115.0f; // m/s, the booster's kick off the launcher
  float motorFrom = 10.0f;    // m flown when the motor lights
  float motorTo = 500.0f;     // m flown when it burns out
  float topSpeed = 295.0f;    // m/s at burnout
  float selfDestruct = 4.5f;  // s
  float reach = 920.0f;       // m flown by then
  // The warhead: how much steel its jet goes through, and its explosive as
  // TNT (est.).
  float jetSteel = 0.26f;     // m
  float warheadTnt = 0.7f;    // kg
  bool operator==(const RocketSpec &) const = default;
};

// What isn't documented, fitted so the flight meets what is: a constant
// thrust while the motor burns, and the drag coefficient (on the widest
// cross-section), so it reaches topSpeed at motorTo and has flown reach
// at selfDestruct (both level, in still air).
struct RocketFit {
  float thrust = 0.0f;          // N
  float dragCoefficient = 0.0f;
};
RocketFit fitRocket(const RocketSpec &spec);

// A rocket in the air. Its nose stays along the air it flies through (the
// fins weathercock it within metres), and the motor pushes along the nose:
// in a crosswind it turns into the wind while it burns.
struct RocketFlight {
  glm::vec3 position{0.0f}, velocity{0.0f};
  glm::vec3 axis{0.0f, 0.0f, 1.0f}; // the way the nose points, unit
  float time = 0.0f;  // s since launch
  float flown = 0.0f; // m along its path
  bool motorOn = false, burntOut = false;
};
RocketFlight launchRocket(glm::vec3 from, glm::vec3 direction, const RocketSpec &spec);
// dt on, in steps of at most a millisecond so it flies the same at any frame
// rate, never past its self-destruct time. wind is the air's velocity (m/s).
void stepRocket(RocketFlight &flight, const RocketSpec &spec, const RocketFit &fit, float dt,
                glm::vec3 wind, glm::vec3 gravity);
bool rocketSpent(const RocketFlight &flight, const RocketSpec &spec);

// How far above the line to a point `range` m off (level) to launch, radians,
// so the rocket drops onto it in still air: the sight's range marks set
// right. Past what it can reach, the lift for about its furthest.
float rocketLift(const RocketSpec &spec, const RocketFit &fit, float range, glm::vec3 gravity);

// The shaped-charge jet. Birkhoff's hydrodynamic rule (1948): how deep a
// jet goes goes with the square root of its density over the target's, so
// a jet good for d of steel goes d sqrt(7850 / rho) into something of
// density rho. It leaves out the target's strength, so soft things come out
// a little generous.
constexpr float kSteelDensity = 7850.0f; // kg/m^3
float jetDepth(float steelLeft, float density);
// The steel's worth of depth a jet uses going through `thickness` of it.
float jetUsed(float thickness, float density);
// kg/m^3 for each material the bullets know, as the physics materials have
// them (dirt est.).
float matterDensity(Matter matter);

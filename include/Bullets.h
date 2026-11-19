#pragma once

#include "Ballistics.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

// Bullets in flight (step 1.6): each is a point mass stepped through the air
// (Ballistics.h), a segment of the world checked every sub-step, and slowed
// by what it goes through by Poncelet's penetration law. No physics here:
// the game answers what each segment hits.

// What a bullet can go into.
enum class Matter { Stone, Wood, Metal, Rubber, Ice, Dirt, Flesh, Count };

// Poncelet's law (1829, still the standard simple model of penetration):
// inside a material a bullet is held back by a + b v^2, so a bullet of mass
// m stops after (m / 2b) ln(1 + b v^2 / a). With b = 0 the force is
// constant and a bullet through t of t_max keeps v sqrt(1 - t / t_max).
struct Resistance {
  float a = 0.0f; // N
  float b = 0.0f; // kg/m
  bool operator==(const Resistance &) const = default;
};

// The law through two documented depths at two speeds (the AKM in pine:
// NSD's 35 cm at 150 m and 25 cm at 500 m). The depths must grow with speed
// and less than the speeds' squares do, or the nearer fit is used.
Resistance ponceletFit(float mass, float depth1, float speed1, float depth2, float speed2);
// Through one documented depth at a speed, with b / a (s^2/m^2) borrowed
// from a fitted material; 0 is the constant force.
Resistance ponceletFrom(float mass, float depth, float speed, float ratio);
// Its b / a, s^2/m^2.
float ratioOf(const Resistance &r);
// m a bullet gets in from this speed (infinite if nothing holds it back).
float stoppingDepth(const Resistance &r, float mass, float speed);
// m/s it comes out at after going through this thickness, 0 if it stops.
float exitSpeed(const Resistance &r, float mass, float speed, float thickness);

// A kind of round: how it flies and how far it gets into each material.
struct Round {
  Projectile projectile;
  float mass = 0.0079f; // kg
  std::array<Resistance, static_cast<size_t>(Matter::Count)> resistance{};
};

struct Bullet {
  const Round *round = nullptr;
  Flight flight;
  float age = 0.0f; // s since it left the muzzle
  int owner = -1;   // who fired it, for the game
  bool tracer = false;
};

// The first thing a segment of a bullet's path enters: where, and how thick
// it is along the path from there (the game finds that with a second ray
// cast back from beyond it).
struct BulletContact {
  float distance = 0.0f; // m along the segment
  glm::vec3 point{0.0f};
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
  uint64_t body = 0;
  Matter matter = Matter::Stone;
  float thickness = 0.0f; // m along the path
};
using BulletWorld =
    std::function<std::optional<BulletContact>(glm::vec3 from, glm::vec3 to)>;

// A bullet going into something: what it hit, how fast it was going in and
// out (0 if it stopped there).
struct BulletImpact {
  glm::vec3 point{0.0f};
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
  glm::vec3 direction{0.0f, 0.0f, 1.0f};
  uint64_t body = 0;
  Matter matter = Matter::Stone;
  float speedIn = 0.0f, speedOut = 0.0f;
  float mass = 0.0f;
  float thickness = 0.0f; // what it went through, or into before it stopped
  bool stopped = false;
  int owner = -1;
  // J it left in what it hit.
  float energy() const { return 0.5f * mass * (speedIn * speedIn - speedOut * speedOut); }
  // N s it pushed what it hit with.
  float impulse() const { return mass * (speedIn - speedOut); }
};

struct BulletRules {
  float subStep = 1.0f / 240.0f; // s, at most
  float slowest = 50.0f;         // m/s: slower is spent
  float longest = 3.0f;          // s in the air at most
  int contactsPerStep = 4;       // things gone through in one sub-step at most
};

// Moves every bullet dt on in sub-steps, asking the world what each sub-
// step's segment hits; what they hit goes into impacts (added to), and the
// spent bullets are dropped.
void stepBullets(std::vector<Bullet> &bullets, float dt, const Air &air, glm::vec3 gravity,
                 const BulletWorld &world, std::vector<BulletImpact> &impacts,
                 const BulletRules &rules = {});

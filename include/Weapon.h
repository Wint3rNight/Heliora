#pragma once

#include "WeaponFile.h"

#include <cstdint>
#include <random>
#include <vector>

// What a gun is doing (step 1.6): its magazines as round counts, the
// chamber, the fire clock, reloads, and where each shot goes off the
// barrel's line. No drawing, no bullets: the game turns shots into both.

enum class FireMode { Auto, Semi };

// How it's held, for the spread: NSD's numbers are prone with a rest;
// standing and moving multiply them (NSD p.162); from the hip is its own.
enum class FiringStance { Supported, Standing, Moving, Hip };

struct Shot {
  float time = 0.0f;        // s into the frame it went off
  glm::vec2 deviation{0.0f}; // mrad off the barrel's line, x to the right, y up
  bool tracer = false;
};

struct WeaponEvents {
  std::vector<Shot> shots;
  bool dry = false;      // the trigger fell on an empty chamber
  bool reloaded = false; // a reload finished
};

// Sigma of the random kick each shot gives the aim, mrad (side, up). The
// aim comes back by e^(-t / recoilReturn), so the rounds after the first
// carry what's left of the kicks before them; the kick is solved so bursts
// of 3 to 5 at the gun's rate (NSD's short bursts) spread as followSpread.
glm::vec2 recoilKick(const WeaponSpec &spec);

// Which way a shot leaves a gun whose frame (in the world) is `gun` (+z its
// barrel, +y its top, +x its left): along the barrel, tilted up by the
// zero angle (radians) so it crosses the sight line at the zeroing range,
// and off by the shot's deviation (mrad, x to the right, y up). Unit.
glm::vec3 shotDirection(const glm::mat3 &gun, float zero, glm::vec2 deviation);

// How a shot moves the gun you see (not where its bullets go, that's
// Shot::deviation): back into the shoulder or hand, its muzzle up and a
// little aside, then back. A critically damped spring per part, so it comes
// back without bouncing: from rest a kick goes x(t) = v t e^(-t / peak),
// highest at `peak` s. Stepped exactly, the same at any frame rate.
struct GunKick {
  glm::vec3 offset{0.0f};   // back (m), up (radians), aside to the left (radians)
  glm::vec3 velocity{0.0f}; // per second
  // A shot: the same parts, as high as they'd go from rest.
  void kick(glm::vec3 amount, float peak);
  void update(float dt, float peak);
};

class Weapon {
public:
  explicit Weapon(const WeaponSpec &spec, uint32_t seed = 1);
  const WeaponSpec &spec() const { return gun; }

  // n full magazines: one in the gun, topped up with a round in the chamber,
  // the rest in the pouch.
  void load(int magazines);
  void setPouch(std::vector<int> magazines);
  bool chambered() const { return chamber; }
  int magazineRounds() const { return magazine; } // -1 with none in
  const std::vector<int> &pouch() const { return spares; }
  int roundsInGun() const;
  int rounds() const; // in the gun and the pouch
  bool slideLocked() const { return locked; }

  FireMode mode() const { return fireMode; }
  void switchMode(); // auto and semi in turn, if it has auto

  // A frame of dt with the trigger held or not.
  void update(float dt, bool trigger, FiringStance stance, WeaponEvents &events);

  // Starts a reload, false if there's nothing to load or no need. A partial
  // magazine goes back in the pouch, the fullest comes out; an empty gun
  // also needs its bolt worked or slide let go, which takes longer.
  bool reload();
  void cancelReload();
  bool reloading() const { return reloadTime > 0.0f; }
  bool emptyReload() const { return reloadEmpty; }
  float reloadLeft() const { return reloadTime; }

  // mrad (side, up) the kicks have the aim off now.
  glm::vec2 recoil() const { return offset; }

private:
  void finishReload();
  glm::vec2 gaussian(glm::vec2 sigma);

  WeaponSpec gun;
  glm::vec2 kick;
  std::mt19937 dice;
  std::normal_distribution<float> normal{0.0f, 1.0f};

  bool chamber = false;
  int magazine = -1;
  std::vector<int> spares;
  bool locked = false;
  FireMode fireMode = FireMode::Auto;

  float ready = 0.0f;   // s until the action can fire again
  bool wasHeld = false; // the trigger, last frame
  bool pullUsed = false; // this pull has fired (semi) or clicked
  int fired = 0;         // rounds since it was loaded, for tracers

  float reloadTime = 0.0f;
  bool reloadEmpty = false;

  glm::vec2 offset{0.0f}; // the kicks' sum, coming back
};

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
enum class Stance { Supported, Standing, Moving, Hip };

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
  void update(float dt, bool trigger, Stance stance, WeaponEvents &events);

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

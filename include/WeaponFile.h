#pragma once

#include "Bullets.h"
#include "SettingsFile.h"

#include <array>
#include <string>

// What a gun is (step 1.6), from its .weapon file (Resources/Weapons/), in
// the settings files' format. Every number there is marked with where it
// comes from: a manual's page, or "estimate".

// How deep a round gets into a material, at a speed: from a source, one
// point or two (two fit Poncelet's law; one borrows the fitted b / a).
struct MatterPenetration {
  float depth = 0.0f, speed = 0.0f;   // m, m/s
  float depth2 = 0.0f, speed2 = 0.0f; // a second point, or 0
  bool operator==(const MatterPenetration &) const = default;
};

struct WeaponSpec {
  std::string name;
  // The round.
  float bulletMass = 0.0079f; // kg
  float muzzleSpeed = 715.0f; // m/s
  DragModel drag = DragModel::G7;
  float coefficient = 0.15f;  // lb/in^2, against the drag model
  float zeroRange = 100.0f;   // m where the bullet crosses the line of sight
  int tracerEvery = 0;        // every nth round lights up; 0 none
  std::array<MatterPenetration, static_cast<size_t>(Matter::Count)> penetration{};
  // The action.
  int magazine = 30;          // rounds in a full magazine
  float rate = 600.0f;        // rounds a minute the action cycles at
  bool automatic = true;      // has an automatic setting
  bool locksOpen = false;     // the slide or bolt stays back after the last round
  // Spread, sigma in mrad (x to the side, y up), prone with a rest: the first
  // round of a burst, and the ones after it; times standing or moving.
  glm::vec2 firstSpread{0.44f, 0.59f};
  glm::vec2 followSpread{1.63f, 1.19f};
  float standing = 1.34f;
  float moving = 1.52f;
  float hipSpread = 15.0f;    // mrad from the hip, both ways
  // Recoil.
  float recoilImpulse = 7.65f; // N s
  float recoilReturn = 0.15f;  // s for the aim to come back to 1/e
  // Reloads, s: with a round still chambered, and empty.
  float reload = 2.5f;
  float emptyReload = 3.2f;
  // Carrying.
  float mass = 3.6f;          // kg loaded
  float length = 0.88f;       // m
  int spareMagazines = 4;     // carried besides the one in the gun
  bool operator==(const WeaponSpec &) const = default;
};

// From a parsed .weapon file; anything missing or unreadable keeps its
// default (logged).
WeaponSpec weaponFromSettings(const SettingsFile &file);
// Into one, every value written.
void weaponToSettings(const WeaponSpec &spec, SettingsFile &file);

// The round it fires: its drag, and each material's resistance from the
// penetration figures. A material with two points is fitted; one with a
// single point borrows the b / a of the first fitted one (or is a constant
// force without one); one with none doesn't hold the bullet back.
Round roundOf(const WeaponSpec &spec);

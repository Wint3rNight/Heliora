#pragma once

#include <glm/glm.hpp>

#include <optional>

// How a character holds a gun (step 1.6), worked out from its stance's
// aiming clip so nothing is placed by hand: an aiming clip aims along the
// character's facing, so the gun lies that way, upright, its trigger where
// the right index finger is. Then the gun rides on the right hand, and the
// left hand (a rifle held in two hands) is put back on the gun: on its
// centre line, under it, where the clip had it along the gun.

// A bone's matrix without its scale (Mixamo rigs scale their armature):
// its axes made unit and square, its place kept.
glm::mat4 rigidFrame(const glm::mat4 &m);

// The left hand in that pose: its palm (what touches the gun) and its
// wrist (what IK moves), and how high on the gun the palm goes (gun space y,
// m; just under a rifle's handguard).
struct OffHand {
  glm::vec3 palm{0.0f}, wrist{0.0f};
  float height = 0.0f;
};

struct WeaponHold {
  glm::mat4 gripInHand{1.0f}; // the gun's frame in the right hand's (rigid) frame
  glm::vec3 offHand{0.0f};    // where the left wrist goes, in gun space
  glm::vec3 offPalm{0.0f};    // and its palm
  bool twoHanded = false;     // the left hand holds the gun away from the right
};

// The palm closer to the trigger than this holds one grip (a pistol), m:
// Remy's left palm is 0.14 m from his trigger finger cupping the pistol and
// over 0.2 m on an AK's handguard.
constexpr float kTwoHandedReach = 0.18f;

// From one pose, all in the same (world) space: the right hand's rigid
// frame, the trigger finger's point, the way the clip aims, the left hand
// if it's on the gun, and up.
WeaponHold holdFrom(const glm::mat4 &hand, glm::vec3 trigger, glm::vec3 forward,
                    std::optional<OffHand> off = std::nullopt,
                    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f));

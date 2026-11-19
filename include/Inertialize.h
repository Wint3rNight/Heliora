#pragma once

#include "Skeleton.h"

#include <vector>

// Inertialization (Bollo, "Inertialization: High-Performance Animation
// Transitions in Gears of War", GDC 2018): at a change of animation the new
// pose plays at once, and the difference to the pose that was showing, with
// how fast it was moving, is eased away. Nothing frozen, nothing blended
// twice.

// An offset x0 moving at v0 brought to rest at 0 by t1, along a quintic with
// no jump in position, speed or acceleration at the start and everything at
// 0 at the end. Unreal's clamps keep it from overshooting: moving away is
// treated as still, a fast approach shortens t1 to -5 x0 / v0, and the
// starting acceleration is chosen so the jerk at t1 is 0, never below 0.
struct QuinticDecay {
  float x0 = 0.0f;
  float v0 = 0.0f;
  float t1 = 0.0f;
  float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f; // the t^5 to t^2 terms

  // x0 must be 0 or more (the caller picks the direction).
  static QuinticDecay make(float x0, float v0, float t1);
  // The offset t seconds in: x0 at 0, 0 from t1 on.
  float at(float t) const;
};

// Every bone's offset from a target pose, eased away over a duration.
// Rotations turn about a fixed axis, translations shift along a fixed
// direction; scales jump to the target's.
class PoseInertialization {
public:
  // last and beforeLast are the last two poses shown, lastDt the time
  // between them (0 if there was only one); target is the new pose now.
  void start(const Pose &last, const Pose &beforeLast, float lastDt, const Pose &target,
             float duration);
  // Adds the offsets, elapsed seconds after start(), to this frame's target.
  void apply(float elapsed, Pose &pose) const;
  float duration() const { return length; }
  bool active(float elapsed) const { return elapsed < length && !bones.empty(); }

private:
  struct BoneOffset {
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    QuinticDecay turn;
    glm::vec3 direction{0.0f, 1.0f, 0.0f};
    QuinticDecay shift;
  };
  std::vector<BoneOffset> bones;
  float length = 0.0f;
};

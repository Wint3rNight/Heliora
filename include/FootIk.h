#pragma once

#include "Skeleton.h"

#include <array>
#include <vector>

// Feet on uneven ground (step 1.5b). After the clips, a ray under each foot
// finds the ground; each foot moves up or down by how far its ground is off
// the floor the clip walks on, the hips drop by the lowest foot, the legs
// reach by two-bone IK and a planted foot turns to the ground's slope. The
// setup is ozz-animation's foot IK sample (ray from 0.5 m above the ankle,
// the pelvis down by the largest correction, the knee bending the way it
// already points), smoothed the way Holden damps (by a half-life).
// No physics here: the game casts the rays.
//
// It also warps the stride (Unreal's Stride Warping): each foot's distance
// ahead of or behind the hips along the way it moves is scaled, so a clip
// played at a person's real cadence still covers the ground the capsule
// does, and a planted foot that ends up out of reach brings the hips down.

struct FootIkTuning {
  bool enabled = true;
  float reach = 0.45f;      // m a foot follows the ground down below the capsule's feet (est.)
  float tilt = 0.5236f;     // radians a planted foot turns toward the slope at most (30 degrees, est.)
  float halfLife = 0.1f;    // s for a change to get half way (Holden's value)
  float probeHeight = 0.5f; // m above the ankle each ray starts (ozz)
  bool strideWarp = true;   // runs at a real cadence, the stride made up here
  bool operator==(const FootIkTuning &) const = default;
};

// The leg bones by Mixamo's names, left then right; -1 where missing.
struct FootIkRig {
  int hips = -1;
  std::array<int, 2> upLeg{-1, -1};
  std::array<int, 2> leg{-1, -1};
  std::array<int, 2> foot{-1, -1};
  // The ankles' height over the floor in the bind pose, in the skeleton's
  // units: a foot this high is planted.
  float ankleHeight = 0.0f;
  bool valid() const;
};
FootIkRig findFootIkRig(const Skeleton &skeleton);

// Where the character is drawn: a point p of the skeleton's space is at
// feet + facing * (scale * p) in the world. feet is the capsule's bottom,
// the floor the clips walk on. stride scales the feet's distance from the
// hips along strideAxis (the skeleton's space, level), 1 leaving it.
struct FootPlacement {
  glm::vec3 feet{0.0f};
  glm::quat facing{1.0f, 0.0f, 0.0f, 0.0f};
  float scale = 1.0f;
  float stride = 1.0f;
  glm::vec3 strideAxis{0.0f, 0.0f, 1.0f};
};

// The two rays to cast, world space: from probeHeight above each ankle
// (where the stride puts it) straight down to reach below the feet.
struct FootRay {
  glm::vec3 from{0.0f};
  glm::vec3 to{0.0f};
};
std::array<FootRay, 2> footRays(const FootIkRig &rig, const std::vector<glm::mat4> &model,
                                const FootPlacement &placement, const FootIkTuning &tuning);

// What a ray found, world space.
struct GroundHit {
  bool hit = false;
  glm::vec3 point{0.0f};
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

// Kept per character between frames.
struct FootIkState {
  bool started = false;
  float weight = 0.0f;                  // IK's share, 0 off to 1 on
  std::array<float, 2> correction{};    // m each foot moves up (down below 0)
  float hipsDrop = 0.0f;                // m, 0 or below
  float stride = 1.0f;                  // the placement's, eased in
  std::array<glm::vec3, 2> normal{glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)};
  std::array<bool, 2> planted{true, true};
  std::array<bool, 2> stepped{};        // a foot came down this frame
};

// Moves pose's hips and legs onto the ground. on: whether the feet should
// follow it now (on the ground, not dodging); off, IK lets go over the
// half-life. Fills state.stepped.
void solveFootIk(const Skeleton &skeleton, const FootIkRig &rig, const FootPlacement &placement,
                 const std::array<GroundHit, 2> &hits, bool on, float dt,
                 const FootIkTuning &tuning, FootIkState &state, Pose &pose);

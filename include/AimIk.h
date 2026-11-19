#pragma once

#include "Skeleton.h"

#include <functional>
#include <vector>

// Aiming by IK (step 1.6): after the clips, the spine turns so the gun's
// barrel points at the aim point. The turn is shared 20/30/50 over Spine,
// Spine1 and Spine2, the upper spine doing most of a trunk's turn (Fujii et
// al. 2007; the split is est.). Turning the spine moves the gun too, so it
// takes a few passes.

struct AimRig {
  std::vector<int> bones;    // Spine, Spine1, Spine2
  std::vector<float> shares; // of the turn, adding up to 1
  bool valid() const { return !bones.empty() && bones.size() == shares.size(); }
};
AimRig findAimRig(const Skeleton &skeleton);

// Where the barrel is in the skeleton's space, for model-space matrices.
struct Barrel {
  glm::vec3 muzzle{0.0f};
  glm::vec3 forward{0.0f, 0.0f, 1.0f};
};
using BarrelFinder = std::function<Barrel(const std::vector<glm::mat4> &model)>;

// Turns the spine so the barrel's line runs through target (the skeleton's
// space), weight of the way (0 to 1), the whole turn held to maxAngle
// radians. Returns the angle the barrel turned.
float aimAt(const Skeleton &skeleton, const AimRig &rig, Pose &pose, const BarrelFinder &barrel,
            glm::vec3 target, float weight, float maxAngle = 1.2f, int passes = 4);

// Turns one bone about its joint so its axis (in its own frame, like a
// head's face) points at target (the skeleton's space), weight of the way,
// at most maxAngle radians. Returns the angle turned.
float turnToward(const Skeleton &skeleton, Pose &pose, int bone, glm::vec3 axis, glm::vec3 target,
                 float weight, float maxAngle);

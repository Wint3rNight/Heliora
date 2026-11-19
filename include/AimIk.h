#pragma once

#include "Skeleton.h"

#include <vector>

// Aiming (step 1.6): after the clips, the chest squares to the target and
// leans with the aim, the turns shared 20/30/50 over Spine, Spine1 and
// Spine2, the upper spine doing most of a trunk's turn (Fujii et al. 2007;
// the split is est.); the gun is then laid along the aim from the body and
// the hands put on it by IK (Guns.cpp).

struct AimRig {
  std::vector<int> bones;    // Spine, Spine1, Spine2
  std::vector<float> shares; // of the turn, adding up to 1
  bool valid() const { return !bones.empty() && bones.size() == shares.size(); }
};
AimRig findAimRig(const Skeleton &skeleton);

// The way a chest faces, level and square to its shoulder line (the left
// shoulder joint's place minus the right's), in the skeleton's space.
glm::vec3 chestFacing(glm::vec3 leftShoulder, glm::vec3 rightShoulder, glm::vec3 up);

// Squares the chest to the aim: turns the rig's spine bones about `up`
// (sharing the turn as for the aim) until the chest faces `blade` radians
// to the right of aim's level direction, weight of the way, at most
// maxAngle radians. A rifle clip's stance can stand nearly side-on, and
// bending a side-on chest to aim up or down tips one shoulder over the
// other; modern Army rifle doctrine (TC 3-22.9) stands squared to the
// target instead. Returns the angle turned.
float squareChest(const Skeleton &skeleton, const AimRig &rig, Pose &pose, int leftShoulder,
                  int rightShoulder, glm::vec3 aim, float blade, float weight, float maxAngle,
                  glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f));

// Bends the rig's spine about the chest's own sideways line (its shoulder
// line, made level), sharing the angle as for the aim: positive leans the
// chest back, so the gun rises. The shoulders stay level however the chest
// faces.
void bendChest(const Skeleton &skeleton, const AimRig &rig, Pose &pose, int leftShoulder,
               int rightShoulder, float angle, glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f));

// A gun's frame (+z its barrel, +y its top, +x its left) laid along aim
// with its top as near up as it goes (not canted), placed so the point
// anchorInGun (in the gun's own frame, like its butt) lands on anchor.
glm::mat4 aimedGun(glm::vec3 aim, glm::vec3 up, glm::vec3 anchorInGun, glm::vec3 anchor);

// Turns one bone about its joint so its axis (in its own frame, like a
// head's face) points at target (the skeleton's space), weight of the way,
// at most maxAngle radians. Returns the angle turned.
float turnToward(const Skeleton &skeleton, Pose &pose, int bone, glm::vec3 axis, glm::vec3 target,
                 float weight, float maxAngle);

#pragma once

#include "Skeleton.h"

#include <vector>

// Mixing poses: two at a time, and the weights for a row of clips by speed
// or for four clips by direction.

// out is a mixed with b by weight (0 gives a, 1 gives b): translations and
// scales linear, rotations the short way, normalised (nlerp).
void blendPoses(const Pose &a, const Pose &b, float weight, Pose &out);

// 1 for firstBone and every bone under it, 0 for the rest: the upper body
// from the first spine bone, so the legs keep walking under it.
std::vector<float> upperBodyMask(const Skeleton &skeleton, int firstBone);
// blendPoses with each bone's weight times its mask (a bone past the mask's
// end gets none).
void blendPosesMasked(const Pose &a, const Pose &b, float weight, const std::vector<float> &mask,
                      Pose &out);
// Turns the bones about an axis in the skeleton's space (unit), the angle
// shared between them: down a chain like the spine the last one ends up
// turned by all of it, and everything they carry turns with them.
void turnBones(const Skeleton &skeleton, Pose &pose, const std::vector<int> &bones, glm::vec3 axis,
               float angle);
// Two-bone IK: turns `upper` and `lower` (its child, whose child is `end`)
// so `end` lands on target, in the skeleton's space, the middle joint
// bending toward `pole` (a direction); out of reach it points straight at
// it. The bones keep their lengths.
void twoBoneIk(const Skeleton &skeleton, Pose &pose, int upper, int lower, int end,
               glm::vec3 target, glm::vec3 pole);

// Weights for clips made at these speeds (in rising order) for one speed:
// the two clips either side share it, nearer gets more, and past the ends
// the end clip has it all. They add up to 1.
std::vector<float> speedBlend(const std::vector<float> &speeds, float speed);

struct DirectionWeights {
  float forward = 0.0f;
  float back = 0.0f;
  float left = 0.0f;
  float right = 0.0f;
};
// Weights for the four directional clips for a direction in the
// character's own frame (x to its right, y ahead), adding up to 1. A
// diagonal is half and half; standing still is all forward.
DirectionWeights directionBlend(glm::vec2 direction);

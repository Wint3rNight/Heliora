#pragma once

#include "Skeleton.h"

#include <vector>

// Mixing poses: two at a time, and the weights for a row of clips by speed
// or for four clips by direction.

// out is a mixed with b by weight (0 gives a, 1 gives b): translations and
// scales linear, rotations the short way, normalised (nlerp).
void blendPoses(const Pose &a, const Pose &b, float weight, Pose &out);

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

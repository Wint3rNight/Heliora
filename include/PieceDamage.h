#pragma once

#include "Stress.h"

#include <vector>

// Damage that builds up in a breakable piece (step 1.6, the user's choice:
// "Damage builds up"). A bullet's few N s never reach the impulse a link
// needs, but the energy it leaves in a piece stays there: at the piece's
// toughness its links give, and before that they hold less and less, so a
// wall riddled near its base comes down under its own weight.

// J a piece takes to break: its volume times its material's energy per
// cubic metre (tuned, est.).
float pieceToughness(float volume, float energyPerVolume);

struct PieceDamage {
  float energy = 0.0f;    // J left in it so far
  float toughness = 0.0f; // J; 0 for a piece damage can't break
  // Adds a hit's energy; true for the hit that broke it.
  bool add(float joules);
  bool broken() const { return toughness > 0.0f && energy >= toughness; }
  // The share of its links' strength left, 1 untouched to 0 at its
  // toughness.
  float strengthLeft() const;
};

// A link between two pieces is as strong as its weaker end.
float linkStrengthLeft(const PieceDamage &a, const PieceDamage &b);

// The limits with their pull, push and slide scaled by the share left;
// friction belongs to the surfaces and stays.
LinkLimits weakened(const LinkLimits &limits, float share);

// A piece's damage shared out by volume between the pieces a cut makes of
// it (equal shares if they have no volume).
std::vector<float> shareDamage(float energy, const std::vector<float> &volumes);

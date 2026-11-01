#pragma once

#include <cmath>

// Turns frame times into a whole number of fixed physics steps. Leftover
// time carries into the next frame, and alpha() is how far the frame sits
// between the last step and the next one, for interpolation.
class FixedStepClock {
public:
  explicit FixedStepClock(double stepSeconds = 1.0 / 60.0,
                          int maxStepsPerFrame = 4)
      : step(stepSeconds), maxSteps(maxStepsPerFrame) {}

  // How many steps to run this frame.
  int advance(double frameSeconds) {
    owed += frameSeconds;
    int steps = static_cast<int>(owed / step);
    if (steps > maxSteps) {
      // A hitch (scene load, breakpoint, slow frame). Running every missed
      // step would make this frame slower still and the next one further
      // behind, so drop the backlog: the game slows down for a moment
      // instead of spiralling.
      steps = maxSteps;
      owed = std::fmod(owed, step);
    } else {
      owed -= steps * step;
    }
    return steps;
  }

  // 0 = exactly at the last step, just under 1 = almost at the next one.
  float alpha() const { return static_cast<float>(owed / step); }

  double stepSeconds() const { return step; }

private:
  double step;
  int maxSteps;
  double owed = 0.0; // time not yet simulated; double so it doesn't drift
};

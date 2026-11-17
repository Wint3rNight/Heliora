#pragma once

#include "CharacterMovement.h"

// A character's brain: what decides what its body does each physics step,
// the player's keys (PlayerController) or, later, AI. Unreal's Controller;
// the body is the Pawn. A CharacterRoster hands it a body to drive
// (possessing), and a body with none stands still.
class Controller {
public:
  virtual ~Controller() = default;

  // The commands for its body's next physics step.
  virtual CharacterInput decide() = 0;

  // The body it drives, or none.
  CharacterMovement *body() const { return possessed; }

protected:
  // Just after it's handed a body, or had it taken away (body() is null).
  virtual void onPossess() {}

private:
  friend class CharacterRoster; // the only one that hands out bodies
  CharacterMovement *possessed = nullptr;
};

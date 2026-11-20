#include "Melee.h"

#include <algorithm>

const char *meleeStateName(MeleeState state) {
  switch (state) {
  case MeleeState::Idle: return "Idle";
  case MeleeState::Attacking: return "Attacking";
  case MeleeState::Blocking: return "Blocking";
  case MeleeState::Staggered: return "Staggered";
  }
  return "?";
}

void timeBlowFromClip(MeleeAttack &blow, float start, float end, float hitFrom, float hitTo) {
  if (!(hitFrom >= start && hitTo > hitFrom && end >= hitTo))
    return;
  blow.windup = hitFrom - start;
  blow.active = hitTo - hitFrom;
  blow.recovery = end - hitTo;
  blow.chainFrom = blow.windup;
}

void MeleeController::stagger(float seconds) {
  state_ = MeleeState::Staggered;
  staggered = std::max(seconds, 0.0f);
  attack_ = -1;
  striking_ = false;
  request = -1;
  time_ = 0.0f;
}

void MeleeController::start(int which, MeleeEvents &e) {
  state_ = MeleeState::Attacking;
  attack_ = which;
  time_ = 0.0f;
  striking_ = false;
  request = -1;
  e.started = which;
}

MeleeEvents MeleeController::update(const MeleeInput &in, float dt, const MeleeTuning &t) {
  MeleeEvents e;
  // The button: a short press is a light blow (on release), held long
  // enough the heavy one (at once). Nothing while the powers hold something.
  const bool attackHeld = in.attack && !in.busy;
  if (attackHeld) {
    downFor = downFor < 0.0f ? 0.0f : downFor + dt;
    if (!heavyThrown && downFor + 1e-5f >= t.heavyHold) {
      heavyThrown = true;
      request = kHeavyAttack;
      requestAge = 0.0f;
    }
  } else {
    if (downFor >= 0.0f && !heavyThrown && !in.busy) {
      request = 0;
      requestAge = 0.0f;
    }
    downFor = -1.0f;
    heavyThrown = false;
  }
  if (request >= 0 && (requestAge += dt) > t.buffer + dt)
    request = -1;

  switch (state_) {
  case MeleeState::Staggered:
    request = -1;
    if ((staggered -= dt) <= 0.0f)
      state_ = MeleeState::Idle;
    break;
  case MeleeState::Blocking:
    request = -1;
    time_ += dt;
    if (!in.block)
      state_ = MeleeState::Idle;
    break;
  case MeleeState::Idle:
    if (in.block) {
      state_ = MeleeState::Blocking;
      time_ = 0.0f;
      request = -1;
    } else if (request >= 0) {
      start(request == kHeavyAttack ? kHeavyAttack : 0, e);
    }
    break;
  case MeleeState::Attacking: {
    time_ += dt;
    const MeleeAttack &a = blow(attack_, t);
    const bool inWindow = time_ >= a.windup && time_ < a.windup + a.active;
    if (inWindow && !striking_)
      e.strikeOpened = true;
    if (!inWindow && striking_)
      e.strikeClosed = true;
    striking_ = inWindow;
    const float end = a.windup + a.active + a.recovery;
    // The next blow: a light one goes on through the combo, the heavy one
    // can follow any light one; after the third or the heavy, wait for the end.
    if (request >= 0 && time_ >= a.chainFrom && attack_ != kHeavyAttack) {
      if (request == kHeavyAttack)
        start(kHeavyAttack, e);
      else if (attack_ < 2)
        start(attack_ + 1, e);
      break;
    }
    // A dodge cuts the recovery short.
    if (in.dodge && time_ >= a.windup + a.active) {
      e.cancelled = true;
      state_ = MeleeState::Idle;
      attack_ = -1;
      striking_ = false;
      break;
    }
    if (time_ >= end) {
      state_ = MeleeState::Idle;
      attack_ = -1;
      striking_ = false;
      // A press waiting from the end of the combo starts it again.
      if (request >= 0 && !in.block)
        start(request == kHeavyAttack ? kHeavyAttack : 0, e);
    }
    break;
  }
  }
  return e;
}

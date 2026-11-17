#include "CharacterController.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979f;
// Timers count down by whole steps, and 0.25 - 15 steps can land a
// rounding error above zero. Anything under this counts as run out.
constexpr float kTimerEpsilon = 1e-4f;
// Below this the stick counts as untouched.
constexpr float kDeadZone = 0.01f;
// Asked to go back more than this far off its heading, faster than walking
// pace, it plants and brakes before it turns.
constexpr float kPlantAngle = 2.0943951f; // 120 degrees
constexpr float kPlantSpeed = 1.5f;       // m/s
// Slower than this it has no heading of its own and goes by its facing.
constexpr float kStill = 1e-3f;

glm::vec3 flat(glm::vec3 v) { return glm::vec3(v.x, 0.0f, v.z); }

// The angle from a to b the short way round, -pi to pi.
float angleBetween(float a, float b) {
  return std::remainder(b - a, 2.0f * kPi);
}

float yawOf(glm::vec3 direction) { return std::atan2(-direction.x, -direction.z); }

// A flat direction turned about +y by angle (positive turns -z towards -x).
glm::vec3 turnFlat(glm::vec3 d, float angle) {
  const float c = std::cos(angle), s = std::sin(angle);
  return glm::vec3(c * d.x + s * d.z, 0.0f, -s * d.x + c * d.z);
}

// Moves from towards to by at most maxStep.
glm::vec3 approach(glm::vec3 from, glm::vec3 to, float maxStep) {
  const glm::vec3 gap = to - from;
  const float distance = glm::length(gap);
  return distance <= maxStep ? to : from + gap * (maxStep / distance);
}

} // namespace

glm::vec3 yawForward(float yaw) {
  return glm::vec3(-std::sin(yaw), 0.0f, -std::cos(yaw));
}

CharacterController::CharacterController(PhysicsWorld &world, glm::vec3 feet,
                                         float yaw,
                                         const CharacterTuning &tuning)
    : world(world), tune(tuning) {
  PhysicsCharacterDesc desc;
  desc.radius = tune.radius;
  desc.height = tune.height;
  desc.mass = tune.mass;
  desc.position = feet;
  id = world.addCharacter(desc);
  place(feet, yaw);
}

CharacterController::~CharacterController() { world.removeCharacter(id); }

void CharacterController::place(glm::vec3 feet, float yaw) {
  world.placeCharacter(id, feet);
  current = {world.characterState(id).position, yaw};
  previous = current;
  horizontal = glm::vec3(0.0f);
  onGround = false;
  jumpedSinceGround = false;
  airTime = 0.0f;
  jumpQueued = 0.0f;
  dodgeLeft = 0.0f;
  dodgeWait = 0.0f;
}

void CharacterController::update(const CharacterInput &input, float dt) {
  const PhysicsCharacterState state = world.characterState(id);
  const float gravity = std::max(0.0f, -world.gravity().y);

  // Standing on walkable ground. Jolt already counts the step after a jump
  // as in the air, even for a small hop (tested), so a rising character
  // never reads as standing here.
  onGround = state.ground == PhysicsGround::OnGround;
  if (onGround) {
    airTime = 0.0f;
    jumpedSinceGround = false;
  } else {
    airTime += dt;
  }

  // Crouch, or stand back up once there's room (setCharacterHeight refuses
  // under a ceiling, so this just tries again every step).
  const float wantedHeight = input.crouch ? tune.crouchHeight : tune.height;
  if (state.height != wantedHeight)
    world.setCharacterHeight(id, wantedHeight);
  const bool crouched = height() < tune.height;

  glm::vec3 move = flat(input.move);
  if (glm::length(move) > 1.0f)
    move = glm::normalize(move);
  const bool moving = glm::length(move) > kDeadZone;
  const float speed = crouched       ? tune.crouchSpeed
                      : input.sprint ? tune.sprintSpeed
                                     : tune.runSpeed;
  const glm::vec3 wanted = move * speed;

  // Dodge: a fixed burst along the stick, or forward without input. Only
  // from the ground, and not again until the cooldown has run out.
  dodgeWait = std::max(0.0f, dodgeWait - dt);
  if (input.dodge && onGround && !dodging() && dodgeWait <= kTimerEpsilon) {
    dodgeDirection = moving ? glm::normalize(move) : yawForward(current.yaw);
    dodgeLeft = tune.dodgeTime;
    dodgeWait = tune.dodgeCooldown;
  }

  const glm::vec3 groundVelocity =
      onGround ? state.groundVelocity : glm::vec3(0.0f);
  glm::vec3 stepHorizontal; // this step's; horizontal carries into the next
  if (dodging()) {
    const glm::vec3 burst =
        dodgeDirection * (tune.dodgeDistance / tune.dodgeTime);
    stepHorizontal = flat(groundVelocity) + burst;
    horizontal = stepHorizontal;
    dodgeLeft -= dt;
    if (dodgeLeft <= kTimerEpsilon) {
      // Out of the dodge after this step, at the speed the stick asks for,
      // so it doesn't slide on.
      dodgeLeft = 0.0f;
      horizontal = flat(groundVelocity) +
                   dodgeDirection * std::min(glm::length(burst), glm::length(wanted));
    }
  } else if (onGround && !input.aim) {
    // Momentum: a heading and a speed, relative to the ground under it.
    const glm::vec3 own = horizontal - flat(groundVelocity);
    float speed = glm::length(own);
    glm::vec3 heading = speed > kStill ? own / speed : yawForward(current.yaw);
    if (!moving) {
      speed = std::max(0.0f, speed - tune.groundBraking * dt);
    } else {
      const glm::vec3 want = glm::normalize(move);
      // Signed angle from the heading to where it's asked to go.
      const float angle = std::atan2(heading.z * want.x - heading.x * want.z,
                                     glm::dot(heading, want));
      if (std::fabs(angle) > kPlantAngle && speed > kPlantSpeed) {
        // Plant and brake along the heading; it turns once slow enough.
        speed = std::max(0.0f, speed - tune.groundBraking * dt);
      } else {
        // Turn, as far as the grip allows at this speed.
        const float rate = std::min(tune.turnRate, tune.turnGrip / std::max(speed, kStill));
        const float turn = std::clamp(angle, -rate * dt, rate * dt);
        heading = turnFlat(heading, turn);
        // Push the way it's heading, fading near the target; facing away
        // from where it's asked to go slows it instead.
        const float target = glm::length(wanted);
        const float along = glm::dot(heading, want);
        if (along < 0.0f)
          speed = std::max(0.0f, speed + tune.groundBraking * along * dt);
        else if (speed < target)
          speed = std::min(target, speed + tune.groundAcceleration *
                                               (1.0f - speed / target) * along * dt);
        else
          speed = std::max(target, speed - tune.groundBraking * dt);
      }
    }
    horizontal = flat(groundVelocity) + heading * speed;
    stepHorizontal = horizontal;
  } else {
    // Aiming or in the air: ease the velocity straight towards what the
    // stick asks for (plus the ground's own motion, so it rides what it
    // stands on).
    const float rate = onGround ? tune.aimAcceleration : tune.airAcceleration;
    horizontal = approach(horizontal, flat(groundVelocity) + wanted, rate * dt);
    stepHorizontal = horizontal;
  }

  // Jump: a press stays queued for jumpBuffer seconds, and the ground still
  // counts for coyoteTime seconds after walking off it.
  if (input.jump)
    jumpQueued = tune.jumpBuffer;
  float vertical;
  const bool canJump =
      !dodging() && (onGround || (airTime <= tune.coyoteTime && !jumpedSinceGround));
  if (jumpQueued > kTimerEpsilon && canJump) {
    // sqrt(2 g h) is the speed that reaches h with smooth gravity. Moving in
    // whole steps, each one applies a step of gravity first, which costs
    // v dt / 2 of height; half a step of extra speed puts it back.
    vertical = groundVelocity.y + std::sqrt(2.0f * gravity * tune.jumpHeight) +
               0.5f * gravity * dt;
    jumpQueued = 0.0f;
    jumpedSinceGround = true;
    onGround = false;
  } else if (onGround) {
    vertical = groundVelocity.y; // the gravity added below keeps it on the ground
  } else {
    vertical = state.velocity.y; // a ceiling has already taken its share
  }
  jumpQueued = std::max(0.0f, jumpQueued - dt);
  vertical -= gravity * dt;

  // Face its heading on the ground, where it's asked to go in the air, the
  // dodge's way while dodging, the camera's way while aiming, turning at
  // turnRate.
  float targetYaw = current.yaw;
  const glm::vec3 ownNow = stepHorizontal - flat(groundVelocity);
  if (input.aim)
    targetYaw = input.aimYaw;
  else if (dodging())
    targetYaw = yawOf(dodgeDirection);
  else if (onGround && glm::length(ownNow) > kStill)
    targetYaw = yawOf(ownNow);
  else if (moving)
    targetYaw = yawOf(move);
  const float turn = angleBetween(current.yaw, targetYaw);
  const float maxTurn = tune.turnRate * dt;
  const float yaw = std::remainder(
      current.yaw + std::clamp(turn, -maxTurn, maxTurn), 2.0f * kPi);

  PhysicsCharacterMove step;
  step.velocity = stepHorizontal + glm::vec3(0.0f, vertical, 0.0f);
  step.stepUp = tune.stepUp;
  step.stickDown = tune.stickDown;
  step.maxSlope = tune.maxSlope;
  step.pushStrength = tune.pushStrength;
  world.moveCharacter(id, step, dt);
  // Keep no more speed than really moved when something blocked it: a wall
  // or a box it pushes takes its share, so it doesn't keep a run stored and
  // shoot off the end of the wall. What the move lost is told apart by its
  // direction: lost to a wall, a box or a slope too steep to walk points
  // more than maxSlope off straight up and comes off the stored speed; lost
  // to the floor or a walkable slope (the step's gravity, a ramp bending
  // the move along itself) doesn't, or every ramp would slow it to a crawl.
  const glm::vec3 requested = step.velocity - groundVelocity;
  const glm::vec3 moved = world.characterState(id).velocity - groundVelocity;
  const glm::vec3 lost = requested - moved;
  const float lostSideways = glm::length(flat(lost));
  if (!dodging() && lostSideways > kStill &&
      lostSideways > std::fabs(lost.y) * std::tan(tune.maxSlope) &&
      glm::length(flat(moved)) < glm::length(flat(requested)))
    horizontal = flat(groundVelocity) + flat(moved);

  previous = current;
  current = {world.characterState(id).position, yaw};
}

glm::vec3 CharacterController::velocity() const {
  return world.characterState(id).velocity;
}

float CharacterController::height() const {
  return world.characterState(id).height;
}

glm::vec3 CharacterController::drawnPosition(float alpha) const {
  return glm::mix(previous.position, current.position, alpha);
}

float CharacterController::drawnYaw(float alpha) const {
  return previous.yaw + angleBetween(previous.yaw, current.yaw) * alpha;
}

CharacterTuning consistent(CharacterTuning tuning) {
  // 5 cm of straight side at the least, so the capsule is still a capsule.
  const float shortest = 2.0f * tuning.radius + 0.05f;
  tuning.height = std::max(tuning.height, shortest);
  tuning.crouchHeight = std::clamp(tuning.crouchHeight, shortest, tuning.height);
  return tuning;
}

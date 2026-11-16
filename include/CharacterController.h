#pragma once

#include "PhysicsWorld.h"

#include <glm/glm.hpp>

// The player's movement: running, sprinting, jumping, crouching, dodging and
// turning, on top of a PhysicsWorld character. No Vulkan and no input
// devices here, so it's tested like the rest of heliora_core.
//
// Yaw is in radians about +y, and yaw 0 faces -z, so the facing rotation is
// glm::angleAxis(yaw, up).

// Every value the Character tab tunes. Radius, heights and mass are read
// when the controller is made; the rest apply from the next step.
struct CharacterTuning {
  float radius = 0.3f;
  float height = 1.8f;
  float crouchHeight = 1.1f;
  float mass = 70.0f; // kg

  float runSpeed = 5.0f; // m/s
  float sprintSpeed = 8.0f;
  float crouchSpeed = 2.0f;
  // m/s^2. Braking is when there's no input. In the air both are the air
  // value, so a jump keeps most of its momentum.
  float groundAcceleration = 30.0f;
  float groundBraking = 40.0f;
  float airAcceleration = 8.0f;
  float turnRate = 12.566371f; // radians/s, two full turns a second

  float jumpHeight = 1.2f; // m
  float coyoteTime = 0.1f; // s after walking off a ledge that a jump still works
  float jumpBuffer = 0.1f; // s before landing that a jump press still counts

  float dodgeDistance = 5.0f; // m
  float dodgeTime = 0.25f;    // s
  float dodgeCooldown = 0.6f; // s from the start of one dodge to the next

  float maxSlope = 0.785398f; // radians (45 degrees)
  float stepUp = 0.3f;        // m, see PhysicsCharacterMove::stepUp
  float stickDown = 0.5f;     // m
  float pushStrength = 100.0f; // N
};

// The same values, made to fit together: the height leaves room for the
// two rounded ends, and the crouch height sits between them and the full
// height. The Character tab and a hand-edited settings file go through it.
CharacterTuning consistent(CharacterTuning tuning);

// What the player asks for in one physics step.
struct CharacterInput {
  // World space and flat, length 0 to 1: WASD turned by the camera's yaw.
  glm::vec3 move{0.0f};
  bool sprint = false;
  bool crouch = false;
  bool aim = false;     // faces aimYaw instead of where it moves
  float aimYaw = 0.0f;  // the camera's yaw
  bool jump = false;    // pressed since the last step
  bool dodge = false;   // pressed since the last step
};

// The direction a yaw faces, flat.
glm::vec3 yawForward(float yaw);

class CharacterController {
public:
  CharacterController(PhysicsWorld &world, glm::vec3 feet, float yaw,
                      const CharacterTuning &tuning = {});
  ~CharacterController(); // takes the character out of the world
  CharacterController(const CharacterController &) = delete;
  CharacterController &operator=(const CharacterController &) = delete;

  // One physics step: works out the velocity and moves the character. Call
  // it right before PhysicsWorld::step(), with the same dt.
  void update(const CharacterInput &input, float dt);
  // Puts it there standing still, drawn there at once (spawning, Stop).
  void place(glm::vec3 feet, float yaw);
  void setTuning(const CharacterTuning &tuning) { tune = tuning; }
  const CharacterTuning &tuning() const { return tune; }

  glm::vec3 position() const { return current.position; } // feet
  float yaw() const { return current.yaw; }
  // Its actual velocity after the last step: walls and floors take away
  // what they blocked.
  glm::vec3 velocity() const;
  float height() const; // drops to crouchHeight while crouched
  bool grounded() const { return onGround; }
  bool dodging() const { return dodgeLeft > 0.0f; }

  // Blended between the last two steps by the physics clock's alpha, so the
  // drawn capsule and the camera move smoothly at any frame rate.
  glm::vec3 drawnPosition(float alpha) const;
  float drawnYaw(float alpha) const;

private:
  struct Pose {
    glm::vec3 position{0.0f};
    float yaw = 0.0f;
  };

  PhysicsWorld &world;
  PhysicsCharacterId id;
  CharacterTuning tune;
  Pose previous;
  Pose current;
  glm::vec3 horizontal{0.0f}; // world space, flat, what the input built up
  bool onGround = false;
  bool jumpedSinceGround = false;
  float airTime = 0.0f;   // s since it last stood on walkable ground
  float jumpQueued = 0.0f; // s a jump press stays buffered
  float dodgeLeft = 0.0f;  // s of the current dodge still to go
  float dodgeWait = 0.0f;  // s until the next dodge is allowed
  glm::vec3 dodgeDirection{0.0f};
};

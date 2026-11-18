#pragma once

#include "AnimationClip.h"
#include "PhysicsWorld.h"

#include <array>

#include <glm/glm.hpp>

// A character's movement: running, sprinting, jumping, crouching, dodging and
// turning, on top of a PhysicsWorld character. Its controller (Controller.h)
// says what it should do each step; this works out how the body does it. No Vulkan and no input
// devices here, so it's tested like the rest of heliora_core.
//
// Yaw is in radians about +y, and yaw 0 faces -z, so the facing rotation is
// glm::angleAxis(yaw, up).

// Every value the Character tab tunes. Radius, heights and mass are read
// when the body is made; the rest apply from the next step.
struct CharacterTuning {
  float radius = 0.3f;
  float height = 1.8f;
  float crouchHeight = 1.1f;
  float mass = 70.0f; // kg

  float runSpeed = 5.0f; // m/s
  float sprintSpeed = 8.0f;
  float crouchSpeed = 2.0f;
  // On the ground it keeps a heading and a speed. The push, m/s^2 from
  // standing, fades as the speed nears the target, like a runner's legs:
  // walking pace in about 0.2 s, nearly the run by 1.2 s, the sprint
  // building on after. Braking is letting go, slowing past the target, and
  // planting to turn back.
  float groundAcceleration = 10.0f;
  float groundBraking = 10.0f;
  // m/s^2 of sideways grip: the heading turns at most turnGrip / speed, so
  // the faster it goes the wider it turns (about 1.5 m at a run, 4 m at a
  // sprint), and at most turnRate.
  float turnGrip = 16.0f;
  float turnRate = 12.566371f; // radians/s, two full turns a second
  // While aiming it eases in a straight line at this, both ways: quick
  // footwork rather than momentum.
  float aimAcceleration = 30.0f;
  // In the air it eases in a straight line at this, so a jump keeps most of
  // its momentum.
  float airAcceleration = 8.0f;
  // With it, the air can steer and slow it but not take it faster than it
  // left the ground at (walking pace at least); without, it speeds up in
  // flight like an arcade game (the climb's "rage" variant).
  bool airSpeedLimit = true;

  float jumpHeight = 1.2f; // m
  float coyoteTime = 0.1f; // s after walking off a ledge that a jump still works
  float jumpBuffer = 0.1f; // s before landing that a jump press still counts

  // Without a clip's path (no character), a dodge is a fixed burst.
  float dodgeDistance = 5.0f; // m
  float dodgeTime = 0.25f;    // s
  float dodgeScale = 1.0f;    // a clip's dodge path is stretched by this
  float dodgeCooldown = 0.6f; // s from the start of one dodge to the next

  float maxSlope = 0.785398f; // radians (45 degrees)
  float stepUp = 0.3f;        // m, see PhysicsCharacterMove::stepUp
  float stickDown = 0.5f;     // m
  float pushStrength = 100.0f; // N
  bool operator==(const CharacterTuning &) const = default;
};

// The same values, made to fit together: the height leaves room for the
// two rounded ends, and the crouch height sits between them and the full
// height. The Character tab and a hand-edited settings file go through it.
CharacterTuning consistent(CharacterTuning tuning);

// What its controller asks for in one physics step.
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

// Which dodge: the roll, or while aiming a step the way it's pressed.
enum class DodgeKind { Roll, Forward, Back, Left, Right };

// The direction a yaw faces, flat.
glm::vec3 yawForward(float yaw);

class CharacterMovement {
public:
  CharacterMovement(PhysicsWorld &world, glm::vec3 feet, float yaw,
                      const CharacterTuning &tuning = {});
  ~CharacterMovement(); // takes the character out of the world
  CharacterMovement(const CharacterMovement &) = delete;
  CharacterMovement &operator=(const CharacterMovement &) = delete;

  // One physics step: works out the velocity and moves the character. Call
  // it right before PhysicsWorld::step(), with the same dt.
  void update(const CharacterInput &input, float dt);
  // Puts it there standing still, drawn there at once (spawning, Stop).
  void place(glm::vec3 feet, float yaw);
  // The body keeps the radius, heights and mass it was made with: the
  // capsule in the world is that size. The rest apply from the next step.
  void setTuning(const CharacterTuning &tuning);
  // The path each dodge moves along, from its clip (rootPath), by
  // DodgeKind. One left empty is the fixed burst.
  void setDodgePaths(const std::array<RootMotion, 5> &paths) { dodgePaths = paths; }
  const CharacterTuning &tuning() const { return tune; }

  glm::vec3 position() const { return current.position; } // feet
  PhysicsCharacterId physicsId() const { return id; }
  float yaw() const { return current.yaw; }
  // Its actual velocity after the last step: walls and floors take away
  // what they blocked.
  glm::vec3 velocity() const;
  float height() const; // drops to crouchHeight while crouched
  bool grounded() const { return onGround; }
  bool dodging() const { return dodgeLeft > 0.0f; }
  bool aiming() const { return aimed; } // the last step's input aimed
  DodgeKind dodgeKind() const { return dodgeType; } // the last one started

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
  // World space, flat, what the input built up (the ground's own motion
  // included); after each step no more than what really moved.
  glm::vec3 horizontal{0.0f};
  bool onGround = false;
  bool aimed = false;
  bool jumpedSinceGround = false;
  float airTime = 0.0f;   // s since it last stood on walkable ground
  float airSpeedCap = 1.5f; // m/s, the most the air may take it to (airSpeedLimit)
  float jumpQueued = 0.0f; // s a jump press stays buffered
  float dodgeLeft = 0.0f;  // s of the current dodge still to go
  float dodgeWait = 0.0f;  // s until the next dodge is allowed
  glm::vec3 dodgeDirection{0.0f};
  std::array<RootMotion, 5> dodgePaths;
  DodgeKind dodgeType = DodgeKind::Roll;
  bool dodgeOnPath = false;  // this dodge follows its clip's path
  float dodgeLength = 0.0f;  // s the current dodge lasts
  float dodgeFacing = 0.0f;  // the yaw its path is laid out from
};

#pragma once

#include "PhysicsWorld.h"

#include <glm/glm.hpp>

// The play camera: orbits a pivot at the character's shoulders, slides over
// the right shoulder while aiming, and pulls in front of walls. No Vulkan
// and no input devices, so it's tested like the rest of heliora_core.
//
// Yaw and pitch are in radians. Yaw is about +y with 0 looking down -z, the
// same as CharacterController, so its yaw can turn WASD into the world.
// Positive pitch looks up.

struct ThirdPersonCameraTuning {
  float pivotHeight = 1.5f; // m above the feet: shoulder height
  float distance = 4.0f;    // m behind the pivot, where the wheel starts
  float minDistance = 2.0f;
  float maxDistance = 8.0f;
  float zoomStep = 0.5f;    // m per wheel notch
  float minPitch = -1.308997f; // radians (-75 degrees)
  float maxPitch = 1.308997f;  // (75 degrees)
  float mouseSensitivity = 0.0025f; // radians per pixel

  // Aiming: the pivot moves this far right and up of the shoulders, and the
  // camera comes this close, over aimTime seconds.
  float aimRight = 0.55f;
  float aimUp = 0.1f;
  float aimDistance = 2.5f;
  float aimTime = 0.12f;

  // How quickly the pivot catches up with the character (the time to close
  // about two thirds of the gap), and the furthest it may trail behind.
  float followTime = 0.08f;
  float maxLag = 0.6f;

  float collisionRadius = 0.2f; // m, the ball swept from the pivot
  // After a wall stops being in the way, the time to close about two thirds
  // of the gap back out. Pulling in is instant.
  float easeOutTime = 0.4f;
};

// The same values, made to fit together: the closest zoom no further than
// the furthest, the starting distance between them, and the pitch limits
// either side of level.
ThirdPersonCameraTuning consistent(ThirdPersonCameraTuning tuning);

glm::vec3 cameraForward(float yaw, float pitch);

class ThirdPersonCamera {
public:
  explicit ThirdPersonCamera(const ThirdPersonCameraTuning &tuning = {});

  // Mouse movement in pixels, screen style: +x right, +y down. Moving right
  // turns right, moving up looks up.
  void orbit(float mouseDx, float mouseDy);
  // Wheel notches; positive comes closer.
  void zoom(float notches);
  // Jumps straight to its place behind the feet, no easing (spawning).
  void snapTo(const PhysicsWorld &world, glm::vec3 feet, float yaw,
              float pitch = -0.25f);
  // Once a frame, with the character's drawn feet position and the frame's
  // time.
  void update(const PhysicsWorld &world, glm::vec3 feet, bool aiming, float dt);

  void setTuning(const ThirdPersonCameraTuning &tuning);
  const ThirdPersonCameraTuning &tuning() const { return tune; }

  glm::vec3 position() const { return eye; }
  glm::vec3 forward() const { return cameraForward(yawAngle, pitchAngle); }
  float yaw() const { return yawAngle; }
  float pitch() const { return pitchAngle; }
  float distance() const { return zoomDistance; } // what the wheel asks for
  glm::vec3 pivot() const { return pivotPoint; }  // where it orbits, smoothed
  glm::mat4 view() const;

private:
  void place(const PhysicsWorld &world, bool snap, float dt);

  ThirdPersonCameraTuning tune;
  float yawAngle = 0.0f;
  float pitchAngle = -0.25f;
  float zoomDistance = 4.0f;
  float aimBlend = 0.0f;      // 0 behind the shoulders, 1 over the right one
  glm::vec3 pivotPoint{0.0f};
  // m the eye may go from the pivot. Drops to a wall at once, and eases back
  // out while easingOut, from a wall going until it has caught up.
  float clearance = 0.0f;
  bool easingOut = false;
  glm::vec3 eye{0.0f};
};

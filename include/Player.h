#pragma once

#include "CharacterController.h"
#include "PhysicsWorld.h"
#include "ThirdPersonCamera.h"

#include <glm/glm.hpp>

// The keys and mouse for one frame, as they are. No input devices here, so
// the mapping is tested like the rest of heliora_core.
struct PlayerControls {
  bool forward = false; // W
  bool back = false;    // S
  bool left = false;    // A
  bool right = false;   // D
  bool sprint = false;
  bool crouch = false;
  bool aim = false;
  bool jump = false;      // went down this frame
  bool dodge = false;     // went down this frame
  glm::vec2 mouse{0.0f};  // pixels, screen style: +x right, +y down
  float wheel = 0.0f;     // notches, positive away from you
};

// Play mode's character and its camera. Each frame: control(), then step()
// before every physics step, then follow().
class Player {
public:
  Player(PhysicsWorld &world, glm::vec3 feet, float yaw,
         const CharacterTuning &character = {},
         const ThirdPersonCameraTuning &camera = {});

  // The mouse and wheel move the camera at once. Jump and dodge presses
  // wait for the next step: at a high frame rate most frames have none.
  void control(const PlayerControls &controls);
  // Moves the character one physics step, walking relative to the camera.
  void step(float dt);
  // The camera follows where the character is drawn this frame.
  void follow(float alpha, float frameSeconds);

  const CharacterController &character() const { return controller; }
  const ThirdPersonCamera &camera() const { return view; }

private:
  PhysicsWorld &world;
  CharacterController controller;
  ThirdPersonCamera view;
  PlayerControls held;     // this frame's keys
  bool jumpWaiting = false;  // pressed, no step has taken it yet
  bool dodgeWaiting = false;
};

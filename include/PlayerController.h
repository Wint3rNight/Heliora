#pragma once

#include "Controller.h"
#include "PhysicsWorld.h"
#include "ThirdPersonCamera.h"

#include <glm/glm.hpp>

#include <string_view>

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

// HELIORA_CAPTURE_KEYS, for checking the character in captures, where
// nothing presses keys: what's pressed on a frame. Space-separated keys
// (w a s d shift ctrl alt rmb space), each held all the time, or only on one
// frame ("space@120"), or over a range of frames ("w@60-200"). Anything
// else is skipped.
PlayerControls scriptedControls(std::string_view script, int frame);

// The player's brain: the keys and mouse drive the body it possesses, turned
// by its third-person camera, which follows that body. Unreal's
// PlayerController. Each frame: control(), then the roster's step() before
// every physics step, then follow().
class PlayerController : public Controller {
public:
  explicit PlayerController(PhysicsWorld &world,
                            const ThirdPersonCameraTuning &camera = {});

  // The mouse and wheel move the camera at once. Jump and dodge presses
  // wait for the next step: at a high frame rate most frames have none.
  void control(const PlayerControls &controls);
  // WASD in the camera's frame, and the presses since the last step.
  CharacterInput decide() override;
  // The camera follows where its body is drawn this frame.
  void follow(float alpha, float frameSeconds);
  void setCameraTuning(const ThirdPersonCameraTuning &camera);

  const ThirdPersonCamera &camera() const { return view; }

protected:
  // Behind the new body, looking the way it faces.
  void onPossess() override;

private:
  PhysicsWorld &world;
  ThirdPersonCamera view;
  PlayerControls held;       // this frame's keys
  bool jumpWaiting = false;  // pressed, no step has taken it yet
  bool dodgeWaiting = false;
};

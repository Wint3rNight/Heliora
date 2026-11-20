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
  bool run = false;     // Shift, held
  bool sprint = false;  // a sprint asked for outright (capture scripts); the
                        // player sprints by pressing Shift again soon after letting go
  bool crouch = false;
  bool aim = false;
  bool jump = false;      // went down this frame
  bool dodge = false;     // went down this frame
  bool fire = false;      // held
  bool cast = false;      // the fire button went down this frame: throw what's held
  bool reveal = false;    // went down this frame (the powers)
  bool burst = false;     // went down this frame (the powers)
  glm::vec2 mouse{0.0f};  // pixels, screen style: +x right, +y down
  float wheel = 0.0f;     // notches, positive away from you
};

// After firing from the hip, it faces the camera this long (est.: then it
// goes back to facing the way it moves).
constexpr float kHipFireHold = 1.0f; // s

// Shift and the gait: held, it runs; pressed again within kDoubleTap of
// letting go, it sprints for as long as it's held; let go, it walks.
constexpr float kDoubleTap = 0.3f; // s (est.; desktops wait up to 0.5 s for a double click)
class GaitKeys {
public:
  Gait update(bool held, float seconds);

private:
  bool wasHeld = false;
  bool sprinting = false;
  float sinceRelease = 1e9f; // s since Shift was let go
};

// HELIORA_CAPTURE_KEYS, for checking the character in captures, where
// nothing presses keys: what's pressed on a frame. Space-separated keys
// (w a s d shift ctrl alt rmb space lmb, sprint for a sprint without the
// double tap; mu md ml mr move the mouse 8 px
// up, down, left, right each frame), each held all the time, or only on one
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
  void control(const PlayerControls &controls, float seconds);
  // Holding a gun ready: firing from the hip then turns it to face the
  // camera, until kHipFireHold after the trigger is let go.
  void setArmed(bool armed);
  // Throwing a grenade: he turns to face the camera, as aiming does.
  void setThrowing(bool on) { throwing = on; }
  // Facing the camera for hip fire, without the right mouse.
  bool hipFiring() const { return armed && hipFire > 0.0f && !held.aim; }
  // WASD in the camera's frame, and the presses since the last step.
  CharacterInput decide() override;
  // The camera follows where its body is drawn this frame.
  void follow(float alpha, float frameSeconds);
  void setCameraTuning(const ThirdPersonCameraTuning &camera);

  const ThirdPersonCamera &camera() const { return view; }
  // A gun's recoil on the view (ThirdPersonCamera::kick).
  void kickView(float yaw, float pitch) { view.kick(yaw, pitch); }

protected:
  // Behind the new body, looking the way it faces.
  void onPossess() override;

private:
  PhysicsWorld &world;
  ThirdPersonCamera view;
  PlayerControls held;       // this frame's keys
  GaitKeys shift;
  Gait gait = Gait::Walk;    // from this frame's keys
  bool armed = false;
  bool throwing = false;
  float hipFire = 0.0f;      // s it keeps facing the camera after firing
  bool jumpWaiting = false;  // pressed, no step has taken it yet
  bool dodgeWaiting = false;
};

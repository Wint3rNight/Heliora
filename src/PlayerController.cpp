#include "PlayerController.h"

#include <algorithm>
#include <charconv>
#include <cmath>

PlayerController::PlayerController(PhysicsWorld &world,
                                   const ThirdPersonCameraTuning &camera)
    : world(world), view(camera) {}

Gait GaitKeys::update(bool held, float seconds) {
  if (held && !wasHeld)
    sprinting = sinceRelease <= kDoubleTap; // pressed again soon enough
  if (held) {
    sinceRelease = 1e9f;
  } else {
    sinceRelease = wasHeld ? 0.0f : sinceRelease + seconds;
    sprinting = false;
  }
  wasHeld = held;
  return held ? (sprinting ? Gait::Sprint : Gait::Run) : Gait::Walk;
}

void PlayerController::control(const PlayerControls &controls, float seconds) {
  held = controls;
  const Gait keys = shift.update(controls.run, seconds);
  gait = controls.sprint ? Gait::Sprint : keys;
  if (armed && controls.fire)
    hipFire = kHipFireHold;
  else
    hipFire = std::max(0.0f, hipFire - seconds);
  // Screen style in, the camera turns right for +x and looks up for -y.
  view.orbit(controls.mouse.x, controls.mouse.y);
  view.zoom(controls.wheel);
  jumpWaiting = jumpWaiting || controls.jump;
  dodgeWaiting = dodgeWaiting || controls.dodge;
}

CharacterInput PlayerController::decide() {
  if (!body())
    return {};
  // WASD in the camera's frame, flattened: W walks where the camera looks.
  const float yaw = view.yaw();
  const glm::vec3 forward = yawForward(yaw);
  const glm::vec3 right(std::cos(yaw), 0.0f, -std::sin(yaw));
  // The movement caps it at length 1, so a diagonal is no faster.
  const glm::vec3 move = forward * (float(held.forward) - float(held.back)) +
                         right * (float(held.right) - float(held.left));

  CharacterInput input;
  input.move = rooted ? glm::vec3(0.0f) : move;
  input.gait = gait;
  input.crouch = held.crouch;
  input.aim = held.aim || (armed && hipFire > 0.0f) || throwing;
  input.aimYaw = yaw;
  input.jump = jumpWaiting && !rooted;
  input.dodge = dodgeWaiting;
  jumpWaiting = dodgeWaiting = false; // this step took them
  return input;
}

void PlayerController::setArmed(bool isArmed) {
  armed = isArmed;
  if (!armed)
    hipFire = 0.0f;
}

void PlayerController::follow(float alpha, float frameSeconds) {
  if (const CharacterMovement *character = body())
    view.update(world, character->drawnPosition(alpha), held.aim, frameSeconds);
}

void PlayerController::setCameraTuning(const ThirdPersonCameraTuning &camera) {
  view.setTuning(camera);
}

void PlayerController::onPossess() {
  if (const CharacterMovement *character = body())
    view.snapTo(world, character->position(), character->yaw());
}

namespace {

// The frames a token's "@N" or "@N-M" covers; all of them without one.
bool coversFrame(std::string_view frames, int frame) {
  if (frames.empty())
    return true;
  int first = 0, last = 0;
  const char *end = frames.data() + frames.size();
  auto [p, error] = std::from_chars(frames.data(), end, first);
  if (error != std::errc())
    return false;
  last = first;
  if (p != end) {
    if (*p != '-')
      return false;
    auto [q, rangeError] = std::from_chars(p + 1, end, last);
    if (rangeError != std::errc() || q != end)
      return false;
  }
  return frame >= first && frame <= last;
}

} // namespace

// Pixels a scripted mouse key moves the mouse each frame it's held.
constexpr float kScriptedMouse = 8.0f;

PlayerControls scriptedControls(std::string_view script, int frame) {
  PlayerControls controls;
  while (!script.empty()) {
    const size_t space = script.find(' ');
    const std::string_view token = script.substr(0, space);
    script = space == std::string_view::npos ? "" : script.substr(space + 1);
    const size_t at = token.find('@');
    const std::string_view key = token.substr(0, at);
    if (!coversFrame(at == std::string_view::npos ? "" : token.substr(at + 1), frame))
      continue;
    if (key == "w") controls.forward = true;
    else if (key == "s") controls.back = true;
    else if (key == "a") controls.left = true;
    else if (key == "d") controls.right = true;
    else if (key == "shift") controls.run = true;
    else if (key == "sprint") controls.sprint = true;
    else if (key == "ctrl") controls.crouch = true;
    else if (key == "rmb") controls.aim = true;
    else if (key == "space") controls.jump = true;
    else if (key == "alt") controls.dodge = true;
    else if (key == "lmb") {
      controls.fire = true;
      // Pressed on the first frame it covers.
      controls.cast = controls.cast || !coversFrame(at == std::string_view::npos ? "" : token.substr(at + 1), frame - 1);
    }
    else if (key == "q") controls.reveal = true;
    else if (key == "e") controls.burst = true;
    else if (key == "r") controls.block = true;
    else if (key == "mu") controls.mouse.y -= kScriptedMouse;
    else if (key == "md") controls.mouse.y += kScriptedMouse;
    else if (key == "ml") controls.mouse.x -= kScriptedMouse;
    else if (key == "mr") controls.mouse.x += kScriptedMouse;
  }
  return controls;
}

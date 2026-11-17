#include "Player.h"

#include <charconv>
#include <cmath>

Player::Player(PhysicsWorld &world, glm::vec3 feet, float yaw,
               const CharacterTuning &character,
               const ThirdPersonCameraTuning &camera)
    : world(world), controller(world, feet, yaw, character), view(camera) {
  // Behind the character, looking the way it faces.
  view.snapTo(world, controller.position(), yaw);
}

void Player::control(const PlayerControls &controls) {
  held = controls;
  // Screen style in, the camera turns right for +x and looks up for -y.
  view.orbit(controls.mouse.x, controls.mouse.y);
  view.zoom(controls.wheel);
  jumpWaiting = jumpWaiting || controls.jump;
  dodgeWaiting = dodgeWaiting || controls.dodge;
}

void Player::step(float dt) {
  // WASD in the camera's frame, flattened: W walks where the camera looks.
  const float yaw = view.yaw();
  const glm::vec3 forward = yawForward(yaw);
  const glm::vec3 right(std::cos(yaw), 0.0f, -std::sin(yaw));
  // The controller caps it at length 1, so a diagonal is no faster.
  const glm::vec3 move = forward * (float(held.forward) - float(held.back)) +
                         right * (float(held.right) - float(held.left));

  CharacterInput input;
  input.move = move;
  input.sprint = held.sprint;
  input.crouch = held.crouch;
  input.aim = held.aim;
  input.aimYaw = yaw;
  input.jump = jumpWaiting;
  input.dodge = dodgeWaiting;
  jumpWaiting = dodgeWaiting = false; // this step took them
  controller.update(input, dt);
}

void Player::follow(float alpha, float frameSeconds) {
  view.update(world, controller.drawnPosition(alpha), held.aim, frameSeconds);
}

void Player::setTuning(const CharacterTuning &character,
                       const ThirdPersonCameraTuning &camera) {
  CharacterTuning tuning = character;
  const CharacterTuning &spawned = controller.tuning();
  tuning.radius = spawned.radius;
  tuning.height = spawned.height;
  tuning.crouchHeight = spawned.crouchHeight;
  tuning.mass = spawned.mass;
  controller.setTuning(tuning);
  view.setTuning(camera);
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
    else if (key == "shift") controls.sprint = true;
    else if (key == "ctrl") controls.crouch = true;
    else if (key == "rmb") controls.aim = true;
    else if (key == "space") controls.jump = true;
    else if (key == "alt") controls.dodge = true;
  }
  return controls;
}

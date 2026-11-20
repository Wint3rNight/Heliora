// VulkanRenderer's view of the played character: where the crosshair lands,
// the camera's shake from a blast, and the movement recorder.

#include "VulkanRenderer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>

glm::vec3 VulkanRenderer::aimPoint(const LiveCharacter &c) const {
  constexpr float kFar = 300.0f; // m
  if (!playerController || !physics)
    return glm::vec3(0.0f);
  const ThirdPersonCamera &view = playerController->camera();
  const glm::vec3 from = view.position();
  const glm::vec3 dir = glm::normalize(cameraForward(view.yaw() - c.viewKick.x, view.pitch() - c.viewKick.y));
  const glm::vec3 body = c.body >= 0 && roster ? roster->body(c.body).position() : from;
  const float start = std::max(0.0f, glm::dot(body - from, dir));
  if (const std::optional<PhysicsRayHit> hit = physics->castRay(from + dir * start, dir, kFar - start))
    return from + dir * (start + hit->distance);
  return from + dir * kFar;
}

void VulkanRenderer::shakeView(int index, float seconds) {
  LiveCharacter &c = characters[index];
  c.shake.update(seconds, 0.05f);
  if (index != playedCharacter || !playerController)
    return;
  const glm::vec2 view(c.shake.offset.z, c.shake.offset.y);
  playerController->kickView(view.x - c.viewKick.x, view.y - c.viewKick.y);
  c.viewKick = view;
}

namespace {

// The recorder keeps this long, and saves by itself when the played
// character rises faster than this without a jump in the last moment (a
// jump of 1.2 m leaves at about 4.9 m/s).
constexpr double kRecordSeconds = 10.0;
constexpr float kThrownUp = 8.0f; // m/s (est.)
constexpr double kJumpGrace = 0.5; // s

const char *groundName(int ground) {
  switch (static_cast<PhysicsGround>(ground)) {
  case PhysicsGround::OnGround: return "ground";
  case PhysicsGround::OnSteepGround: return "steep";
  case PhysicsGround::NotSupported: return "unsupported";
  case PhysicsGround::InAir: return "air";
  }
  return "?";
}

} // namespace

void VulkanRenderer::recordMovement(float seconds) {
  if (playedCharacter < 0 || playedCharacter >= static_cast<int>(characters.size()) || !roster ||
      characters[playedCharacter].body < 0) {
    moveRecording.clear();
    return;
  }
  const PhysicsCharacterState &state = roster->body(characters[playedCharacter].body).physicsState();
  MoveRecord r;
  r.time = playSeconds;
  r.seconds = seconds;
  r.position = state.position;
  r.velocity = state.velocity;
  r.ground = static_cast<int>(state.ground);
  r.groundUserData = state.groundUserData;
  r.groundBody = state.groundBody.value;
  r.groundVelocity = state.groundVelocity;
  r.groundNormal = state.groundNormal;
  r.jump = playerControls.jump;
  if (r.jump)
    lastJumpAt = r.time;
  moveRecording.push_back(r);
  while (!moveRecording.empty() && moveRecording.front().time < r.time - kRecordSeconds)
    moveRecording.pop_front();
  // Thrown up without a jump: save what led to it, once in a while.
  sinceThrownSave += seconds;
  if (r.velocity.y > kThrownUp && r.time - lastJumpAt > kJumpGrace && sinceThrownSave > 5.0) {
    sinceThrownSave = 0.0;
    spdlog::warn("[recorder] the character shot up at {:.1f} m/s without a jump (standing on {}, "
                 "its velocity {:.1f}, {:.1f}, {:.1f})",
                 r.velocity.y, groundName(r.ground), r.groundVelocity.x, r.groundVelocity.y,
                 r.groundVelocity.z);
    saveMovementRecording("thrown up");
  }
}

std::string VulkanRenderer::saveMovementRecording(const char *why) {
  if (moveRecording.empty())
    return "";
  std::error_code error;
  std::filesystem::create_directories("recordings", error);
  const std::time_t now = std::time(nullptr);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
  const std::string path = std::string("recordings/movement-") + stamp + ".csv";
  std::ofstream out(path);
  if (!out) {
    spdlog::error("[recorder] can't write {}", path);
    return "";
  }
  out << "# Heliora movement recording (" << why << "): the played character, a row a frame\n";
  out << "time,frame_s,x,y,z,vx,vy,vz,ground,ground_body,ground_kind,ground_vx,ground_vy,ground_vz,"
         "normal_x,normal_y,normal_z,jump\n";
  for (const MoveRecord &r : moveRecording) {
    const char *kind = r.groundUserData & kBreakBodyFlag    ? "breakable"
                       : r.groundUserData & kRubbleBodyFlag ? "rubble"
                       : r.groundBody != 0                  ? "body"
                                                            : "";
    out << std::format("{:.4f},{:.4f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{:.3f},{:.3f},"
                       "{:.3f},{:.3f},{:.3f},{:.3f},{}\n",
                       r.time, r.seconds, r.position.x, r.position.y, r.position.z, r.velocity.x,
                       r.velocity.y, r.velocity.z, groundName(r.ground), r.groundBody, kind,
                       r.groundVelocity.x, r.groundVelocity.y, r.groundVelocity.z, r.groundNormal.x,
                       r.groundNormal.y, r.groundNormal.z, r.jump ? 1 : 0);
  }
  spdlog::info("[recorder] the last {:.1f} s of movement saved to {}", moveRecording.back().time -
               moveRecording.front().time, std::filesystem::absolute(path).string());
  return path;
}

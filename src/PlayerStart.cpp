#include "PlayerStart.h"

#include "CharacterMovement.h"
#include "Spawning.h"

#include <glm/gtc/constants.hpp>

#include <cmath>

namespace {

constexpr glm::vec3 kUp(0.0f, 1.0f, 0.0f);
constexpr int kSegments = 24; // per full circle

} // namespace

std::optional<float> rayHitsPlayerStart(const PickRay &ray,
                                        const ScenePlayerStart &start,
                                        float radius, float height) {
  // The rounded ends' centres sit a radius in from the feet and the head.
  return rayHitsCapsule(ray, start.position + kUp * radius,
                        start.position + kUp * (height - radius), radius);
}

void appendPlayerStartLines(std::vector<PhysicsDebugVertex> &lines,
                            const ScenePlayerStart &start, float radius,
                            float height, uint32_t color) {
  const glm::vec3 forward = yawForward(glm::radians(start.yaw));
  const glm::vec3 right = glm::cross(forward, kUp);
  const glm::vec3 low = start.position + kUp * radius;
  const glm::vec3 high = start.position + kUp * (height - radius);
  auto line = [&](glm::vec3 from, glm::vec3 to) {
    lines.push_back({from, color});
    lines.push_back({to, color});
  };
  // The point at angle a on the circle spanned by u and v around centre.
  auto around = [&](glm::vec3 centre, glm::vec3 u, glm::vec3 v, float a) {
    return centre + radius * (std::cos(a) * u + std::sin(a) * v);
  };
  const float step = glm::two_pi<float>() / kSegments;
  for (int i = 0; i < kSegments; ++i) {
    const float a = i * step;
    // A ring where each rounded end meets the side.
    for (const glm::vec3 &centre : {low, high})
      line(around(centre, right, forward, a),
           around(centre, right, forward, a + step));
    // Half circles over the top and under the bottom, front to back and
    // side to side: the first half of the turn goes up, the second down.
    const glm::vec3 &end = i < kSegments / 2 ? high : low;
    for (const glm::vec3 &across : {right, forward})
      line(around(end, across, kUp, a), around(end, across, kUp, a + step));
  }
  // The side, at the front, back, left and right.
  for (const glm::vec3 &out : {right, -right, forward, -forward})
    line(low + out * radius, high + out * radius);

  // An arrow on the ground from the front of the capsule, 2 cm up so the
  // floor doesn't hide it.
  const glm::vec3 ground = start.position + kUp * 0.02f;
  const glm::vec3 tip = ground + forward * (radius + 0.5f);
  line(ground + forward * radius, tip);
  line(tip, tip - forward * 0.15f + right * 0.1f);
  line(tip, tip - forward * 0.15f - right * 0.1f);
}

ScenePlayerStart playerStartAt(const PickRay &ray,
                               const std::optional<PhysicsRayHit> &hit,
                               float radius, float height) {
  ScenePlayerStart start;
  start.position = spawnPosition(ray, hit, glm::vec3(-radius, 0.0f, -radius),
                                 glm::vec3(radius, height, radius));
  // The yaw whose yawForward, (-sin, 0, -cos), is the ray's way along the
  // ground. Looking straight down there's none, so it faces -z.
  const glm::vec2 along(ray.direction.x, ray.direction.z);
  if (glm::dot(along, along) > 1e-8f)
    start.yaw = glm::degrees(std::atan2(-along.x, -along.y));
  return start;
}

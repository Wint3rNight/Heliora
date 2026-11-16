#include "CharacterController.h"
#include "PlayerStart.h"
#include "TestRunner.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr float kRadius = 0.3f;
constexpr float kHeight = 1.8f;

ScenePlayerStart startAt(glm::vec3 feet, float yawDegrees) {
  ScenePlayerStart start;
  start.position = feet;
  start.yaw = yawDegrees;
  return start;
}

PhysicsRayHit surface(float distance, glm::vec3 normal) {
  PhysicsRayHit hit{};
  hit.distance = distance;
  hit.normal = normal;
  return hit;
}

// The way a start's yaw faces, flat on the ground.
glm::vec3 facing(const ScenePlayerStart &start) {
  return yawForward(glm::radians(start.yaw));
}

} // namespace

TEST(clicking_the_player_start_picks_it) {
  const ScenePlayerStart start = startAt(glm::vec3(2.0f, 1.0f, 0.0f), 30.0f);
  // At chest height from the side, and from above onto its head.
  const PickRay side{glm::vec3(2.0f, 2.0f, 10.0f), glm::vec3(0, 0, -1)};
  CHECK_NEAR(rayHitsPlayerStart(side, start, kRadius, kHeight).value_or(-1.0f),
             9.7, 1e-4);
  const PickRay down{glm::vec3(2.0f, 10.0f, 0.0f), glm::vec3(0, -1, 0)};
  CHECK_NEAR(rayHitsPlayerStart(down, start, kRadius, kHeight).value_or(-1.0f),
             10.0 - 2.8, 1e-4);
  // Just over its head, and beside it.
  const PickRay over{glm::vec3(2.0f, 2.85f, 10.0f), glm::vec3(0, 0, -1)};
  CHECK(!rayHitsPlayerStart(over, start, kRadius, kHeight));
  const PickRay beside{glm::vec3(2.5f, 2.0f, 10.0f), glm::vec3(0, 0, -1)};
  CHECK(!rayHitsPlayerStart(beside, start, kRadius, kHeight));
}

TEST(player_start_outline_is_the_characters_size) {
  const ScenePlayerStart start = startAt(glm::vec3(1.0f, 2.0f, 3.0f), 0.0f);
  std::vector<PhysicsDebugVertex> lines;
  appendPlayerStartLines(lines, start, kRadius, kHeight, 0xffffffffu);
  CHECK(!lines.empty());
  CHECK(lines.size() % 2 == 0); // pairs of ends
  float low = std::numeric_limits<float>::max();
  float high = std::numeric_limits<float>::lowest();
  for (const PhysicsDebugVertex &v : lines) {
    low = std::min(low, v.position.y);
    high = std::max(high, v.position.y);
    CHECK(v.color == 0xffffffffu);
  }
  CHECK_NEAR(low, 2.0, 1e-4);  // from the feet
  CHECK_NEAR(high, 3.8, 1e-4); // to the top of the head
}

TEST(player_start_arrow_points_where_it_faces) {
  // Whichever way it's turned, the point furthest out from the capsule's
  // axis is the arrow's tip, out in front.
  for (const float yaw : {0.0f, 90.0f, -135.0f}) {
    const ScenePlayerStart start = startAt(glm::vec3(-4.0f, 0.0f, 1.0f), yaw);
    std::vector<PhysicsDebugVertex> lines;
    appendPlayerStartLines(lines, start, kRadius, kHeight, 0u);
    glm::vec3 tip(0.0f);
    float furthest = 0.0f;
    for (const PhysicsDebugVertex &v : lines) {
      const glm::vec3 out = v.position - start.position;
      const float distance = std::hypot(out.x, out.z);
      if (distance > furthest) {
        furthest = distance;
        tip = glm::vec3(out.x, 0.0f, out.z);
      }
    }
    CHECK(furthest > 2.0f * kRadius); // well clear of the capsule
    CHECK_NEAR(glm::dot(glm::normalize(tip), facing(start)), 1.0, 1e-4);
  }
}

TEST(player_start_lands_on_the_floor_facing_away_from_the_camera) {
  // The camera looks down at 45 degrees towards +x.
  const PickRay ray{glm::vec3(0.0f, 4.0f, 1.0f),
                    glm::normalize(glm::vec3(1.0f, -1.0f, 0.0f))};
  const ScenePlayerStart start =
      playerStartAt(ray, surface(4.0f * std::sqrt(2.0f), glm::vec3(0, 1, 0)),
                    kRadius, kHeight);
  CHECK_NEAR(glm::distance(start.position, glm::vec3(4.0f, 0.0f, 1.0f)), 0.0,
             1e-4);
  CHECK_NEAR(glm::distance(facing(start), glm::vec3(1, 0, 0)), 0.0, 1e-4);
}

TEST(player_start_stands_off_a_wall) {
  const PickRay ray{glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0, 0, -1)};
  const ScenePlayerStart start =
      playerStartAt(ray, surface(5.0f, glm::vec3(0, 0, 1)), kRadius, kHeight);
  CHECK_NEAR(start.position.z, -5.0 + kRadius, 1e-4);
  CHECK_NEAR(glm::distance(facing(start), glm::vec3(0, 0, -1)), 0.0, 1e-4);
}

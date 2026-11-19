#include "HitZones.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace {

// A man 1.78 m to the crown in a T-pose, facing +z with his left along +x,
// named like a Mixamo rig.
Skeleton standing() {
  Skeleton s;
  auto add = [&](const char *name, const char *parent, glm::vec3 offset) {
    Bone b;
    b.name = std::string("mixamorig:") + name;
    b.parent = -1;
    if (parent)
      for (int i = 0; i < static_cast<int>(s.bones.size()); ++i)
        if (s.bones[i].name == std::string("mixamorig:") + parent)
          b.parent = i;
    b.bind.translation = offset;
    s.bones.push_back(b);
  };
  add("Hips", nullptr, {0.0f, 1.0f, 0.0f});
  add("Spine", "Hips", {0.0f, 0.1f, 0.0f});
  add("Spine1", "Spine", {0.0f, 0.12f, 0.0f});
  add("Spine2", "Spine1", {0.0f, 0.12f, 0.0f});
  add("Neck", "Spine2", {0.0f, 0.16f, 0.0f});
  add("Head", "Neck", {0.0f, 0.08f, 0.0f});
  add("HeadTop_End", "Head", {0.0f, 0.2f, 0.0f});
  for (float side : {1.0f, -1.0f}) {
    const std::string s1 = side > 0.0f ? "Left" : "Right";
    add((s1 + "Shoulder").c_str(), "Spine2", {0.06f * side, 0.1f, 0.0f});
    add((s1 + "Arm").c_str(), (s1 + "Shoulder").c_str(), {0.12f * side, 0.0f, 0.0f});
    add((s1 + "ForeArm").c_str(), (s1 + "Arm").c_str(), {0.28f * side, 0.0f, 0.0f});
    add((s1 + "Hand").c_str(), (s1 + "ForeArm").c_str(), {0.25f * side, 0.0f, 0.0f});
    add((s1 + "UpLeg").c_str(), "Hips", {0.1f * side, -0.05f, 0.0f});
    add((s1 + "Leg").c_str(), (s1 + "UpLeg").c_str(), {0.0f, -0.45f, 0.0f});
    add((s1 + "Foot").c_str(), (s1 + "Leg").c_str(), {0.0f, -0.42f, 0.0f});
  }
  return s;
}

std::vector<HitCapsule> placed(const Skeleton &s, const glm::mat4 &where = glm::mat4(1.0f)) {
  return placeHitCapsules(findHitCapsules(s), modelSpace(s, bindPose(s)), where);
}

// A shot along -z, from 5 m in front of him, at this height and side.
std::optional<CharacterHit> shotAt(const std::vector<HitCapsule> &capsules, float x, float y) {
  return firstHit(capsules, glm::vec3(x, y, 5.0f), glm::vec3(x, y, -5.0f));
}

} // namespace

TEST(a_ray_through_a_capsules_middle_enters_and_leaves_its_sides) {
  // A capsule along y, radius 0.1: a ray along x through its axis crosses
  // 0.2 m of it; 6 cm off the axis, a chord of 2 sqrt(0.1^2 - 0.06^2).
  const glm::vec3 a(0.0f, 0.0f, 0.0f), b(0.0f, 1.0f, 0.0f);
  auto through = rayCapsule({-1.0f, 0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, 5.0f, a, b, 0.1f);
  CHECK(through.has_value());
  if (through) {
    CHECK_NEAR(through->entry, 0.9, 1e-5);
    CHECK_NEAR(through->exit, 1.1, 1e-5);
  }
  auto off = rayCapsule({-1.0f, 0.5f, 0.06f}, {1.0f, 0.0f, 0.0f}, 5.0f, a, b, 0.1f);
  CHECK(off.has_value());
  if (off)
    CHECK_NEAR(off->exit - off->entry, 2.0 * std::sqrt(0.01 - 0.0036), 1e-4);
}

TEST(a_ray_meets_a_capsules_rounded_ends_and_runs_along_it) {
  const glm::vec3 a(0.0f, 0.0f, 0.0f), b(0.0f, 1.0f, 0.0f);
  // Straight down onto the top cap, along the axis: in at 1.1, out at -0.1.
  auto down = rayCapsule({0.0f, 2.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, 5.0f, a, b, 0.1f);
  CHECK(down.has_value());
  if (down) {
    CHECK_NEAR(down->entry, 0.9, 1e-5);
    CHECK_NEAR(down->exit, 2.1, 1e-5);
  }
  // Past the top cap's edge: a miss. Too short, or pointing away: a miss.
  CHECK(!rayCapsule({-1.0f, 1.11f, 0.0f}, {1.0f, 0.0f, 0.0f}, 5.0f, a, b, 0.1f));
  CHECK(!rayCapsule({-1.0f, 0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, 0.8f, a, b, 0.1f));
  CHECK(!rayCapsule({-1.0f, 0.5f, 0.0f}, {-1.0f, 0.0f, 0.0f}, 5.0f, a, b, 0.1f));
  // Starting inside: in at 0.
  auto inside = rayCapsule({0.0f, 0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, 5.0f, a, b, 0.1f);
  CHECK(inside.has_value());
  if (inside) {
    CHECK_NEAR(inside->entry, 0.0, 1e-6);
    CHECK_NEAR(inside->exit, 0.1, 1e-5);
  }
}

TEST(the_body_parts_are_found_by_mixamo_names) {
  const std::vector<HitCapsuleDef> defs = findHitCapsules(standing());
  int counts[static_cast<int>(HitZone::Count)] = {};
  for (const HitCapsuleDef &d : defs)
    ++counts[static_cast<int>(d.zone)];
  CHECK(counts[int(HitZone::Head)] == 1);
  CHECK(counts[int(HitZone::Chest)] == 1);
  CHECK(counts[int(HitZone::Belly)] == 1);
  CHECK(counts[int(HitZone::Arm)] == 4);
  CHECK(counts[int(HitZone::Leg)] == 4);
  Skeleton nothing;
  nothing.bones.resize(2);
  CHECK(findHitCapsules(nothing).empty());
}

TEST(a_shot_hits_the_part_it_passes_through) {
  const std::vector<HitCapsule> body = placed(standing());
  struct Case {
    float x, y;
    HitZone zone;
  } cases[] = {{0.0f, 1.68f, HitZone::Head}, {0.0f, 1.35f, HitZone::Chest},
               {0.0f, 1.1f, HitZone::Belly}, {0.6f, 1.44f, HitZone::Arm},
               {-0.3f, 1.44f, HitZone::Arm}, {0.1f, 0.7f, HitZone::Leg},
               {-0.1f, 0.3f, HitZone::Leg}};
  for (const Case &c : cases) {
    const std::optional<CharacterHit> hit = shotAt(body, c.x, c.y);
    CHECK(hit.has_value());
    if (hit)
      CHECK(hit->zone == c.zone);
  }
  CHECK(!shotAt(body, 0.0f, 1.95f)); // over his head
  CHECK(!shotAt(body, 0.4f, 0.7f));  // between his legs and arm
}

TEST(a_shot_through_the_chest_says_where_and_how_much_body_it_crossed) {
  const std::vector<HitCapsule> body = placed(standing());
  const std::optional<CharacterHit> hit = shotAt(body, 0.0f, 1.35f);
  CHECK(hit.has_value());
  if (!hit)
    return;
  CHECK_NEAR(hit->point.z, kChestRadius, 1e-4);
  CHECK_NEAR(hit->distance, 5.0 - kChestRadius, 1e-4);
  CHECK(glm::distance(hit->normal, glm::vec3(0.0f, 0.0f, 1.0f)) < 1e-4f);
  CHECK_NEAR(hit->thickness, 2.0 * kChestRadius, 1e-4);
}

TEST(the_capsules_follow_where_he_stands_and_faces) {
  // Turned to face +x and moved: a shot along -x at his chest hits it.
  const glm::mat4 where = glm::rotate(glm::translate(glm::mat4(1.0f), {10.0f, 0.0f, -3.0f}),
                                      glm::radians(90.0f), glm::vec3(0, 1, 0));
  const std::vector<HitCapsule> body = placed(standing(), where);
  const std::optional<CharacterHit> hit =
      firstHit(body, glm::vec3(20.0f, 1.35f, -3.0f), glm::vec3(0.0f, 1.35f, -3.0f));
  CHECK(hit.has_value());
  if (hit) {
    CHECK(hit->zone == HitZone::Chest);
    CHECK_NEAR(hit->point.x, 10.0 + kChestRadius, 1e-3);
  }
}

TEST(a_wound_takes_health_by_energy_and_where_it_lands) {
  // The same energy hurts most in the head and least in a limb; any of
  // our rounds through the head kills.
  CHECK(woundPoints(HitZone::Head, 100.0f) > woundPoints(HitZone::Chest, 100.0f));
  CHECK(woundPoints(HitZone::Chest, 100.0f) > woundPoints(HitZone::Belly, 100.0f));
  CHECK(woundPoints(HitZone::Belly, 100.0f) > woundPoints(HitZone::Leg, 100.0f));
  CHECK(woundPoints(HitZone::Leg, 100.0f) > woundPoints(HitZone::Arm, 100.0f));
  CHECK(woundPoints(HitZone::Head, 150.0f) >= Health{}.points);
  CHECK_NEAR(woundPoints(HitZone::Chest, 0.0f), 0.0, 1e-6);
  Health h;
  CHECK(!h.wound(40.0f));
  CHECK_NEAR(h.points, 60.0, 1e-5);
  CHECK(h.wound(70.0f)); // this one killed him
  CHECK(h.dead());
  CHECK(!h.wound(10.0f)); // dead already: not again
  CHECK_NEAR(h.points, 0.0, 1e-6);
}

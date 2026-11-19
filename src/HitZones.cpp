#include "HitZones.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <string_view>

namespace {

// The bone named this after Mixamo's prefix ("mixamorig:Head"), or -1.
int boneNamed(const Skeleton &skeleton, std::string_view name) {
  for (int i = 0; i < static_cast<int>(skeleton.bones.size()); ++i) {
    const std::string_view bone = skeleton.bones[i].name;
    if (bone.size() < name.size() || bone.substr(bone.size() - name.size()) != name)
      continue;
    if (bone.size() == name.size() ||
        !std::isalnum(static_cast<unsigned char>(bone[bone.size() - name.size() - 1])))
      return i;
  }
  return -1;
}

glm::vec3 closestOnSegment(glm::vec3 a, glm::vec3 b, glm::vec3 p) {
  const glm::vec3 ab = b - a;
  const float length2 = glm::dot(ab, ab);
  const float t = length2 > 0.0f ? std::clamp(glm::dot(p - a, ab) / length2, 0.0f, 1.0f) : 0.0f;
  return a + ab * t;
}

} // namespace

std::vector<HitCapsuleDef> findHitCapsules(const Skeleton &skeleton) {
  struct Want {
    HitZone zone;
    const char *from, *to;
    float radius;
    bool inset;
  };
  const Want wants[] = {
      {HitZone::Head, "Head", "HeadTop_End", kHeadRadius, true},
      {HitZone::Chest, "Spine1", "Neck", kChestRadius, false},
      {HitZone::Belly, "Hips", "Spine1", kBellyRadius, false},
      {HitZone::Arm, "LeftArm", "LeftForeArm", kUpperArmRadius, false},
      {HitZone::Arm, "LeftForeArm", "LeftHand", kForearmRadius, false},
      {HitZone::Arm, "RightArm", "RightForeArm", kUpperArmRadius, false},
      {HitZone::Arm, "RightForeArm", "RightHand", kForearmRadius, false},
      {HitZone::Leg, "LeftUpLeg", "LeftLeg", kThighRadius, false},
      {HitZone::Leg, "LeftLeg", "LeftFoot", kShinRadius, false},
      {HitZone::Leg, "RightUpLeg", "RightLeg", kThighRadius, false},
      {HitZone::Leg, "RightLeg", "RightFoot", kShinRadius, false},
  };
  std::vector<HitCapsuleDef> defs;
  for (const Want &w : wants) {
    const int from = boneNamed(skeleton, w.from), to = boneNamed(skeleton, w.to);
    if (from >= 0 && to >= 0)
      defs.push_back({w.zone, from, to, w.radius, w.inset});
  }
  // Without a head and a chest it isn't a body we know.
  const bool head = std::any_of(defs.begin(), defs.end(),
                                [](const HitCapsuleDef &d) { return d.zone == HitZone::Head; });
  const bool chest = std::any_of(defs.begin(), defs.end(),
                                 [](const HitCapsuleDef &d) { return d.zone == HitZone::Chest; });
  if (!head || !chest)
    defs.clear();
  return defs;
}

std::vector<HitCapsule> placeHitCapsules(const std::vector<HitCapsuleDef> &defs,
                                         const std::vector<glm::mat4> &model,
                                         const glm::mat4 &modelToWorld) {
  std::vector<HitCapsule> capsules;
  capsules.reserve(defs.size());
  for (const HitCapsuleDef &d : defs) {
    if (d.from < 0 || d.to < 0 || d.from >= static_cast<int>(model.size()) ||
        d.to >= static_cast<int>(model.size()))
      continue;
    HitCapsule c;
    c.zone = d.zone;
    c.radius = d.radius;
    c.a = glm::vec3(modelToWorld * glm::vec4(glm::vec3(model[d.from][3]), 1.0f));
    c.b = glm::vec3(modelToWorld * glm::vec4(glm::vec3(model[d.to][3]), 1.0f));
    if (d.inset) {
      const glm::vec3 along = c.b - c.a;
      const float length = glm::length(along);
      if (length > 2.0f * d.radius)
        c.b -= along / length * d.radius;
      else
        c.b = c.a;
    }
    capsules.push_back(c);
  }
  return capsules;
}

std::optional<RaySpan> rayCapsule(glm::vec3 origin, glm::vec3 dir, float length, glm::vec3 a,
                                  glm::vec3 b, float radius) {
  // The capsule is its cylinder and its two end spheres; it's convex, so
  // the ray is inside it for one span: the earliest way in of the three to
  // the latest way out.
  float entry = INFINITY, exit = -INFINITY;
  const glm::vec3 ba = b - a, oa = origin - a;
  const float baba = glm::dot(ba, ba), bard = glm::dot(ba, dir), baoa = glm::dot(ba, oa);
  const float k2 = baba - bard * bard;
  if (baba > 0.0f && k2 > 1e-9f * baba) {
    const float k1 = baba * glm::dot(oa, dir) - baoa * bard;
    const float k0 = baba * glm::dot(oa, oa) - baoa * baoa - radius * radius * baba;
    const float h = k1 * k1 - k2 * k0;
    if (h >= 0.0f) {
      const float root = std::sqrt(h);
      const float t0 = (-k1 - root) / k2, t1 = (-k1 + root) / k2;
      const float y0 = baoa + t0 * bard, y1 = baoa + t1 * bard;
      if (y0 >= 0.0f && y0 <= baba)
        entry = std::min(entry, t0);
      if (y1 >= 0.0f && y1 <= baba)
        exit = std::max(exit, t1);
    }
  }
  for (const glm::vec3 &centre : {a, b}) {
    const glm::vec3 oc = origin - centre;
    const float half = glm::dot(oc, dir);
    const float h = half * half - (glm::dot(oc, oc) - radius * radius);
    if (h < 0.0f)
      continue;
    const float root = std::sqrt(h);
    entry = std::min(entry, -half - root);
    exit = std::max(exit, -half + root);
  }
  if (!(exit >= 0.0f) || !(entry <= length) || entry > exit)
    return std::nullopt;
  return RaySpan{std::max(entry, 0.0f), exit};
}

std::optional<CharacterHit> firstHit(const std::vector<HitCapsule> &capsules, glm::vec3 from,
                                     glm::vec3 to) {
  const glm::vec3 d = to - from;
  const float length = glm::length(d);
  if (length < 1e-6f)
    return std::nullopt;
  const glm::vec3 dir = d / length;
  std::optional<CharacterHit> best;
  for (size_t i = 0; i < capsules.size(); ++i) {
    const HitCapsule &c = capsules[i];
    const std::optional<RaySpan> span = rayCapsule(from, dir, length, c.a, c.b, c.radius);
    if (!span || (best && span->entry >= best->distance))
      continue;
    CharacterHit hit;
    hit.capsule = static_cast<int>(i);
    hit.zone = c.zone;
    hit.distance = span->entry;
    hit.point = from + dir * span->entry;
    const glm::vec3 out = hit.point - closestOnSegment(c.a, c.b, hit.point);
    hit.normal = glm::length(out) > 1e-6f ? glm::normalize(out) : -dir;
    hit.thickness = span->exit - span->entry;
    best = hit;
  }
  return best;
}

float woundPoints(HitZone zone, float energy) {
  float weight = 1.0f;
  switch (zone) {
  case HitZone::Head: weight = 10.0f; break;
  case HitZone::Chest: weight = 1.5f; break;
  case HitZone::Belly: weight = 1.0f; break;
  case HitZone::Arm: weight = 0.4f; break;
  case HitZone::Leg: weight = 0.5f; break;
  case HitZone::Count: break;
  }
  return std::max(energy, 0.0f) / 10.0f * weight;
}

bool Health::wound(float amount) {
  if (dead() || !(amount > 0.0f))
    return false;
  points = std::max(0.0f, points - amount);
  return dead();
}

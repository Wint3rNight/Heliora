#pragma once

#include "Skeleton.h"

#include <optional>
#include <vector>

// Where a character can be hit (step 1.6): a capsule on each body part,
// following his bones, tested against bullets' paths; and his health. Until
// step 1.9 brings real wounding (the FBI's 1989 report), how much a hit
// hurts is an estimate.

enum class HitZone { Head, Chest, Belly, Arm, Leg, Count };

// Capsule radii, m (est., from typical adult men's sizes: head breadth
// about 15.5 cm, chest about 33 by 25 cm, thigh 59 cm round, calf 39 cm,
// upper arm 31 cm, forearm 28 cm; to check against ANSUR II).
constexpr float kHeadRadius = 0.09f;
constexpr float kChestRadius = 0.15f;
constexpr float kBellyRadius = 0.14f;
constexpr float kUpperArmRadius = 0.05f;
constexpr float kForearmRadius = 0.045f;
constexpr float kThighRadius = 0.09f;
constexpr float kShinRadius = 0.06f;

// A capsule from one bone's joint to another's. inset pulls the end at `to`
// in by the radius, so the round end stops at that joint (the crown).
struct HitCapsuleDef {
  HitZone zone = HitZone::Chest;
  int from = -1, to = -1;
  float radius = 0.1f;
  bool inset = false;
};
// The capsules of a Mixamo-named skeleton (head, chest, belly, two on each
// arm and leg); empty if it isn't one.
std::vector<HitCapsuleDef> findHitCapsules(const Skeleton &skeleton);

// A capsule in the world.
struct HitCapsule {
  HitZone zone = HitZone::Chest;
  glm::vec3 a{0.0f}, b{0.0f};
  float radius = 0.1f;
};
// The capsules where the pose puts the bones (model: modelSpace's matrices),
// placed by modelToWorld; radii stay in metres whatever the model's scale.
std::vector<HitCapsule> placeHitCapsules(const std::vector<HitCapsuleDef> &defs,
                                         const std::vector<glm::mat4> &model,
                                         const glm::mat4 &modelToWorld);

// Where a ray (dir of unit length) is inside a capsule: in and out, metres
// along it, in at 0 if it starts inside; nothing if it misses before length.
struct RaySpan {
  float entry = 0.0f, exit = 0.0f;
};
std::optional<RaySpan> rayCapsule(glm::vec3 origin, glm::vec3 dir, float length, glm::vec3 a,
                                  glm::vec3 b, float radius);

// The first body part on the way from `from` to `to`.
struct CharacterHit {
  int capsule = 0;
  HitZone zone = HitZone::Chest;
  float distance = 0.0f; // m from `from`
  glm::vec3 point{0.0f};
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
  float thickness = 0.0f; // m of that part along the path
};
std::optional<CharacterHit> firstHit(const std::vector<HitCapsule> &capsules, glm::vec3 from,
                                     glm::vec3 to);

// The first body part a ball of this radius (a fist, a foot) touches on its
// way from `from` to `to`; still, whether it's touching one where it is.
// The point is on the capsule's surface, widened by the radius.
std::optional<CharacterHit> sweptHit(const std::vector<HitCapsule> &capsules, glm::vec3 from,
                                     glm::vec3 to, float radius);

// How far to move a set of points (the hands on a gun) so each is at least
// margin m outside every capsule: pushed out the way the deepest one
// points, a few times over; or, given `along` (unit), only that way (a gun
// carried in front goes forward, not sideways off the body). Zero if
// they're all clear.
glm::vec3 clearOfBody(const std::vector<HitCapsule> &body, const std::vector<glm::vec3> &points,
                      float margin, glm::vec3 along = glm::vec3(0.0f));

// Health points a wound takes: the energy the bullet left in him (J) over 10,
// times its zone's weight (est.: head 10, chest 1.5, belly 1, leg 0.5,
// arm 0.4), so anything through the head kills.
float woundPoints(HitZone zone, float energy);

struct Health {
  float points = 100.0f;
  // True for the wound that kills; a dead man takes no more.
  bool wound(float amount);
  bool dead() const { return points <= 0.0f; }
};

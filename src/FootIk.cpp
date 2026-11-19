#include "FootIk.h"

#include "PoseBlend.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <string_view>

namespace {

const glm::vec3 kUp(0.0f, 1.0f, 0.0f);

// A planted foot's ankle is within this of its standing height; lifted past
// kLifted it's off the ground (the gap stops one landing counting twice).
constexpr float kPlanted = 0.02f; // m
constexpr float kLifted = 0.05f;  // m

// The bone named this after Mixamo's prefix ("mixamorig:LeftFoot"), or -1.
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

glm::vec3 originOf(const glm::mat4 &m) { return glm::vec3(m[3]); }

// A model matrix's rotation, its scale taken out.
glm::quat rotationOf(const glm::mat4 &m) {
  return glm::normalize(glm::quat_cast(glm::mat3(glm::normalize(glm::vec3(m[0])),
                                                 glm::normalize(glm::vec3(m[1])),
                                                 glm::normalize(glm::vec3(m[2])))));
}

glm::vec3 toWorld(const FootPlacement &p, glm::vec3 local) {
  return p.feet + p.facing * (p.scale * local);
}

// A world-space offset in the skeleton's space.
glm::vec3 offsetToModel(const FootPlacement &p, glm::vec3 offset) {
  return glm::inverse(p.facing) * offset / std::max(p.scale, 1e-9f);
}

// Holden's exact damper: the share of the gap closed in dt for a half-life.
float closeShare(float dt, float halfLife) {
  if (halfLife <= 0.0f)
    return 1.0f;
  return 1.0f - std::exp(-0.69314718f * dt / halfLife);
}

} // namespace

bool FootIkRig::valid() const {
  for (int k = 0; k < 2; ++k)
    if (upLeg[k] < 0 || leg[k] < 0 || foot[k] < 0)
      return false;
  return hips >= 0;
}

FootIkRig findFootIkRig(const Skeleton &skeleton) {
  FootIkRig rig;
  rig.hips = boneNamed(skeleton, "Hips");
  const char *sides[2] = {"Left", "Right"};
  for (int k = 0; k < 2; ++k) {
    const std::string side = sides[k];
    rig.upLeg[k] = boneNamed(skeleton, side + "UpLeg");
    rig.leg[k] = boneNamed(skeleton, side + "Leg");
    rig.foot[k] = boneNamed(skeleton, side + "Foot");
  }
  if (rig.valid()) {
    const std::vector<glm::mat4> bind = modelSpace(skeleton, bindPose(skeleton));
    rig.ankleHeight = 0.5f * (originOf(bind[rig.foot[0]]).y + originOf(bind[rig.foot[1]]).y);
  }
  return rig;
}

std::array<FootRay, 2> footRays(const FootIkRig &rig, const std::vector<glm::mat4> &model,
                                const FootPlacement &placement, const FootIkTuning &tuning) {
  std::array<FootRay, 2> rays{};
  if (!rig.valid())
    return rays;
  for (int k = 0; k < 2; ++k) {
    const glm::vec3 ankle = toWorld(placement, originOf(model[rig.foot[k]]));
    rays[k].from = ankle + kUp * tuning.probeHeight;
    rays[k].to = glm::vec3(ankle.x, placement.feet.y - tuning.reach, ankle.z);
  }
  return rays;
}

void solveFootIk(const Skeleton &skeleton, const FootIkRig &rig, const FootPlacement &placement,
                 const std::array<GroundHit, 2> &hits, bool on, float dt,
                 const FootIkTuning &tuning, FootIkState &state, Pose &pose) {
  state.stepped = {false, false};
  if (!rig.valid() || static_cast<int>(pose.local.size()) != static_cast<int>(skeleton.bones.size()))
    return;
  const bool following = on && tuning.enabled;
  const std::vector<glm::mat4> model = modelSpace(skeleton, pose);
  const float restHeight = rig.ankleHeight * placement.scale;

  // --- What the ground asks for, and where each foot is in the clip.
  std::array<float, 2> wanted{};
  std::array<glm::vec3, 2> wantedNormal{kUp, kUp};
  std::array<float, 2> plantedShare{};
  std::array<glm::vec3, 2> ankle{};
  for (int k = 0; k < 2; ++k) {
    ankle[k] = originOf(model[rig.foot[k]]);
    // How far the clip has lifted the foot off its standing height.
    const float lift = toWorld(placement, ankle[k]).y - placement.feet.y - restHeight;
    plantedShare[k] = std::clamp(1.0f - lift / kLifted, 0.0f, 1.0f);
    if (state.planted[k] && lift > kLifted) {
      state.planted[k] = false;
    } else if (!state.planted[k] && lift < kPlanted) {
      state.planted[k] = true;
      // Not gated on the IK's weight: the first foot down after a landing
      // counts while the IK is still fading back in.
      state.stepped[k] = following && hits[k].hit;
    }
    if (following && hits[k].hit) {
      wanted[k] = std::clamp(hits[k].point.y - placement.feet.y, -tuning.reach, tuning.reach);
      wantedNormal[k] = glm::normalize(hits[k].normal);
    }
  }
  // The hips only ever come down, by the lowest foot's drop.
  const float wantedDrop = std::min(0.0f, std::min(wanted[0], wanted[1]));

  // --- Damped toward it by the half-life; the first frame starts there.
  const float share = state.started ? closeShare(dt, tuning.halfLife) : 1.0f;
  state.started = true;
  state.weight += ((following ? 1.0f : 0.0f) - state.weight) * share;
  state.hipsDrop += (wantedDrop - state.hipsDrop) * share;
  for (int k = 0; k < 2; ++k) {
    state.correction[k] += (wanted[k] - state.correction[k]) * share;
    state.normal[k] = glm::normalize(state.normal[k] + (wantedNormal[k] - state.normal[k]) * share);
  }

  const float w = state.weight;
  const float drop = w * state.hipsDrop;
  const std::array<float, 2> correction = {w * state.correction[0], w * state.correction[1]};
  std::array<float, 2> tilt{};
  std::array<glm::vec3, 2> tiltAxis{kUp, kUp};
  bool anything = std::abs(drop) > 1e-6f;
  for (int k = 0; k < 2; ++k) {
    anything = anything || std::abs(correction[k]) > 1e-6f;
    const glm::vec3 normal = glm::normalize(glm::inverse(placement.facing) * state.normal[k]);
    const float angle = std::acos(std::clamp(glm::dot(kUp, normal), -1.0f, 1.0f));
    const glm::vec3 axis = glm::cross(kUp, normal);
    if (angle > 1e-5f && glm::length(axis) > 1e-7f) {
      tilt[k] = std::min(angle, tuning.tilt) * w * plantedShare[k];
      tiltAxis[k] = glm::normalize(axis);
      anything = anything || tilt[k] > 1e-6f;
    }
  }
  if (!anything)
    return;

  // --- The hips down, through their parent's frame.
  const int hipsParent = skeleton.bones[rig.hips].parent;
  const glm::vec3 hipsShift = offsetToModel(placement, kUp * drop);
  pose.local[rig.hips].translation +=
      hipsParent >= 0 ? glm::inverse(glm::mat3(model[hipsParent])) * hipsShift : hipsShift;

  // --- Each leg reaches its foot's ground, the foot keeping the turn it had
  // in the clip, then turned to the slope.
  for (int k = 0; k < 2; ++k) {
    const glm::quat footTurn = rotationOf(model[rig.foot[k]]);
    const glm::vec3 hip = originOf(model[rig.upLeg[k]]);
    const glm::vec3 knee = originOf(model[rig.leg[k]]);
    glm::vec3 pole = knee - 0.5f * (hip + ankle[k]);
    if (glm::length(pole) < 1e-6f)
      pole = glm::vec3(0.0f, 0.0f, 1.0f); // Mixamo characters face +z
    const glm::vec3 target = ankle[k] + offsetToModel(placement, kUp * correction[k]);
    twoBoneIk(skeleton, pose, rig.upLeg[k], rig.leg[k], rig.foot[k], target, glm::normalize(pole));

    const glm::quat slope = glm::angleAxis(tilt[k], tiltAxis[k]);
    const std::vector<glm::mat4> after = modelSpace(skeleton, pose);
    const glm::quat parent = rotationOf(after[rig.leg[k]]);
    pose.local[rig.foot[k]].rotation = glm::normalize(glm::inverse(parent) * slope * footTurn);
  }
}

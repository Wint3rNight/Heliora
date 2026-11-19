#include "FootIk.h"
#include "PoseBlend.h"
#include "TestRunner.h"

#include <glm/gtc/quaternion.hpp>

#include <cmath>

namespace {

constexpr float kDt = 1.0f / 60.0f;
const glm::vec3 kUp(0.0f, 1.0f, 0.0f);

enum : int { kArmature, kHips, kLUp, kLLeg, kLFoot, kLToe, kRUp, kRLeg, kRFoot, kRToe };

// Legs 0.9 m long, the knees 3 cm forward, the ankles 5 cm over the floor
// and the hips 1 m up, in the skeleton's space. unit scales the bones (100
// for a rig in centimetres). The armature is turned a quarter about z and
// the hips turned back, so the hips' own frame isn't the skeleton's: moving
// them down has to go through their parent.
Skeleton legRig(float unit = 1.0f) {
  Skeleton s;
  auto add = [&](const char *name, int parent, glm::vec3 t) {
    Bone b;
    b.name = name;
    b.parent = parent;
    b.bind.translation = t * unit;
    s.bones.push_back(b);
  };
  const glm::quat quarter = glm::angleAxis(glm::radians(90.0f), glm::vec3(0, 0, 1));
  add("Armature", -1, glm::vec3(0.0f));
  s.bones.back().bind.rotation = quarter;
  add("mixamorig:Hips", kArmature, glm::inverse(quarter) * glm::vec3(0.0f, 1.0f, 0.0f));
  s.bones.back().bind.rotation = glm::inverse(quarter);
  for (float side : {1.0f, -1.0f}) {
    const bool left = side > 0.0f;
    const int up = left ? kLUp : kRUp;
    add(left ? "mixamorig:LeftUpLeg" : "mixamorig:RightUpLeg", kHips,
        glm::vec3(0.1f * side, -0.05f, 0.0f));
    add(left ? "mixamorig:LeftLeg" : "mixamorig:RightLeg", up, glm::vec3(0.0f, -0.45f, 0.03f));
    add(left ? "mixamorig:LeftFoot" : "mixamorig:RightFoot", up + 1,
        glm::vec3(0.0f, -0.45f, -0.03f));
    add(left ? "mixamorig:LeftToeBase" : "mixamorig:RightToeBase", up + 2,
        glm::vec3(0.0f, -0.04f, 0.12f));
  }
  return s;
}

GroundHit ground(float y, glm::vec3 normal = kUp) {
  GroundHit h;
  h.hit = true;
  h.point = glm::vec3(0.0f, y, 0.0f);
  h.normal = glm::normalize(normal);
  return h;
}

glm::vec3 world(const FootPlacement &p, glm::vec3 local) {
  return p.feet + p.facing * (p.scale * local);
}

glm::vec3 boneWorld(const Skeleton &s, const Pose &pose, const FootPlacement &p, int bone) {
  return world(p, glm::vec3(modelSpace(s, pose)[bone][3]));
}

// Solves for this long against the same ground, from the bind pose each
// frame (the clips would give a fresh pose each frame too).
Pose settle(const Skeleton &s, const FootIkRig &rig, const FootPlacement &p,
            std::array<GroundHit, 2> hits, FootIkState &state, float seconds = 2.0f,
            bool on = true, FootIkTuning tuning = {}) {
  Pose pose = bindPose(s);
  for (int i = 0; i < int(std::lround(seconds / kDt)); ++i) {
    pose = bindPose(s);
    solveFootIk(s, rig, p, hits, on, kDt, tuning, state, pose);
  }
  return pose;
}

// How far a foot's up has turned from straight up, degrees.
float footTilt(const Skeleton &s, const Pose &pose, const FootPlacement &p, int foot) {
  const glm::mat3 m = glm::mat3(modelSpace(s, pose)[foot]);
  const glm::vec3 up = glm::normalize(p.facing * glm::normalize(m[1]));
  return glm::degrees(std::acos(std::min(1.0f, up.y)));
}

} // namespace

TEST(the_leg_bones_are_found_by_mixamo_names) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  CHECK(rig.valid());
  CHECK(rig.hips == kHips);
  CHECK(rig.upLeg[0] == kLUp && rig.leg[0] == kLLeg && rig.foot[0] == kLFoot);
  CHECK(rig.upLeg[1] == kRUp && rig.leg[1] == kRLeg && rig.foot[1] == kRFoot);
  CHECK_NEAR(rig.ankleHeight, 0.05, 1e-5);
  Skeleton noLegs;
  noLegs.bones.resize(1);
  CHECK(!findFootIkRig(noLegs).valid());
}

TEST(the_rays_start_above_each_ankle_and_end_below_the_feet) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootPlacement p;
  p.feet = glm::vec3(5.0f, 2.0f, -3.0f);
  const auto rays = footRays(rig, modelSpace(s, bindPose(s)), p, FootIkTuning{});
  CHECK(glm::distance(rays[0].from, glm::vec3(5.1f, 2.0f + 0.05f + 0.5f, -3.0f)) < 1e-4f);
  CHECK(glm::distance(rays[0].to, glm::vec3(5.1f, 2.0f - 0.45f, -3.0f)) < 1e-4f);
  CHECK(glm::distance(rays[1].from, glm::vec3(4.9f, 2.55f, -3.0f)) < 1e-4f);
}

TEST(flat_ground_leaves_the_pose_as_it_is) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  const Pose pose = settle(s, rig, FootPlacement{}, {ground(0.0f), ground(0.0f)}, state);
  const Pose bind = bindPose(s);
  for (size_t b = 0; b < bind.local.size(); ++b) {
    CHECK(glm::distance(pose.local[b].translation, bind.local[b].translation) < 1e-5f);
    CHECK(std::abs(glm::dot(pose.local[b].rotation, bind.local[b].rotation)) > 1.0f - 1e-6f);
  }
}

TEST(a_step_down_drops_the_hips_and_puts_each_foot_on_its_ground) {
  // Left on the floor, right 20 cm lower: the hips come down 20 cm so the
  // right foot reaches, and the left knee bends to stay on the floor. The
  // same in centimetres, scaled and turned like a Mixamo character.
  for (float unit : {1.0f, 100.0f}) {
    const Skeleton s = legRig(unit);
    const FootIkRig rig = findFootIkRig(s);
    FootPlacement p;
    p.scale = 1.0f / unit;
    if (unit > 1.0f)
      p.facing = glm::angleAxis(glm::radians(90.0f), kUp);
    FootIkState state;
    const Pose pose = settle(s, rig, p, {ground(0.0f), ground(-0.2f)}, state);
    CHECK_NEAR(boneWorld(s, pose, p, kHips).y, 0.8, 1e-3);
    CHECK_NEAR(boneWorld(s, pose, p, kLFoot).y, 0.05, 2e-3);
    CHECK_NEAR(boneWorld(s, pose, p, kRFoot).y, -0.15, 2e-3);
  }
}

TEST(higher_ground_lifts_the_foot_but_never_the_hips) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  const FootPlacement p;
  const Pose pose = settle(s, rig, p, {ground(0.1f), ground(0.0f)}, state);
  CHECK_NEAR(boneWorld(s, pose, p, kHips).y, 1.0, 1e-4);
  CHECK_NEAR(boneWorld(s, pose, p, kLFoot).y, 0.15, 2e-3);
  CHECK_NEAR(boneWorld(s, pose, p, kRFoot).y, 0.05, 2e-3);
  // The knee bent forward, the way it pointed.
  CHECK(boneWorld(s, pose, p, kLLeg).z > 0.05f);
}

TEST(a_planted_foot_turns_to_the_slope_up_to_its_limit) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  const FootPlacement p;
  const glm::vec3 x(1.0f, 0.0f, 0.0f);
  FootIkState a;
  const Pose gentle = settle(s, rig, p,
                             {ground(0.0f, glm::angleAxis(glm::radians(20.0f), x) * kUp),
                              ground(0.0f)},
                             a);
  CHECK_NEAR(footTilt(s, gentle, p, kLFoot), 20.0, 0.5);
  CHECK_NEAR(footTilt(s, gentle, p, kRFoot), 0.0, 0.5);
  FootIkState b;
  const Pose steep = settle(s, rig, p,
                            {ground(0.0f, glm::angleAxis(glm::radians(45.0f), x) * kUp),
                             ground(0.0f)},
                            b);
  CHECK_NEAR(footTilt(s, steep, p, kLFoot), 30.0, 0.5);
}

TEST(ground_past_the_reach_is_followed_only_as_far_as_the_reach) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  const FootPlacement p;
  const Pose pose = settle(s, rig, p, {ground(0.0f), ground(-0.8f)}, state);
  CHECK_NEAR(boneWorld(s, pose, p, kHips).y, 1.0 - 0.45, 1e-3);
}

TEST(a_change_of_ground_gets_half_way_in_the_half_life) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  const FootPlacement p;
  settle(s, rig, p, {ground(0.0f), ground(0.0f)}, state, 0.5f);
  const Pose pose = settle(s, rig, p, {ground(0.0f), ground(-0.2f)}, state, 0.1f);
  CHECK_NEAR(boneWorld(s, pose, p, kHips).y, 0.9, 2e-3);
}

TEST(off_the_ground_the_feet_let_go) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  const FootPlacement p;
  settle(s, rig, p, {ground(0.0f), ground(-0.2f)}, state, 1.0f);
  const Pose pose = settle(s, rig, p, {ground(0.0f), ground(-0.2f)}, state, 2.0f, false);
  CHECK_NEAR(boneWorld(s, pose, p, kHips).y, 1.0, 1e-3);
  CHECK_NEAR(boneWorld(s, pose, p, kRFoot).y, 0.05, 1e-3);
}

TEST(turned_off_in_the_tuning_it_does_nothing) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  FootIkTuning off;
  off.enabled = false;
  const FootPlacement p;
  const Pose pose = settle(s, rig, p, {ground(0.0f), ground(-0.2f)}, state, 1.0f, true, off);
  CHECK_NEAR(boneWorld(s, pose, p, kHips).y, 1.0, 1e-5);
}

TEST(a_foot_coming_down_is_one_footstep) {
  // The left hip swings the foot up off the floor, then back down: one
  // step on the frame it lands, none for the right foot that stayed.
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  const FootPlacement p;
  const std::array<GroundHit, 2> flat = {ground(0.0f), ground(0.0f)};
  settle(s, rig, p, flat, state, 0.5f);
  int leftSteps = 0, rightSteps = 0;
  for (int frame = 0; frame < 40; ++frame) {
    Pose pose = bindPose(s);
    if (frame < 20) // lifted about 12 cm
      pose.local[kLUp].rotation = glm::angleAxis(glm::radians(-30.0f), glm::vec3(1, 0, 0));
    solveFootIk(s, rig, p, flat, true, kDt, FootIkTuning{}, state, pose);
    leftSteps += state.stepped[0] ? 1 : 0;
    rightSteps += state.stepped[1] ? 1 : 0;
    if (frame == 20)
      CHECK(state.stepped[0]);
  }
  CHECK(leftSteps == 1);
  CHECK(rightSteps == 0);
}

TEST(the_first_foot_down_after_a_landing_is_a_footstep) {
  // In the air the IK has let go and the clip lifts the feet; the frame it
  // lands and a foot comes down counts, though the IK is only fading back in.
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  const FootPlacement p;
  const std::array<GroundHit, 2> flat = {ground(0.0f), ground(0.0f)};
  for (int frame = 0; frame < 60; ++frame) {
    Pose pose = bindPose(s);
    pose.local[kLUp].rotation = glm::angleAxis(glm::radians(-30.0f), glm::vec3(1, 0, 0));
    solveFootIk(s, rig, p, flat, false, kDt, FootIkTuning{}, state, pose);
  }
  Pose pose = bindPose(s);
  solveFootIk(s, rig, p, flat, true, kDt, FootIkTuning{}, state, pose);
  CHECK(state.stepped[0]);
}

namespace {

// Running: the hips 10 cm lower and the knees bent, the left foot planted
// ahead of them by reach (m) and the right behind, both on the floor.
Pose stride(const Skeleton &s, float reach) {
  Pose pose = bindPose(s);
  const glm::quat armature = s.bones[kArmature].bind.rotation;
  pose.local[kHips].translation += glm::inverse(armature) * glm::vec3(0.0f, -0.1f, 0.0f);
  twoBoneIk(s, pose, kLUp, kLLeg, kLFoot, glm::vec3(0.1f, 0.05f, reach), glm::vec3(0, 0, 1));
  twoBoneIk(s, pose, kRUp, kRLeg, kRFoot, glm::vec3(-0.1f, 0.05f, -reach), glm::vec3(0, 0, 1));
  return pose;
}

Pose solveStride(const Skeleton &s, const FootIkRig &rig, const FootPlacement &p, float reach) {
  FootIkState state;
  Pose pose = stride(s, reach);
  solveFootIk(s, rig, p, {ground(0.0f), ground(0.0f)}, true, kDt, FootIkTuning{}, state, pose);
  return pose;
}

} // namespace

TEST(a_longer_stride_puts_the_feet_further_from_the_hips) {
  // Stride warping: each foot's distance ahead of or behind the hips along
  // the way it moves grows by the stride's scale, on the same ground.
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootPlacement p;
  p.stride = 1.2f;
  const Pose pose = solveStride(s, rig, p, 0.2f);
  const glm::vec3 left = boneWorld(s, pose, p, kLFoot), right = boneWorld(s, pose, p, kRFoot);
  CHECK_NEAR(left.z, 0.24, 2e-3);
  CHECK_NEAR(right.z, -0.24, 2e-3);
  CHECK_NEAR(left.y, 0.05, 2e-3);
  CHECK_NEAR(left.x, 0.1, 2e-3);
  // Shorter, and along a turned axis: only the part along it scales.
  p.stride = 0.5f;
  p.strideAxis = glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f));
  const Pose shorter = solveStride(s, rig, p, 0.2f);
  const glm::vec3 moved = boneWorld(s, shorter, p, kLFoot) - glm::vec3(0.1f, 0.05f, 0.2f);
  const float along = glm::dot(glm::vec3(0.1f, 0.0f, 0.2f), p.strideAxis);
  CHECK(glm::distance(moved, -0.5f * along * p.strideAxis) < 3e-3f);
}

TEST(the_rays_follow_the_longer_stride) {
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootPlacement p;
  p.stride = 1.2f;
  const auto rays = footRays(rig, modelSpace(s, stride(s, 0.2f)), p, FootIkTuning{});
  CHECK_NEAR(rays[0].from.z, 0.24, 2e-3);
  CHECK_NEAR(rays[1].to.z, -0.24, 2e-3);
}

TEST(a_planted_foot_out_of_reach_brings_the_hips_down) {
  // 54 cm ahead of a hip joint 75 cm over the ankle is past the 0.9 m leg:
  // the hips come down about 3 cm until each planted foot reaches its spot.
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootPlacement p;
  p.stride = 1.8f;
  const Pose pose = solveStride(s, rig, p, 0.3f);
  CHECK(glm::distance(boneWorld(s, pose, p, kLFoot), glm::vec3(0.1f, 0.05f, 0.54f)) < 3e-3f);
  CHECK(glm::distance(boneWorld(s, pose, p, kRFoot), glm::vec3(-0.1f, 0.05f, -0.54f)) < 3e-3f);
  CHECK_NEAR(boneWorld(s, pose, p, kHips).y, 0.85 - 0.029, 3e-3);
}

TEST(a_change_of_stride_eases_in_over_the_half_life) {
  // From 1 to 1.4 at once: half way (1.2) after the half-life.
  const Skeleton s = legRig();
  const FootIkRig rig = findFootIkRig(s);
  FootIkState state;
  FootPlacement p;
  const std::array<GroundHit, 2> flat = {ground(0.0f), ground(0.0f)};
  Pose pose;
  for (int i = 0; i < 30; ++i) {
    pose = stride(s, 0.2f);
    solveFootIk(s, rig, p, flat, true, kDt, FootIkTuning{}, state, pose);
  }
  p.stride = 1.4f;
  for (int i = 0; i < 6; ++i) {
    pose = stride(s, 0.2f);
    solveFootIk(s, rig, p, flat, true, kDt, FootIkTuning{}, state, pose);
  }
  CHECK_NEAR(boneWorld(s, pose, p, kLFoot).z, 0.24, 3e-3);
}

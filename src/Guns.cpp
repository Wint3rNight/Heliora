// VulkanRenderer's guns on characters (step 1.6, part 6c): the gun assets
// (a .weapon file, its parts and their models), the played character's kit,
// and each frame the gun in the hands or put away, the reach for a draw,
// the aim by IK and the off hand on the gun. The maths is heliora_core's
// (Loadout.h, WeaponHold.h, AimIk.h).

#include "VulkanRenderer.h"

#include "PoseBlend.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string_view>

namespace {

const char *kWeaponsFolder = "Resources/Weapons/";
const char *kPistolName = "p38"; // the pistol everyone carries

// From the ready, a gun comes onto the aim point in 0.51 to 0.83 s (Force
// Science): eased in over the shorter.
constexpr float kAimIn = 0.5f; // s

// Where a put-away gun rides, in the character's own frame (+x its left, +y
// up, +z ahead), metres (est., checked by eye): the pistol's trigger on the
// right hip, muzzle down, a little behind the hip joint's side; the long
// gun flat across the back, muzzle up over the left shoulder.
constexpr glm::vec3 kHolster(-0.22f, -0.09f, -0.03f);
constexpr glm::vec3 kSling(0.0f, -0.03f, -0.12f);
constexpr float kSlingMiddle = 0.1f; // m along the gun from its trigger to its middle
// The trigger, m to the left of the right index finger's middle joint (the
// finger lies along the gun's right side), and how high on the gun the left
// palm goes (gun space, the trigger's height: just under a rifle's
// handguard); est.
constexpr float kFingerToTrigger = 0.015f;
constexpr float kPalmHeight = 0.0f;

// A reload's clip eases in and out over its first and last tenth (as the
// animator does); the magazine is out of the gun, in the pouch, for the
// middle of it (est.: no clip of the left hand's own timing yet).
constexpr float kReloadEase = 0.1f;
constexpr float kMagazineOut = 0.3f, kMagazineIn = 0.65f;

// Carried relaxed. The rifle at the Army's low ready (FM 3-22.9: the butt in
// the shoulder pocket, the muzzle down about 45 degrees), a little across
// the body (est.); the pistol hanging in the right hand standing, its
// muzzle down and this much ahead (est.), and crouched at a two-handed low
// ready, the arms lowered from the aim this far (est.).
constexpr float kLowReadyDown = glm::radians(45.0f);
constexpr float kLowReadyAcross = glm::radians(30.0f);
constexpr float kPistolAhead = 0.35f;
constexpr float kPistolLowReady = glm::radians(45.0f);
// Carried relaxed, the hands stay at least this far outside the body's
// capsules (HitZones.h), so they don't sink into the belly (est.: the
// capsules run a little inside the shirt).
constexpr float kBodyMargin = 0.05f; // m

// Aiming. The chest squares to the target: the rifle with its support
// shoulder this far forward (modern Army doctrine, TC 3-22.9, stands
// squared; the angle is est.), the pistol square (the isosceles stance).
// The spine leans with the aim this share of its pitch, the arms doing the
// rest (est.). The rifle's butt goes into the shoulder pocket, in front of
// and inside the shoulder joint (in the chest's frame: +x its left, +y up,
// +z ahead; est.), and the neck brings the eye down to the sight line (gun
// space height; est.). The pistol is held out with its sights at the eye,
// its trigger this far under the eye's line (est.). The head turns to look
// at the aim point.
constexpr float kRifleBlade = glm::radians(10.0f);
// The stance clips stand side-on at the hips too (36 to 57 degrees off the
// aim, measured), which leaves the squared chest twisted over them; the
// hips turn to this far off the aim (est.: a slight blade, the support
// foot forward) and the spine squares the rest.
constexpr float kHipsBlade = glm::radians(25.0f);
constexpr float kMaxHipsTurn = glm::radians(45.0f);
constexpr float kMaxSquare = glm::radians(60.0f);
constexpr float kBendShare = 0.5f;
constexpr float kMaxBend = glm::radians(35.0f);
constexpr glm::vec3 kPocket(0.04f, -0.03f, 0.06f);
constexpr float kRifleSightLine = 0.08f;
constexpr float kNeckTurn = glm::radians(25.0f);
constexpr float kPistolSightDrop = 0.06f;
constexpr float kHeadTurn = glm::radians(30.0f);
// Firing: faster than this he fires on the move (est.); the view shows the
// recoil this many times what the bullets get (est.).
constexpr float kMovingFire = 0.5f; // m/s
// The player comes back this long after dying (the design's 4 s).
constexpr float kComeBack = 4.0f; // s
// A weapons crate is in reach this close to its middle, across the ground
// (est.: half a crate, a step and an arm).
constexpr float kCrateReach = 2.0f; // m
// The kick each shot gives what you see (est.): the camera a third of the
// gun's muzzle rise, the chest rocking back a quarter of it; a pistol turns
// about the web of the hand (gun space, under and behind the trigger), a
// rifle about its butt.
constexpr float kCameraKick = 0.5f;
constexpr float kChestKick = 0.4f;
constexpr glm::vec3 kPistolPivot(0.0f, -0.04f, -0.03f);
// Carried relaxed, the gun follows this share of the chest's turn, this far
// behind (s), and breathes (15 breaths a minute, est.; half a degree).
constexpr float kCarryFollow = 0.5f;
// How far the AK's bolt carrier and the P38's slide run back (est.).
constexpr float kBoltTravel = 0.12f, kSlideTravel = 0.06f; // m
// A carry clip holding the rifle at the ready (level, as the Pro Rifle
// pack's walk and run do) has both arms turned down until it's no higher
// than this (est., the low ready's 45 degrees, FM 3-22.9; Mixamo's "Rifle
// Idle" holds it 31 to 39 down), at most this far.
constexpr float kCarryPitch = glm::radians(-40.0f);
constexpr float kMostLowered = glm::radians(60.0f);
constexpr float kCarryLag = 0.08f;
constexpr float kBreathsPerSecond = 0.25f;
constexpr float kBreathTurn = glm::radians(0.5f);
// A rocket launcher. Its tube sits on top of the right shoulder, nearer the
// neck than the joint (in the chest's frame from the right arm's joint: x
// left, y up, z ahead; est.), and is carried there relaxed with its muzzle
// up a little (est.). It's fired only once it's raised this far.
constexpr glm::vec3 kLauncherPocket(0.07f, 0.09f, -0.02f);
constexpr float kLauncherCarryUp = glm::radians(15.0f);
constexpr float kLauncherRaised = 0.95f;
// Loaded by hand at the muzzle (est., no clip): the tube comes off the
// shoulder to the right hip, held by its grip with the muzzle forward, up
// and across him (so the left hand reaches it); the left hand goes back to
// the pack, brings a rocket to the muzzle and pushes it in, then goes back
// to the rear grip. As shares of the reload: the rocket comes out of the
// pack at kRocketOut, is at the muzzle kRocketCarry later, starting this
// far out, and is home by kRocketIn; each hand move takes kHandMove.
constexpr glm::vec3 kLoadingGrip(0.0f, -0.35f, 0.25f); // from the right arm's joint, chest frame
constexpr float kLoadingAcross = glm::radians(40.0f), kLoadingUp = glm::radians(25.0f);
constexpr float kRocketOut = 0.3f, kRocketCarry = 0.15f, kRocketIn = 0.75f;
constexpr float kRocketStart = 0.35f; // m
constexpr float kHandMove = 0.15f;

// Between two rigid frames: the place along a line, the turn the short way.
glm::mat4 between(const glm::mat4 &a, const glm::mat4 &b, float t) {
  const glm::quat q = glm::slerp(glm::normalize(glm::quat_cast(glm::mat3(a))),
                                 glm::normalize(glm::quat_cast(glm::mat3(b))), t);
  glm::mat4 m = glm::mat4_cast(q);
  m[3] = glm::vec4(glm::mix(glm::vec3(a[3]), glm::vec3(b[3]), t), 1.0f);
  return m;
}

// How far through a reload a gun is, 0 to 1, or below 0 for none.
float reloadProgress(const Weapon &weapon) {
  if (!weapon.reloading())
    return -1.0f;
  const WeaponSpec &spec = weapon.spec();
  const float total = std::max(weapon.emptyReload() ? spec.emptyReload : spec.reload, 1e-3f);
  return std::clamp(1.0f - weapon.reloadLeft() / total, 0.0f, 1.0f);
}

// A path as written or one folder up (the game usually runs from build/).
std::string resolved(const std::string &path) {
  if (std::filesystem::exists(path) || !std::filesystem::exists("../" + path))
    return path;
  return "../" + path;
}

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

glm::quat rotationOf(const glm::mat4 &m) {
  return glm::normalize(glm::quat_cast(glm::mat3(m)));
}

glm::vec3 pointOf(const glm::mat4 &m) { return glm::vec3(m[3]); }

glm::mat4 frame(glm::vec3 x, glm::vec3 y, glm::vec3 z, glm::vec3 at) {
  return glm::mat4(glm::vec4(x, 0.0f), glm::vec4(y, 0.0f), glm::vec4(z, 0.0f), glm::vec4(at, 1.0f));
}

float smooth(float t) {
  t = std::clamp(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

// Two-bone IK toward target, pulled weight of the way from where the end
// is now, the middle joint bending the way it already does.
void reach(const Skeleton &skeleton, Pose &pose, std::vector<glm::mat4> &model, int upper,
           int lower, int end, glm::vec3 target, float weight) {
  if (weight <= 0.0f)
    return;
  const glm::vec3 a = pointOf(model[upper]), b = pointOf(model[lower]), c = pointOf(model[end]);
  glm::vec3 pole = b - 0.5f * (a + c);
  if (glm::length(pole) < 1e-6f)
    pole = glm::vec3(0.0f, -1.0f, 0.0f);
  twoBoneIk(skeleton, pose, upper, lower, end, glm::mix(c, target, std::min(weight, 1.0f)),
            glm::normalize(pole));
  model = modelSpace(skeleton, pose);
}

// Turns `bone` so its turn in the skeleton's space goes weight of the way to
// `wanted`; what it carries turns with it.
void turnTo(const Skeleton &skeleton, Pose &pose, std::vector<glm::mat4> &model, int bone,
            glm::quat wanted, float weight) {
  if (weight <= 0.0f || bone < 0)
    return;
  const glm::quat now = rotationOf(rigidFrame(model[bone]));
  const glm::quat to = glm::slerp(now, glm::dot(now, wanted) < 0.0f ? -wanted : wanted,
                                  std::min(weight, 1.0f));
  const int parent = skeleton.bones[bone].parent;
  const glm::quat parentTurn =
      parent >= 0 ? rotationOf(rigidFrame(model[parent])) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  pose.local[bone].rotation = glm::normalize(glm::inverse(parentTurn) * to);
  model = modelSpace(skeleton, pose);
}

// Every bone under `hand` (its fingers), with its own turn in `pose`.
std::vector<std::pair<int, glm::quat>> fingersOf(const Skeleton &skeleton, const Pose &pose,
                                                 int hand) {
  std::vector<std::pair<int, glm::quat>> fingers;
  std::vector<char> under(skeleton.bones.size(), 0);
  for (int i = 0; i < static_cast<int>(skeleton.bones.size()); ++i) {
    const int parent = skeleton.bones[i].parent;
    if (parent >= 0 && (parent == hand || under[parent])) {
      under[i] = 1;
      fingers.emplace_back(i, pose.local[i].rotation);
    }
  }
  return fingers;
}

// The fingers turned weight of the way to a grip's.
void wrapFingers(Pose &pose, const std::vector<std::pair<int, glm::quat>> &grip, float weight) {
  if (weight <= 0.0f)
    return;
  for (const auto &[bone, turn] : grip) {
    glm::quat &now = pose.local[bone].rotation;
    now = glm::slerp(now, glm::dot(now, turn) < 0.0f ? -turn : turn, std::min(weight, 1.0f));
  }
}

// The mirror image of a turn across x = 0 (a body's or a gun's upright
// middle plane): M R M with M = diag(-1, 1, 1).
glm::quat mirrorX(glm::quat q) { return glm::quat(q.w, q.x, -q.y, -q.z); }

// The bone on the other side ("RightHandIndex1" for "LeftHandIndex1" and
// the other way), or -1.
int otherSide(const Skeleton &skeleton, int bone) {
  if (bone < 0)
    return -1;
  std::string name = skeleton.bones[bone].name;
  if (const size_t at = name.find("Right"); at != std::string::npos)
    name.replace(at, 5, "Left");
  else if (const size_t at2 = name.find("Left"); at2 != std::string::npos)
    name.replace(at2, 4, "Right");
  else
    return -1;
  for (int i = 0; i < static_cast<int>(skeleton.bones.size()); ++i)
    if (skeleton.bones[i].name == name)
      return i;
  return -1;
}

// The stance a gun is carried relaxed in.
Stance carryStance(Stance aiming) {
  return aiming == Stance::Pistol ? Stance::PistolCarry : Stance::RifleCarry;
}

// Whether the animator's move is one the gun's carry stance has a clip for
// (so the clip's hands hold the gun), rather than his own.
bool clipCarries(const CharacterAnimator &animator, const AnimationLibrary &library, Stance gun) {
  const Stance carry = carryStance(gun);
  if (animator.activeStanceShown() != carry)
    return false;
  switch (animator.state()) {
  case AnimState::Locomotion: return library.clip(ClipSlot::Idle, carry);
  case AnimState::Crouch: return library.clip(ClipSlot::CrouchIdle, carry);
  case AnimState::Jump: return library.clip(ClipSlot::Jump, carry);
  case AnimState::Fall: return library.clip(ClipSlot::Fall, carry);
  case AnimState::Land: return library.clip(ClipSlot::Land, carry);
  case AnimState::Dodge:
  case AnimState::Dead: return false;
  }
  return false;
}

} // namespace

const VulkanRenderer::WeaponAsset *VulkanRenderer::weaponAsset(const std::string &name) {
  if (const auto it = weaponAssets.find(name); it != weaponAssets.end())
    return it->second.get();
  auto asset = std::make_unique<WeaponAsset>();
  asset->name = name;
  const std::string specPath = resolved(kWeaponsFolder + name + ".weapon");
  if (!std::filesystem::exists(specPath)) {
    spdlog::warn("No gun called {} ({} isn't there)", name, specPath);
    return nullptr; // not kept: the next Play looks again
  }
  asset->spec = weaponFromSettings(SettingsFile::load(specPath));
  if (!asset->spec.model.empty()) {
    asset->parts = weaponPartsFrom(SettingsFile::load(resolved(asset->spec.model)));
    const std::filesystem::path folder = std::filesystem::path(asset->spec.model).parent_path();
    for (const WeaponPart &part : asset->parts.parts) {
      int model = -1;
      try {
        model = createMeshModel((folder / part.file).generic_string());
      } catch (const std::exception &e) {
        spdlog::warn("Gun {}: couldn't load its {}: {}", name, part.name, e.what());
      }
      asset->partModels.push_back(model);
    }
  }
  if (asset->parts.parts.empty()) {
    spdlog::warn("Gun {} has no model ({}): run tools/weapon_convert/convert_guns.sh", name,
                 asset->spec.model);
    return nullptr; // not kept: the next Play looks again
  }
  asset->round = roundOf(asset->spec);
  if (asset->spec.rocket) {
    asset->rocketFit = fitRocket(*asset->spec.rocket);
    spdlog::info("Gun {}: its rocket fitted to {:.0f} N of thrust, a drag coefficient of {:.2f}", asset->spec.name,
                 asset->rocketFit.thrust, asset->rocketFit.dragCoefficient);
  }
  asset->zero = zeroAngle(asset->round.projectile, asset->spec.muzzleSpeed, asset->spec.zeroRange,
                          Air{}, glm::vec3(0.0f, -9.81f, 0.0f));
  spdlog::info("Gun {}: {} parts, {} sockets, zeroed at {:.0f} m ({:.2f} mrad up)", asset->spec.name,
               asset->parts.parts.size(), asset->parts.sockets.size(), asset->spec.zeroRange,
               asset->zero * 1000.0f);
  return (weaponAssets[name] = std::move(asset)).get();
}

// The scene's kit: the pistol and the long gun it names, each with its parts
// as nodes, its grip worked out from its stance's aiming pose and its place
// when put away from the bind pose. Nothing in the hands to start with.
void VulkanRenderer::armCharacter(LiveCharacter &c) {
  const CharacterAsset *asset = c.copy >= 0 ? c.file->asset : nullptr;
  const SceneKit kit = currentScene.kit.value_or(SceneKit{});
  if (!asset || (!kit.pistol && kit.longGun.empty() && kit.launcher.empty()))
    return;
  const Skeleton &skeleton = asset->skeleton;
  GunBones &b = c.gunBones;
  b.rightArm = boneNamed(skeleton, "RightArm");
  b.rightForeArm = boneNamed(skeleton, "RightForeArm");
  b.rightHand = boneNamed(skeleton, "RightHand");
  b.rightIndex = boneNamed(skeleton, "RightHandIndex2");
  b.leftArm = boneNamed(skeleton, "LeftArm");
  b.leftForeArm = boneNamed(skeleton, "LeftForeArm");
  b.leftHand = boneNamed(skeleton, "LeftHand");
  b.leftKnuckle = boneNamed(skeleton, "LeftHandMiddle1");
  b.hips = boneNamed(skeleton, "Hips");
  b.spine2 = boneNamed(skeleton, "Spine2");
  b.leftLeg = boneNamed(skeleton, "LeftUpLeg");
  b.rightLeg = boneNamed(skeleton, "RightUpLeg");
  b.neck = boneNamed(skeleton, "Neck");
  b.head = boneNamed(skeleton, "Head");
  b.rightEye = boneNamed(skeleton, "RightEye");
  if (!b.valid()) {
    spdlog::warn("The character's skeleton has no Mixamo arms and spine: no guns");
    return;
  }
  c.aimRig = findAimRig(skeleton);
  c.hitCapsules = findHitCapsules(skeleton);

  // Calibrated in metres with the character at the origin facing +z (its
  // model's way), its left +x.
  const glm::mat4 toMetres = glm::scale(glm::mat4(1.0f), glm::vec3(c.scale));
  const std::vector<glm::mat4> bind = modelSpace(skeleton, bindPose(skeleton));
  // The face looks along the model's +z in the bind pose.
  if (b.head >= 0)
    b.headFace = glm::inverse(rotationOf(rigidFrame(bind[b.head]))) * glm::vec3(0.0f, 0.0f, 1.0f);
  // The unarmed idle's pose, where a pistol hangs at the side.
  Pose rest = bindPose(skeleton);
  if (const CharacterClipData *idle = asset->library.clip(ClipSlot::Idle))
    sampleClip(idle->clip, skeleton, idle->start, idle->loop, rest);
  const std::vector<glm::mat4> hanging = modelSpace(skeleton, rest);
  // His chest standing at rest, square (the frame the relaxed carries are
  // worked out in), and where his right eye is in a pose.
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  auto restAt = [&](int bone) { return pointOf(toMetres * hanging[bone]); };
  const glm::vec3 restAhead = chestFacing(restAt(b.leftArm), restAt(b.rightArm), up);
  const glm::mat3 restChest(glm::cross(up, restAhead), up, restAhead);
  auto eyeIn = [&](const std::vector<glm::mat4> &m) {
    if (b.rightEye >= 0)
      return pointOf(toMetres * m[b.rightEye]);
    return pointOf(toMetres * m[b.head]) + glm::vec3(-0.03f, 0.08f, 0.08f); // est.
  };
  for (int slot = 0; slot < 3; ++slot) {
    const bool pistol = slot == 0, launcher = slot == 2;
    const std::string &name = pistol ? std::string(kPistolName) : launcher ? kit.launcher : kit.longGun;
    if (pistol ? !kit.pistol : name.empty())
      continue;
    const WeaponAsset *weapon = weaponAsset(name);
    if (!weapon || weapon->parts.parts.empty())
      continue;
    CarriedGun gun;
    gun.asset = weapon;
    gun.stance = pistol ? Stance::Pistol : Stance::Rifle; // a launcher's held up like a rifle
    gun.launcher = launcher && weapon->spec.rocket.has_value();

    // The grip, from the stance's aiming pose: its aim_idle, else its idle.
    const CharacterClipData *clip = asset->library.clip(ClipSlot::AimIdle, gun.stance);
    if (!clip)
      clip = asset->library.clip(ClipSlot::Idle, gun.stance);
    Pose pose = bindPose(skeleton);
    if (clip)
      sampleClip(clip->clip, skeleton, clip->start, clip->loop, pose);
    const std::vector<glm::mat4> held = modelSpace(skeleton, pose);
    auto at = [&](int bone) { return pointOf(toMetres * held[bone]); };
    const glm::mat4 hand = rigidFrame(toMetres * held[b.rightHand]);
    // The aiming clip aims along the character's facing, +z here; the
    // finger's middle joint lies along the gun's right side, so the trigger
    // is a little to its left (+x).
    const glm::vec3 trigger =
        (b.rightIndex >= 0 ? at(b.rightIndex) : at(b.rightHand)) + glm::vec3(kFingerToTrigger, 0, 0);
    // The left palm, where the hand meets the gun: between the wrist and the
    // middle knuckle.
    OffHand off;
    off.wrist = at(b.leftHand);
    off.palm = b.leftKnuckle >= 0 ? glm::mix(off.wrist, at(b.leftKnuckle), 0.5f) : off.wrist;
    off.height = kPalmHeight;
    gun.hold = holdFrom(hand, trigger, glm::vec3(0.0f, 0.0f, 1.0f), off);
    gun.offHandTurn = glm::inverse(rotationOf(hand * gun.hold.gripInHand)) *
                      rotationOf(rigidFrame(toMetres * held[b.leftHand]));
    gun.rightGrip = fingersOf(skeleton, pose, b.rightHand);
    gun.leftGrip = fingersOf(skeleton, pose, b.leftHand);
    spdlog::info("Carrying the {} ({}-handed: the left palm {:.2f} m from the trigger in {})",
                 weapon->spec.name, gun.hold.twoHanded ? "two" : "one",
                 glm::distance(off.palm, trigger),
                 clip ? "its stance's aiming pose" : "the bind pose");

    const glm::mat4 aimedHold = hand * gun.hold.gripInHand;
    gun.leftInGun = glm::inverse(aimedHold) * rigidFrame(toMetres * held[b.leftHand]);
    // The hands' joints, in the gun's frame, to keep clear of the body.
    const std::optional<WeaponSocket> muzzle = weapon->parts.socket("muzzle");
    gun.clearPoints.clear();
    for (const auto *grip : {&gun.rightGrip, &gun.leftGrip})
      for (const auto &[bone, turn] : *grip)
        gun.clearPoints.push_back(pointOf(glm::inverse(aimedHold) * toMetres * held[bone]));

    // Carried relaxed, worked out on his square chest at rest, and kept in a
    // level frame facing his way at his shoulder (so a run's or a crouch's
    // forward lean doesn't tip the gun). A rifle at low ready: aimed
    // straight ahead with its butt in the shoulder pocket, then turned about
    // the butt, muzzle down and a little across. A pistol: in the hanging
    // hand with the aiming grip, the wrist turned so the muzzle points down
    // and a little ahead; crouched, held out as when aiming and lowered from
    // the shoulders.
    gun.butt = glm::vec3(0.0f, 0.0f, (muzzle ? muzzle->position.z : 0.5f) - weapon->parts.length);
    const glm::mat4 restLevel = frame(restChest[0], up, restAhead, restAt(b.rightArm));
    c.restChest = glm::inverse(rotationOf(restLevel)) * rotationOf(rigidFrame(toMetres * hanging[b.spine2]));
    if (gun.launcher) {
      // The tube's underside on top of the shoulder, the eye at its sights.
      if (const std::optional<WeaponSocket> s = weapon->parts.socket("shoulder"))
        gun.butt = s->position;
      if (const std::optional<WeaponSocket> s = weapon->parts.socket("sight"))
        gun.sight = s->position;
      gun.pocket = kLauncherPocket;
      // The left hand on the rear grip, holding it as the right holds the
      // trigger's grip: the right hand's frame on the gun mirrored across
      // the gun's middle and moved back by the grips' spacing. A bone
      // mirrored needs its local axes put back the left side's way, a turn
      // fixed per bone and taken from the bind pose, where the two sides
      // mirror each other exactly.
      auto bindTurn = [&](int bone) { return rotationOf(rigidFrame(bind[bone])); };
      auto sideFix = [&](int right) {
        const int left = otherSide(skeleton, right);
        return left < 0 ? glm::quat(1.0f, 0.0f, 0.0f, 0.0f)
                        : glm::inverse(mirrorX(bindTurn(right))) * bindTurn(left);
      };
      const std::optional<WeaponSocket> front = weapon->parts.socket("grip");
      const std::optional<WeaponSocket> rear = weapon->parts.socket("rear_grip");
      const glm::vec3 spacing = front && rear ? rear->position - front->position : glm::vec3(0.0f, 0.0f, -0.17f);
      const glm::mat4 rightInGun = glm::inverse(gun.hold.gripInHand);
      glm::vec3 place = pointOf(rightInGun);
      place.x = -place.x;
      gun.leftInGun = glm::translate(glm::mat4(1.0f), place + spacing) *
                      glm::mat4_cast(mirrorX(rotationOf(rightInGun)) * sideFix(b.rightHand));
      // Its fingers wrapped as the right's are.
      gun.leftGrip.clear();
      for (const auto &[bone, turn] : gun.rightGrip) {
        const int left = otherSide(skeleton, bone), parent = skeleton.bones[bone].parent;
        if (left >= 0 && parent >= 0)
          gun.leftGrip.emplace_back(left,
                                    glm::normalize(glm::inverse(sideFix(parent)) * mirrorX(turn) * sideFix(bone)));
      }
      gun.clearPoints.clear(); // it rests on the shoulder: nothing to push off the chest
      // Carried on the shoulder as it's aimed, the muzzle up a little.
      const glm::vec3 pocket = restAt(b.rightArm) + restChest * gun.pocket;
      const glm::mat4 carried =
          aimedGun(restAhead, up, gun.butt, pocket) * glm::translate(glm::mat4(1.0f), gun.butt) *
          glm::rotate(glm::mat4(1.0f), -kLauncherCarryUp, glm::vec3(1, 0, 0)) *
          glm::translate(glm::mat4(1.0f), -gun.butt);
      gun.lowReady = glm::inverse(restLevel) * carried;
      // Being loaded: held by its grip at his right hip, the muzzle forward,
      // up and across in front of him.
      const glm::vec3 across = glm::angleAxis(kLoadingAcross, up) * restAhead;
      const glm::vec3 line = glm::normalize(across * std::cos(kLoadingUp) + up * std::sin(kLoadingUp));
      gun.loadingHold = glm::inverse(restLevel) *
                        aimedGun(line, up, glm::vec3(0.0f), restAt(b.rightArm) + restChest * kLoadingGrip);
    } else if (gun.hold.twoHanded) {
      gun.sight = glm::vec3(0.0f, kRifleSightLine, 0.0f);
      gun.pocket = kPocket;
      const glm::vec3 pocket = restAt(b.rightArm) + restChest * gun.pocket;
      const glm::mat4 lowered =
          aimedGun(restAhead, up, gun.butt, pocket) * glm::translate(glm::mat4(1.0f), gun.butt) *
          glm::rotate(glm::mat4(1.0f), kLowReadyAcross, glm::vec3(0, 1, 0)) *
          glm::rotate(glm::mat4(1.0f), kLowReadyDown, glm::vec3(1, 0, 0)) *
          glm::translate(glm::mat4(1.0f), -gun.butt);
      gun.lowReady = glm::inverse(restLevel) * lowered;
    } else {
      gun.atSide = true;
      // Held out as when aiming: its trigger this far along the eye's line.
      gun.eyeReach = std::max(0.3f, glm::dot(pointOf(aimedHold) - eyeIn(held), glm::vec3(0, 0, 1)));
      const glm::vec3 eye = eyeIn(hanging);
      const glm::vec3 shoulders = 0.5f * (restAt(b.leftArm) + restAt(b.rightArm));
      const glm::mat4 outAhead =
          aimedGun(restAhead, up, glm::vec3(0.0f), eye + restAhead * gun.eyeReach - up * kPistolSightDrop);
      const glm::mat4 lowered = glm::translate(glm::mat4(1.0f), shoulders) *
                                glm::rotate(glm::mat4(1.0f), kPistolLowReady, restChest[0]) *
                                glm::translate(glm::mat4(1.0f), -shoulders) * outAhead;
      gun.lowReady = glm::inverse(restLevel) * lowered;
      // At the side: the gun's barrel down and a little ahead, its top
      // ahead (the thumb's side), and the hand turned to hold it so.
      const glm::vec3 barrel = glm::normalize(-up + kPistolAhead * restAhead);
      const glm::vec3 top = glm::normalize(restAhead - glm::dot(restAhead, barrel) * barrel);
      const glm::quat gunTurn = glm::quat_cast(glm::mat3(glm::cross(top, barrel), top, barrel));
      const glm::quat handTurn = gunTurn * glm::inverse(rotationOf(gun.hold.gripInHand));
      const glm::quat restHand = rotationOf(rigidFrame(toMetres * hanging[b.rightHand]));
      gun.sideHand = glm::inverse(rotationOf(rigidFrame(toMetres * hanging[b.rightForeArm]))) * handTurn;
      spdlog::info("Pistol at the side: the wrist turns {:.0f} degrees from the idle's hang, "
                   "held out {:.2f} m from the eye when aiming",
                   glm::degrees(glm::angle(glm::inverse(restHand) * handTurn)), gun.eyeReach);
    }

    // Put away: the pistol in a holster on the right hip, muzzle down, its
    // top ahead; the long gun across the back, muzzle up over the left
    // shoulder, its top facing back; a launcher across it the other way,
    // muzzle up over the right shoulder, outside the long gun.
    glm::mat4 stow;
    if (gun.launcher) {
      gun.stowBone = b.spine2;
      const glm::vec3 z = glm::normalize(glm::vec3(-0.5f, 0.866f, 0.0f));
      const glm::vec3 x(0.0f, 0.0f, -1.0f);
      const glm::vec3 back = pointOf(toMetres * bind[b.spine2]) + kSling + glm::vec3(0.0f, 0.0f, -0.07f);
      const std::optional<WeaponSocket> v = weapon->parts.socket("venturi");
      const float middle = 0.5f * ((muzzle ? muzzle->position.z : 0.2f) + (v ? v->position.z : -0.8f));
      stow = frame(x, glm::cross(z, x), z, back - z * middle);
    } else if (pistol) {
      gun.stowBone = b.hips;
      stow = frame({1, 0, 0}, {0, 0, 1}, {0, -1, 0}, pointOf(toMetres * bind[b.hips]) + kHolster);
    } else {
      gun.stowBone = b.spine2;
      // Flat on the back: its left side facing out, its top toward the
      // right shoulder.
      const glm::vec3 z = glm::normalize(glm::vec3(0.5f, 0.866f, 0.0f));
      const glm::vec3 x(0.0f, 0.0f, -1.0f);
      const glm::vec3 back = pointOf(toMetres * bind[b.spine2]) + kSling;
      stow = frame(x, glm::cross(z, x), z, back - z * kSlingMiddle);
    }
    gun.stowInBone = glm::inverse(rigidFrame(toMetres * bind[gun.stowBone])) * stow;

    for (int model : weapon->partModels) {
      auto node = std::make_unique<SceneNode>();
      node->setModelId(model);
      gun.parts.push_back(node.get());
      rootNode.addChild(std::move(node));
    }
    for (size_t k = 0; k < weapon->parts.parts.size(); ++k) {
      if (weapon->parts.parts[k].name == "magazine")
        gun.magazinePart = static_cast<int>(k);
      if (weapon->parts.parts[k].name == "bolt" || weapon->parts.parts[k].name == "slide")
        gun.boltPart = static_cast<int>(k);
      if (weapon->parts.parts[k].name == "rocket")
        gun.rocketPart = static_cast<int>(k);
    }
    // Loaded as the kit says: the one in the gun and its spares.
    gun.weapon = std::make_shared<Weapon>(weapon->spec);
    gun.weapon->load(weapon->spec.spareMagazines + 1);
    c.guns[slot] = std::move(gun);
  }
  c.loadout = Loadout();
  c.loadout.give(c.guns[0].has_value(), c.guns[1].has_value(), c.guns[2].has_value());
  c.grenades = kit.grenades;
  c.charges = kit.charges;
}

// Before the clips: the gun keys (the played character's), the draw moving
// on, and the stance the clips play in: the gun's once it's in the hand.
void VulkanRenderer::gunsBeforeClips(int index, AnimatorInput &input, float seconds) {
  LiveCharacter &c = characters[index];
  if (!c.guns[0] && !c.guns[1] && !c.guns[2])
    return;
  const Holdable before = c.loadout.inHand();
  if (index == playedCharacter) {
    if (playerControls.pistol)
      c.loadout.select(Holdable::Pistol);
    if (playerControls.longGun)
      c.loadout.select(Holdable::Long);
    if (playerControls.launcher)
      c.loadout.select(Holdable::Launcher);
    if (playerControls.holster)
      c.loadout.select(Holdable::None);
  }
  // At a weapons crate: E gives every gun the kit's magazines again.
  if (index == playedCharacter && c.body >= 0 && roster) {
    c.atCrate = false;
    const glm::vec3 feet = roster->body(c.body).position();
    if (currentScene.kit)
      for (const glm::vec3 &crate : currentScene.kit->crates)
        if (glm::length(glm::vec2(crate.x - feet.x, crate.z - feet.z)) <= kCrateReach &&
            std::abs(crate.y - feet.y) < 2.0f)
          c.atCrate = true;
    if (c.atCrate && playerControls.use && !c.health.dead() && currentScene.kit) {
      c.grenades = currentScene.kit->grenades;
      c.charges = currentScene.kit->charges;
    }
    if (c.atCrate && playerControls.use && !c.health.dead())
      for (std::optional<CarriedGun> &g : c.guns)
        if (g && g->weapon) {
          g->weapon->cancelReload();
          g->weapon->load(g->asset->spec.spareMagazines + 1);
          spdlog::info("Refilled the {} at the crate", g->asset->spec.name);
        }
  }
  // The player, dead 4 s, comes back at the start with the kit again.
  if (index == playedCharacter && c.health.dead() && c.body >= 0 && roster) {
    c.deadFor += seconds;
    if (c.deadFor >= kComeBack && currentScene.player) {
      c.health = Health{};
      c.deadFor = 0.0f;
      c.throwTime = -1.0f;
      c.pinOut = false;
      roster->body(c.body).place(currentScene.player->position, glm::radians(currentScene.player->yaw));
      if (currentScene.kit) {
        c.grenades = currentScene.kit->grenades;
        c.charges = currentScene.kit->charges;
      }
      for (std::optional<CarriedGun> &g : c.guns)
        if (g && g->weapon) {
          g->weapon->cancelReload();
          g->weapon->load(g->asset->spec.spareMagazines + 1);
        }
      c.loadout.select(Holdable::None);
      spdlog::info("The player comes back at the start");
    }
  }
  // Dead, whatever he was throwing is dropped (no live grenade).
  if (c.health.dead()) {
    c.throwTime = -1.0f;
    c.pinOut = false;
    c.letGo = false;
  }
  // A grenade: G pulls the pin (a pistol goes back in its holster, the
  // throwing hand needs to be free; a rifle hangs at the low ready);
  // letting go throws it, let go of at the clip's release moment.
  if (index == playedCharacter && !c.health.dead()) {
    if (playerControls.grenade && !c.pinOut && c.throwTime < 0.0f && c.grenades > 0) {
      c.pinOut = true;
      if (c.loadout.inHand() == Holdable::Pistol)
        c.loadout.select(Holdable::None);
    }
    if (c.pinOut && !playerControls.grenade) {
      c.pinOut = false;
      // Let go before the pistol was away (it was drawn again): the pin
      // goes back in, nothing thrown.
      if (c.loadout.inHand() != Holdable::Pistol)
        c.throwTime = 0.0f;
    }
  }
  // A charge on what's in front, and the detonator.
  if (index == playedCharacter && !c.health.dead()) {
    if (playerControls.charge)
      placeCharge(index);
    if (playerControls.detonate)
      fireCharges(index);
  }
  if (c.throwTime >= 0.0f) {
    const float before = c.throwTime;
    c.throwTime += seconds;
    if (before < kThrowRelease && c.throwTime >= kThrowRelease)
      c.letGo = true;
    if (c.throwTime >= kThrowEnd)
      c.throwTime = -1.0f;
  }
  input.throwing = c.throwTime >= 0.0f ? c.throwTime / kThrowEnd : -1.0f;
  c.loadout.update(seconds);
  const Holdable held = c.loadout.inHand();
  CarriedGun *gun = gunSlot(held) >= 0 && c.guns[gunSlot(held)] ? &*c.guns[gunSlot(held)] : nullptr;
  // Carried relaxed, he moves with his own clips; aiming or firing from the
  // hip, with the gun's (for the legs and body: the arms are placed on the
  // gun by IK after the clips). Crouched too: his own crouch is a deep
  // stoop that would bury a rifle, the gun's is a kneel (the gun still
  // relaxed unless he aims).
  const bool raised = gun && c.loadout.ready() && c.body >= 0 && roster &&
                      roster->body(c.body).aiming() && c.throwTime < 0.0f && !c.pinOut;
  c.gunCrouched = gun && c.loadout.ready() && input.crouched;
  // Not aiming, the gun's carry stance if the character has one: captured
  // clips of it held relaxed (the moves it has no clip for are his own).
  const CharacterAsset *asset = c.copy >= 0 ? c.file->asset : nullptr;
  const bool carried = gun && asset && asset->library.clip(ClipSlot::Idle, carryStance(gun->stance));
  // (A launcher's carried with the rifle's clips for the legs and body, its
  // tube held on the shoulder by IK after them.)
  input.stance = raised          ? gun->stance
                 : carried       ? carryStance(gun->stance)
                 : c.gunCrouched ? gun->stance
                                 : Stance::Unarmed;
  input.gunInRightHand = gun && gun->atSide && c.loadout.drawn() >= 0.5f;
  // Reloading: R with a gun ready; putting it away or swapping stops it.
  for (std::optional<CarriedGun> &g : c.guns)
    if (g && g->weapon && (&*g != gun || held != before || !c.loadout.ready()))
      g->weapon->cancelReload();
  if (gun && gun->weapon && c.loadout.ready() && !c.health.dead()) {
    const bool player = index == playedCharacter && c.throwTime < 0.0f && !c.pinOut;
    if (player && playerControls.reload)
      gun->weapon->reload();
    if (player && playerControls.fireMode)
      gun->weapon->switchMode();
    // The trigger: the left mouse. From the hip without the right; moving
    // or standing (NSD's spreads for each) otherwise.
    const bool moving =
        c.body >= 0 && roster && glm::length(glm::vec2(roster->body(c.body).velocity().x,
                                                       roster->body(c.body).velocity().z)) > kMovingFire;
    const FiringStance firing = player && !playerControls.aim && !gun->launcher ? FiringStance::Hip
                                : moving                                         ? FiringStance::Moving
                                                                                 : FiringStance::Standing;
    // A launcher only goes off from the shoulder, raised.
    const bool trigger =
        player && playerControls.fire && (!gun->launcher || (playerControls.aim && c.aimWeight >= kLauncherRaised));
    WeaponEvents events;
    gun->weapon->update(seconds, trigger, firing, events);
    gun->shots.insert(gun->shots.end(), events.shots.begin(), events.shots.end());
    // The recoil on the view: the gun's kick, shown bigger than the
    // bullets feel it (the gun only moves once they've left), and coming
    // back with it. The aim point leaves it out.

    // (A launcher's loaded by IK alone: the rifle's magazine change would
    // play on his upper body.)
    if (!gun->launcher) {
      input.reload = reloadProgress(*gun->weapon);
      input.reloadStance = gun->stance;
    }
  }
  // Every gun's kick springs back. The camera takes part of the one in his
  // hands (the bullets' own recoil is in their deviation) and comes back
  // with it; the aim point leaves it out.
  for (std::optional<CarriedGun> &g : c.guns)
    if (g) {
      g->kick.update(seconds, g->asset->spec.kickPeak);
      g->sinceShot += seconds;
    }
  c.shake.update(seconds, 0.05f);
  if (index == playedCharacter && playerController) {
    const glm::vec2 view =
        (gun ? glm::vec2(gun->kick.offset.z, gun->kick.offset.y) * kCameraKick : glm::vec2(0.0f)) +
        glm::vec2(c.shake.offset.z, c.shake.offset.y);
    playerController->kickView(view.x - c.viewKick.x, view.y - c.viewKick.y);
    c.viewKick = view;
    playerController->setArmed(c.loadout.ready());
    playerController->setThrowing(c.pinOut || c.throwTime >= 0.0f);
  }
}

// Where the played character aims: what the camera looks at, from about
// the character on (nothing between the camera and it counts), or far off.
glm::vec3 VulkanRenderer::aimPoint(const LiveCharacter &c) const {
  constexpr float kFar = 300.0f; // m
  if (!playerController || !physics)
    return glm::vec3(0.0f);
  const ThirdPersonCamera &view = playerController->camera();
  const glm::vec3 from = view.position();
  const glm::vec3 dir = glm::normalize(cameraForward(view.yaw() - c.viewKick.x, view.pitch() - c.viewKick.y));
  const glm::vec3 body = c.body >= 0 && roster ? roster->body(c.body).position() : from;
  const float start = std::max(0.0f, glm::dot(body - from, dir));
  if (const std::optional<PhysicsRayHit> hit =
          physics->castRay(from + dir * start, dir, kFar - start))
    return from + dir * (start + hit->distance);
  return from + dir * kFar;
}

// After the clips and the feet: the right hand reaching for a gun being
// drawn or put away, the spine aiming the gun, the left hand onto it, then
// every carried gun's parts where they ride.
void VulkanRenderer::gunsAfterClips(int index, const glm::mat4 &placement, float seconds,
                                    Pose &pose, std::vector<glm::mat4> &model) {
  LiveCharacter &c = characters[index];
  if ((!c.guns[0] && !c.guns[1] && !c.guns[2]) || !c.gunBones.valid())
    return;
  const CharacterAsset *asset = c.copy >= 0 ? c.file->asset : nullptr;
  if (!asset)
    return;
  const Skeleton &skeleton = asset->skeleton;
  const GunBones &b = c.gunBones;
  const glm::mat4 toModel = glm::inverse(placement);
  auto world = [&](int bone) { return placement * model[bone]; };
  auto inModel = [&](glm::vec3 p) { return glm::vec3(toModel * glm::vec4(p, 1.0f)); };

  const Holdable held = c.loadout.inHand();
  CarriedGun *gun = gunSlot(held) >= 0 && c.guns[gunSlot(held)] ? &*c.guns[gunSlot(held)] : nullptr;
  const float drawn = c.loadout.drawn();
  auto stowed = [&](const CarriedGun &g) {
    return rigidFrame(world(g.stowBone)) * g.stowInBone;
  };
  auto inHand = [&](const CarriedGun &g) {
    return rigidFrame(world(b.rightHand)) * g.hold.gripInHand;
  };

  // Aiming, or firing from the hip, with a gun ready: eased in. Crouched
  // with one, the same.
  const bool aiming = gun && c.loadout.ready() && index == playedCharacter && c.body >= 0 &&
                      roster && roster->body(c.body).aiming() && c.throwTime < 0.0f && !c.pinOut;
  const float step = seconds / kAimIn;
  c.aimWeight = std::clamp(c.aimWeight + (aiming ? step : -step), 0.0f, 1.0f);
  const float raise = smooth(c.aimWeight);
  c.crouchWeight = std::clamp(c.crouchWeight + (gun && c.gunCrouched ? step : -step), 0.0f, 1.0f);
  const float crouch = smooth(c.crouchWeight);
  const float drawnIn = smooth((drawn - 0.5f) / 0.5f); // from the put-away gun into the hands

  // A hand to `handAt` (a rigid frame in the world), placed and turned
  // weight of the way there.
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  const glm::quat toModelTurn = glm::inverse(rotationOf(rigidFrame(placement)));
  auto holdWith = [&](int arm, int foreArm, int hand, const glm::mat4 &handAt, float weight) {
    reach(skeleton, pose, model, arm, foreArm, hand, inModel(pointOf(handAt)), weight);
    turnTo(skeleton, pose, model, hand, toModelTurn * rotationOf(handAt), weight);
  };
  auto at = [&](int bone) { return pointOf(world(bone)); };

  // Aiming: the chest squares to the target and leans with the aim, then
  // the gun is placed from the body: the rifle's butt in the shoulder
  // pocket, the pistol out along the eye's line, both laid along the line
  // to the aim point. The rifleman's neck brings his eye down to the
  // sights; the head turns to look at the aim point.
  glm::mat4 aimedAt(1.0f);
  if (gun && raise > 0.0f) {
    const glm::vec3 target = aimPoint(c);
    const glm::vec3 chest = 0.5f * (at(b.leftArm) + at(b.rightArm));
    const glm::vec3 toTarget = glm::normalize(target - chest);
    const glm::vec3 aimInModel = glm::mat3(toModel) * toTarget;
    if (b.leftLeg >= 0 && b.rightLeg >= 0) {
      const AimRig hips{{b.hips}, {1.0f}};
      squareChest(skeleton, hips, pose, b.leftLeg, b.rightLeg, aimInModel, kHipsBlade, raise,
                  kMaxHipsTurn);
    }
    squareChest(skeleton, c.aimRig, pose, b.leftArm, b.rightArm, aimInModel,
                gun->atSide ? 0.0f : kRifleBlade, raise, kMaxSquare);
    const float pitch = std::asin(std::clamp(glm::dot(toTarget, up), -1.0f, 1.0f));
    bendChest(skeleton, c.aimRig, pose, b.leftArm, b.rightArm,
              raise * std::clamp(kBendShare * pitch, -kMaxBend, kMaxBend) +
                  kChestKick * gun->kick.offset.y);
    model = modelSpace(skeleton, pose);
    if (gun->atSide) {
      if (b.head >= 0) {
        turnToward(skeleton, pose, b.head, b.headFace, inModel(target), raise, kHeadTurn);
        model = modelSpace(skeleton, pose);
      }
      const glm::vec3 eye = b.rightEye >= 0 ? at(b.rightEye) : at(b.head);
      const glm::vec3 line = glm::normalize(target - eye);
      aimedAt = aimedGun(line, up, glm::vec3(0.0f), eye + line * gun->eyeReach - up * kPistolSightDrop);
    } else {
      const glm::vec3 ahead = chestFacing(at(b.leftArm), at(b.rightArm), up);
      const glm::vec3 pocket =
          at(b.rightArm) + glm::mat3(glm::cross(up, ahead), up, ahead) * gun->pocket;
      aimedAt = aimedGun(glm::normalize(target - pocket), up, gun->butt, pocket);
      const glm::vec3 sight(gun->sight.x, gun->sight.y, 0.0f);
      // The cheek on the stock: the neck tips the head so the eye comes to
      // the sight line.
      if (b.neck >= 0 && b.rightEye >= 0) {
        const glm::vec3 sightFrom = glm::vec3(aimedAt * glm::vec4(sight, 1.0f));
        const glm::vec3 sightDir = glm::vec3(aimedAt[2]);
        const glm::vec3 eye = at(b.rightEye);
        const glm::vec3 onSight = sightFrom + glm::dot(eye - sightFrom, sightDir) * sightDir;
        const glm::vec3 neckToEye = inModel(eye) - pointOf(model[b.neck]);
        const glm::vec3 axis = glm::inverse(rotationOf(rigidFrame(model[b.neck]))) * glm::normalize(neckToEye);
        turnToward(skeleton, pose, b.neck, axis, inModel(onSight), raise, kNeckTurn);
        model = modelSpace(skeleton, pose);
      }
      if (b.head >= 0) {
        turnToward(skeleton, pose, b.head, b.headFace, inModel(target), raise, kHeadTurn);
        model = modelSpace(skeleton, pose);
      }
      // Then the rifle meets the eye: its sight line through the eye to the
      // aim point, slid along that line until its butt is as near the
      // shoulder pocket as it gets.
      if (b.rightEye >= 0) {
        const glm::vec3 eye = at(b.rightEye);
        const glm::vec3 line = glm::normalize(target - eye);
        const glm::mat4 turn = aimedGun(line, up, glm::vec3(0.0f), glm::vec3(0.0f));
        const glm::vec3 buttFromSight = glm::mat3(turn) * (gun->butt - sight);
        const float along = glm::dot(pocket - (eye + buttFromSight), line);
        aimedAt = aimedGun(line, up, sight, eye + along * line);
      }
    }
  }

  // A reload's clip moves the hands: the left one lets go of the gun (eased
  // over the clip's first and last tenth), and the gun goes into the clip's
  // right hand. Without a reload clip for its stance (the pistol's),
  // nothing moves, so the hand stays on and the magazine stays in.
  auto reloadShown = [&](const CarriedGun &g) {
    return g.weapon && !g.launcher && asset->library.clip(ClipSlot::Reload, g.stance)
               ? reloadProgress(*g.weapon)
               : -1.0f;
  };
  const float reloading = gun ? reloadShown(*gun) : -1.0f;
  // A launcher loaded by hand (no clip): how far, and the tube's tip and
  // the left hand's leaving eased in and out over its first and last tenth.
  const float loading = gun && gun->launcher && gun->weapon ? reloadProgress(*gun->weapon) : -1.0f;
  const float loadingEase =
      loading < 0.0f ? 0.0f : smooth(std::clamp(std::min(loading, 1.0f - loading) / 0.1f, 0.0f, 1.0f));
  const float reloadWeight =
      reloading < 0.0f
          ? 0.0f
          : smooth(std::clamp(std::min(reloading, 1.0f - reloading) / kReloadEase, 0.0f, 1.0f));

  // Carried relaxed: the pistol standing in the hanging hand, the hand
  // turned so its muzzle points down; otherwise at low ready on the chest,
  // kept clear of the body.
  // (A pistol carry clip, standing, already has the hand down at the side.)
  const bool pistolClip = gun && gun->atSide && c.animator->activeStanceShown() == Stance::PistolCarry &&
                          c.animator->state() == AnimState::Locomotion && c.animator->clipWeight(ClipSlot::Idle) > 0.5f;
  const float sideWeight = gun && gun->atSide && !pistolClip ? 1.0f - std::max(raise, crouch) : 0.0f;
  if (gun && gun->atSide && drawn >= 0.5f && sideWeight > 0.0f)
    turnTo(skeleton, pose, model, b.rightHand,
           rotationOf(rigidFrame(model[b.rightForeArm])) * gun->sideHand, sideWeight);
  // The trunk only (chest and belly): pushed off the thighs, a gun held low
  // while kneeling would be shoved up to the chin.
  std::vector<HitCapsule> trunk;
  // Following his chest part of the way, a little behind, breathing.
  {
    const glm::vec3 ahead = glm::normalize(glm::vec3(placement[2]) - glm::dot(glm::vec3(placement[2]), up) * up);
    const glm::quat level = rotationOf(frame(glm::cross(up, ahead), up, ahead, glm::vec3(0.0f)));
    const glm::quat chestTurn = glm::inverse(level) * rotationOf(rigidFrame(world(b.spine2))) *
                                glm::inverse(c.restChest);
    const glm::quat wanted = glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), chestTurn, kCarryFollow);
    c.carryTurn = glm::slerp(c.carryTurn, glm::dot(c.carryTurn, wanted) < 0.0f ? -wanted : wanted,
                             1.0f - std::exp(-seconds / kCarryLag));
  }
  const glm::quat breath = glm::angleAxis(
      kBreathTurn * std::sin(2.0f * glm::pi<float>() * kBreathsPerSecond * static_cast<float>(playSeconds)),
      glm::vec3(1.0f, 0.0f, 0.0f));
  auto lowReadyAt = [&](const CarriedGun &g) {
    const glm::vec3 ahead = glm::normalize(glm::vec3(placement[2]) - glm::dot(glm::vec3(placement[2]), up) * up);
    glm::mat4 low = frame(glm::cross(up, ahead), up, ahead, at(b.rightArm)) *
                    glm::mat4_cast(c.carryTurn * breath) * g.lowReady;
    if (trunk.empty())
      for (const HitCapsule &part : placeHitCapsules(c.hitCapsules, model, placement))
        if (part.zone == HitZone::Chest || part.zone == HitZone::Belly)
          trunk.push_back(part);
    std::vector<glm::vec3> points;
    for (const glm::vec3 &p : g.clearPoints)
      points.push_back(glm::vec3(low * glm::vec4(p, 1.0f)));
    // Straight ahead, never sideways off the body.
    low[3] += glm::vec4(clearOfBody(trunk, points, kBodyMargin, ahead), 0.0f);
    return low;
  };

  // Where the gun in the hands is: relaxed, raised to the aim by the aim's
  // weight, in the reload clip's hand while that plays; over a draw's
  // second half on its way from where it was put away, over the first half
  // still there while the right hand goes to take it.
  glm::mat4 heldAt(1.0f);
  // Throwing a grenade: the hands leave the gun for the throw's clip (eased
  // like it, over its first and last tenth), the rifle hanging at the low
  // ready as on a sling.
  const float throwing = c.throwTime >= 0.0f ? c.throwTime / kThrowEnd : -1.0f;
  const float handsOff =
      throwing < 0.0f ? 0.0f : smooth(std::clamp(std::min(throwing, 1.0f - throwing) / 0.1f, 0.0f, 1.0f));
  const float handsOn = 1.0f - handsOff;
  if (gun) {
    // The rifle in the carry clip's hands when its stance has a clip for
    // this move; at the low ready by IK when it doesn't (a dodge, a throw).
    const bool clipHolds = !gun->atSide && !gun->launcher && handsOff <= 0.0f &&
                           clipCarries(*c.animator, asset->library, gun->stance);
    // Held at the ready by the clip: both arms turned down together about
    // the line through the shoulders, so the gun stays in both hands and
    // tips to the low ready, the clip's own arm motion kept.
    if (clipHolds && raise < 1.0f) {
      const float pitch = std::asin(std::clamp(glm::normalize(glm::vec3(inHand(*gun)[2])).y, -1.0f, 1.0f));
      const float lower = std::clamp(pitch - kCarryPitch, 0.0f, kMostLowered) * (1.0f - raise);
      if (lower > 1e-3f) {
        const glm::vec3 line = glm::normalize(glm::mat3(toModel) * (at(b.leftArm) - at(b.rightArm)));
        const glm::quat down = glm::angleAxis(lower, line);
        for (const int arm : {b.rightArm, b.leftArm})
          turnTo(skeleton, pose, model, arm, down * rotationOf(rigidFrame(model[arm])), 1.0f);
      }
    }
    glm::mat4 relaxed = gun->atSide || clipHolds ? inHand(*gun) : lowReadyAt(*gun);
    if (gun->atSide && crouch > 0.0f)
      relaxed = between(relaxed, lowReadyAt(*gun), crouch);
    heldAt = raise > 0.0f ? between(relaxed, aimedAt, raise) : relaxed;
    gun->aimedFrom = heldAt;
    // The shot's kick: back along the barrel, the muzzle up and aside,
    // turning about the butt or the hand; the hands follow it by IK.
    if (glm::length(gun->kick.offset) > 1e-6f) {
      const glm::vec3 pivot = gun->atSide ? kPistolPivot : gun->butt;
      heldAt = heldAt * glm::translate(glm::mat4(1.0f), pivot) *
               glm::rotate(glm::mat4(1.0f), gun->kick.offset.z, glm::vec3(0.0f, 1.0f, 0.0f)) *
               glm::rotate(glm::mat4(1.0f), -gun->kick.offset.y, glm::vec3(1.0f, 0.0f, 0.0f)) *
               glm::translate(glm::mat4(1.0f), -pivot - glm::vec3(0.0f, 0.0f, gun->kick.offset.x));
    }
    if (reloadWeight > 0.0f)
      heldAt = between(heldAt, inHand(*gun), reloadWeight);
    // A launcher being loaded: off the shoulder to the loading hold.
    if (loading >= 0.0f) {
      const glm::vec3 ahead = glm::normalize(glm::vec3(placement[2]) - glm::dot(glm::vec3(placement[2]), up) * up);
      heldAt = between(heldAt, frame(glm::cross(up, ahead), up, ahead, at(b.rightArm)) * gun->loadingHold, loadingEase);
    }
    if (drawn < 0.5f) {
      holdWith(b.rightArm, b.rightForeArm, b.rightHand,
               stowed(*gun) * glm::inverse(gun->hold.gripInHand), smooth(drawn / 0.5f));
      heldAt = stowed(*gun);
    } else {
      if (drawn < 1.0f)
        heldAt = between(stowed(*gun), heldAt, drawnIn);
      // The right hand holds it there (at the side, the hand already is).
      if (sideWeight < 1.0f || drawn < 1.0f || reloadWeight > 0.0f)
        holdWith(b.rightArm, b.rightForeArm, b.rightHand,
                 heldAt * glm::inverse(gun->hold.gripInHand), handsOn);
    }
  }

  // The left hand: on a rifle's handguard always (let go while a reload's
  // clip moves it); on a pistol, wrapped round the right hand when it's
  // held out or low.
  float leftWeight = 0.0f;
  std::optional<glm::mat4> rocketAt; // a rocket being loaded: its part's frame, in the hand or sliding in
  if (gun && drawn >= 0.5f) {
    if (gun->launcher) {
      // On the rear grip; while loading, off to the pack on his back, then
      // with the rocket at the muzzle, pushing it home, and back.
      leftWeight = drawnIn * handsOn;
      glm::mat4 hand = heldAt * gun->leftInGun;
      if (loading >= 0.0f && gun->rocketPart >= 0) {
        // The hand closed under the rocket's middle as it closes on the
        // rear grip (the wrist as far from it), the rocket slid out of the
        // muzzle by `out` while it goes in.
        const WeaponParts &parts = gun->asset->parts;
        const glm::vec3 pivot = parts.parts[gun->rocketPart].pivot;
        const std::optional<WeaponSocket> nose = parts.socket("nose"), rear = parts.socket("rear_grip");
        const float middle = 0.5f * ((nose ? nose->position.z : pivot.z + 0.7f) - pivot.z);
        const glm::vec3 wristToGrip = rear ? rear->position - pointOf(gun->leftInGun) : glm::vec3(0.0f);
        const glm::mat4 handOnRocket =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.045f, middle) - wristToGrip) *
            glm::mat4_cast(rotationOf(gun->leftInGun));
        const float home = kRocketOut + kRocketCarry;
        const float out = kRocketStart * (1.0f - smooth((loading - home) / (kRocketIn - home)));
        const glm::mat4 inTube = heldAt * glm::translate(glm::mat4(1.0f), pivot + glm::vec3(0.0f, 0.0f, out));
        // The pack on his back, left of the spine.
        glm::mat4 atPack = hand;
        atPack[3] = glm::vec4(at(b.spine2) + rotationOf(placement) * glm::vec3(0.15f, -0.15f, -0.2f), 1.0f);
        if (loading < kRocketOut) {
          hand = between(hand, atPack, smooth(loading / kHandMove));
        } else if (loading < home) {
          // Bringing it to the muzzle, the rocket in the hand.
          hand = between(atPack, inTube * handOnRocket, smooth((loading - kRocketOut) / kRocketCarry));
          rocketAt = hand * glm::inverse(handOnRocket);
        } else if (loading < kRocketIn) {
          hand = inTube * handOnRocket;
          rocketAt = inTube;
        } else {
          hand = between(inTube * handOnRocket, hand, smooth((loading - kRocketIn) / kHandMove));
        }
      }
      holdWith(b.leftArm, b.leftForeArm, b.leftHand, hand, leftWeight);
    } else if (gun->hold.twoHanded) {
      leftWeight = drawnIn * (1.0f - reloadWeight) * handsOn;
      const glm::vec3 grip = glm::vec3(heldAt * glm::vec4(gun->hold.offHand, 1.0f));
      reach(skeleton, pose, model, b.leftArm, b.leftForeArm, b.leftHand, inModel(grip), leftWeight);
      turnTo(skeleton, pose, model, b.leftHand, toModelTurn * rotationOf(heldAt) * gun->offHandTurn,
             leftWeight);
    } else {
      leftWeight = drawnIn * std::max(raise, crouch) * handsOn;
      holdWith(b.leftArm, b.leftForeArm, b.leftHand, heldAt * gun->leftInGun, leftWeight);
    }
  }

  // The fingers round the gun as the aiming clip has them (the right hand's
  // from the grab on), except where a reload's clip moves them.
  if (gun) {
    wrapFingers(pose, gun->rightGrip,
                (drawn < 0.5f ? smooth(drawn / 0.5f) : 1.0f) * (1.0f - reloadWeight) * handsOn);
    wrapFingers(pose, gun->leftGrip, leftWeight);
    model = modelSpace(skeleton, pose);
  }

  // Every carried gun's parts: the one in the hands where it's held, the
  // other where it rides.
  for (std::optional<CarriedGun> &g : c.guns) {
    if (!g)
      continue;
    const glm::mat4 at = &*g == gun ? heldAt : stowed(*g);
    g->at = at;
    const glm::quat turn = rotationOf(at);
    // The magazine is out (in the pouch) for the middle of a reload.
    const float progress = reloadShown(*g);
    const bool magazineOut = progress >= kMagazineOut && progress < kMagazineIn;
    // The bolt or slide: back and forward over the cycle after a shot,
    // held back once the last round's gone on a gun that locks open.
    float boltBack = 0.0f;
    if (g->boltPart >= 0) {
      const float cycle = 60.0f / std::max(g->asset->spec.rate, 60.0f);
      const float travel = g->atSide ? kSlideTravel : kBoltTravel;
      if (g->weapon && g->weapon->slideLocked())
        boltBack = travel;
      else if (g->sinceShot < cycle)
        boltBack = travel * std::sin(glm::pi<float>() * g->sinceShot / cycle);
    }
    // A launcher's rocket: in the tube while it's loaded; while loading, in
    // the left hand from the moment it's out of the pack, sliding home.
    const float loadingNow = g->launcher && g->weapon ? reloadProgress(*g->weapon) : -1.0f;
    const bool rocketShown = g->weapon && (g->weapon->chambered() || loadingNow >= kRocketOut);
    for (size_t k = 0; k < g->parts.size() && k < g->asset->parts.parts.size(); ++k) {
      const bool hidden = (static_cast<int>(k) == g->magazinePart && magazineOut) ||
                          (static_cast<int>(k) == g->rocketPart && !rocketShown);
      const int model = hidden ? -1 : g->asset->partModels[k];
      if (g->parts[k]->getModelId() != model)
        g->parts[k]->setModelId(model);
      glm::vec3 pivot = g->asset->parts.parts[k].pivot;
      if (static_cast<int>(k) == g->boltPart)
        pivot.z -= boltBack;
      if (static_cast<int>(k) == g->rocketPart && &*g == gun && rocketAt) {
        g->parts[k]->setPose(pointOf(*rocketAt), rotationOf(*rocketAt));
        continue;
      }
      g->parts[k]->setPose(glm::vec3(at * glm::vec4(pivot, 1.0f)), turn);
    }
  }
  // A grenade let go this frame leaves the right hand.
  if (c.letGo) {
    c.letGo = false;
    throwGrenade(index, at(b.rightHand));
  }
  // The shots this frame's trigger let off leave the muzzle where the gun
  // is now.
  for (std::optional<CarriedGun> &g : c.guns)
    if (g && !g->shots.empty()) {
      if (&*g == gun && drawn >= 1.0f)
        fireShots(index, *g);
      else
        g->shots.clear();
    }
}

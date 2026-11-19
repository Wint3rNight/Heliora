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
// the shoulder pocket, the muzzle down about 45 degrees), and about 20
// degrees across the body (est.); the pistol in the hanging hand, its
// muzzle down and this much ahead (est.).
constexpr float kLowReadyDown = glm::radians(45.0f);
constexpr float kLowReadyAcross = glm::radians(20.0f);
constexpr float kPistolAhead = 0.35f;
// Aiming, the head turns to look at the aim point, at most this far (est.).
constexpr float kHeadTurn = glm::radians(30.0f);

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
  spdlog::info("Gun {}: {} parts, {} sockets", asset->spec.name, asset->parts.parts.size(),
               asset->parts.sockets.size());
  return (weaponAssets[name] = std::move(asset)).get();
}

// The scene's kit: the pistol and the long gun it names, each with its parts
// as nodes, its grip worked out from its stance's aiming pose and its place
// when put away from the bind pose. Nothing in the hands to start with.
void VulkanRenderer::armCharacter(LiveCharacter &c) {
  const CharacterAsset *asset = c.copy >= 0 ? c.file->asset : nullptr;
  const SceneKit kit = currentScene.kit.value_or(SceneKit{});
  if (!asset || (!kit.pistol && kit.longGun.empty()))
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
  b.head = boneNamed(skeleton, "Head");
  if (!b.valid()) {
    spdlog::warn("The character's skeleton has no Mixamo arms and spine: no guns");
    return;
  }
  c.aimRig = findAimRig(skeleton);

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
  for (int slot = 0; slot < 2; ++slot) {
    const bool pistol = slot == 0;
    if (pistol ? !kit.pistol : kit.longGun.empty())
      continue;
    const WeaponAsset *weapon = weaponAsset(pistol ? kPistolName : kit.longGun);
    if (!weapon || weapon->parts.parts.empty())
      continue;
    CarriedGun gun;
    gun.asset = weapon;
    gun.stance = pistol ? Stance::Pistol : Stance::Rifle;

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

    // Carried relaxed. A rifle: the aiming hold turned about its butt, the
    // muzzle dipped and brought across, kept where it is to the chest. A
    // pistol: in the hanging hand of the unarmed idle.
    if (gun.hold.twoHanded) {
      const std::optional<WeaponSocket> muzzle = weapon->parts.socket("muzzle");
      const glm::vec3 butt(0.0f, 0.0f, (muzzle ? muzzle->position.z : 0.5f) - weapon->parts.length);
      const glm::mat4 lowered =
          hand * gun.hold.gripInHand * glm::translate(glm::mat4(1.0f), butt) *
          glm::rotate(glm::mat4(1.0f), kLowReadyAcross, glm::vec3(0, 1, 0)) *
          glm::rotate(glm::mat4(1.0f), kLowReadyDown, glm::vec3(1, 0, 0)) *
          glm::translate(glm::mat4(1.0f), -butt);
      gun.carry = glm::inverse(rigidFrame(toMetres * held[b.spine2])) * lowered;
      gun.carryInHand = false;
    } else {
      auto restAt = [&](int bone) { return pointOf(toMetres * hanging[bone]); };
      const glm::vec3 down = glm::normalize(restAt(b.rightHand) - restAt(b.rightForeArm));
      const glm::vec3 restTrigger = b.rightIndex >= 0 ? restAt(b.rightIndex) : restAt(b.rightHand);
      gun.carry = holdFrom(rigidFrame(toMetres * hanging[b.rightHand]), restTrigger,
                           glm::normalize(down + kPistolAhead * glm::vec3(0.0f, 0.0f, 1.0f)),
                           std::nullopt, glm::vec3(0.0f, 0.0f, 1.0f))
                      .gripInHand;
      gun.carryInHand = true;
    }

    // Put away: the pistol in a holster on the right hip, muzzle down, its
    // top ahead; the long gun across the back, muzzle up over the left
    // shoulder, its top facing back.
    glm::mat4 stow;
    if (pistol) {
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
    for (size_t k = 0; k < weapon->parts.parts.size(); ++k)
      if (weapon->parts.parts[k].name == "magazine")
        gun.magazinePart = static_cast<int>(k);
    // Loaded as the kit says: the one in the gun and its spares.
    gun.weapon = std::make_shared<Weapon>(weapon->spec);
    gun.weapon->load(weapon->spec.spareMagazines + 1);
    c.guns[slot] = std::move(gun);
  }
  c.loadout = Loadout();
  c.loadout.give(c.guns[0].has_value(), c.guns[1].has_value());
}

// Before the clips: the gun keys (the played character's), the draw moving
// on, and the stance the clips play in: the gun's once it's in the hand.
void VulkanRenderer::gunsBeforeClips(int index, AnimatorInput &input, float seconds) {
  LiveCharacter &c = characters[index];
  if (!c.guns[0] && !c.guns[1])
    return;
  const Holdable before = c.loadout.inHand();
  if (index == playedCharacter) {
    if (playerControls.pistol)
      c.loadout.select(Holdable::Pistol);
    if (playerControls.longGun)
      c.loadout.select(Holdable::Long);
    if (playerControls.holster)
      c.loadout.select(Holdable::None);
  }
  c.loadout.update(seconds);
  const Holdable held = c.loadout.inHand();
  std::optional<CarriedGun> &slot = c.guns[held == Holdable::Long ? 1 : 0];
  CarriedGun *gun = held != Holdable::None && slot ? &*slot : nullptr;
  // Carried relaxed, he moves with his own clips; aiming or firing from the
  // hip, with the gun's. Crouched too: his own squat would bury a rifle or
  // drag a pistol on the ground. The rifle's crouch holds it at the ready;
  // the pistol has none, so its arms go over his squat, pointing it at the
  // ground ahead.
  const bool raised = gun && c.loadout.ready() && c.body >= 0 && roster &&
                      roster->body(c.body).aiming();
  const bool crouchedReady = gun && c.loadout.ready() && input.crouched;
  c.stanceHolds = raised || crouchedReady;
  input.stance = c.stanceHolds ? gun->stance : Stance::Unarmed;
  input.gunInRightHand = gun && gun->carryInHand && c.loadout.drawn() >= 0.5f;
  // Reloading: R with a gun ready; putting it away or swapping stops it.
  for (std::optional<CarriedGun> &g : c.guns)
    if (g && g->weapon && (&*g != gun || held != before || !c.loadout.ready()))
      g->weapon->cancelReload();
  if (gun && gun->weapon && c.loadout.ready()) {
    if (index == playedCharacter && playerControls.reload)
      gun->weapon->reload();
    WeaponEvents events;
    gun->weapon->update(seconds, false, FiringStance::Standing, events);
    input.reload = reloadProgress(*gun->weapon);
    input.reloadStance = gun->stance;
  }
  if (index == playedCharacter && playerController)
    playerController->setArmed(c.loadout.ready());
}

// Where the played character aims: what the camera looks at, from about
// the character on (nothing between the camera and it counts), or far off.
glm::vec3 VulkanRenderer::aimPoint(const LiveCharacter &c) const {
  constexpr float kFar = 300.0f; // m
  if (!playerController || !physics)
    return glm::vec3(0.0f);
  const ThirdPersonCamera &view = playerController->camera();
  const glm::vec3 from = view.position(), dir = glm::normalize(view.forward());
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
  if ((!c.guns[0] && !c.guns[1]) || !c.gunBones.valid())
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
  CarriedGun *gun = nullptr;
  if (held != Holdable::None && c.guns[held == Holdable::Long ? 1 : 0])
    gun = &*c.guns[held == Holdable::Long ? 1 : 0];
  const float drawn = c.loadout.drawn();
  auto stowed = [&](const CarriedGun &g) {
    return rigidFrame(world(g.stowBone)) * g.stowInBone;
  };
  auto inHand = [&](const CarriedGun &g) {
    return rigidFrame(world(b.rightHand)) * g.hold.gripInHand;
  };

  auto relaxed = [&](const CarriedGun &g) {
    return rigidFrame(world(g.carryInHand ? b.rightHand : b.spine2)) * g.carry;
  };

  // Aiming, or firing from the hip, with a gun ready: eased in.
  const bool aiming = gun && c.loadout.ready() && index == playedCharacter && c.body >= 0 &&
                      roster && roster->body(c.body).aiming();
  const float step = seconds / kAimIn;
  c.aimWeight = std::clamp(c.aimWeight + (aiming ? step : -step), 0.0f, 1.0f);
  const float raise = smooth(c.aimWeight);
  c.holdWeight = std::clamp(c.holdWeight + (gun && c.stanceHolds ? step : -step), 0.0f, 1.0f);

  // A hand to `handAt` (a rigid frame in the world), placed and turned
  // weight of the way there.
  const glm::quat toModelTurn = glm::inverse(rotationOf(rigidFrame(placement)));
  auto holdWith = [&](int arm, int foreArm, int hand, const glm::mat4 &handAt, float weight) {
    reach(skeleton, pose, model, arm, foreArm, hand, inModel(pointOf(handAt)), weight);
    turnTo(skeleton, pose, model, hand, toModelTurn * rotationOf(handAt), weight);
  };

  // Aiming: the spine brings the barrel onto the aim point, and the head
  // turns to look there.
  if (gun && c.aimWeight > 0.0f) {
    const glm::vec3 target = inModel(aimPoint(c));
    const std::optional<WeaponSocket> muzzle = gun->asset->parts.socket("muzzle");
    const glm::mat4 muzzleInGun = muzzle ? socketMatrix(*muzzle) : glm::mat4(1.0f);
    const glm::mat3 turnToModel = glm::mat3(toModel);
    BarrelFinder barrel = [&](const std::vector<glm::mat4> &m) {
      const glm::mat4 gunAt = rigidFrame(placement * m[b.rightHand]) * gun->hold.gripInHand;
      const glm::mat4 tip = gunAt * muzzleInGun;
      return Barrel{inModel(pointOf(tip)),
                    glm::normalize(turnToModel * glm::vec3(gunAt[2]))};
    };
    aimAt(skeleton, c.aimRig, pose, barrel, target, raise);
    if (b.head >= 0)
      turnToward(skeleton, pose, b.head, b.headFace, target, raise, kHeadTurn);
    model = modelSpace(skeleton, pose);
  }

  // A reload's clip moves the hands: the left one lets go of the gun (eased
  // over the clip's first and last tenth), and the gun comes up into the
  // clip's right hand. Without a reload clip for its stance (the pistol's),
  // nothing moves, so the hand stays on and the magazine stays in.
  auto reloadShown = [&](const CarriedGun &g) {
    return g.weapon && asset->library.clip(ClipSlot::Reload, g.stance)
               ? reloadProgress(*g.weapon)
               : -1.0f;
  };
  const float reloading = gun ? reloadShown(*gun) : -1.0f;
  const float reloadWeight =
      reloading < 0.0f
          ? 0.0f
          : std::clamp(std::min(reloading, 1.0f - reloading) / kReloadEase, 0.0f, 1.0f);

  // In the hands: carried relaxed, or in the clip's hands while the gun's
  // stance clips play (aiming, crouched) or a reload's does.
  const float up = std::max({raise, smooth(c.holdWeight), smooth(reloadWeight)});
  // Drawing or putting away. Over the first half the right hand goes to the
  // put-away gun's grip and takes it; over the second the gun comes from
  // there into the hands. A pistol's hand brings it back the way the clip
  // holds the hand; a rifle comes up to where it's held, and its right hand
  // holds it by IK until the clip's hand does (raised all the way the two
  // are the same).
  glm::mat4 heldAt(1.0f);
  if (gun) {
    const glm::mat4 hand = rigidFrame(world(b.rightHand));
    const glm::mat4 handsAt = between(relaxed(*gun), inHand(*gun), up);
    const glm::mat4 grabAt = stowed(*gun) * glm::inverse(gun->hold.gripInHand);
    const float back = smooth((drawn - 0.5f) / 0.5f); // 0 at the put-away gun, 1 in the hands
    if (drawn < 0.5f) {
      holdWith(b.rightArm, b.rightForeArm, b.rightHand, grabAt, smooth(drawn / 0.5f));
      heldAt = stowed(*gun);
    } else if (drawn < 1.0f && gun->carryInHand) {
      const glm::mat4 handAt = between(grabAt, hand, back);
      holdWith(b.rightArm, b.rightForeArm, b.rightHand, handAt, 1.0f);
      heldAt = handAt * between(gun->hold.gripInHand, glm::inverse(hand) * handsAt, back);
    } else {
      heldAt = drawn < 1.0f ? between(stowed(*gun), handsAt, back) : handsAt;
    }
    if (!gun->carryInHand && drawn >= 0.5f && up < 1.0f)
      holdWith(b.rightArm, b.rightForeArm, b.rightHand,
               heldAt * glm::inverse(gun->hold.gripInHand), 1.0f);
  }

  // Two hands on a rifle: the left one on its handguard.
  if (gun && gun->hold.twoHanded && drawn >= 0.5f) {
    const float weight = smooth((drawn - 0.5f) / 0.5f) * (1.0f - reloadWeight);
    const glm::vec3 grip = glm::vec3(heldAt * glm::vec4(gun->hold.offHand, 1.0f));
    reach(skeleton, pose, model, b.leftArm, b.leftForeArm, b.leftHand, inModel(grip), weight);
    turnTo(skeleton, pose, model, b.leftHand, toModelTurn * rotationOf(heldAt) * gun->offHandTurn,
           weight);
  }

  // The fingers round the gun, where the clip doesn't already have them
  // there: all the way once it's in the hand (the right hand's from the
  // grab on), less as it's raised into the aiming clip's own.
  if (gun) {
    const float inHandWeight = drawn < 0.5f ? smooth(drawn / 0.5f) : 1.0f;
    wrapFingers(pose, gun->rightGrip, inHandWeight * (1.0f - up));
    if (gun->hold.twoHanded && drawn >= 0.5f)
      wrapFingers(pose, gun->leftGrip,
                  smooth((drawn - 0.5f) / 0.5f) * (1.0f - reloadWeight) * (1.0f - up));
    model = modelSpace(skeleton, pose);
  }

  // Every carried gun's parts: the one in the hands where it's held, the
  // other where it rides.
  for (const std::optional<CarriedGun> &g : c.guns) {
    if (!g)
      continue;
    const glm::mat4 at = &*g == gun ? heldAt : stowed(*g);
    const glm::quat turn = rotationOf(at);
    // The magazine is out (in the pouch) for the middle of a reload.
    const float progress = reloadShown(*g);
    const bool magazineOut = progress >= kMagazineOut && progress < kMagazineIn;
    for (size_t k = 0; k < g->parts.size() && k < g->asset->parts.parts.size(); ++k) {
      const bool hidden = static_cast<int>(k) == g->magazinePart && magazineOut;
      const int model = hidden ? -1 : g->asset->partModels[k];
      if (g->parts[k]->getModelId() != model)
        g->parts[k]->setModelId(model);
      g->parts[k]->setPose(glm::vec3(at * glm::vec4(g->asset->parts.parts[k].pivot, 1.0f)), turn);
    }
  }
}

#include "WeaponFile.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <vector>

namespace {

const char *kMatterNames[] = {"stone", "wood", "metal", "rubber", "ice", "dirt", "flesh"};
static_assert(std::size(kMatterNames) == static_cast<size_t>(Matter::Count));

// Keys for the penetration figures, made once so the bindings can point at
// their text.
struct PenetrationKeys {
  std::array<std::array<std::string, 4>, static_cast<size_t>(Matter::Count)> keys;
  PenetrationKeys() {
    for (size_t m = 0; m < keys.size(); ++m) {
      const std::string name = kMatterNames[m];
      keys[m] = {name + "_depth", name + "_speed", name + "_depth2", name + "_speed2"};
    }
  }
};
const PenetrationKeys &penetrationKeys() {
  static const PenetrationKeys k;
  return k;
}

// Every number in the file, bound to the spec.
std::vector<SettingBinding> bindings(WeaponSpec &s) {
  std::vector<SettingBinding> b = {
      {"round", "bullet_mass", &s.bulletMass, 0.0001, 10},
      {"round", "muzzle_speed", &s.muzzleSpeed, 1, 3000},
      {"round", "coefficient", &s.coefficient, 0.001, 5},
      {"round", "zero_range", &s.zeroRange, 0, 3000},
      {"round", "tracer_every", &s.tracerEvery, 0, 1000},
      {"action", "magazine", &s.magazine, 0, 1000000},
      {"action", "rate", &s.rate, 1, 10000},
      {"action", "automatic", &s.automatic},
      {"action", "locks_open", &s.locksOpen},
      {"spread", "first_side", &s.firstSpread.x, 0, 1000},
      {"spread", "first_up", &s.firstSpread.y, 0, 1000},
      {"spread", "follow_side", &s.followSpread.x, 0, 1000},
      {"spread", "follow_up", &s.followSpread.y, 0, 1000},
      {"spread", "standing", &s.standing, 0, 100},
      {"spread", "moving", &s.moving, 0, 100},
      {"spread", "hip", &s.hipSpread, 0, 1000},
      {"recoil", "impulse", &s.recoilImpulse, 0, 1000},
      {"recoil", "return", &s.recoilReturn, 0.001, 10},
      {"recoil", "kick_back", &s.kickBack, 0, 1},
      {"recoil", "kick_up", &s.kickUp, 0, 90},
      {"recoil", "kick_aside", &s.kickAside, 0, 90},
      {"recoil", "kick_peak", &s.kickPeak, 0.001, 1},
      {"reload", "partial", &s.reload, 0, 60},
      {"reload", "empty", &s.emptyReload, 0, 60},
      {"carry", "mass", &s.mass, 0, 1000},
      {"carry", "length", &s.length, 0, 100},
      {"carry", "spare_magazines", &s.spareMagazines, 0, 1000},
      {"sounds", "loudness", &s.loudness, 0, 250},
  };
  const PenetrationKeys &k = penetrationKeys();
  for (size_t m = 0; m < k.keys.size(); ++m) {
    MatterPenetration &p = s.penetration[m];
    b.push_back({"penetration", k.keys[m][0], &p.depth, 0, 100});
    b.push_back({"penetration", k.keys[m][1], &p.speed, 0, 3000});
    b.push_back({"penetration", k.keys[m][2], &p.depth2, 0, 100});
    b.push_back({"penetration", k.keys[m][3], &p.speed2, 0, 3000});
  }
  return b;
}

// A launcher's rocket and warhead.
std::vector<SettingBinding> rocketBindings(RocketSpec &r) {
  return {
      {"rocket", "mass", &r.mass, 0.01, 1000},
      {"rocket", "diameter", &r.diameter, 0.001, 10},
      {"rocket", "launch_speed", &r.launchSpeed, 1, 3000},
      {"rocket", "motor_from", &r.motorFrom, 0, 10000},
      {"rocket", "motor_to", &r.motorTo, 0, 100000},
      {"rocket", "top_speed", &r.topSpeed, 1, 3000},
      {"rocket", "self_destruct", &r.selfDestruct, 0.1, 1000},
      {"rocket", "reach", &r.reach, 1, 100000},
      {"warhead", "steel", &r.jetSteel, 0, 10},
      {"warhead", "tnt", &r.warheadTnt, 0, 1000},
  };
}

const char *dragName(DragModel model) { return model == DragModel::G1 ? "G1" : "G7"; }

} // namespace

WeaponSpec weaponFromSettings(const SettingsFile &file) {
  WeaponSpec spec;
  if (const std::optional<std::string> name = file.text("gun", "name"))
    spec.name = *name;
  if (const std::optional<std::string> model = file.text("gun", "model"))
    spec.model = *model;
  if (const std::optional<std::string> drag = file.text("round", "drag")) {
    if (*drag == "G1")
      spec.drag = DragModel::G1;
    else if (*drag == "G7")
      spec.drag = DragModel::G7;
    else
      spdlog::warn("Weapon {}: no drag model '{}' (G1 or G7), keeping {}", spec.name, *drag,
                   dragName(spec.drag));
  }
  if (const std::optional<std::string> shot = file.text("sounds", "shot"))
    spec.shotSound = *shot;
  if (const std::optional<std::string> far = file.text("sounds", "shot_far"))
    spec.farShotSound = *far;
  // The numbers; the drag model and the sounds were text, so they aren't
  // reported as unknown.
  SettingsFile numbers = file;
  numbers.remove("round", "drag");
  numbers.remove("sounds", "shot");
  numbers.remove("sounds", "shot_far");
  const std::vector<SettingBinding> b = bindings(spec);
  for (const char *section : {"round", "penetration", "action", "spread", "recoil", "reload",
                              "carry", "sounds"})
    readSettings(numbers, b, section);
  const std::vector<std::string> sections = file.sectionNames();
  if (std::find(sections.begin(), sections.end(), "rocket") != sections.end()) {
    spec.rocket = RocketSpec{};
    const std::vector<SettingBinding> r = rocketBindings(*spec.rocket);
    readSettings(file, r, "rocket");
    readSettings(file, r, "warhead");
  }
  return spec;
}

void weaponToSettings(const WeaponSpec &spec, SettingsFile &file) {
  WeaponSpec copy = spec;
  file.setString("gun", "name", spec.name);
  if (!spec.model.empty())
    file.setString("gun", "model", spec.model);
  if (!spec.shotSound.empty())
    file.setString("sounds", "shot", spec.shotSound);
  if (!spec.farShotSound.empty())
    file.setString("sounds", "shot_far", spec.farShotSound);
  file.setString("round", "drag", dragName(spec.drag));
  const std::vector<SettingBinding> b = bindings(copy);
  for (const char *section : {"round", "penetration", "action", "spread", "recoil", "reload",
                              "carry", "sounds"})
    writeSettings(file, b, section);
  if (copy.rocket) {
    const std::vector<SettingBinding> r = rocketBindings(*copy.rocket);
    writeSettings(file, r, "rocket");
    writeSettings(file, r, "warhead");
  }
}

Round roundOf(const WeaponSpec &spec) {
  Round round;
  round.projectile = {spec.drag, spec.coefficient};
  round.mass = spec.bulletMass;
  // The fitted ratio a single figure borrows.
  float ratio = 0.0f;
  for (const MatterPenetration &p : spec.penetration)
    if (p.depth > 0.0f && p.depth2 > 0.0f) {
      ratio = ratioOf(ponceletFit(spec.bulletMass, p.depth, p.speed, p.depth2, p.speed2));
      break;
    }
  for (size_t m = 0; m < spec.penetration.size(); ++m) {
    const MatterPenetration &p = spec.penetration[m];
    if (!(p.depth > 0.0f) || !(p.speed > 0.0f))
      continue;
    round.resistance[m] = p.depth2 > 0.0f && p.speed2 > 0.0f
                              ? ponceletFit(spec.bulletMass, p.depth, p.speed, p.depth2, p.speed2)
                              : ponceletFrom(spec.bulletMass, p.depth, p.speed, ratio);
  }
  return round;
}

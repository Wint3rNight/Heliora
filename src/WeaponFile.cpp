#include "WeaponFile.h"

#include <spdlog/spdlog.h>

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
      {"action", "magazine", &s.magazine, 1, 1000000},
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
      {"reload", "partial", &s.reload, 0, 60},
      {"reload", "empty", &s.emptyReload, 0, 60},
      {"carry", "mass", &s.mass, 0, 1000},
      {"carry", "length", &s.length, 0, 100},
      {"carry", "spare_magazines", &s.spareMagazines, 0, 1000},
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

const char *dragName(DragModel model) { return model == DragModel::G1 ? "G1" : "G7"; }

} // namespace

WeaponSpec weaponFromSettings(const SettingsFile &file) {
  WeaponSpec spec;
  if (const std::optional<std::string> name = file.text("gun", "name"))
    spec.name = *name;
  if (const std::optional<std::string> drag = file.text("round", "drag")) {
    if (*drag == "G1")
      spec.drag = DragModel::G1;
    else if (*drag == "G7")
      spec.drag = DragModel::G7;
    else
      spdlog::warn("Weapon {}: no drag model '{}' (G1 or G7), keeping {}", spec.name, *drag,
                   dragName(spec.drag));
  }
  // The numbers; the drag model was text, so it isn't reported as unknown.
  SettingsFile numbers = file;
  numbers.remove("round", "drag");
  const std::vector<SettingBinding> b = bindings(spec);
  for (const char *section : {"round", "penetration", "action", "spread", "recoil", "reload",
                              "carry"})
    readSettings(numbers, b, section);
  return spec;
}

void weaponToSettings(const WeaponSpec &spec, SettingsFile &file) {
  WeaponSpec copy = spec;
  file.setString("gun", "name", spec.name);
  file.setString("round", "drag", dragName(spec.drag));
  const std::vector<SettingBinding> b = bindings(copy);
  for (const char *section : {"round", "penetration", "action", "spread", "recoil", "reload",
                              "carry"})
    writeSettings(file, b, section);
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

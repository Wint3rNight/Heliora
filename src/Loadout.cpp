#include "Loadout.h"

#include <algorithm>

Loadout::Loadout(LoadoutTiming timing) : timing(timing) {}

void Loadout::give(bool hasPistol, bool hasLong, bool hasLauncher) {
  pistol = hasPistol;
  longGun = hasLong;
  launcher = hasLauncher;
  if (!has(current)) {
    current = Holdable::None;
    amount = 0.0f;
  }
  if (!has(wanted))
    wanted = Holdable::None;
}

bool Loadout::has(Holdable which) const {
  switch (which) {
  case Holdable::None: return true;
  case Holdable::Pistol: return pistol;
  case Holdable::Long: return longGun;
  case Holdable::Launcher: return launcher;
  }
  return false;
}

void Loadout::select(Holdable which) {
  if (has(which))
    wanted = which;
}

float Loadout::drawTime(Holdable which) const {
  return which == Holdable::Pistol     ? timing.pistolDraw
         : which == Holdable::Launcher ? timing.launcherDraw
                                       : timing.longDraw;
}

float Loadout::holsterTime(Holdable which) const {
  return which == Holdable::Pistol     ? timing.pistolHolster
         : which == Holdable::Launcher ? timing.launcherHolster
                                       : timing.longHolster;
}

void Loadout::update(float dt) {
  if (!(dt > 0.0f))
    return;
  float left = dt;
  // A frame can finish putting one away and start the next.
  for (int guard = 0; guard < 4 && left > 0.0f; ++guard) {
    if (current == Holdable::None) {
      if (wanted == Holdable::None)
        return;
      current = wanted;
      amount = 0.0f;
    }
    if (wanted == current) {
      const float rate = 1.0f / std::max(drawTime(current), 1e-3f);
      const float need = (1.0f - amount) / rate;
      amount = std::min(1.0f, amount + rate * left);
      left = std::max(0.0f, left - need);
      return;
    }
    // Putting it away first.
    const float rate = 1.0f / std::max(holsterTime(current), 1e-3f);
    const float need = amount / rate;
    if (need > left) {
      amount -= rate * left;
      return;
    }
    left -= need;
    amount = 0.0f;
    current = Holdable::None;
  }
}

#include "Weapon.h"

#include <algorithm>
#include <cmath>

glm::vec2 recoilKick(const WeaponSpec &spec) {
  const double cycle = 60.0 / std::max(spec.rate, 1.0f);
  const double back = std::max(spec.recoilReturn, 1e-3f);
  // What a follow-up round carries of the kicks before it, averaged over
  // the follow-ups of bursts of 3, 4 and 5: sum over m of e^(-2 m cycle / back).
  double carried = 0.0;
  int rounds = 0;
  for (int burst = 3; burst <= 5; ++burst)
    for (int k = 2; k <= burst; ++k, ++rounds)
      for (int m = 1; m < k; ++m)
        carried += std::exp(-2.0 * m * cycle / back);
  carried /= rounds;
  auto solve = [&](float follow, float first) {
    const double extra = static_cast<double>(follow) * follow - static_cast<double>(first) * first;
    return static_cast<float>(std::sqrt(std::max(extra, 0.0) / carried));
  };
  return {solve(spec.followSpread.x, spec.firstSpread.x),
          solve(spec.followSpread.y, spec.firstSpread.y)};
}

Weapon::Weapon(const WeaponSpec &spec, uint32_t seed)
    : gun(spec), kick(recoilKick(spec)), dice(seed),
      fireMode(spec.automatic ? FireMode::Auto : FireMode::Semi) {}

void Weapon::load(int magazines) {
  const int n = std::max(magazines, 0);
  chamber = n > 0;
  magazine = n > 0 ? gun.magazine : -1;
  spares.assign(static_cast<size_t>(std::max(n - 1, 0)), gun.magazine);
  locked = false;
  reloadTime = 0.0f;
  fired = 0;
  offset = glm::vec2(0.0f);
}

void Weapon::setPouch(std::vector<int> magazines) { spares = std::move(magazines); }

int Weapon::roundsInGun() const { return (chamber ? 1 : 0) + std::max(magazine, 0); }

int Weapon::rounds() const {
  int total = roundsInGun();
  for (int m : spares)
    total += m;
  return total;
}

void Weapon::switchMode() {
  if (gun.automatic)
    fireMode = fireMode == FireMode::Auto ? FireMode::Semi : FireMode::Auto;
}

glm::vec2 Weapon::gaussian(glm::vec2 sigma) {
  const float x = normal(dice);
  const float y = normal(dice);
  return {x * sigma.x, y * sigma.y};
}

void Weapon::update(float dt, bool trigger, FiringStance stance, WeaponEvents &events) {
  if (!(dt > 0.0f))
    return;
  if (trigger && !wasHeld)
    pullUsed = false;
  wasHeld = trigger;
  const float back = std::max(gun.recoilReturn, 1e-3f);
  float now = 0.0f; // how far into the frame the kicks have come back
  auto comeBack = [&](float to) {
    offset *= std::exp(-(to - now) / back);
    now = to;
  };

  if (reloadTime > 0.0f) {
    reloadTime -= dt;
    if (reloadTime <= 0.0f) {
      reloadTime = 0.0f;
      finishReload();
      events.reloaded = true;
    }
  } else if (trigger) {
    const float cycle = 60.0f / std::max(gun.rate, 1.0f);
    const float factor = stance == FiringStance::Standing ? gun.standing
                         : stance == FiringStance::Supported ? 1.0f
                                                       : gun.moving;
    const glm::vec2 spread =
        stance == FiringStance::Hip ? glm::vec2(gun.hipSpread) : gun.firstSpread * factor;
    float at = std::max(ready, 0.0f);
    while (at < dt) {
      if (fireMode == FireMode::Semi && pullUsed)
        break;
      if (!chamber) {
        if (!pullUsed)
          events.dry = true;
        pullUsed = true;
        break;
      }
      comeBack(at);
      Shot shot;
      shot.time = at;
      shot.deviation = offset + gaussian(spread);
      ++fired;
      shot.tracer = gun.tracerEvery > 0 && fired % gun.tracerEvery == 0;
      events.shots.push_back(shot);
      offset += gaussian(kick * factor);
      // The action cycles: the next round in, or empty (held open, if it is).
      chamber = false;
      if (magazine > 0) {
        --magazine;
        chamber = true;
      } else if (gun.locksOpen) {
        locked = true;
      }
      pullUsed = true;
      at += cycle;
    }
    ready = at;
  }
  ready = std::max(0.0f, ready - dt);
  comeBack(dt);
}

bool Weapon::reload() {
  if (reloadTime > 0.0f)
    return false;
  if (chamber && magazine >= gun.magazine)
    return false;
  const auto fullest = std::max_element(spares.begin(), spares.end());
  if (fullest == spares.end() || *fullest <= 0)
    return false;
  reloadEmpty = !chamber;
  reloadTime = reloadEmpty ? gun.emptyReload : gun.reload;
  if (reloadTime <= 0.0f) {
    finishReload();
    return true;
  }
  return true;
}

void Weapon::cancelReload() { reloadTime = 0.0f; }

void Weapon::finishReload() {
  // The fullest out of the pouch, the partial one back in its place (an
  // empty one is dropped), and a round chambered if there wasn't one.
  const auto fullest = std::max_element(spares.begin(), spares.end());
  if (fullest == spares.end())
    return;
  const int out = magazine;
  magazine = *fullest;
  spares.erase(fullest);
  if (out > 0)
    spares.push_back(out);
  if (!chamber && magazine > 0) {
    --magazine;
    chamber = true;
  }
  locked = false;
}

#include "PieceDamage.h"

#include <algorithm>

float pieceToughness(float volume, float energyPerVolume) {
  return std::max(volume, 0.0f) * std::max(energyPerVolume, 0.0f);
}

bool PieceDamage::add(float joules) {
  if (!(joules > 0.0f))
    return false;
  const bool was = broken();
  energy += joules;
  return !was && broken();
}

float PieceDamage::strengthLeft() const {
  if (!(toughness > 0.0f))
    return 1.0f;
  return std::clamp(1.0f - energy / toughness, 0.0f, 1.0f);
}

float linkStrengthLeft(const PieceDamage &a, const PieceDamage &b) {
  return std::min(a.strengthLeft(), b.strengthLeft());
}

LinkLimits weakened(const LinkLimits &limits, float share) {
  const float s = std::clamp(share, 0.0f, 1.0f);
  LinkLimits out = limits;
  out.pull *= s;
  out.push *= s;
  out.slide *= s;
  return out;
}

std::vector<float> shareDamage(float energy, const std::vector<float> &volumes) {
  std::vector<float> shares(volumes.size(), 0.0f);
  if (volumes.empty())
    return shares;
  float total = 0.0f;
  for (float v : volumes)
    total += std::max(v, 0.0f);
  for (size_t i = 0; i < volumes.size(); ++i)
    shares[i] = total > 0.0f ? energy * std::max(volumes[i], 0.0f) / total
                             : energy / static_cast<float>(volumes.size());
  return shares;
}

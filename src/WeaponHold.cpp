#include "WeaponHold.h"

#include <cmath>

glm::mat4 rigidFrame(const glm::mat4 &m) {
  const glm::vec3 x = glm::normalize(glm::vec3(m[0]));
  glm::vec3 y = glm::vec3(m[1]);
  y = glm::normalize(y - x * glm::dot(y, x));
  const glm::vec3 z = glm::cross(x, y);
  return glm::mat4(glm::vec4(x, 0.0f), glm::vec4(y, 0.0f), glm::vec4(z, 0.0f), m[3]);
}

WeaponHold holdFrom(const glm::mat4 &hand, glm::vec3 trigger, glm::vec3 forward,
                    std::optional<OffHand> off, glm::vec3 up) {
  WeaponHold hold;
  const glm::vec3 f = glm::normalize(forward);
  // Upright: up made square to the barrel (or any square way if they line up).
  glm::vec3 u = up - f * glm::dot(up, f);
  if (glm::length(u) < 1e-4f)
    u = std::abs(f.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
  u = glm::normalize(u - f * glm::dot(u, f));
  const glm::vec3 left = glm::cross(u, f);
  const glm::mat4 gun(glm::vec4(left, 0.0f), glm::vec4(u, 0.0f), glm::vec4(f, 0.0f),
                      glm::vec4(trigger, 1.0f));
  hold.gripInHand = glm::inverse(rigidFrame(hand)) * gun;
  if (off) {
    const glm::mat4 toGun = glm::inverse(gun);
    const glm::vec3 palm = glm::vec3(toGun * glm::vec4(off->palm, 1.0f));
    const glm::vec3 wrist = glm::vec3(toGun * glm::vec4(off->wrist, 1.0f));
    hold.twoHanded = glm::distance(off->palm, trigger) > kTwoHandedReach;
    // The palm onto the centre line at its height, the wrist moved with it.
    hold.offPalm = glm::vec3(0.0f, off->height, palm.z);
    hold.offHand = wrist + (hold.offPalm - palm);
  }
  return hold;
}

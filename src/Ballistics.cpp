#include "Ballistics.h"

#include <glm/gtc/constants.hpp>

#include <cmath>

glm::vec3 throwVelocity(glm::vec3 from, glm::vec3 to, float speed, glm::vec3 gravity) {
  const float g = glm::length(gravity);
  const glm::vec3 up = g > 0.0f ? -gravity / g : glm::vec3(0.0f, 1.0f, 0.0f);
  const glm::vec3 d = to - from;
  const float y = glm::dot(d, up);
  const glm::vec3 across = d - up * y;
  const float x = glm::length(across);
  if (x < 1e-4f || g <= 0.0f) {
    const float length = glm::length(d);
    return (length > 1e-6f ? d / length : up) * speed;
  }
  // The two angles that reach (x, y) at this speed; the smaller is the low arc.
  const float v2 = speed * speed;
  const float root = v2 * v2 - g * (g * x * x + 2.0f * y * v2);
  const float angle = root < 0.0f ? 0.25f * glm::pi<float>()
                                  : std::atan((v2 - std::sqrt(root)) / (g * x));
  return (across / x * std::cos(angle) + up * std::sin(angle)) * speed;
}

#pragma once

#include <glm/glm.hpp>

// A critically damped spring for each of three parts: a jolt that peaks and
// comes back without bouncing, like a blast's shake on the camera or a
// blow's on the body. Stepped exactly, so it's the same at any frame rate.
// No Vulkan, so it's tested like the rest of heliora_core.
struct SpringKick {
  glm::vec3 offset{0.0f};   // each part's own units (m, radians)
  glm::vec3 velocity{0.0f}; // per second
  // A jolt: the same parts, as high as they'd go from rest, at peak s.
  void kick(glm::vec3 amount, float peak);
  void update(float dt, float peak);
};

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// The renderer's camera projection. Reversed-Z: depth is 1 at the near plane
// and 0 at the far one, the opposite of the usual way round. Float depth has
// most of its precision near 0, and perspective squeezes far depths
// together, so reversing puts the precision where the squeeze is: far
// surfaces stop flickering against each other even with a near plane a few
// centimetres out. Y is flipped for Vulkan. Needs
// GLM_FORCE_DEPTH_ZERO_TO_ONE, which heliora_core sets.
inline glm::mat4 cameraProjection(float fovYRadians, float aspect,
                                  float nearPlane, float farPlane) {
  // glm's 0-to-1 perspective with near and far swapped is exactly that.
  glm::mat4 projection =
      glm::perspective(fovYRadians, aspect, farPlane, nearPlane);
  projection[1][1] *= -1.0f;
  return projection;
}

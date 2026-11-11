#include "Picking.h"

#include <cmath>

PickRay rayFromScreen(glm::vec2 pixel, glm::vec2 viewport,
                      const glm::mat4 &invViewProj) {
  // Vulkan's clip space has y pointing down like window pixels (the
  // projection's flip handles the rest), so no flip here.
  const glm::vec2 ndc = pixel / viewport * 2.0f - 1.0f;
  auto unproject = [&](float depth) {
    const glm::vec4 p = invViewProj * glm::vec4(ndc, depth, 1.0f);
    return glm::vec3(p) / p.w;
  };
  const glm::vec3 nearPoint = unproject(0.0f);
  const glm::vec3 farPoint = unproject(1.0f);
  return {nearPoint, glm::normalize(farPoint - nearPoint)};
}

std::optional<float> rayHitsSphere(const PickRay &ray, glm::vec3 centre,
                                   float radius) {
  // Points on the ray are origin + t * direction. Solving
  // |origin + t * direction - centre| = radius for t gives
  // t = b -+ sqrt(b^2 - c), with b and c as below (direction is unit length).
  const glm::vec3 toCentre = centre - ray.origin;
  const float b = glm::dot(toCentre, ray.direction);
  const float c = glm::dot(toCentre, toCentre) - radius * radius;
  const float discriminant = b * b - c;
  if (discriminant < 0.0f)
    return std::nullopt; // passes beside it
  const float root = std::sqrt(discriminant);
  if (b - root >= 0.0f)
    return b - root; // enters in front
  if (b + root >= 0.0f)
    return b + root; // starts inside: the far side
  return std::nullopt; // all of it is behind
}

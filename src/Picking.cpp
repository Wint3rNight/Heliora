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
  // Reversed-Z (cameraProjection): depth 1 is the near plane, 0 the far.
  const glm::vec3 nearPoint = unproject(1.0f);
  const glm::vec3 farPoint = unproject(0.0f);
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

std::optional<float> rayHitsCapsule(const PickRay &ray, glm::vec3 a,
                                    glm::vec3 b, float radius) {
  // A capsule is its two end spheres plus the cylinder between them, and a
  // ray from outside enters it where it enters the first of those.
  std::optional<float> nearest;
  auto keep = [&](std::optional<float> distance) {
    if (distance && (!nearest || *distance < *nearest))
      nearest = distance;
  };
  keep(rayHitsSphere(ray, a, radius));
  keep(rayHitsSphere(ray, b, radius));
  const float length = glm::distance(a, b);
  if (length <= 0.0f)
    return nearest;
  // The side: with the axis's part taken out of both, the ray's distance to
  // the axis is radius where |sideOrigin + t * sideDirection| = radius. A
  // ray along the axis has no side part and only meets the ends.
  const glm::vec3 axis = (b - a) / length;
  const glm::vec3 fromA = ray.origin - a;
  const glm::vec3 sideDirection =
      ray.direction - glm::dot(ray.direction, axis) * axis;
  const glm::vec3 sideOrigin = fromA - glm::dot(fromA, axis) * axis;
  const float qa = glm::dot(sideDirection, sideDirection);
  if (qa < 1e-12f)
    return nearest;
  const float qb = glm::dot(sideDirection, sideOrigin);
  const float qc = glm::dot(sideOrigin, sideOrigin) - radius * radius;
  const float discriminant = qb * qb - qa * qc;
  if (discriminant < 0.0f)
    return nearest;
  const float t = (-qb - std::sqrt(discriminant)) / qa; // where it enters
  const float along = glm::dot(fromA + t * ray.direction, axis);
  if (t >= 0.0f && along >= 0.0f && along <= length)
    keep(t);
  return nearest;
}

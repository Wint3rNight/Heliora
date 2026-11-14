#include "GizmoMath.h"

#include <cmath>

std::optional<glm::vec3> closestPointOnAxis(const PickRay &ray,
                                            glm::vec3 axisOrigin,
                                            glm::vec3 axisDir) {
  // Points on the axis are axisOrigin + s * axisDir, points on the ray are
  // ray.origin + t * ray.direction. At the closest pair, the line between
  // them is at right angles to both, which gives two equations in s and t.
  // Both directions are unit length, so they reduce to the ones below.
  const glm::vec3 between = axisOrigin - ray.origin;
  const float cosAngle = glm::dot(axisDir, ray.direction);
  const float alongAxis = glm::dot(axisDir, between);
  const float alongRay = glm::dot(ray.direction, between);
  // 1 - cos^2 is sin^2 of the angle between them: 0 when they're parallel,
  // and then every point is equally close.
  const float denominator = 1.0f - cosAngle * cosAngle;
  if (denominator < 1e-6f)
    return std::nullopt;
  const float s = (cosAngle * alongRay - alongAxis) / denominator;
  const float t = (alongRay - cosAngle * alongAxis) / denominator;
  if (t < 0.0f)
    return std::nullopt; // behind the camera
  return axisOrigin + s * axisDir;
}

std::optional<float> angleAroundAxis(const PickRay &ray, glm::vec3 centre,
                                     glm::vec3 axis) {
  // Where the ray meets the plane: dot(origin + t * direction - centre,
  // axis) = 0.
  const float facing = glm::dot(ray.direction, axis);
  if (std::abs(facing) < 1e-4f)
    return std::nullopt; // runs along the plane
  const float t = glm::dot(centre - ray.origin, axis) / facing;
  if (t < 0.0f)
    return std::nullopt;
  const glm::vec3 offset = ray.origin + t * ray.direction - centre;
  // Two directions in the plane, at right angles: u, then v a quarter turn
  // further the right-hand way, since cross(u, v) = axis.
  const glm::vec3 helper = std::abs(axis.y) < 0.9f ? glm::vec3(0, 1, 0)
                                                   : glm::vec3(1, 0, 0);
  const glm::vec3 u = glm::normalize(glm::cross(axis, helper));
  const glm::vec3 v = glm::cross(axis, u);
  return std::atan2(glm::dot(offset, v), glm::dot(offset, u));
}

float snap(float value, float step) {
  if (step <= 0.0f)
    return value;
  return std::round(value / step) * step;
}

std::optional<glm::vec2> screenFromWorld(glm::vec3 point, glm::vec2 viewport,
                                         const glm::mat4 &viewProj) {
  const glm::vec4 clip = viewProj * glm::vec4(point, 1.0f);
  // w is the distance in front of the camera; at or behind it, the divide
  // below would mirror the point onto the screen.
  if (clip.w <= 1e-6f)
    return std::nullopt;
  // No y flip, as in rayFromScreen: the projection already flips it.
  const glm::vec2 ndc = glm::vec2(clip) / clip.w;
  return (ndc + 1.0f) * 0.5f * viewport;
}

float distanceToSegment(glm::vec2 point, glm::vec2 a, glm::vec2 b) {
  const glm::vec2 ab = b - a;
  const float lengthSquared = glm::dot(ab, ab);
  // How far along the segment the nearest point is, 0 at a and 1 at b.
  const float along =
      lengthSquared > 0.0f
          ? glm::clamp(glm::dot(point - a, ab) / lengthSquared, 0.0f, 1.0f)
          : 0.0f;
  return glm::distance(point, a + along * ab);
}

float scaleFromDrag(glm::vec2 centre, glm::vec2 grabbed, glm::vec2 now) {
  const float from = glm::distance(grabbed, centre);
  if (from < 1.0f)
    return 1.0f; // less than a pixel out
  return glm::distance(now, centre) / from;
}

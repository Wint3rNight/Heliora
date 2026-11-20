#include "EffectGeometry.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

namespace {

// Two unit directions square to u (unit) and to each other.
void across(glm::vec3 u, glm::vec3 &side, glm::vec3 &up) {
  const glm::vec3 ref = std::fabs(u.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
  side = glm::normalize(glm::cross(u, ref));
  up = glm::cross(side, u);
}

// A strip running within about 3 degrees of straight at the eye keeps the
// way across it had before: square to both is no direction at all there.
constexpr float kEndOn = 0.05f;

EffectVertex vertex(glm::vec3 at, glm::vec4 core, glm::vec4 edge, glm::vec2 uv) {
  return {at, core, edge, uv};
}

// An angle round a circle as a distance along it from 0 either way, so the
// flow coordinate meets itself where the circle closes instead of jumping
// from all the way round back to 0.
float round(float angle) { return std::min(angle, glm::two_pi<float>() - angle); }

// Two triangles over the quad a0 b0 b1 a1 (a the side with uv.y 0).
void quad(std::vector<EffectVertex> &out, const EffectVertex &a0, const EffectVertex &b0,
          const EffectVertex &a1, const EffectVertex &b1) {
  out.insert(out.end(), {a0, b0, b1, a0, b1, a1});
}

} // namespace

void addFacingStrip(std::vector<EffectVertex> &out, const std::vector<glm::vec3> &points,
                    glm::vec3 eye, float width0, float width1, glm::vec4 core0, glm::vec4 core1,
                    glm::vec4 edge) {
  const size_t n = points.size();
  if (n < 2)
    return;
  std::vector<float> along(n, 0.0f);
  for (size_t i = 1; i < n; ++i)
    along[i] = along[i - 1] + glm::distance(points[i - 1], points[i]);
  const float total = along.back();
  std::vector<EffectVertex> left(n), right(n);
  glm::vec3 last(0.0f);
  for (size_t i = 0; i < n; ++i) {
    // Across the strip: square to the way it runs here and to the eye, so it
    // shows its face. Seen end on that's ill-defined and flips as the path
    // wobbles across the line of sight, so it keeps the last one's way round,
    // and where it runs almost straight at the eye it keeps the last one.
    const glm::vec3 tangent = points[std::min(i + 1, n - 1)] - points[i > 0 ? i - 1 : 0];
    const glm::vec3 view = eye - points[i];
    glm::vec3 side = glm::cross(tangent, view);
    const float sine = glm::length(side) / std::max(glm::length(tangent) * glm::length(view), 1e-12f);
    if (sine < kEndOn && i > 0) {
      side = last;
    } else if (glm::length(side) < 1e-9f) {
      glm::vec3 up;
      across(glm::length(tangent) > 1e-6f ? glm::normalize(tangent) : glm::vec3(1, 0, 0), side, up);
    }
    side = glm::normalize(side);
    if (i > 0 && glm::dot(side, last) < 0.0f)
      side = -side;
    last = side;
    const float f = total > 0.0f ? along[i] / total : static_cast<float>(i) / (n - 1);
    const float half = 0.5f * glm::mix(width0, width1, f);
    const glm::vec4 core = glm::mix(core0, core1, f);
    left[i] = vertex(points[i] - side * half, core, edge, {along[i], 0.0f});
    right[i] = vertex(points[i] + side * half, core, edge, {along[i], 1.0f});
  }
  out.reserve(out.size() + (n - 1) * 6);
  for (size_t i = 0; i + 1 < n; ++i)
    quad(out, left[i], right[i], left[i + 1], right[i + 1]);
}

void addCone(std::vector<EffectVertex> &out, glm::vec3 apex, glm::vec3 axis, float length,
             float halfAngle, glm::vec4 core, glm::vec4 edge, int segments) {
  segments = std::max(segments, 3);
  glm::vec3 side, up;
  across(axis, side, up);
  // Its rim a full length from the apex along its side, so it reaches as
  // far every way it covers.
  const float rim = length * std::sin(halfAngle);
  const float ahead = length * std::cos(halfAngle);
  auto at = [&](int k, bool atRim) {
    const float a = glm::two_pi<float>() * k / segments;
    const glm::vec3 p = atRim ? apex + axis * ahead + (std::cos(a) * side + std::sin(a) * up) * rim
                              : apex;
    return vertex(p, core, edge, {round(a) * rim, atRim ? 1.0f : 0.0f});
  };
  out.reserve(out.size() + segments * 6);
  for (int k = 0; k < segments; ++k)
    quad(out, at(k, false), at(k, true), at(k + 1, false), at(k + 1, true));
}

std::vector<glm::vec3> ribbonStrand(glm::vec3 from, glm::vec3 to, int strand, float time,
                                    int segments) {
  segments = std::max(segments, 1);
  const glm::vec3 d = to - from;
  const float length = glm::length(d);
  std::vector<glm::vec3> points(segments + 1, from);
  if (length < 1e-5f)
    return points;
  glm::vec3 side, up;
  across(d / length, side, up);
  // Each strand bows its own way round the line (about 120 degrees apart),
  // swaying, and a ripple runs along it toward what it holds.
  const float s = static_cast<float>(strand);
  const float angle = 2.1f * s + 0.35f * std::sin(1.3f * time + 1.7f * s);
  const glm::vec3 bowWay = std::cos(angle) * up + std::sin(angle) * side;
  const glm::vec3 rippleWay = -std::sin(angle) * up + std::cos(angle) * side;
  const float bow = length * (0.10f + 0.03f * std::sin(2.3f * time + s));
  for (int i = 0; i <= segments; ++i) {
    const float t = static_cast<float>(i) / segments;
    const float pinned = std::sin(glm::pi<float>() * t); // 0 at both ends
    const float ripple = 0.02f * length * pinned * std::sin(9.0f * t - 6.0f * time + 2.0f * s);
    points[i] = from + d * t + bowWay * (bow * 4.0f * t * (1.0f - t)) + rippleWay * ripple;
  }
  points.back() = to;
  return points;
}

void addGroundRing(std::vector<EffectVertex> &out, glm::vec3 centre, float radius, float width,
                   glm::vec4 core, glm::vec4 edge, int segments) {
  segments = std::max(segments, 3);
  const float inner = std::max(0.0f, radius - 0.5f * width), outer = radius + 0.5f * width;
  auto at = [&](int k, float r, float v) {
    const float a = glm::two_pi<float>() * k / segments;
    return vertex(centre + glm::vec3(std::cos(a), 0.0f, std::sin(a)) * r, core, edge,
                  {round(a) * radius, v});
  };
  out.reserve(out.size() + segments * 6);
  for (int k = 0; k < segments; ++k)
    quad(out, at(k, inner, 0.0f), at(k, outer, 1.0f), at(k + 1, inner, 0.0f),
         at(k + 1, outer, 1.0f));
}

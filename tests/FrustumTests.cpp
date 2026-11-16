#include "Frustum.h"
#include "ShadowCascades.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace {

// The sandbox's default sun, and the cascades' far plane (as in the cascade
// tests).
const glm::vec3 kSun = glm::normalize(glm::vec3(-0.282f, -0.941f, 0.188f));
constexpr float kCsmFar = 80.0f;

struct TestCamera {
  glm::vec3 position;
  glm::vec3 target;
  float fovDeg;
  float drawDistance;
};

const TestCamera kCameras[] = {
    {{0.2f, 16.7f, 21.3f}, {0, 0, 0}, 45.4f, 20000.0f},
    {{0.0f, 2.0f, 0.0f}, {0, 2, -10}, 65.0f, 200.0f},
    {{0.0f, 30.0f, 50.0f}, {0, 0, 0}, 65.0f, 500.0f}, // the 5k scene's camera
};

glm::mat4 viewOf(const TestCamera &c) {
  return glm::lookAt(c.position, c.target, glm::vec3(0, 1, 0));
}

glm::mat4 projectionOf(const TestCamera &c) {
  glm::mat4 p = glm::perspective(glm::radians(c.fovDeg), 16.0f / 9.0f, 0.1f,
                                 c.drawDistance);
  p[1][1] *= -1.0f;
  return p;
}

ShadowCascades cascadesOf(const TestCamera &c) {
  return fitShadowCascades(viewOf(c), projectionOf(c), kSun, kCsmFar,
                           c.drawDistance, 2048);
}

bool sphereKept(const glm::mat4 &viewProj, glm::vec3 center, float radius) {
  glm::vec4 planes[6];
  extractFrustumPlanes(viewProj, planes);
  return sphereInFrustum(planes, center, radius);
}

// Inside what the rasterizer keeps: x and y from -1 to 1, depth 0 to 1.
bool insideClipVolume(const glm::mat4 &viewProj, glm::vec3 point) {
  const glm::vec4 clip = viewProj * glm::vec4(point, 1.0f);
  const glm::vec3 p = glm::vec3(clip) / clip.w;
  return std::abs(p.x) <= 1 && std::abs(p.y) <= 1 && p.z >= 0 && p.z <= 1;
}

// Every point the rasterizer keeps: x and y from -1 to 1, depth 0 to 1.
// Returns false at the first one the sphere test would drop.
bool keepsEverythingDrawn(const glm::mat4 &viewProj) {
  const glm::mat4 toWorld = glm::inverse(viewProj);
  for (float x : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f})
    for (float y : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f})
      for (float z : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        const glm::vec4 w = toWorld * glm::vec4(x, y, z, 1.0f);
        // A millimetre of slack for float error on the boundary itself.
        if (!sphereKept(viewProj, glm::vec3(w) / w.w, 0.001f))
          return false;
      }
  return true;
}

// The same for the camera, with its points built from its own geometry:
// rebuilding them from the perspective matrix puts far points metres off
// (the depth is that non-linear with a 0.1 m near plane). The far plane
// itself is only good to about 0.2% at 20 km with that near plane (float
// cancellation in the matrix), so the farthest points sit at 99%.
bool keepsEverythingInView(const TestCamera &c) {
  const glm::mat4 viewProj = projectionOf(c) * viewOf(c);
  const glm::vec3 forward = glm::normalize(c.target - c.position);
  const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
  const glm::vec3 up = glm::cross(right, forward);
  const float tanY = std::tan(glm::radians(c.fovDeg) * 0.5f);
  const float tanX = tanY * 16.0f / 9.0f;
  for (float depth : {0.1f, 1.0f, 10.0f, 0.5f * c.drawDistance,
                      0.99f * c.drawDistance})
    for (float x : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f})
      for (float y : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}) {
        const glm::vec3 point = c.position + forward * depth +
                                right * (x * depth * tanX) +
                                up * (y * depth * tanY);
        // Slack for float error, growing with the distance.
        if (!sphereKept(viewProj, point, 0.001f + 1e-5f * depth))
          return false;
      }
  return true;
}

} // namespace

// A shadow caster up towards the sun from what a cascade covers casts into
// it: the cascade's depth range has to reach it (each box reaches csmFar
// towards the sun past its slice) and the caster culling has to keep it.
TEST(casters_between_the_sun_and_a_cascade_are_kept) {
  for (const TestCamera &c : kCameras) {
    const ShadowCascades cascades = cascadesOf(c);
    // Every cascade is a sphere around the camera, so it covers the camera.
    for (float upTowardsSun : {5.0f, 40.0f, 75.0f}) {
      const glm::vec3 caster = c.position - kSun * upTowardsSun;
      for (int i = 0; i < kShadowCascades; ++i) {
        CHECK(insideClipVolume(cascades.lightSpace[i], caster));
        CHECK(sphereKept(cascades.lightSpace[i], caster, 0.5f));
      }
    }
  }
}

TEST(casters_outside_a_cascades_footprint_are_dropped) {
  // Sideways to the sun: a caster there can't shade anything in the cascade.
  const glm::vec3 sideways = glm::normalize(glm::cross(kSun, glm::vec3(0, 0, 1)));
  for (const TestCamera &c : kCameras) {
    const ShadowCascades cascades = cascadesOf(c);
    const glm::vec3 caster = c.position + sideways * 500.0f;
    for (int i = 0; i < kShadowCascades; ++i)
      CHECK(!sphereKept(cascades.lightSpace[i], caster, 0.5f));
  }
}

// The culling may keep extra, but never drops what would be drawn: for the
// shadow cascades and for the camera.
TEST(frustum_culling_keeps_everything_the_rasterizer_draws) {
  for (const TestCamera &c : kCameras) {
    const ShadowCascades cascades = cascadesOf(c);
    for (int i = 0; i < kShadowCascades; ++i)
      CHECK(keepsEverythingDrawn(cascades.lightSpace[i]));
    CHECK(keepsEverythingInView(c));
  }
}

TEST(spheres_behind_the_camera_are_dropped) {
  for (const TestCamera &c : kCameras) {
    const glm::vec3 forward = glm::normalize(c.target - c.position);
    CHECK(!sphereKept(projectionOf(c) * viewOf(c), c.position - forward * 20.0f,
                      1.0f));
  }
}

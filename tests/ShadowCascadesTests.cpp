#include "ShadowCascades.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {

// The sandbox's default sun.
const glm::vec3 kSun = glm::normalize(glm::vec3(-0.282f, -0.941f, 0.188f));
constexpr float kCsmFar = 80.0f;

struct TestCamera {
  const char *name;
  glm::vec3 position;
  glm::vec3 target;
  float fovDeg;
  float drawDistance;
};

const TestCamera kCameras[] = {
    // The user's screenshot on 2026-11-01: the floor in front went dark.
    {"high, narrow FOV, far draw distance", {0.2f, 16.7f, 21.3f}, {0, 0, 0}, 45.4f, 20000.0f},
    {"high, wide FOV", {0.2f, 16.7f, 21.3f}, {0, 0, 0}, 65.0f, 200.0f},
    {"standing in Sponza", {0.0f, 2.0f, 0.0f}, {0, 2, -10}, 65.0f, 200.0f},
    {"far away, zoomed in", {0.0f, 40.0f, 40.0f}, {0, 0, 0}, 30.0f, 500.0f},
};

// The renderer's camera: depth 0 to 1 and Y flipped for Vulkan.
void matricesFor(const TestCamera &c, glm::mat4 &view, glm::mat4 &projection) {
  view = glm::lookAt(c.position, c.target, glm::vec3(0, 1, 0));
  projection = glm::perspective(glm::radians(c.fovDeg), 16.0f / 9.0f, 0.1f,
                                c.drawDistance);
  projection[1][1] *= -1.0f;
}

bool insideShadowMap(const glm::mat4 &lightSpace, glm::vec3 point) {
  const glm::vec4 clip = lightSpace * glm::vec4(point, 1.0f);
  const glm::vec3 p = glm::vec3(clip) / clip.w;
  const glm::vec2 uv = glm::vec2(p) * 0.5f + 0.5f;
  return uv.x >= 0 && uv.x <= 1 && uv.y >= 0 && uv.y <= 1 && p.z >= 0 &&
         p.z <= 1;
}

} // namespace

TEST(every_cascade_contains_its_own_slice) {
  for (const TestCamera &c : kCameras) {
    glm::mat4 view, projection;
    matricesFor(c, view, projection);
    const ShadowCascades cascades =
        fitShadowCascades(view, projection, kSun, kCsmFar, c.drawDistance, 2048);
    const glm::mat4 invViewProj = glm::inverse(projection * view);
    auto ndcDepth = [&](float viewDepth) {
      const glm::vec4 clip = projection * glm::vec4(0, 0, -viewDepth, 1);
      return clip.z / clip.w;
    };

    float sliceNear = 0.1f;
    for (int i = 0; i < kShadowCascades; ++i) {
      // The slice's 8 corners, pulled 5% towards its centre, plus the centre.
      glm::vec3 corners[8];
      glm::vec3 centre(0.0f);
      int k = 0;
      for (float x : {-1.0f, 1.0f})
        for (float y : {-1.0f, 1.0f})
          for (float z : {ndcDepth(sliceNear), ndcDepth(cascades.splits[i])}) {
            const glm::vec4 w = invViewProj * glm::vec4(x, y, z, 1.0f);
            corners[k] = glm::vec3(w) / w.w;
            centre += corners[k++] / 8.0f;
          }
      int outside = insideShadowMap(cascades.lightSpace[i], centre) ? 0 : 1;
      for (const glm::vec3 &corner : corners)
        if (!insideShadowMap(cascades.lightSpace[i],
                             glm::mix(corner, centre, 0.05f)))
          ++outside;
      if (outside > 0)
        std::printf("    %s, cascade %d: %d of 9 points outside\n", c.name, i,
                    outside);
      CHECK(outside == 0);
      sliceNear = cascades.splits[i];
    }
  }
}

TEST(floor_in_front_of_a_narrow_camera_gets_a_cascade) {
  // The exact case from the screenshot: this floor point was outside all four
  // cascades, and lit.frag treats that as fully in shadow.
  const TestCamera &c = kCameras[0];
  glm::mat4 view, projection;
  matricesFor(c, view, projection);
  const ShadowCascades cascades =
      fitShadowCascades(view, projection, kSun, kCsmFar, c.drawDistance, 2048);
  bool covered = false;
  for (const glm::mat4 &lightSpace : cascades.lightSpace)
    covered = covered || insideShadowMap(lightSpace, glm::vec3(0, 0, 4));
  CHECK(covered);
}

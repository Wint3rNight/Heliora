#include "Picking.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {

constexpr glm::vec2 kViewport(1600.0f, 900.0f);

// The renderer's camera: depth 0 to 1 (GLM_FORCE_DEPTH_ZERO_TO_ONE) and y
// flipped for Vulkan, as in VulkanRenderer::rebuildProjection.
glm::mat4 invViewProj(glm::vec3 eye, glm::vec3 forward) {
  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), kViewport.x / kViewport.y, 1.0f, 200.0f);
  projection[1][1] *= -1.0f;
  const glm::mat4 view =
      glm::lookAt(eye, eye + forward, glm::vec3(0.0f, 1.0f, 0.0f));
  return glm::inverse(projection * view);
}

} // namespace

TEST(centre_pixel_ray_points_forward) {
  const glm::vec3 eye(1.0f, 2.0f, 3.0f);
  const glm::vec3 forward = glm::normalize(glm::vec3(1.0f, 0.0f, -2.0f));
  const PickRay ray =
      rayFromScreen(kViewport * 0.5f, kViewport, invViewProj(eye, forward));
  CHECK_NEAR(glm::length(ray.direction), 1.0, 1e-5);
  CHECK_NEAR(glm::dot(ray.direction, forward), 1.0, 1e-4);
  // It starts on the near plane, 1 m in front of the camera.
  CHECK_NEAR(glm::distance(ray.origin, eye + forward), 0.0, 1e-3);
}

TEST(top_pixel_ray_points_up_and_left_pixel_points_left) {
  // Window pixels count down from the top; the flipped projection must not
  // turn that upside down.
  const glm::mat4 inv = invViewProj(glm::vec3(0.0f), glm::vec3(0, 0, -1));
  const PickRay top = rayFromScreen(glm::vec2(800.0f, 0.0f), kViewport, inv);
  CHECK(top.direction.y > 0.3f); // half of the 45 degree field of view
  const PickRay left = rayFromScreen(glm::vec2(0.0f, 450.0f), kViewport, inv);
  CHECK(left.direction.x < -0.5f);
}

TEST(ray_hits_sphere_in_front) {
  const PickRay ray{glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
  const std::optional<float> distance =
      rayHitsSphere(ray, glm::vec3(0.0f, 0.0f, -10.0f), 2.0f);
  CHECK(distance.has_value());
  CHECK_NEAR(distance.value_or(-1.0f), 8.0, 1e-4);
}

TEST(ray_misses_sphere_behind) {
  const PickRay ray{glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
  CHECK(!rayHitsSphere(ray, glm::vec3(0.0f, 0.0f, 10.0f), 2.0f));
}

TEST(ray_misses_sphere_off_to_the_side) {
  const PickRay ray{glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
  CHECK(!rayHitsSphere(ray, glm::vec3(5.0f, 0.0f, -10.0f), 2.0f));
}

TEST(ray_from_inside_sphere_hits_its_far_side) {
  // So a big object around the camera doesn't win every click: anything
  // nearer than its far side still gets picked first.
  const PickRay ray{glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
  const std::optional<float> distance =
      rayHitsSphere(ray, glm::vec3(0.0f, 0.0f, -1.0f), 3.0f);
  CHECK_NEAR(distance.value_or(-1.0f), 4.0, 1e-4);
}

#include "CameraProjection.h"
#include "GizmoMath.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <numbers>

namespace {

constexpr glm::vec2 kViewport(1600.0f, 900.0f);

// The renderer's camera (reversed-Z, y flipped for Vulkan).
glm::mat4 viewProj(glm::vec3 eye, glm::vec3 target) {
  const glm::mat4 projection = cameraProjection(
      glm::radians(45.0f), kViewport.x / kViewport.y, 0.1f, 200.0f);
  return projection * glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

// The cursor's ray when it's over the point `to`.
PickRay rayThrough(glm::vec3 from, glm::vec3 to) {
  return {from, glm::normalize(to - from)};
}

} // namespace

TEST(drag_along_x_moves_only_x) {
  // A camera up and off to the side, and the cursor going from half a metre
  // along the x handle to 4.5 m along it.
  const glm::vec3 eye(-2.0f, 6.0f, 10.0f);
  const glm::vec3 origin(1.0f, 2.0f, 3.0f);
  const glm::vec3 x(1.0f, 0.0f, 0.0f);
  const std::optional<glm::vec3> grabbed = closestPointOnAxis(
      rayThrough(eye, origin + glm::vec3(0.5f, 0.0f, 0.0f)), origin, x);
  const std::optional<glm::vec3> now = closestPointOnAxis(
      rayThrough(eye, origin + glm::vec3(4.5f, 0.0f, 0.0f)), origin, x);
  CHECK(grabbed.has_value());
  CHECK(now.has_value());
  if (!grabbed || !now)
    return;
  const glm::vec3 moved = *now - *grabbed;
  CHECK_NEAR(moved.x, 4.0, 1e-3);
  CHECK(moved.y == 0.0f);
  CHECK(moved.z == 0.0f);
}

TEST(ray_along_the_axis_gives_no_point) {
  const PickRay ray{glm::vec3(-5.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)};
  CHECK(!closestPointOnAxis(ray, glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
}

TEST(angle_around_y_matches_the_mouse) {
  // The cursor goes 30 degrees around a y ring, seen from above at an angle.
  const glm::vec3 eye(3.0f, 8.0f, 6.0f);
  const glm::vec3 centre(1.0f, 0.5f, -2.0f);
  const glm::vec3 y(0.0f, 1.0f, 0.0f);
  const float turn = glm::radians(30.0f);
  const glm::vec3 first(2.0f, 0.0f, 0.0f);
  const glm::vec3 second(2.0f * std::cos(turn), 0.0f, -2.0f * std::sin(turn));
  const std::optional<float> a = angleAroundAxis(
      rayThrough(eye, centre + first), centre, y);
  const std::optional<float> b = angleAroundAxis(
      rayThrough(eye, centre + second), centre, y);
  CHECK(a.has_value());
  CHECK(b.has_value());
  if (!a || !b)
    return;
  const float change = std::remainder(*b - *a, 2.0f * std::numbers::pi_v<float>);
  CHECK_NEAR(change, turn, 1e-4);
  // Turning by that change carries the point that was under the cursor to
  // where the cursor is now.
  const glm::vec3 carried = glm::angleAxis(change, y) * first;
  CHECK_NEAR(glm::distance(carried, second), 0.0, 1e-3);
}

TEST(ray_in_the_ring_plane_gives_no_angle) {
  const PickRay ray{glm::vec3(-5.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)};
  CHECK(!angleAroundAxis(ray, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)));
}

TEST(snap_rounds_to_the_nearest_step) {
  CHECK_NEAR(snap(0.37f, 0.25f), 0.25, 1e-6);
  CHECK_NEAR(snap(0.38f, 0.25f), 0.5, 1e-6);
  CHECK_NEAR(snap(-0.13f, 0.25f), -0.25, 1e-6);
  CHECK_NEAR(snap(52.0f, 15.0f), 45.0, 1e-5);
  CHECK_NEAR(snap(1.26f, 0.1f), 1.3, 1e-6);
  CHECK(snap(0.37f, 0.0f) == 0.37f);
}

TEST(point_on_screen_lies_on_the_ray_through_its_pixel) {
  const glm::mat4 vp = viewProj(glm::vec3(4.0f, 3.0f, 8.0f), glm::vec3(0, 1, 0));
  const glm::vec3 point(1.5f, 0.5f, -1.0f);
  const std::optional<glm::vec2> pixel = screenFromWorld(point, kViewport, vp);
  CHECK(pixel.has_value());
  if (!pixel)
    return;
  const PickRay ray = rayFromScreen(*pixel, kViewport, glm::inverse(vp));
  const glm::vec3 toPoint = point - ray.origin;
  const glm::vec3 offRay =
      toPoint - ray.direction * glm::dot(toPoint, ray.direction);
  CHECK_NEAR(glm::length(offRay), 0.0, 1e-3);
}

TEST(point_behind_the_camera_has_no_pixel) {
  const glm::mat4 vp = viewProj(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
  CHECK(!screenFromWorld(glm::vec3(0.0f, 0.0f, 5.0f), kViewport, vp));
}

TEST(distance_to_segment_clamps_to_its_ends) {
  const glm::vec2 a(0.0f, 0.0f);
  const glm::vec2 b(10.0f, 0.0f);
  CHECK_NEAR(distanceToSegment(glm::vec2(5.0f, 3.0f), a, b), 3.0, 1e-5);
  CHECK_NEAR(distanceToSegment(glm::vec2(13.0f, 4.0f), a, b), 5.0, 1e-5);
  CHECK_NEAR(distanceToSegment(glm::vec2(-3.0f, -4.0f), a, b), 5.0, 1e-5);
  // A segment with no length is a point.
  CHECK_NEAR(distanceToSegment(glm::vec2(3.0f, 4.0f), a, a), 5.0, 1e-5);
}

TEST(dragging_twice_as_far_out_doubles_the_scale) {
  const glm::vec2 centre(800.0f, 450.0f);
  CHECK_NEAR(scaleFromDrag(centre, centre + glm::vec2(50.0f, 0.0f),
                           centre + glm::vec2(0.0f, -100.0f)),
             2.0, 1e-5);
  CHECK_NEAR(scaleFromDrag(centre, centre + glm::vec2(80.0f, 60.0f),
                           centre + glm::vec2(40.0f, 30.0f)),
             0.5, 1e-5);
  // Grabbed on the centre itself, there's no distance to compare with.
  CHECK(scaleFromDrag(centre, centre, centre + glm::vec2(30.0f, 0.0f)) ==
        1.0f);
}

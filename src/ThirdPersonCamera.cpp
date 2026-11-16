#include "ThirdPersonCamera.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979f;
const glm::vec3 kUp(0.0f, 1.0f, 0.0f);

// The share of the gap that smoothing with this time constant closes in dt.
float smoothing(float dt, float timeConstant) {
  return timeConstant > 0.0f ? 1.0f - std::exp(-dt / timeConstant) : 1.0f;
}

} // namespace

glm::vec3 cameraForward(float yaw, float pitch) {
  const float flat = std::cos(pitch);
  return glm::vec3(-std::sin(yaw) * flat, std::sin(pitch),
                   -std::cos(yaw) * flat);
}

ThirdPersonCamera::ThirdPersonCamera(const ThirdPersonCameraTuning &tuning)
    : tune(tuning),
      zoomDistance(std::clamp(tuning.distance, tuning.minDistance,
                              tuning.maxDistance)) {}

void ThirdPersonCamera::orbit(float mouseDx, float mouseDy) {
  yawAngle = std::remainder(yawAngle - mouseDx * tune.mouseSensitivity,
                            2.0f * kPi);
  pitchAngle = std::clamp(pitchAngle - mouseDy * tune.mouseSensitivity,
                          tune.minPitch, tune.maxPitch);
}

void ThirdPersonCamera::zoom(float notches) {
  zoomDistance = std::clamp(zoomDistance - notches * tune.zoomStep,
                            tune.minDistance, tune.maxDistance);
}

void ThirdPersonCamera::setTuning(const ThirdPersonCameraTuning &tuning) {
  tune = tuning;
  zoomDistance = std::clamp(zoomDistance, tune.minDistance, tune.maxDistance);
  pitchAngle = std::clamp(pitchAngle, tune.minPitch, tune.maxPitch);
}

void ThirdPersonCamera::snapTo(const PhysicsWorld &world, glm::vec3 feet,
                               float yaw, float pitch) {
  yawAngle = std::remainder(yaw, 2.0f * kPi);
  pitchAngle = std::clamp(pitch, tune.minPitch, tune.maxPitch);
  aimBlend = 0.0f;
  pivotPoint = feet + kUp * tune.pivotHeight;
  place(world, true, 0.0f);
}

void ThirdPersonCamera::update(const PhysicsWorld &world, glm::vec3 feet,
                               bool aiming, float dt) {
  const float aimStep = tune.aimTime > 0.0f ? dt / tune.aimTime : 1.0f;
  aimBlend = std::clamp(aimBlend + (aiming ? aimStep : -aimStep), 0.0f, 1.0f);

  // Catch up with the shoulders smoothly, but never trail so far that the
  // character slides out of the middle of the view.
  const glm::vec3 shoulders = feet + kUp * tune.pivotHeight;
  pivotPoint += (shoulders - pivotPoint) * smoothing(dt, tune.followTime);
  const glm::vec3 trail = pivotPoint - shoulders;
  if (glm::length(trail) > tune.maxLag)
    pivotPoint = shoulders + glm::normalize(trail) * tune.maxLag;

  place(world, false, dt);
}

void ThirdPersonCamera::place(const PhysicsWorld &world, bool snap, float dt) {
  // Where the eye wants to be: behind the pivot, or over the right shoulder
  // and closer while aiming. Smoothstep eases the slide in and out.
  const float slide = aimBlend * aimBlend * (3.0f - 2.0f * aimBlend);
  const glm::vec3 right(std::cos(yawAngle), 0.0f, -std::sin(yawAngle));
  const glm::vec3 shoulder =
      pivotPoint + (right * tune.aimRight + kUp * tune.aimUp) * slide;
  const float distance =
      glm::mix(zoomDistance, std::min(zoomDistance, tune.aimDistance), slide);
  const glm::vec3 toEye = shoulder - forward() * distance - pivotPoint;
  const float length = glm::length(toEye);
  const glm::vec3 direction = toEye / length;

  // Sweep a ball from the pivot towards it: the first thing it touches is as
  // far as the eye may go, with the ball's radius kept clear around the eye.
  const std::optional<PhysicsRayHit> hit =
      world.castSphere(pivotPoint, direction, tune.collisionRadius, length);
  const float target = hit ? std::min(hit->distance, length) : length;
  if (snap || target <= clearance || !easingOut)
    clearance = target; // pulled in at once, or nothing to recover from
  else
    clearance = target + (clearance - target) * std::exp(-dt / tune.easeOutTime);
  easingOut = (hit && hit->distance < length) || clearance < target - 1e-3f;

  eye = pivotPoint + direction * clearance;
}

glm::mat4 ThirdPersonCamera::view() const {
  return glm::lookAt(eye, eye + forward(), kUp);
}

ThirdPersonCameraTuning consistent(ThirdPersonCameraTuning tuning) {
  tuning.maxDistance = std::max(tuning.maxDistance, tuning.minDistance);
  tuning.distance =
      std::clamp(tuning.distance, tuning.minDistance, tuning.maxDistance);
  tuning.minPitch = std::min(tuning.minPitch, 0.0f);
  tuning.maxPitch = std::max(tuning.maxPitch, 0.0f);
  return tuning;
}

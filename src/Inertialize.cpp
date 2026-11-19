#include "Inertialize.h"

#include <algorithm>
#include <cmath>

QuinticDecay QuinticDecay::make(float x0, float v0, float t1) {
  QuinticDecay q;
  q.x0 = x0;
  // Moving away from the target counts as standing still (Bollo).
  q.v0 = std::min(v0, 0.0f);
  q.t1 = t1;
  // A fast approach would overshoot inside t1: end sooner instead.
  if (x0 > 0.0f && q.v0 < 0.0f)
    q.t1 = std::min(t1, -5.0f * x0 / q.v0);
  // Nothing to ease, or too short to ease (t1^5 would underflow): gone at
  // once.
  if (x0 <= 0.0f || q.t1 < 1e-4f) {
    q.x0 = 0.0f;
    q.v0 = 0.0f;
    q.t1 = 0.0f;
    return q;
  }
  const float t = q.t1;
  const float v = q.v0;
  // The starting acceleration that leaves no jerk at t1, never negative.
  const float a0 = std::max(0.0f, (-8.0f * v * t - 20.0f * x0) / (t * t));
  // x(t1) = v(t1) = a(t1) = 0 solved for the remaining terms.
  q.a = -(a0 * t * t + 6.0f * v * t + 12.0f * x0) / (2.0f * t * t * t * t * t);
  q.b = (3.0f * a0 * t * t + 16.0f * v * t + 30.0f * x0) / (2.0f * t * t * t * t);
  q.c = -(3.0f * a0 * t * t + 12.0f * v * t + 20.0f * x0) / (2.0f * t * t * t);
  q.d = 0.5f * a0;
  return q;
}

float QuinticDecay::at(float t) const {
  if (t1 <= 0.0f || t >= t1)
    return 0.0f;
  if (t <= 0.0f)
    return x0;
  return (((((a * t + b) * t + c) * t + d) * t + v0) * t) + x0;
}

namespace {

// q as a turn of `angle` (0 to pi) about a unit axis; no turn gives +y.
void axisAngle(glm::quat q, glm::vec3 &axis, float &angle) {
  if (q.w < 0.0f)
    q = -q; // the short way round
  const glm::vec3 v(q.x, q.y, q.z);
  const float s = glm::length(v);
  angle = 2.0f * std::atan2(s, q.w);
  axis = s > 1e-7f ? v / s : glm::vec3(0.0f, 1.0f, 0.0f);
}

// How far q turns about the axis (its twist), radians.
float twistAbout(glm::quat q, glm::vec3 axis) {
  if (q.w < 0.0f)
    q = -q;
  return 2.0f * std::atan2(glm::dot(glm::vec3(q.x, q.y, q.z), axis), q.w);
}

} // namespace

void PoseInertialization::start(const Pose &last, const Pose &beforeLast, float lastDt,
                                const Pose &target, float duration) {
  length = std::max(duration, 0.0f);
  bones.clear();
  if (length <= 0.0f)
    return;
  const size_t count = std::min(last.local.size(), target.local.size());
  const bool moving = lastDt > 0.0f && beforeLast.local.size() >= count;
  bones.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const BoneTransform &now = last.local[i];
    const BoneTransform &to = target.local[i];
    BoneOffset &o = bones[i];

    // The turn from the target to where it was, about one axis for the
    // whole blend; its speed is how fast that twist was shrinking.
    float angle = 0.0f;
    axisAngle(glm::normalize(now.rotation) * glm::inverse(glm::normalize(to.rotation)), o.axis,
              angle);
    float turnSpeed = 0.0f;
    if (moving) {
      const glm::quat before =
          glm::normalize(beforeLast.local[i].rotation) * glm::inverse(glm::normalize(to.rotation));
      turnSpeed = (angle - twistAbout(before, o.axis)) / lastDt;
    }
    o.turn = QuinticDecay::make(angle, turnSpeed, length);

    // The same for the shift, along one direction.
    const glm::vec3 offset = now.translation - to.translation;
    const float distance = glm::length(offset);
    o.direction = distance > 1e-7f ? offset / distance : glm::vec3(0.0f, 1.0f, 0.0f);
    float shiftSpeed = 0.0f;
    if (moving)
      shiftSpeed =
          (distance - glm::dot(beforeLast.local[i].translation - to.translation, o.direction)) /
          lastDt;
    o.shift = QuinticDecay::make(distance, shiftSpeed, length);
  }
}

void PoseInertialization::apply(float elapsed, Pose &pose) const {
  if (!active(elapsed))
    return;
  const size_t count = std::min(bones.size(), pose.local.size());
  for (size_t i = 0; i < count; ++i) {
    const BoneOffset &o = bones[i];
    BoneTransform &bone = pose.local[i];
    if (const float angle = o.turn.at(elapsed); angle != 0.0f)
      bone.rotation = glm::normalize(glm::angleAxis(angle, o.axis) * bone.rotation);
    if (const float distance = o.shift.at(elapsed); distance != 0.0f)
      bone.translation += o.direction * distance;
  }
}

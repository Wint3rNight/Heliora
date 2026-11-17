#include "AnimationClip.h"

#include <algorithm>
#include <cmath>

namespace {

// The value at time from keys sorted by time, mixed between the two around
// it; the ends hold outside them, and no keys gives the fallback.
template <typename T, typename Mix>
T sampleKeys(const std::vector<Key<T>> &keys, float time, const T &fallback,
             Mix mix) {
  if (keys.empty())
    return fallback;
  if (time <= keys.front().time)
    return keys.front().value;
  if (time >= keys.back().time)
    return keys.back().value;
  const auto next = std::upper_bound(
      keys.begin(), keys.end(), time,
      [](float t, const Key<T> &key) { return t < key.time; });
  const auto prev = next - 1;
  const float span = next->time - prev->time;
  const float along = span > 0.0f ? (time - prev->time) / span : 0.0f;
  return mix(prev->value, next->value, along);
}

glm::vec3 lerp(const glm::vec3 &a, const glm::vec3 &b, float t) {
  return glm::mix(a, b, t);
}

// glm's slerp goes the short way: it flips one end when they're more than
// a half turn apart.
glm::quat slerp(const glm::quat &a, const glm::quat &b, float t) {
  return glm::normalize(glm::slerp(a, b, t));
}

} // namespace

void sampleClip(const AnimationClip &clip, const Skeleton &skeleton, float time,
                bool loop, Pose &pose) {
  if (loop && clip.duration > 0.0f) {
    time = std::fmod(time, clip.duration);
    if (time < 0.0f)
      time += clip.duration;
  }
  pose.local.resize(skeleton.bones.size());
  for (size_t i = 0; i < skeleton.bones.size(); ++i) {
    const BoneTransform &bind = skeleton.bones[i].bind;
    if (i >= clip.tracks.size()) {
      pose.local[i] = bind;
      continue;
    }
    const BoneTrack &track = clip.tracks[i];
    pose.local[i].translation =
        sampleKeys(track.positions, time, bind.translation, lerp);
    pose.local[i].rotation = sampleKeys(track.rotations, time, bind.rotation, slerp);
    pose.local[i].scale = sampleKeys(track.scales, time, bind.scale, lerp);
  }
}

namespace {

// The root's parent's bind transform, from the root's key space to the
// skeleton's (rotation and scale).
glm::mat3 parentToSkeleton(const Skeleton &skeleton, int rootBone) {
  if (const int parent = skeleton.bones[rootBone].parent; parent >= 0)
    return glm::mat3(modelSpace(skeleton, bindPose(skeleton))[parent]);
  return glm::mat3(1.0f);
}

// The skeleton's up, seen from that space.
glm::vec3 upInParent(const glm::mat3 &toSkeleton) {
  return glm::normalize(glm::inverse(toSkeleton) * glm::vec3(0.0f, 1.0f, 0.0f));
}

} // namespace

glm::vec2 RootMotion::total() const {
  return offsets.empty() ? glm::vec2(0.0f)
                         : offsets.back().value - offsets.front().value;
}

float RootMotion::speed() const {
  return duration > 0.0f ? glm::length(total()) / duration : 0.0f;
}

RootMotion removeRootMotion(AnimationClip &clip, const Skeleton &skeleton,
                            int rootBone, RootMotionMode mode) {
  RootMotion motion;
  motion.duration = clip.duration;
  if (rootBone < 0 || rootBone >= static_cast<int>(clip.tracks.size()) ||
      rootBone >= static_cast<int>(skeleton.bones.size()))
    return motion;
  std::vector<Key<glm::vec3>> &keys = clip.tracks[rootBone].positions;
  if (keys.empty())
    return motion;
  // The root's keys are in its parent's space; the skeleton's up, seen from
  // there, says which part of a key is height and which is ground.
  const glm::mat3 toSkeleton = parentToSkeleton(skeleton, rootBone);
  const glm::vec3 up = upInParent(toSkeleton);
  auto ground = [&](const glm::vec3 &p) { return p - glm::dot(p, up) * up; };
  const glm::vec3 start = ground(keys.front().value);
  const glm::vec3 end = ground(keys.back().value);
  const float first = keys.front().time;
  const float last = keys.back().time;
  for (Key<glm::vec3> &key : keys) {
    glm::vec3 removed; // along the ground, from the start, parent space
    if (mode == RootMotionMode::Path) {
      removed = ground(key.value) - start;
    } else {
      const float along = last > first ? (key.time - first) / (last - first) : 0.0f;
      removed = (end - start) * along;
    }
    key.value -= removed;
    const glm::vec3 inSkeleton = toSkeleton * removed;
    motion.offsets.push_back({key.time, glm::vec2(inSkeleton.x, inSkeleton.z)});
  }
  return motion;
}

glm::vec2 rootDisplacement(const RootMotion &motion, float from, float to,
                           bool loop) {
  if (motion.offsets.empty())
    return glm::vec2(0.0f);
  auto offsetAt = [&](float t) {
    return sampleKeys(motion.offsets, t, glm::vec2(0.0f),
                      [](const glm::vec2 &a, const glm::vec2 &b, float w) {
                        return glm::mix(a, b, w);
                      });
  };
  const float length = motion.duration;
  if (!loop || length <= 0.0f) {
    from = std::clamp(from, 0.0f, std::max(length, 0.0f));
    to = std::clamp(to, 0.0f, std::max(length, 0.0f));
    return offsetAt(to) - offsetAt(from);
  }
  // Whole passes of the seam each add the clip's total.
  const float fromLoops = std::floor(from / length);
  const float toLoops = std::floor(to / length);
  return offsetAt(to - toLoops * length) - offsetAt(from - fromLoops * length) +
         motion.total() * (toLoops - fromLoops);
}

void limitRootHeight(AnimationClip &clip, const Skeleton &skeleton, int rootBone) {
  if (rootBone < 0 || rootBone >= static_cast<int>(clip.tracks.size()) ||
      rootBone >= static_cast<int>(skeleton.bones.size()))
    return;
  std::vector<Key<glm::vec3>> &keys = clip.tracks[rootBone].positions;
  if (keys.empty())
    return;
  const glm::vec3 up = upInParent(parentToSkeleton(skeleton, rootBone));
  const float start = glm::dot(keys.front().value, up);
  for (Key<glm::vec3> &key : keys)
    if (const float height = glm::dot(key.value, up); height > start)
      key.value -= (height - start) * up;
}

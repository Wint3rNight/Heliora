#pragma once

#include "Skeleton.h"

#include <string>
#include <vector>

// Keyframed animation for a skeleton: what each bone does over time, and the
// pose at any moment.

template <typename T> struct Key {
  float time; // seconds from the clip's start
  T value;
};

// One bone's keys. A channel with no keys leaves that part of the bone at
// its bind pose.
struct BoneTrack {
  std::vector<Key<glm::vec3>> positions;
  std::vector<Key<glm::quat>> rotations;
  std::vector<Key<glm::vec3>> scales;
};

struct AnimationClip {
  std::string name;
  float duration = 0.0f;         // seconds
  std::vector<BoneTrack> tracks; // by skeleton bone; missing ones are empty
};

// Sets pose to the clip at time. Between keys, positions and scales are
// linear and rotations turn the short way (slerp). Before the first key and
// after the last the ends hold; a looping clip wraps time into its length
// first. Bones and channels the clip leaves out get the bind pose.
void sampleClip(const AnimationClip &clip, const Skeleton &skeleton, float time,
                bool loop, Pose &pose);

// How removeRootMotion treats the root's movement along the ground.
enum class RootMotionMode {
  // Locomotion loops: only the steady progress comes out (a straight line
  // from the first key to the last), so the hips keep their sway.
  Progression,
  // One-off moves such as a roll: the whole path along the ground comes
  // out, so the hips stay over the capsule while it follows the path.
  Path,
};

// What came out: the root's offset along the ground from the clip's start,
// in the skeleton's space (x and z), at the root's position key times.
struct RootMotion {
  std::vector<Key<glm::vec2>> offsets;
  float duration = 0.0f;
  glm::vec2 total() const; // from the start to the end
  float speed() const;     // the length of total over the duration
};

// Takes the root bone's movement along the ground out of the clip and
// returns it. "Along the ground" is the skeleton's x and z, worked out in
// the root's parent's space, so a parent turned for z-up still works.
// The height is never touched.
RootMotion removeRootMotion(AnimationClip &clip, const Skeleton &skeleton,
                            int rootBone, RootMotionMode mode);

// How far the root went between two times. A looping clip adds its total
// for every pass of the seam; otherwise the times are held to the clip.
glm::vec2 rootDisplacement(const RootMotion &motion, float from, float to,
                           bool loop);

// Keeps the root bone from rising above where it starts: dips stay, rises
// go. For a jump clip, whose own jump would add to the capsule's.
void limitRootHeight(AnimationClip &clip, const Skeleton &skeleton, int rootBone);

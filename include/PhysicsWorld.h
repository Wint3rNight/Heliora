#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <memory>
#include <vector>

// Rigid-body physics on Jolt. Jolt and Vulkan stay out of this header, so the
// tests run without a GPU and files that include it skip Jolt's headers.
//
// Not thread-safe: call it from the main thread. Jolt's workers only run
// inside step().

// Wraps Jolt's BodyID (index plus a sequence number), so a stale handle never
// points at a new body.
struct PhysicsBodyId {
  uint32_t value = 0xffffffffu; // Jolt's "no body"
  bool isValid() const { return value != 0xffffffffu; }
};

enum class PhysicsShape : uint8_t { Box, Sphere, Capsule };

// Kinematic bodies are moved by our code and push dynamic bodies aside.
enum class PhysicsMotion : uint8_t { Static, Dynamic, Kinematic };

struct PhysicsBodyDesc {
  PhysicsShape shape = PhysicsShape::Box;
  glm::vec3 halfExtents{0.5f}; // box
  float radius = 0.5f;         // sphere and capsule
  float halfHeight = 0.5f;     // capsule: half the length of its straight part
  PhysicsMotion motion = PhysicsMotion::Dynamic;
  float mass = 0.0f;           // kg; 0 = from the volume at 1000 kg/m^3
  float friction = 0.2f;
  float restitution = 0.0f;    // bounciness, 0 to 1
  glm::vec3 position{0.0f};
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  uint64_t userData = 0; // ours, handed back by movedPoses()
};

struct PhysicsPose {
  glm::vec3 position;
  glm::quat rotation;
};

struct PhysicsMovedPose {
  uint64_t userData;
  PhysicsPose pose;
};

// Pose between two physics steps, alpha 0 to 1.
PhysicsPose blendPoses(const PhysicsPose &from, const PhysicsPose &to,
                       float alpha);

class PhysicsWorld {
public:
  struct Config {
    // Jolt preallocates for these, so a step never allocates. The defaults
    // are what Jolt's samples use for about 10,000 bodies. Contacts past the
    // limits get dropped (see droppedContactSteps()).
    uint32_t maxBodies = 16384;
    uint32_t maxBodyPairs = 65536;
    uint32_t maxContactConstraints = 20480;
    // Scratch memory for one step. Jolt aborts if it runs out, so the
    // constructor checks it against maxContactConstraints.
    uint32_t tempAllocatorBytes = 32u * 1024u * 1024u;
    // The main thread helps while it waits, so a step uses workerThreads + 1.
    uint32_t workerThreads = 4;
  };

  PhysicsWorld();
  explicit PhysicsWorld(const Config &config);
  ~PhysicsWorld();
  PhysicsWorld(const PhysicsWorld &) = delete;
  PhysicsWorld &operator=(const PhysicsWorld &) = delete;

  // Invalid id if the world is full (maxBodies).
  PhysicsBodyId addBody(const PhysicsBodyDesc &desc);
  void removeBody(PhysicsBodyId id);

  // Call after adding a lot of bodies at once, like a scene load. Adding them
  // one by one leaves the broad phase tree unbalanced.
  void optimizeBroadPhase();

  void step(float dt);

  // Latest pose of the body's origin.
  PhysicsPose pose(PhysicsBodyId id) const;
  // The pose to draw: blended between the last two steps by the clock's
  // alpha, so motion looks smooth at any frame rate.
  PhysicsPose interpolatedPose(PhysicsBodyId id, float alpha) const;
  // Bodies whose drawn pose can change: everything that moved in the last
  // step. Sleeping bodies never show up, so their nodes cost nothing.
  const std::vector<PhysicsBodyId> &movedBodies() const;
  // The same bodies, with their interpolated pose and userData, in one pass.
  // This is what the game reads every frame. Stays valid until the next call.
  const std::vector<PhysicsMovedPose> &movedPoses(float alpha);
  // False once the body's asleep.
  bool isActive(PhysicsBodyId id) const;

  uint32_t bodyCount() const;
  uint32_t activeBodyCount() const;
  // Steps where Jolt dropped contacts because a Config limit was too small.
  // Bodies sink into each other when that happens, so this should stay 0.
  uint32_t droppedContactSteps() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

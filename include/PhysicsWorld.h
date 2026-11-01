#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <memory>

// Rigid-body physics on Jolt. No Vulkan and no Jolt types in this header:
// the test program runs it without a GPU, and files that include it don't
// pay for Jolt's headers. The Jolt objects live in PhysicsWorld.cpp.
//
// Not thread-safe: call everything from one thread (the main thread). Jolt's
// own worker threads run only inside step().

// A body handle. It wraps Jolt's BodyID (index plus a sequence number), so a
// handle to a removed body never silently refers to a new one.
struct PhysicsBodyId {
  uint32_t value = 0xffffffffu; // Jolt's "no body" value
  bool isValid() const { return value != 0xffffffffu; }
};

enum class PhysicsShape : uint8_t { Box, Sphere, Capsule };

// Static bodies never move. Dynamic bodies are moved by the simulation.
// Kinematic bodies are moved by code and push dynamic bodies out of the way.
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
};

struct PhysicsPose {
  glm::vec3 position;
  glm::quat rotation;
};

class PhysicsWorld {
public:
  struct Config {
    // Jolt preallocates for these limits, so nothing is allocated per step.
    // The defaults are the ones Jolt's own samples use for scenes of about
    // 10,000 bodies. Contacts beyond the limits are dropped, and counted by
    // droppedContactSteps().
    uint32_t maxBodies = 16384;
    uint32_t maxBodyPairs = 65536;
    uint32_t maxContactConstraints = 20480;
    // Scratch memory for one step. Running out aborts the program, so the
    // constructor checks it against the contact limit above.
    uint32_t tempAllocatorBytes = 32u * 1024u * 1024u;
    // The main thread also works while it waits, so a step runs on
    // workerThreads + 1 threads.
    uint32_t workerThreads = 4;
  };

  PhysicsWorld();
  explicit PhysicsWorld(const Config &config);
  ~PhysicsWorld();
  PhysicsWorld(const PhysicsWorld &) = delete;
  PhysicsWorld &operator=(const PhysicsWorld &) = delete;

  // Returns an invalid id if the world is full (maxBodies).
  PhysicsBodyId addBody(const PhysicsBodyDesc &desc);
  void removeBody(PhysicsBodyId id);

  // Call once after adding many bodies at once (a scene load): rebuilds the
  // broad phase's tree, which adding bodies one by one leaves unbalanced.
  void optimizeBroadPhase();

  // Advances the simulation by dt seconds.
  void step(float dt);

  // Position and rotation of the body's origin.
  PhysicsPose pose(PhysicsBodyId id) const;
  // False once the body is asleep (at rest). Sleeping bodies cost nothing.
  bool isActive(PhysicsBodyId id) const;

  uint32_t bodyCount() const;
  uint32_t activeBodyCount() const;
  // Steps where Jolt dropped contacts because a limit in Config was too
  // small. Bodies then sink into each other, so this should stay 0.
  uint32_t droppedContactSteps() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

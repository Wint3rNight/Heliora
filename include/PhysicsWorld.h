#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <memory>
#include <optional>
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

// Hulls and TriangleMesh are built from a model's own triangles.
enum class PhysicsShape : uint8_t { Box, Sphere, Capsule, Hulls, TriangleMesh };

// Kinematic bodies are moved by our code and push dynamic bodies aside.
enum class PhysicsMotion : uint8_t { Static, Dynamic, Kinematic };

// One part of a model, in the model's own space: its vertex positions and
// its triangles, three indices each.
struct PhysicsMeshPart {
  std::vector<glm::vec3> positions;
  std::vector<uint32_t> indices;
};

struct PhysicsMeshParts {
  std::vector<PhysicsMeshPart> parts;
};

struct PhysicsBodyDesc {
  PhysicsShape shape = PhysicsShape::Box;
  glm::vec3 halfExtents{0.5f}; // box
  float radius = 0.5f;         // sphere and capsule
  float halfHeight = 0.5f;     // capsule: half the length of its straight part
  // Hulls: one shrink-wrapped hull per part, for anything. TriangleMesh:
  // every triangle as it is, for static bodies only. Both are scaled by
  // scale, and bodies with the same parts share one Jolt shape.
  std::shared_ptr<const PhysicsMeshParts> parts;
  glm::vec3 scale{1.0f};
  PhysicsMotion motion = PhysicsMotion::Dynamic;
  float mass = 0.0f;           // kg; 0 = volume times density
  float density = 1000.0f;     // kg/m^3, used when mass is 0
  float friction = 0.2f;
  float restitution = 0.0f;    // bounciness, 0 to 1
  glm::vec3 position{0.0f};
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 linearVelocity{0.0f}; // m/s at the start, for thrown things
  uint64_t userData = 0;           // ours, handed back by movedPoses()
};

struct PhysicsPose {
  glm::vec3 position;
  glm::quat rotation;
};

struct PhysicsRayHit {
  PhysicsBodyId body;
  uint64_t userData; // the body's
  float distance;    // m from the ray's start
};

struct PhysicsMovedPose {
  uint64_t userData;
  PhysicsPose pose;
};

// One end of a debug line, laid out to go straight into a vertex buffer.
struct PhysicsDebugVertex {
  glm::vec3 position;
  uint32_t color; // RGBA8
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
    // constructor raises it when maxContactConstraints needs more.
    uint32_t tempAllocatorBytes = 32u * 1024u * 1024u;
    // The main thread helps while it waits, so a step uses workerThreads + 1.
    // 6 was the fastest of 2, 4 and 6 on the 6-core laptop: 5,000 awake
    // boxes step in 3.55 ms, against 4.25 ms with 4 and 6.2 ms with 2
    // (2026-11-01). Nothing else runs during the step, so the extra threads
    // cost the renderer nothing yet.
    uint32_t workerThreads = 6;

    // These can also change while the world runs, through applySettings().
    // The defaults are Jolt's own.
    glm::vec3 gravity{0.0f, -9.81f, 0.0f};
    // Solver passes per step. More passes make stacks steadier and cost
    // time. Friction needs at least 2 velocity passes.
    int velocitySteps = 10;
    int positionSteps = 2;
    float sleepDelay = 0.5f;  // seconds a body must stay slow before it sleeps
    float sleepSpeed = 0.03f; // m/s; slower than this counts as resting
    // How far bodies may sink into each other before being pushed apart, so
    // resting contacts don't jitter.
    float penetrationSlop = 0.02f; // m
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

  // Takes gravity and the solver and sleep values from config; the limits
  // and threads only apply to a new world. A gravity change wakes every
  // body, or sleeping ones would hang in the air until something hit them.
  void applySettings(const Config &config);

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
  // In kg. 0 for static bodies, which never move.
  float mass(PhysicsBodyId id) const;

  // The nearest body the ray hits within maxDistance, for clicking on
  // things. direction needn't be unit length.
  std::optional<PhysicsRayHit> castRay(glm::vec3 origin, glm::vec3 direction,
                                       float maxDistance) const;

  uint32_t bodyCount() const;
  uint32_t activeBodyCount() const;
  // Wireframes of every body's collision shape, as Jolt sees it, in pairs of
  // end points. Coloured by state: awake and asleep differ. Jolt draws the
  // latest step, so a moving body's wireframe can lead its mesh by up to one
  // step. Builds the list on every call; only call it when drawing it.
  const std::vector<PhysicsDebugVertex> &debugLines(glm::vec3 cameraPosition);

  // Steps where Jolt dropped contacts because a Config limit was too small.
  // Bodies sink into each other when that happens, so this should stay 0.
  uint32_t droppedContactSteps() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

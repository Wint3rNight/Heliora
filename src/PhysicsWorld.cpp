#include "PhysicsWorld.h"

// Jolt.h must come before any other Jolt header.
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/ContactConstraintManager.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

// Object layers say what a body is. Static bodies never test against each
// other, which removes most pairs in a world that is mostly static.
namespace Layers {
constexpr JPH::ObjectLayer NonMoving = 0;
constexpr JPH::ObjectLayer Moving = 1;
} // namespace Layers

// Jolt keeps one bounding-volume tree per broad-phase layer, so the tree of
// static bodies is not touched when dynamic bodies move.
namespace BroadPhaseLayers {
constexpr JPH::BroadPhaseLayer NonMoving(0);
constexpr JPH::BroadPhaseLayer Moving(1);
constexpr JPH::uint Count = 2;
} // namespace BroadPhaseLayers

class BroadPhaseLayerMap final : public JPH::BroadPhaseLayerInterface {
public:
  JPH::uint GetNumBroadPhaseLayers() const override {
    return BroadPhaseLayers::Count;
  }
  JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
    return layer == Layers::NonMoving ? BroadPhaseLayers::NonMoving
                                      : BroadPhaseLayers::Moving;
  }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
  const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
    return layer == BroadPhaseLayers::NonMoving ? "NonMoving" : "Moving";
  }
#endif
};

class ObjectVsBroadPhaseFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
  bool ShouldCollide(JPH::ObjectLayer layer,
                     JPH::BroadPhaseLayer broadPhaseLayer) const override {
    // A static body only has to look in the tree of moving bodies.
    return layer != Layers::NonMoving ||
           broadPhaseLayer == BroadPhaseLayers::Moving;
  }
};

class ObjectPairFilter final : public JPH::ObjectLayerPairFilter {
public:
  bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
    return a == Layers::Moving || b == Layers::Moving;
  }
};

void traceToLog(const char *format, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  spdlog::warn("[physics] {}", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
bool assertToLog(const char *expression, const char *message, const char *file,
                 JPH::uint line) {
  // Dropped contacts are reported and counted by step(), so they don't stop a
  // Debug build. If a Jolt update rewords this message, the assert simply
  // stops the program again, which is the safe direction.
  if (message &&
      std::strstr(message, "error occurred during the physics update"))
    return false;
  spdlog::error("[physics] Jolt assert {}:{}: ({}) {}", file, line, expression,
                message ? message : "");
  return true; // every other assert is a bug: stop, in the debugger if attached
}
#endif

// Jolt's allocator hooks, factory and type registry are process-wide, so
// every PhysicsWorld shares them: the first world sets them up and the last
// one tears them down.
std::mutex joltRuntimeMutex;
int joltRuntimeUsers = 0;

struct JoltRuntimeUser {
  JoltRuntimeUser() {
    std::lock_guard lock(joltRuntimeMutex);
    if (joltRuntimeUsers++ == 0) {
      JPH::RegisterDefaultAllocator();
      JPH::Trace = traceToLog;
      JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = assertToLog;)
      JPH::Factory::sInstance = new JPH::Factory();
      JPH::RegisterTypes();
    }
  }
  ~JoltRuntimeUser() {
    std::lock_guard lock(joltRuntimeMutex);
    if (--joltRuntimeUsers == 0) {
      JPH::UnregisterTypes();
      delete JPH::Factory::sInstance;
      JPH::Factory::sInstance = nullptr;
    }
  }
};

JPH::Vec3 toJolt(glm::vec3 v) { return JPH::Vec3(v.x, v.y, v.z); }
JPH::RVec3 toJoltPosition(glm::vec3 v) { return JPH::RVec3(v.x, v.y, v.z); }
// Jolt checks that rotations are unit length (to 1e-5); glm results can be a
// rounding error away from that.
JPH::Quat toJolt(glm::quat q) {
  return JPH::Quat(q.x, q.y, q.z, q.w).Normalized();
}
glm::vec3 toGlm(JPH::RVec3Arg v) {
  return glm::vec3(static_cast<float>(v.GetX()), static_cast<float>(v.GetY()),
                   static_cast<float>(v.GetZ()));
}
glm::quat toGlm(JPH::QuatArg q) {
  return glm::quat::wxyz(q.GetW(), q.GetX(), q.GetY(), q.GetZ());
}

JPH::ShapeRefC makeShape(const PhysicsBodyDesc &desc) {
  switch (desc.shape) {
  case PhysicsShape::Sphere:
    return new JPH::SphereShape(desc.radius);
  case PhysicsShape::Capsule:
    return new JPH::CapsuleShape(desc.halfHeight, desc.radius);
  case PhysicsShape::Box:
    break;
  }
  // Jolt rounds a box's edges by a convex radius that must fit inside it.
  const glm::vec3 h = desc.halfExtents;
  const float convexRadius =
      std::min(JPH::cDefaultConvexRadius, std::min({h.x, h.y, h.z}));
  return new JPH::BoxShape(toJolt(h), convexRadius);
}

} // namespace

struct PhysicsWorld::Impl {
  // Members are destroyed in reverse order: the physics system goes first,
  // then what it uses, and the Jolt runtime last.
  JoltRuntimeUser runtime;
  BroadPhaseLayerMap broadPhaseLayers;
  ObjectVsBroadPhaseFilter objectVsBroadPhase;
  ObjectPairFilter objectPairs;
  std::unique_ptr<JPH::TempAllocatorImpl> tempAllocator;
  std::unique_ptr<JPH::JobSystemThreadPool> jobSystem;
  JPH::PhysicsSystem system;

  uint32_t droppedContactSteps = 0;
  uint32_t loggedErrors = 0; // EPhysicsUpdateError bits already logged
  bool warnedFull = false;
};

PhysicsWorld::PhysicsWorld() : PhysicsWorld(Config{}) {}

PhysicsWorld::PhysicsWorld(const Config &config)
    : impl(std::make_unique<Impl>()) {
  // Every step the contact solver takes a buffer for maxContactConstraints
  // from the scratch memory, and running out of scratch memory aborts the
  // program. Insist on room for the rest of the step as well (islands, body
  // pairs), so a bad config fails here with a message instead.
  const size_t contactBytes =
      size_t(config.maxContactConstraints) *
      (JPH::ContactConstraintManager::cMaxConstraintSize + sizeof(uint32_t));
  if (contactBytes * 2 > config.tempAllocatorBytes)
    throw std::invalid_argument(
        "PhysicsWorld: tempAllocatorBytes must be at least twice the contact "
        "buffer (" + std::to_string(contactBytes) + " bytes)");

  impl->tempAllocator =
      std::make_unique<JPH::TempAllocatorImpl>(config.tempAllocatorBytes);
  impl->jobSystem = std::make_unique<JPH::JobSystemThreadPool>(
      JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers,
      static_cast<int>(config.workerThreads));
  impl->system.Init(config.maxBodies, 0, config.maxBodyPairs,
                    config.maxContactConstraints, impl->broadPhaseLayers,
                    impl->objectVsBroadPhase, impl->objectPairs);
}

PhysicsWorld::~PhysicsWorld() = default;

PhysicsBodyId PhysicsWorld::addBody(const PhysicsBodyDesc &desc) {
  JPH::EMotionType motion = JPH::EMotionType::Dynamic;
  if (desc.motion == PhysicsMotion::Static)
    motion = JPH::EMotionType::Static;
  else if (desc.motion == PhysicsMotion::Kinematic)
    motion = JPH::EMotionType::Kinematic;
  const bool isStatic = motion == JPH::EMotionType::Static;

  JPH::BodyCreationSettings settings(
      makeShape(desc), toJoltPosition(desc.position), toJolt(desc.rotation),
      motion, isStatic ? Layers::NonMoving : Layers::Moving);
  settings.mFriction = desc.friction;
  settings.mRestitution = desc.restitution;
  if (motion == JPH::EMotionType::Dynamic && desc.mass > 0.0f) {
    settings.mOverrideMassProperties =
        JPH::EOverrideMassProperties::CalculateInertia;
    settings.mMassPropertiesOverride.mMass = desc.mass;
  }

  // The no-lock interface is safe because everything outside step() runs on
  // one thread, and it skips a mutex per call.
  const JPH::BodyID id =
      impl->system.GetBodyInterfaceNoLock().CreateAndAddBody(
          settings, isStatic ? JPH::EActivation::DontActivate
                             : JPH::EActivation::Activate);
  if (id.IsInvalid() && !impl->warnedFull) {
    spdlog::warn("[physics] world is full ({} bodies): raise maxBodies",
                 impl->system.GetMaxBodies());
    impl->warnedFull = true;
  }
  return PhysicsBodyId{id.GetIndexAndSequenceNumber()};
}

void PhysicsWorld::removeBody(PhysicsBodyId id) {
  JPH::BodyInterface &bodies = impl->system.GetBodyInterfaceNoLock();
  const JPH::BodyID bodyId(id.value);
  // IsAdded is false for invalid handles and for handles to removed bodies.
  if (!id.isValid() || !bodies.IsAdded(bodyId))
    return;
  bodies.RemoveBody(bodyId);
  bodies.DestroyBody(bodyId);
}

void PhysicsWorld::optimizeBroadPhase() { impl->system.OptimizeBroadPhase(); }

void PhysicsWorld::step(float dt) {
  constexpr int collisionSteps = 1; // one collision pass per 1/60 s step
  const JPH::EPhysicsUpdateError errors =
      impl->system.Update(dt, collisionSteps, impl->tempAllocator.get(),
                          impl->jobSystem.get());
  if (errors == JPH::EPhysicsUpdateError::None)
    return;
  ++impl->droppedContactSteps;
  // Log each kind of error once; after that it is only counted.
  const uint32_t newErrors =
      static_cast<uint32_t>(errors) & ~impl->loggedErrors;
  if (newErrors != 0) {
    spdlog::warn("[physics] Jolt dropped contacts (error bits {:#x}): raise "
                 "maxBodyPairs or maxContactConstraints",
                 newErrors);
    impl->loggedErrors |= newErrors;
  }
}

PhysicsPose PhysicsWorld::pose(PhysicsBodyId id) const {
  JPH::RVec3 position;
  JPH::Quat rotation;
  // Gives the origin and identity for an invalid or removed handle.
  impl->system.GetBodyInterfaceNoLock().GetPositionAndRotation(
      JPH::BodyID(id.value), position, rotation);
  return {toGlm(position), toGlm(rotation)};
}

bool PhysicsWorld::isActive(PhysicsBodyId id) const {
  return impl->system.GetBodyInterfaceNoLock().IsActive(JPH::BodyID(id.value));
}

uint32_t PhysicsWorld::bodyCount() const { return impl->system.GetNumBodies(); }

uint32_t PhysicsWorld::activeBodyCount() const {
  return impl->system.GetNumActiveBodies(JPH::EBodyType::RigidBody);
}

uint32_t PhysicsWorld::droppedContactSteps() const {
  return impl->droppedContactSteps;
}

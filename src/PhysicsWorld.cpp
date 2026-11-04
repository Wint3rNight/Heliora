#include "PhysicsWorld.h"

// Jolt.h has to come before any other Jolt header.
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/ContactConstraintManager.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>
#ifdef JPH_DEBUG_RENDERER
#include <Jolt/Renderer/DebugRendererSimple.h>
#endif

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <stdexcept>
#include <string>

namespace {

// Static bodies never test against each other, which cuts most pairs in a
// world that's mostly static.
namespace Layers {
constexpr JPH::ObjectLayer NonMoving = 0;
constexpr JPH::ObjectLayer Moving = 1;
} // namespace Layers

// One broad phase tree per layer, so moving bodies never touch the static
// tree.
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
    // A static body only needs to look in the moving tree.
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
  // step() already counts and logs dropped contacts, so don't stop a Debug
  // build for them. If Jolt ever rewords this message the assert just stops
  // the program again, which is the safe way to fail.
  if (message &&
      std::strstr(message, "error occurred during the physics update"))
    return false;
  spdlog::error("[physics] Jolt assert {}:{}: ({}) {}", file, line, expression,
                message ? message : "");
  return true; // anything else is a bug: stop (in the debugger if attached)
}
#endif

// Jolt's allocator hooks, factory and type registry are process-wide. The
// first world sets them up, the last one tears them down.
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
// Jolt wants rotations unit length to within 1e-5, and glm results can be a
// rounding error off.
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
  case PhysicsShape::Hulls:        // built from parts, see makeHulls
  case PhysicsShape::TriangleMesh: // and makeTriangleMesh
    break;
  }
  // Jolt rounds box edges by a convex radius, which has to fit inside.
  const glm::vec3 h = desc.halfExtents;
  const float convexRadius =
      std::min(JPH::cDefaultConvexRadius, std::min({h.x, h.y, h.z}));
  return new JPH::BoxShape(toJolt(h), convexRadius);
}

// The 8 corners of a part's bounds, at least 1 cm thick on every axis, so
// even a flat part or a single point gives a hull with some volume.
JPH::Array<JPH::Vec3> boundsCorners(const PhysicsMeshPart &part) {
  constexpr float kMinHalfThickness = 0.005f;
  glm::vec3 lo(std::numeric_limits<float>::max());
  glm::vec3 hi(std::numeric_limits<float>::lowest());
  for (const glm::vec3 &p : part.positions) {
    lo = glm::min(lo, p);
    hi = glm::max(hi, p);
  }
  const glm::vec3 centre = (lo + hi) * 0.5f;
  const glm::vec3 half = glm::max((hi - lo) * 0.5f, glm::vec3(kMinHalfThickness));
  JPH::Array<JPH::Vec3> corners;
  for (int i = 0; i < 8; ++i)
    corners.push_back(toJolt(centre + half * glm::vec3(i & 1 ? 1 : -1,
                                                       i & 2 ? 1 : -1,
                                                       i & 4 ? 1 : -1)));
  return corners;
}

// One part shrink-wrapped. Jolt keeps at most 256 of the points. Points it
// can't wrap (too few, in a line) become the box around them, and so do
// flat ones: Jolt wraps those, but the hull has no volume and so no mass.
JPH::ShapeRefC makeHull(const PhysicsMeshPart &part) {
  JPH::Array<JPH::Vec3> points;
  points.reserve(part.positions.size());
  for (const glm::vec3 &p : part.positions)
    points.push_back(toJolt(p));
  const JPH::ShapeSettings::ShapeResult hull =
      JPH::ConvexHullShapeSettings(points).Create();
  constexpr float kMinVolume = 1e-9f; // m^3
  if (hull.IsValid() && hull.Get()->GetVolume() > kMinVolume)
    return hull.Get();
  spdlog::warn("[physics] couldn't wrap a hull with volume around {} points "
               "({}), using the box around them",
               points.size(),
               hull.IsValid() ? "it's flat" : hull.GetError().c_str());
  return JPH::ConvexHullShapeSettings(boundsCorners(part)).Create().Get();
}

// A hull per part. Several become one compound shape, so the gaps between
// parts stay open.
JPH::ShapeRefC makeHulls(const PhysicsMeshParts &parts) {
  JPH::StaticCompoundShapeSettings compound;
  JPH::ShapeRefC only;
  for (const PhysicsMeshPart &part : parts.parts) {
    if (part.positions.empty())
      continue;
    only = makeHull(part);
    compound.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), only);
  }
  if (compound.mSubShapes.size() == 1)
    return only;
  if (compound.mSubShapes.empty()) {
    spdlog::warn("[physics] a model with no points, using a 10 cm box");
    return new JPH::BoxShape(JPH::Vec3::sReplicate(0.05f), 0.0f);
  }
  return compound.Create().Get();
}

// Every triangle of every part, as one static mesh.
JPH::ShapeRefC makeTriangleMesh(const PhysicsMeshParts &parts) {
  JPH::VertexList vertices;
  JPH::IndexedTriangleList triangles;
  for (const PhysicsMeshPart &part : parts.parts) {
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    for (const glm::vec3 &p : part.positions)
      vertices.push_back(JPH::Float3(p.x, p.y, p.z));
    for (size_t i = 0; i + 2 < part.indices.size(); i += 3)
      triangles.push_back(JPH::IndexedTriangle(base + part.indices[i],
                                               base + part.indices[i + 1],
                                               base + part.indices[i + 2], 0));
  }
  const JPH::ShapeSettings::ShapeResult mesh =
      JPH::MeshShapeSettings(std::move(vertices), std::move(triangles))
          .Create();
  if (mesh.IsValid())
    return mesh.Get();
  spdlog::warn("[physics] couldn't build a triangle mesh ({}), using hulls",
               mesh.GetError().c_str());
  return makeHulls(parts);
}

#ifdef JPH_DEBUG_RENDERER
// Collects Jolt's debug drawing as line end points. Triangles arrive as
// their three edges (DebugRendererSimple does that), text is dropped.
class LineCollector final : public JPH::DebugRendererSimple {
public:
  std::vector<PhysicsDebugVertex> vertices;

  void DrawLine(JPH::RVec3Arg from, JPH::RVec3Arg to,
                JPH::ColorArg color) override {
    vertices.push_back({toGlm(from), color.GetUInt32()});
    vertices.push_back({toGlm(to), color.GetUInt32()});
  }
  void DrawText3D(JPH::RVec3Arg, const std::string_view &, JPH::ColorArg,
                  float) override {}
};
#endif

} // namespace

PhysicsPose blendPoses(const PhysicsPose &from, const PhysicsPose &to,
                       float alpha) {
  // nlerp, not slerp: over one step's worth of rotation they agree to
  // 0.0005 rad even at 30 rad/s of spin, and nlerp skips acos and sin.
  // q and -q are the same rotation, so flip to take the short way round.
  const glm::quat target =
      glm::dot(from.rotation, to.rotation) < 0.0f ? -to.rotation : to.rotation;
  return {glm::mix(from.position, to.position, alpha),
          glm::normalize(glm::lerp(from.rotation, target, alpha))};
}

struct PhysicsWorld::Impl {
  // Destroyed bottom up: the physics system first, the Jolt runtime last.
  JoltRuntimeUser runtime;
  BroadPhaseLayerMap broadPhaseLayers;
  ObjectVsBroadPhaseFilter objectVsBroadPhase;
  ObjectPairFilter objectPairs;
  std::unique_ptr<JPH::TempAllocatorImpl> tempAllocator;
  std::unique_ptr<JPH::JobSystemThreadPool> jobSystem;
  JPH::PhysicsSystem system;

  // The two poses each body is drawn between, indexed by Jolt's body index.
  // Sized to maxBodies up front, so recording them never allocates.
  struct DrawnPose {
    PhysicsPose previous;
    PhysicsPose current;
  };
  std::vector<DrawnPose> drawn;
  std::vector<uint64_t> userData; // PhysicsBodyDesc::userData, same indexing
  // Bodies active after the last step. activeStamp holds the step a body was
  // last active in, which spots the ones that just fell asleep without a
  // search.
  std::vector<JPH::BodyID> lastActive;
  std::vector<JPH::BodyID> active;
  std::vector<uint32_t> activeStamp;
  uint32_t stepCount = 0;
  std::vector<PhysicsBodyId> moved;
  std::vector<PhysicsMovedPose> movedPoseBuffer; // what movedPoses() returns

#ifdef JPH_DEBUG_RENDERER
  // Made on first use. Jolt keeps one debug renderer per process, and it
  // goes before the system it draws.
  std::unique_ptr<LineCollector> debugRenderer;
#endif
  std::vector<PhysicsDebugVertex> noLines; // when Jolt has no debug renderer

  // Bodies with the same shape and size share one Jolt shape: a grid of
  // 5,000 boxes holds one box shape, not 5,000. Keyed by shape and up to
  // three sizes.
  std::map<std::tuple<PhysicsShape, float, float, float>, JPH::ShapeRefC>
      shapes;
  // Shapes built from a model's parts, keyed by the parts and the kind. The
  // entry holds the parts too, so their address can't be reused by others.
  struct PartsShape {
    std::shared_ptr<const PhysicsMeshParts> parts;
    JPH::ShapeRefC shape;
  };
  std::map<std::pair<const PhysicsMeshParts *, PhysicsShape>, PartsShape>
      partsShapes;

  uint32_t droppedContactSteps = 0;
  uint32_t loggedErrors = 0; // EPhysicsUpdateError bits already logged
  bool warnedFull = false;

  JPH::ShapeRefC shapeFor(const PhysicsBodyDesc &desc) {
    const bool fromParts = desc.shape == PhysicsShape::Hulls ||
                           desc.shape == PhysicsShape::TriangleMesh;
    if (fromParts && desc.parts) {
      PhysicsShape kind = desc.shape;
      if (kind == PhysicsShape::TriangleMesh &&
          desc.motion != PhysicsMotion::Static) {
        // Jolt can't collide a moving mesh with other meshes or terrain.
        spdlog::warn("[physics] a moving body can't use a triangle mesh, "
                     "using hulls");
        kind = PhysicsShape::Hulls;
      }
      PartsShape &entry = partsShapes[{desc.parts.get(), kind}];
      if (entry.shape == nullptr) {
        entry.parts = desc.parts;
        entry.shape = kind == PhysicsShape::Hulls
                          ? makeHulls(*desc.parts)
                          : makeTriangleMesh(*desc.parts);
      }
      if (desc.scale == glm::vec3(1.0f))
        return entry.shape;
      return new JPH::ScaledShape(entry.shape, toJolt(desc.scale));
    }
    if (fromParts)
      spdlog::warn("[physics] a hull or mesh body without parts, using a box");

    std::tuple<PhysicsShape, float, float, float> key{PhysicsShape::Box,
                                                      desc.halfExtents.x,
                                                      desc.halfExtents.y,
                                                      desc.halfExtents.z};
    if (desc.shape == PhysicsShape::Sphere)
      key = {desc.shape, desc.radius, 0.0f, 0.0f};
    else if (desc.shape == PhysicsShape::Capsule)
      key = {desc.shape, desc.radius, desc.halfHeight, 0.0f};
    JPH::ShapeRefC &shape = shapes[key];
    if (shape == nullptr)
      shape = makeShape(desc);
    return shape;
  }

  PhysicsPose readPose(JPH::BodyID id) const {
    JPH::RVec3 position;
    JPH::Quat rotation;
    // Origin and identity for an invalid or removed handle.
    system.GetBodyInterfaceNoLock().GetPositionAndRotation(id, position,
                                                           rotation);
    return {toGlm(position), toGlm(rotation)};
  }

  void recordMovedBodies();
};

// Runs after every step. Only active bodies are touched, so a world full of
// sleeping bodies costs nothing here.
void PhysicsWorld::Impl::recordMovedBodies() {
  ++stepCount;
  const JPH::uint count = system.GetNumActiveBodies(JPH::EBodyType::RigidBody);
  const JPH::BodyID *ids =
      system.GetActiveBodiesUnsafe(JPH::EBodyType::RigidBody);
  active.assign(ids, ids + count);

  moved.clear();
  for (JPH::BodyID id : active) {
    DrawnPose &pose = drawn[id.GetIndex()];
    pose.previous = pose.current;
    pose.current = readPose(id);
    activeStamp[id.GetIndex()] = stepCount;
    moved.push_back({id.GetIndexAndSequenceNumber()});
  }
  // Fell asleep this step: from now on draw exactly the resting pose. That
  // skips blending the last sub-millimetre move, and in exchange a sleeping
  // body can't twitch as alpha changes.
  for (JPH::BodyID id : lastActive) {
    if (activeStamp[id.GetIndex()] == stepCount)
      continue;
    DrawnPose &pose = drawn[id.GetIndex()];
    pose.current = readPose(id);
    pose.previous = pose.current;
    moved.push_back({id.GetIndexAndSequenceNumber()});
  }
  std::swap(lastActive, active);
}

PhysicsWorld::PhysicsWorld() : PhysicsWorld(Config{}) {}

PhysicsWorld::PhysicsWorld(const Config &config)
    : impl(std::make_unique<Impl>()) {
  // Every step the contact solver grabs a buffer for maxContactConstraints
  // from scratch memory, and running out aborts. Want room for the rest of
  // the step too (islands, body pairs), so fail here with a message instead.
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

  impl->drawn.resize(config.maxBodies);
  impl->userData.resize(config.maxBodies, 0);
  impl->movedPoseBuffer.reserve(config.maxBodies);
  impl->activeStamp.resize(config.maxBodies, 0);
  impl->lastActive.reserve(config.maxBodies);
  impl->active.reserve(config.maxBodies);
  impl->moved.reserve(config.maxBodies);
}

PhysicsWorld::~PhysicsWorld() = default;

PhysicsBodyId PhysicsWorld::addBody(const PhysicsBodyDesc &desc) {
  JPH::EMotionType motion = JPH::EMotionType::Dynamic;
  if (desc.motion == PhysicsMotion::Static)
    motion = JPH::EMotionType::Static;
  else if (desc.motion == PhysicsMotion::Kinematic)
    motion = JPH::EMotionType::Kinematic;
  const bool isStatic = motion == JPH::EMotionType::Static;

  const JPH::ShapeRefC shape = impl->shapeFor(desc);

  JPH::BodyCreationSettings settings(
      shape, toJoltPosition(desc.position), toJolt(desc.rotation),
      motion, isStatic ? Layers::NonMoving : Layers::Moving);
  settings.mFriction = desc.friction;
  settings.mRestitution = desc.restitution;
  settings.mLinearVelocity = toJolt(desc.linearVelocity);
  if (motion == JPH::EMotionType::Dynamic) {
    // Bodies share shapes, so a body's density can't live on its shape.
    // Instead the shape's mass properties, worked out at Jolt's default of
    // 1000 kg/m^3, are scaled to the mass this body should have. Scaling
    // keeps the inertia matched to it.
    constexpr float kShapeDensity = 1000.0f;
    JPH::MassProperties massProperties = shape->GetMassProperties();
    massProperties.ScaleToMass(desc.mass > 0.0f
                                   ? desc.mass
                                   : massProperties.mMass * desc.density /
                                         kShapeDensity);
    settings.mOverrideMassProperties =
        JPH::EOverrideMassProperties::MassAndInertiaProvided;
    settings.mMassPropertiesOverride = massProperties;
  }

  // No-lock is fine because everything outside step() runs on one thread,
  // and it saves a mutex per call.
  const JPH::BodyID id =
      impl->system.GetBodyInterfaceNoLock().CreateAndAddBody(
          settings, isStatic ? JPH::EActivation::DontActivate
                             : JPH::EActivation::Activate);
  if (id.IsInvalid()) {
    if (!impl->warnedFull) {
      spdlog::warn("[physics] world is full ({} bodies), raise maxBodies",
                   impl->system.GetMaxBodies());
      impl->warnedFull = true;
    }
    return {};
  }
  // Drawn at its spawn pose until the first step moves it.
  const PhysicsPose spawn = impl->readPose(id);
  impl->drawn[id.GetIndex()] = {spawn, spawn};
  impl->userData[id.GetIndex()] = desc.userData;
  return PhysicsBodyId{id.GetIndexAndSequenceNumber()};
}

void PhysicsWorld::removeBody(PhysicsBodyId id) {
  JPH::BodyInterface &bodies = impl->system.GetBodyInterfaceNoLock();
  const JPH::BodyID bodyId(id.value);
  // IsAdded is false for invalid handles and already removed bodies.
  if (!id.isValid() || !bodies.IsAdded(bodyId))
    return;
  bodies.RemoveBody(bodyId);
  bodies.DestroyBody(bodyId);
  // Its index can go to a new body, so it must not stay in these lists.
  std::erase(impl->lastActive, bodyId);
  std::erase_if(impl->moved,
                [&](PhysicsBodyId moved) { return moved.value == id.value; });
}

void PhysicsWorld::optimizeBroadPhase() { impl->system.OptimizeBroadPhase(); }

void PhysicsWorld::step(float dt) {
  constexpr int collisionSteps = 1; // one collision pass per 1/60 s step
  const JPH::EPhysicsUpdateError errors =
      impl->system.Update(dt, collisionSteps, impl->tempAllocator.get(),
                          impl->jobSystem.get());
  if (errors != JPH::EPhysicsUpdateError::None) {
    ++impl->droppedContactSteps;
    // Each kind gets logged once, after that it's only counted.
    const uint32_t newErrors =
        static_cast<uint32_t>(errors) & ~impl->loggedErrors;
    if (newErrors != 0) {
      spdlog::warn("[physics] Jolt dropped contacts (error bits {:#x}), raise "
                   "maxBodyPairs or maxContactConstraints",
                   newErrors);
      impl->loggedErrors |= newErrors;
    }
  }
  impl->recordMovedBodies();
}

PhysicsPose PhysicsWorld::pose(PhysicsBodyId id) const {
  return impl->readPose(JPH::BodyID(id.value));
}

PhysicsPose PhysicsWorld::interpolatedPose(PhysicsBodyId id,
                                           float alpha) const {
  const JPH::BodyID bodyId(id.value);
  // A removed body's slot may belong to a new body by now.
  if (!id.isValid() || !impl->system.GetBodyInterfaceNoLock().IsAdded(bodyId))
    return {glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f)};
  const Impl::DrawnPose &pose = impl->drawn[bodyId.GetIndex()];
  return blendPoses(pose.previous, pose.current, alpha);
}

const std::vector<PhysicsBodyId> &PhysicsWorld::movedBodies() const {
  return impl->moved;
}

const std::vector<PhysicsMovedPose> &PhysicsWorld::movedPoses(float alpha) {
  // No IsAdded check like interpolatedPose(): removeBody() takes bodies out
  // of the moved list, so every id here is live.
  impl->movedPoseBuffer.clear();
  for (PhysicsBodyId id : impl->moved) {
    const uint32_t index = JPH::BodyID(id.value).GetIndex();
    const Impl::DrawnPose &pose = impl->drawn[index];
    impl->movedPoseBuffer.push_back(
        {impl->userData[index], blendPoses(pose.previous, pose.current, alpha)});
  }
  return impl->movedPoseBuffer;
}

bool PhysicsWorld::isActive(PhysicsBodyId id) const {
  return impl->system.GetBodyInterfaceNoLock().IsActive(JPH::BodyID(id.value));
}

float PhysicsWorld::mass(PhysicsBodyId id) const {
  JPH::BodyLockRead lock(impl->system.GetBodyLockInterfaceNoLock(),
                         JPH::BodyID(id.value));
  if (!lock.Succeeded() || !lock.GetBody().IsDynamic())
    return 0.0f;
  return 1.0f / lock.GetBody().GetMotionProperties()->GetInverseMass();
}

uint32_t PhysicsWorld::bodyCount() const { return impl->system.GetNumBodies(); }

uint32_t PhysicsWorld::activeBodyCount() const {
  return impl->system.GetNumActiveBodies(JPH::EBodyType::RigidBody);
}

uint32_t PhysicsWorld::droppedContactSteps() const {
  return impl->droppedContactSteps;
}

const std::vector<PhysicsDebugVertex> &
PhysicsWorld::debugLines(glm::vec3 cameraPosition) {
#ifdef JPH_DEBUG_RENDERER
  if (!impl->debugRenderer)
    impl->debugRenderer = std::make_unique<LineCollector>();
  LineCollector &lines = *impl->debugRenderer;
  lines.vertices.clear();
  // Jolt picks a shape's level of detail by distance to the camera.
  lines.SetCameraPos(toJoltPosition(cameraPosition));
  JPH::BodyManager::DrawSettings settings;
  settings.mDrawShape = true;
  settings.mDrawShapeColor = JPH::BodyManager::EShapeColor::SleepColor;
  impl->system.DrawBodies(settings, &lines);
  return lines.vertices;
#else
  (void)cameraPosition;
  return impl->noLines;
#endif
}

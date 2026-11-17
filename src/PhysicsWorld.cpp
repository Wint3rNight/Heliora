#include "PhysicsWorld.h"

// Jolt.h has to come before any other Jolt header.
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/ContactConstraintManager.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
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
#include <mutex>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <string>

namespace {

// Static bodies never test against each other, which cuts most pairs in a
// world that's mostly static. Character is the body inside each character:
// only moving bodies hit it, and queries pass through it.
namespace Layers {
constexpr JPH::ObjectLayer NonMoving = 0;
constexpr JPH::ObjectLayer Moving = 1;
constexpr JPH::ObjectLayer Character = 2;
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
    // Static bodies and characters' bodies only meet moving bodies, so they
    // only look in the moving tree.
    return layer == Layers::Moving ||
           broadPhaseLayer == BroadPhaseLayers::Moving;
  }
};

class ObjectPairFilter final : public JPH::ObjectLayerPairFilter {
public:
  bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
    return a == Layers::Moving || b == Layers::Moving;
  }
};

// For rays and sweeps that should pass through characters.
class IgnoreCharactersFilter final : public JPH::ObjectLayerFilter {
public:
  bool ShouldCollide(JPH::ObjectLayer layer) const override {
    return layer != Layers::Character;
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

// A capsule standing on the origin, so a character's position is its feet.
JPH::ShapeRefC makeCharacterShape(float radius, float height) {
  const float halfCylinder = std::max(0.5f * height - radius, 0.01f);
  return JPH::RotatedTranslatedShapeSettings(
             JPH::Vec3(0.0f, halfCylinder + radius, 0.0f),
             JPH::Quat::sIdentity(), new JPH::CapsuleShape(halfCylinder, radius))
      .Create()
      .Get();
}

// Dynamic bodies don't shove characters: Jolt would carry a character along
// at their speed, so a thrown ball took a 70 kg character with it. They hit
// its body instead, and it still pushes them.
constexpr float kMinImpactSpeed = 1.0f; // m/s

// New contacts on bodies that report impacts: the closing speed along the
// normal times the pair's reduced mass. Jolt calls this from its worker
// threads during the step, so the list is locked.
class ImpactContacts final : public JPH::ContactListener {
public:
  const std::vector<uint8_t> *reports = nullptr; // by body index
  const std::vector<uint64_t> *userData = nullptr;
  std::mutex mutex;
  std::vector<PhysicsImpact> impacts;

  void OnContactAdded(const JPH::Body &one, const JPH::Body &two,
                      const JPH::ContactManifold &manifold, JPH::ContactSettings &) override {
    const bool reportOne = (*reports)[one.GetID().GetIndex()] != 0;
    const bool reportTwo = (*reports)[two.GetID().GetIndex()] != 0;
    if ((!reportOne && !reportTwo) || manifold.mRelativeContactPointsOn1.empty())
      return;
    JPH::RVec3 point = manifold.GetWorldSpaceContactPointOn1(0);
    for (JPH::uint i = 1; i < manifold.mRelativeContactPointsOn1.size(); ++i)
      point += manifold.GetWorldSpaceContactPointOn1(i);
    point /= static_cast<float>(manifold.mRelativeContactPointsOn1.size());
    const JPH::Vec3 normal = manifold.mWorldSpaceNormal; // from one towards two
    // How fast they came together: the whole relative speed when it heads
    // into the contact at more than a shallow angle. The normal alone can
    // come out tilted where a compound's pieces meet (their hulls' rounded
    // edges), which would count a straight hit as a glancing one.
    const JPH::Vec3 relative = one.GetPointVelocity(point) - two.GetPointVelocity(point);
    const float along = relative.Dot(normal);
    const float speed = relative.Length();
    const float closing = along > 0.2f * speed ? speed : along;
    const float inverseMass =
        (one.IsDynamic() ? one.GetMotionProperties()->GetInverseMass() : 0.0f) +
        (two.IsDynamic() ? two.GetMotionProperties()->GetInverseMass() : 0.0f);
    // Bodies settling onto each other close slowly, and heavy ones would
    // still add up to a big impulse: only real hits count.
    if (closing < kMinImpactSpeed || inverseMass <= 0.0f)
      return;
    const float impulse = closing / inverseMass;
    if (impulse < 1.0f)
      return;
    const glm::vec3 at = toGlm(point);
    std::lock_guard lock(mutex);
    const PhysicsBodyId oneId{one.GetID().GetIndexAndSequenceNumber()};
    const PhysicsBodyId twoId{two.GetID().GetIndexAndSequenceNumber()};
    if (reportOne)
      impacts.push_back({oneId, (*userData)[one.GetID().GetIndex()], at, toGlm(normal), impulse,
                         twoId, toGlm(two.GetLinearVelocity())});
    if (reportTwo)
      impacts.push_back({twoId, (*userData)[two.GetID().GetIndex()], at, -toGlm(normal), impulse,
                         oneId, toGlm(one.GetLinearVelocity())});
  }
};

class CharacterContacts final : public JPH::CharacterContactListener {
public:
  void OnContactAdded(const JPH::CharacterVirtual *,
                      const JPH::CharacterContact &contact,
                      JPH::CharacterContactSettings &settings) override {
    settings.mCanPushCharacter = contact.mMotionTypeB != JPH::EMotionType::Dynamic;
  }
  void OnContactPersisted(const JPH::CharacterVirtual *character,
                          const JPH::CharacterContact &contact,
                          JPH::CharacterContactSettings &settings) override {
    OnContactAdded(character, contact, settings);
  }
};

// Jolt moves a character by the velocity it's given and keeps that velocity
// as it was, even pushing into a wall. Take away what pushed into anything
// it collided with, relative to that thing's own motion, so a fall stops on
// landing and a jump stops at a ceiling.
JPH::Vec3 velocityAfterContacts(const JPH::CharacterVirtual &character) {
  JPH::Vec3 velocity = character.GetLinearVelocity();
  for (const JPH::CharacterContact &contact : character.GetActiveContacts()) {
    if (!contact.mHadCollision || contact.mWasDiscarded)
      continue;
    const float into =
        (velocity - contact.mLinearVelocity).Dot(contact.mContactNormal);
    if (into < 0.0f)
      velocity -= into * contact.mContactNormal;
  }
  return velocity;
}

PhysicsGround toGround(JPH::CharacterBase::EGroundState state) {
  switch (state) {
  case JPH::CharacterBase::EGroundState::OnGround:
    return PhysicsGround::OnGround;
  case JPH::CharacterBase::EGroundState::OnSteepGround:
    return PhysicsGround::OnSteepGround;
  case JPH::CharacterBase::EGroundState::NotSupported:
    return PhysicsGround::NotSupported;
  case JPH::CharacterBase::EGroundState::InAir:
    break;
  }
  return PhysicsGround::InAir;
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

  ImpactContacts impactContacts;
  std::vector<uint8_t> reportsImpacts; // by body index

  // Declared after system, so they go first: each removes its body from it.
  CharacterContacts characterContacts;
  struct CharacterSlot {
    JPH::Ref<JPH::CharacterVirtual> character; // null once removed
    float radius = 0.0f;
    float height = 0.0f;
    glm::vec3 velocity{0.0f}; // after the last move, see velocityAfterContacts
  };
  std::vector<CharacterSlot> characters; // PhysicsCharacterId is the index
  // By body index: 1 for a character's body. Those are left out of the moved
  // list, since the character, not the body, says where it is.
  std::vector<uint8_t> isCharacterBody;
  int characterBodies = 0;
  IgnoreCharactersFilter ignoreCharacters;

  CharacterSlot *slot(PhysicsCharacterId id) {
    if (!id.isValid() || id.value >= characters.size() ||
        characters[id.value].character == nullptr)
      return nullptr;
    return &characters[id.value];
  }
  const CharacterSlot *slot(PhysicsCharacterId id) const {
    return const_cast<Impl *>(this)->slot(id);
  }

  // What a character collides with: everything, its own body excepted
  // (Jolt skips that itself).
  JPH::DefaultBroadPhaseLayerFilter characterBroadPhase() const {
    return system.GetDefaultBroadPhaseLayerFilter(Layers::Moving);
  }
  JPH::DefaultObjectLayerFilter characterLayers() const {
    return system.GetDefaultLayerFilter(Layers::Moving);
  }

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
  if (characterBodies > 0)
    std::erase_if(active, [&](JPH::BodyID id) {
      return isCharacterBody[id.GetIndex()] != 0;
    });

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
  // from scratch memory, and running out aborts. Twice that leaves room for
  // the rest of the step (islands, body pairs).
  const size_t contactBytes =
      size_t(config.maxContactConstraints) *
      (JPH::ContactConstraintManager::cMaxConstraintSize + sizeof(uint32_t));
  const size_t tempBytes =
      std::max<size_t>(config.tempAllocatorBytes, contactBytes * 2);
  if (tempBytes > config.tempAllocatorBytes)
    spdlog::info("[physics] scratch memory raised to {} MB for {} contacts",
                 tempBytes >> 20, config.maxContactConstraints);

  impl->tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(
      static_cast<JPH::uint>(tempBytes));
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
  impl->isCharacterBody.resize(config.maxBodies, 0);
  impl->reportsImpacts.resize(config.maxBodies, 0);
  impl->impactContacts.reports = &impl->reportsImpacts;
  impl->impactContacts.userData = &impl->userData;
  impl->system.SetContactListener(&impl->impactContacts);
  impl->lastActive.reserve(config.maxBodies);
  impl->active.reserve(config.maxBodies);
  impl->moved.reserve(config.maxBodies);
  applySettings(config);
}

PhysicsWorld::~PhysicsWorld() = default;

glm::vec3 PhysicsWorld::linearVelocity(PhysicsBodyId id) const {
  JPH::BodyLockRead lock(impl->system.GetBodyLockInterfaceNoLock(), JPH::BodyID(id.value));
  return lock.Succeeded() ? toGlm(lock.GetBody().GetLinearVelocity()) : glm::vec3(0.0f);
}

glm::vec3 PhysicsWorld::angularVelocity(PhysicsBodyId id) const {
  JPH::BodyLockRead lock(impl->system.GetBodyLockInterfaceNoLock(), JPH::BodyID(id.value));
  return lock.Succeeded() ? toGlm(lock.GetBody().GetAngularVelocity()) : glm::vec3(0.0f);
}

void PhysicsWorld::setLinearVelocity(PhysicsBodyId id, glm::vec3 velocity) {
  JPH::BodyInterface &bodies = impl->system.GetBodyInterfaceNoLock();
  const JPH::BodyID bodyId(id.value);
  if (!id.isValid() || !bodies.IsAdded(bodyId) ||
      bodies.GetMotionType(bodyId) != JPH::EMotionType::Dynamic)
    return;
  bodies.SetLinearVelocity(bodyId, toJolt(velocity));
  bodies.ActivateBody(bodyId);
}

std::vector<PhysicsImpact> PhysicsWorld::takeImpacts() {
  std::lock_guard lock(impl->impactContacts.mutex);
  return std::exchange(impl->impactContacts.impacts, {});
}

void PhysicsWorld::applySettings(const Config &config) {
  const JPH::Vec3 gravity = toJolt(config.gravity);
  const bool gravityChanged = gravity != impl->system.GetGravity();
  impl->system.SetGravity(gravity);
  JPH::PhysicsSettings settings = impl->system.GetPhysicsSettings();
  settings.mNumVelocitySteps = static_cast<JPH::uint>(config.velocitySteps);
  settings.mNumPositionSteps = static_cast<JPH::uint>(config.positionSteps);
  settings.mTimeBeforeSleep = config.sleepDelay;
  settings.mPointVelocitySleepThreshold = config.sleepSpeed;
  settings.mPenetrationSlop = config.penetrationSlop;
  impl->system.SetPhysicsSettings(settings);
  if (gravityChanged) {
    JPH::BodyIDVector bodies;
    impl->system.GetBodies(bodies);
    // Static bodies can't be woken, and Jolt skips them.
    impl->system.GetBodyInterfaceNoLock().ActivateBodies(
        bodies.data(), static_cast<int>(bodies.size()));
  }
}

glm::vec3 PhysicsWorld::gravity() const {
  return toGlm(impl->system.GetGravity());
}

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
  settings.mAngularVelocity = toJolt(desc.angularVelocity);
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
  impl->reportsImpacts[id.GetIndex()] = desc.reportImpacts ? 1 : 0;
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
  impl->reportsImpacts[bodyId.GetIndex()] = 0;
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
  // What characters stand on may have moved or sped up in the step.
  for (Impl::CharacterSlot &slot : impl->characters)
    if (slot.character)
      slot.character->UpdateGroundVelocity();
}

void PhysicsWorld::setPose(PhysicsBodyId id, const PhysicsPose &pose) {
  JPH::BodyInterface &bodies = impl->system.GetBodyInterfaceNoLock();
  const JPH::BodyID bodyId(id.value);
  if (!id.isValid() || !bodies.IsAdded(bodyId))
    return;
  // Static bodies can't be woken; asking would trip a Jolt assert.
  const JPH::EActivation activation =
      bodies.GetMotionType(bodyId) == JPH::EMotionType::Static
          ? JPH::EActivation::DontActivate
          : JPH::EActivation::Activate;
  bodies.SetPositionAndRotation(bodyId, toJoltPosition(pose.position),
                                toJolt(pose.rotation), activation);
  const PhysicsPose placed = impl->readPose(bodyId);
  impl->drawn[bodyId.GetIndex()] = {placed, placed};
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

std::optional<PhysicsRayHit> PhysicsWorld::castRay(glm::vec3 origin,
                                                   glm::vec3 direction,
                                                   float maxDistance) const {
  // Jolt's ray is as long as its direction vector, and a hit comes back as
  // the fraction of that length.
  const glm::vec3 ray = glm::normalize(direction) * maxDistance;
  const JPH::RRayCast cast(toJoltPosition(origin), toJolt(ray));
  JPH::RayCastResult hit;
  if (!impl->system.GetNarrowPhaseQueryNoLock().CastRay(
          cast, hit, {}, impl->ignoreCharacters))
    return std::nullopt;
  PhysicsRayHit result{PhysicsBodyId{hit.mBodyID.GetIndexAndSequenceNumber()},
                       impl->userData[hit.mBodyID.GetIndex()],
                       hit.mFraction * maxDistance};
  // The normal of the face it hit: the sub-shape ID says which part of the
  // shape (a compound's child, a mesh's triangle) that was.
  JPH::BodyLockRead lock(impl->system.GetBodyLockInterfaceNoLock(),
                         hit.mBodyID);
  if (lock.Succeeded()) {
    const JPH::Vec3 n = lock.GetBody().GetWorldSpaceSurfaceNormal(
        hit.mSubShapeID2, cast.GetPointOnRay(hit.mFraction));
    result.normal = glm::vec3(n.GetX(), n.GetY(), n.GetZ());
    // A triangle hit from behind reports the side facing away.
    if (glm::dot(result.normal, ray) > 0.0f)
      result.normal = -result.normal;
  }
  return result;
}

std::optional<PhysicsRayHit> PhysicsWorld::castSphere(glm::vec3 origin,
                                                      glm::vec3 direction,
                                                      float radius,
                                                      float maxDistance) const {
  const glm::vec3 sweep = glm::normalize(direction) * maxDistance;
  JPH::SphereShape sphere(radius);
  sphere.SetEmbedded(); // on the stack: no reference counting
  const JPH::RShapeCast cast = JPH::RShapeCast::sFromWorldTransform(
      &sphere, JPH::Vec3::sOne(),
      JPH::RMat44::sTranslation(toJoltPosition(origin)), toJolt(sweep));
  JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> closest;
  impl->system.GetNarrowPhaseQueryNoLock().CastShape(
      cast, JPH::ShapeCastSettings(), JPH::RVec3::sZero(), closest, {},
      impl->ignoreCharacters);
  if (!closest.HadHit())
    return std::nullopt;
  const JPH::ShapeCastResult &hit = closest.mHit;
  // The penetration axis points from the ball into what it hit.
  const JPH::Vec3 normal = -hit.mPenetrationAxis.NormalizedOr(-JPH::Vec3::sAxisY());
  return PhysicsRayHit{PhysicsBodyId{hit.mBodyID2.GetIndexAndSequenceNumber()},
                       impl->userData[hit.mBodyID2.GetIndex()],
                       hit.mFraction * maxDistance, toGlm(normal)};
}

PhysicsCharacterId PhysicsWorld::addCharacter(const PhysicsCharacterDesc &desc) {
  JPH::Ref<JPH::CharacterVirtualSettings> settings =
      new JPH::CharacterVirtualSettings();
  settings->mShape = makeCharacterShape(desc.radius, desc.height);
  settings->mInnerBodyShape = settings->mShape;
  settings->mInnerBodyLayer = Layers::Character;
  settings->mMass = desc.mass;
  // Only contacts below the centre of the bottom sphere hold it up; the
  // rest are walls.
  settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -desc.radius);
  JPH::Ref<JPH::CharacterVirtual> character = new JPH::CharacterVirtual(
      settings, toJoltPosition(desc.position), JPH::Quat::sIdentity(), 0,
      &impl->system);
  character->SetListener(&impl->characterContacts);
  const JPH::BodyID body = character->GetInnerBodyID();
  if (body.IsInvalid())
    spdlog::warn("[physics] world is full, a character got no body: thrown "
                 "things will pass through it");
  else {
    impl->isCharacterBody[body.GetIndex()] = 1;
    ++impl->characterBodies;
  }
  impl->characters.push_back({character, desc.radius, desc.height});
  const PhysicsCharacterId id{static_cast<uint32_t>(impl->characters.size() - 1)};
  placeCharacter(id, desc.position);
  return id;
}

void PhysicsWorld::removeCharacter(PhysicsCharacterId id) {
  Impl::CharacterSlot *slot = impl->slot(id);
  if (!slot)
    return;
  const JPH::BodyID body = slot->character->GetInnerBodyID();
  if (!body.IsInvalid()) {
    impl->isCharacterBody[body.GetIndex()] = 0;
    --impl->characterBodies;
  }
  slot->character = nullptr; // its destructor removes its body
}

void PhysicsWorld::moveCharacter(PhysicsCharacterId id,
                                 const PhysicsCharacterMove &move, float dt) {
  Impl::CharacterSlot *slot = impl->slot(id);
  if (!slot)
    return;
  JPH::CharacterVirtual &character = *slot->character;
  character.SetMaxSlopeAngle(move.maxSlope);
  character.SetMaxStrength(move.pushStrength);
  character.SetLinearVelocity(toJolt(move.velocity));
  // Update (slide along what it hits), then climb a step if it walked into
  // one, then snap down to the ground if it walked off a small drop.
  JPH::CharacterVirtual::ExtendedUpdateSettings settings;
  settings.mStickToFloorStepDown = JPH::Vec3(0.0f, -move.stickDown, 0.0f);
  settings.mWalkStairsStepUp = JPH::Vec3(0.0f, move.stepUp, 0.0f);
  character.ExtendedUpdate(dt, impl->system.GetGravity(), settings,
                           impl->characterBroadPhase(), impl->characterLayers(),
                           JPH::BodyFilter(), JPH::ShapeFilter(),
                           *impl->tempAllocator);
  slot->velocity = toGlm(velocityAfterContacts(character));
}

void PhysicsWorld::placeCharacter(PhysicsCharacterId id, glm::vec3 position) {
  Impl::CharacterSlot *slot = impl->slot(id);
  if (!slot)
    return;
  slot->character->SetPosition(toJoltPosition(position));
  slot->character->SetLinearVelocity(JPH::Vec3::sZero());
  slot->velocity = glm::vec3(0.0f);
  // Finds what it touches there, so it starts out standing, not falling.
  slot->character->RefreshContacts(impl->characterBroadPhase(),
                                   impl->characterLayers(), JPH::BodyFilter(),
                                   JPH::ShapeFilter(), *impl->tempAllocator);
}

bool PhysicsWorld::setCharacterHeight(PhysicsCharacterId id, float height) {
  Impl::CharacterSlot *slot = impl->slot(id);
  if (!slot)
    return false;
  if (height == slot->height)
    return true;
  const JPH::ShapeRefC shape = makeCharacterShape(slot->radius, height);
  // Refuses if the new capsule would sink further into something than
  // contacts normally do.
  const float maxPenetration =
      1.5f * impl->system.GetPhysicsSettings().mPenetrationSlop;
  if (!slot->character->SetShape(shape, maxPenetration,
                                 impl->characterBroadPhase(),
                                 impl->characterLayers(), JPH::BodyFilter(),
                                 JPH::ShapeFilter(), *impl->tempAllocator))
    return false;
  slot->character->SetInnerBodyShape(shape);
  slot->height = height;
  return true;
}

PhysicsCharacterState PhysicsWorld::characterState(PhysicsCharacterId id) const {
  const Impl::CharacterSlot *slot = impl->slot(id);
  if (!slot)
    return {};
  const JPH::CharacterVirtual &character = *slot->character;
  PhysicsCharacterState state;
  state.position = toGlm(character.GetPosition());
  state.velocity = slot->velocity;
  state.ground = toGround(character.GetGroundState());
  state.groundNormal = toGlm(character.GetGroundNormal());
  state.groundVelocity = toGlm(character.GetGroundVelocity());
  state.height = slot->height;
  return state;
}

uint32_t PhysicsWorld::bodyCount() const { return impl->system.GetNumBodies(); }

uint32_t PhysicsWorld::bodyLimit() const {
  return impl->system.GetMaxBodies();
}

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

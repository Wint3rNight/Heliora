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
  glm::vec3 angularVelocity{0.0f}; // radians/s at the start
  uint64_t userData = 0;           // ours, handed back by movedPoses()
  // Things hitting it are reported by takeImpacts(), for breakable objects.
  bool reportImpacts = false;
};

// Something hit a body that reports impacts: where, the contact normal (out
// of the body), and the impulse (N s): how fast they came together (the
// whole relative speed unless it only grazes) times the pair's reduced mass.
struct PhysicsImpact {
  PhysicsBodyId body;
  uint64_t userData = 0; // the body's
  glm::vec3 point{0.0f};
  glm::vec3 normal{0.0f};
  float impulse = 0.0f;
  // What hit it, and that body's velocity just before the contact (the
  // solver may since have stopped or bounced it).
  PhysicsBodyId other;
  glm::vec3 otherVelocity{0.0f};
};

struct PhysicsPose {
  glm::vec3 position;
  glm::quat rotation;
};

struct PhysicsRayHit {
  PhysicsBodyId body;
  uint64_t userData; // the body's
  float distance;    // m from the ray's start
  glm::vec3 normal{0.0f, 1.0f, 0.0f}; // unit, the way the surface faces
};

struct PhysicsMovedPose {
  uint64_t userData;
  PhysicsPose pose;
};

// A walking character: a capsule that slides along walls, climbs steps and
// stands on slopes instead of tumbling like a body. Jolt's CharacterVirtual.
// Ids are never reused, so a removed character's id stays dead.
struct PhysicsCharacterId {
  uint32_t value = 0xffffffffu;
  bool isValid() const { return value != 0xffffffffu; }
};

struct PhysicsCharacterDesc {
  float radius = 0.3f;
  float height = 1.8f;   // feet to the top of the head, at least 2 * radius
  float mass = 70.0f;    // kg, what it weighs on bodies it stands on
  glm::vec3 position{0.0f}; // its feet
};

enum class PhysicsGround : uint8_t {
  OnGround,      // standing on ground it can walk on
  OnSteepGround, // touching ground too steep to stand on: it slides off
  NotSupported,  // touching something, but nothing underneath (a ledge edge)
  InAir,
};

// What one move does. Everything here can change from step to step, so the
// tuning values can be dragged while playing.
struct PhysicsCharacterMove {
  // m/s for this step. The caller adds gravity: the character only goes
  // where it's told, with walls, floors and steps in the way.
  glm::vec3 velocity{0.0f};
  // m, how far it lifts itself to get onto a step. The round bottom slides
  // over about 10 cm more, so 0.3 climbs steps up to about 40 cm.
  float stepUp = 0.3f;
  float stickDown = 0.5f;  // m, how far it snaps down to keep to slopes and
                           // stairs while walking; 0 = never
  float maxSlope = 0.785398f; // radians (45 degrees); steeper can't be walked up
  float pushStrength = 100.0f; // N, the most it pushes a dynamic body with
};

struct PhysicsCharacterState {
  glm::vec3 position{0.0f}; // its feet
  glm::vec3 velocity{0.0f}; // after the last move: walls and floors take
                            // away what they blocked
  PhysicsGround ground = PhysicsGround::InAir;
  glm::vec3 groundNormal{0.0f, 1.0f, 0.0f};
  glm::vec3 groundVelocity{0.0f}; // of what it stands on, to ride along
  float height = 0.0f;            // the capsule's current height
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
  glm::vec3 gravity() const;

  // Moves the body there at once, for the editor. It's drawn there too, not
  // blended in from where it was. Wakes it if it can move.
  void setPose(PhysicsBodyId id, const PhysicsPose &pose);
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
  glm::vec3 linearVelocity(PhysicsBodyId id) const;
  // Sets a moving body's velocity and wakes it; static bodies ignore it.
  void setLinearVelocity(PhysicsBodyId id, glm::vec3 velocity);
  glm::vec3 angularVelocity(PhysicsBodyId id) const; // radians/s

  // New contacts on bodies that report impacts since the last call, closing
  // at 1 m/s or more and harder than a knock (1 N s), gathered during
  // step(). Settling and resting contacts don't count.
  std::vector<PhysicsImpact> takeImpacts();

  // The nearest body the ray hits within maxDistance, for clicking on
  // things. direction needn't be unit length. Characters don't block it.
  std::optional<PhysicsRayHit> castRay(glm::vec3 origin, glm::vec3 direction,
                                       float maxDistance) const;
  // Sweeps a ball of the radius from origin along direction and returns the
  // first body it touches; distance is how far its centre got. A ball that
  // starts touching something hits at 0. Characters don't block it, so a
  // camera behind one can cast from inside its capsule.
  std::optional<PhysicsRayHit> castSphere(glm::vec3 origin,
                                          glm::vec3 direction, float radius,
                                          float maxDistance) const;

  // Characters. Each one also gets a body of its own shape, so thrown and
  // falling bodies hit it; that body never shows up in movedPoses().
  PhysicsCharacterId addCharacter(const PhysicsCharacterDesc &desc);
  void removeCharacter(PhysicsCharacterId id);
  // Moves the character by move.velocity * dt right away, sliding along
  // whatever it hits. Call it before step(), once per step, like Jolt's
  // samples do.
  void moveCharacter(PhysicsCharacterId id, const PhysicsCharacterMove &move,
                     float dt);
  // Puts its feet at position, standing still, as if it had always been
  // there (for spawning).
  void placeCharacter(PhysicsCharacterId id, glm::vec3 position);
  // Changes the capsule's height, keeping its feet where they are, for
  // crouching. False, and nothing changes, if the taller capsule wouldn't
  // fit (a ceiling over a crouched character).
  bool setCharacterHeight(PhysicsCharacterId id, float height);
  // Up to date after moveCharacter() and step() (what it stands on can move
  // in the step).
  PhysicsCharacterState characterState(PhysicsCharacterId id) const;

  uint32_t bodyCount() const;
  // The most bodies this world holds (Config::maxBodies when it was made).
  uint32_t bodyLimit() const;
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

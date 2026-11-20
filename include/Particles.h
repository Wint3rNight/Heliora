#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <deque>
#include <vector>

// The particles (sparks, smoke, dust; built in step 1.4a): what the CPU asks
// for each frame. The GPU moves and draws them (ParticlePass, Shaders/particles.*).
// No Vulkan here, so it's tested like the rest of heliora_core.

// One particle as it starts, laid out as the GPU keeps it (80 bytes).
struct ParticleSpawn {
  glm::vec3 position{0.0f};
  float age = 0.0f; // seconds; the GPU counts it up
  glm::vec3 velocity{0.0f};
  float life = 1.0f; // seconds it lasts
  glm::vec3 colour{1.0f}; // HDR
  float size = 0.05f;     // m across
  float drag = 1.0f;      // 1/s: how fast its speed dies away
  float swirl = 0.0f;     // m/s^2 of the curl field pushing it round
  float gravity = 0.0f;   // its share of gravity: grit falls, smoke floats
  // 0: it glows, added on top (sparks, fire). Above 0: smoke or dust, this
  // opaque at its middle, laid over what's behind (premultiplied alpha).
  float opacity = 0.0f;
  // Its size at the end of its life over its size at the start, reached
  // fast at first, then slower (particleSizeAt). Under 1 it shrinks (sparks,
  // grit); over 1 it's smoke spreading into the air, thinning as it goes
  // (smokeCover).
  float spread = 0.6f;
  float unused[3] = {0.0f, 0.0f, 0.0f};
};
static_assert(sizeof(ParticleSpawn) == 80, "matches Shaders/particles.comp and particles.vert");

// A small, fast random generator (xorshift32): the same seed gives the same
// particles.
class ParticleRandom {
public:
  explicit ParticleRandom(uint32_t seed = 0x9e3779b9u) : state(seed ? seed : 1u) {}
  float next();             // 0 to 1
  glm::vec3 onSphere();     // a unit direction
  glm::vec3 inCone(glm::vec3 axis, float halfAngle); // a unit direction within halfAngle of axis

private:
  uint32_t state;
};

// How many to start this frame at rate a second, the fraction carried to the
// next frame in carry.
int spawnCount(float rate, float dt, float &carry);

// Its size (m across) a share t (0 to 1) of the way through its life. The
// particle shaders work it out the same way.
float particleSizeAt(const ParticleSpawn &p, float t);
// How much of what's behind a puff of smoke covers at its middle, having
// spread to `spread` times its starting width, when it started covering
// `opacity`: it keeps its smoke, so its optical depth goes as one over its
// area (Beer-Lambert). The particle shaders work it out the same way.
float smokeCover(float opacity, float spread);
// A particle's velocity after dt with drag (1/s) pulling it toward the
// air's: smoke drifts with the wind. particles.comp steps them the same way.
glm::vec3 airDrag(glm::vec3 velocity, glm::vec3 wind, float drag, float dt);

// Particles like kind laid every `every` m along from to to (a trail
// behind something moving), carry holding the distance since the last one,
// so the spacing is the same at any frame rate.
void emitTrail(std::vector<ParticleSpawn> &out, glm::vec3 from, glm::vec3 to, float every, float &carry,
               const ParticleSpawn &kind);

// count particles like kind at random places along the path (a polyline),
// each moving along it at speed toward its end.
void emitAlongPath(std::vector<ParticleSpawn> &out, const std::vector<glm::vec3> &path, int count,
                   float speed, const ParticleSpawn &kind, ParticleRandom &random);
// count particles like kind at random places in the box lo to hi, each
// moving away from its middle at speed.
void emitFromBox(std::vector<ParticleSpawn> &out, glm::vec3 lo, glm::vec3 hi, int count, float speed,
                 const ParticleSpawn &kind, ParticleRandom &random);
// count particles like kind from apex, each moving within halfAngle of axis
// (unit) at a speed between slowest and fastest.
void emitInCone(std::vector<ParticleSpawn> &out, glm::vec3 apex, glm::vec3 axis, float halfAngle,
                int count, float slowest, float fastest, const ParticleSpawn &kind,
                ParticleRandom &random);

// Which of a ring's slots can still hold a living particle. Each frame's new
// ones go in the next slots from the head, so the living ones are the last
// few frames' worth: from the oldest frame not yet past its longest life on
// to the head (counting a short-lived frame behind a long-lived one, never
// leaving a living one out). The particles are drawn over just those.
class LiveRing {
public:
  explicit LiveRing(uint32_t capacity) : capacity(capacity) {}
  // A frame: dt on, then count new ones (lasting at most longest s) at the head.
  void step(float dt, uint32_t count, float longest);
  void clear();
  uint32_t head() const { return next; }
  uint32_t live() const; // slots that may be alive, ending at the head
  uint32_t first() const { return (next + capacity - live() % capacity) % capacity; }

private:
  struct Batch {
    uint32_t count;
    float until; // s on the ring's clock
  };
  uint32_t capacity;
  uint32_t next = 0;
  float clock = 0.0f;
  uint64_t counted = 0; // the batches' counts together
  std::deque<Batch> batches;
};

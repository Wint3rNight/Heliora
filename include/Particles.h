#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

// The particles (sparks, smoke, dust; built in step 1.4a): what the CPU asks
// for each frame. The GPU moves and draws them (ParticlePass, Shaders/particles.*).
// No Vulkan here, so it's tested like the rest of heliora_core.

// One particle as it starts, laid out as the GPU keeps it (64 bytes).
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
};
static_assert(sizeof(ParticleSpawn) == 64, "matches Shaders/particles.comp");

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

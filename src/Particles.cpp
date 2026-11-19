#include "Particles.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

float ParticleRandom::next() {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<float>(state >> 8) * (1.0f / 16777216.0f); // the top 24 bits
}

glm::vec3 ParticleRandom::onSphere() {
  const float z = 2.0f * next() - 1.0f;
  const float a = glm::two_pi<float>() * next();
  const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
  return {r * std::cos(a), r * std::sin(a), z};
}

glm::vec3 ParticleRandom::inCone(glm::vec3 axis, float halfAngle) {
  // Even over the cap of the sphere the cone cuts out.
  const float cosHalf = std::cos(halfAngle);
  const float z = 1.0f - next() * (1.0f - cosHalf);
  const float a = glm::two_pi<float>() * next();
  const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
  const glm::vec3 ref = std::fabs(axis.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
  const glm::vec3 side = glm::normalize(glm::cross(axis, ref));
  const glm::vec3 up = glm::cross(side, axis);
  return glm::normalize(axis * z + (side * std::cos(a) + up * std::sin(a)) * r);
}

int spawnCount(float rate, float dt, float &carry) {
  carry += std::max(0.0f, rate * dt);
  const int count = static_cast<int>(carry);
  carry -= static_cast<float>(count);
  return count;
}

void emitAlongPath(std::vector<ParticleSpawn> &out, const std::vector<glm::vec3> &path, int count,
                   float speed, const ParticleSpawn &kind, ParticleRandom &random) {
  if (path.size() < 2 || count <= 0)
    return;
  std::vector<float> along(path.size(), 0.0f);
  for (size_t i = 1; i < path.size(); ++i)
    along[i] = along[i - 1] + glm::distance(path[i - 1], path[i]);
  const float total = along.back();
  if (total <= 0.0f)
    return;
  out.reserve(out.size() + count);
  for (int k = 0; k < count; ++k) {
    const float at = random.next() * total;
    size_t leg = 1;
    while (leg + 1 < path.size() && along[leg] < at)
      ++leg;
    const float length = along[leg] - along[leg - 1];
    const float t = length > 0.0f ? (at - along[leg - 1]) / length : 0.0f;
    ParticleSpawn p = kind;
    p.position = glm::mix(path[leg - 1], path[leg], t);
    p.velocity = length > 0.0f ? (path[leg] - path[leg - 1]) / length * speed : glm::vec3(0.0f);
    out.push_back(p);
  }
}

void emitFromBox(std::vector<ParticleSpawn> &out, glm::vec3 lo, glm::vec3 hi, int count, float speed,
                 const ParticleSpawn &kind, ParticleRandom &random) {
  if (count <= 0)
    return;
  const glm::vec3 middle = 0.5f * (lo + hi);
  out.reserve(out.size() + count);
  for (int k = 0; k < count; ++k) {
    ParticleSpawn p = kind;
    p.position = lo + (hi - lo) * glm::vec3(random.next(), random.next(), random.next());
    const glm::vec3 away = p.position - middle;
    p.velocity = (glm::length(away) > 1e-5f ? glm::normalize(away) : random.onSphere()) * speed;
    out.push_back(p);
  }
}

void emitInCone(std::vector<ParticleSpawn> &out, glm::vec3 apex, glm::vec3 axis, float halfAngle,
                int count, float slowest, float fastest, const ParticleSpawn &kind,
                ParticleRandom &random) {
  if (count <= 0)
    return;
  out.reserve(out.size() + count);
  for (int k = 0; k < count; ++k) {
    ParticleSpawn p = kind;
    p.position = apex;
    p.velocity = random.inCone(axis, halfAngle) * glm::mix(slowest, fastest, random.next());
    out.push_back(p);
  }
}

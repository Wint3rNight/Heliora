#pragma once

// Static floors, boxes and ramps shared by the character tests.

#include "PhysicsWorld.h"

#include <glm/gtc/quaternion.hpp>

#include <cmath>

namespace test {

inline PhysicsBodyId addStaticBox(PhysicsWorld &world, glm::vec3 position,
                                  glm::vec3 halfExtents,
                                  glm::quat rotation = glm::quat(1, 0, 0, 0)) {
  PhysicsBodyDesc box;
  box.shape = PhysicsShape::Box;
  box.halfExtents = halfExtents;
  box.motion = PhysicsMotion::Static;
  box.position = position;
  box.rotation = rotation;
  return world.addBody(box);
}

// A static floor whose top surface is at y = 0.
inline void addFloor(PhysicsWorld &world) {
  addStaticBox(world, glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(50.0f, 0.5f, 50.0f));
}

// A 8 m long, 20 cm thick ramp rising along +x at the angle. Its top surface
// meets the floor at about x = 4 - 0.1 / sin(angle).
inline void addRamp(PhysicsWorld &world, float degrees) {
  addStaticBox(world, glm::vec3(4.0f, 0.0f, 0.0f), glm::vec3(4.0f, 0.1f, 2.0f),
               glm::angleAxis(glm::radians(degrees), glm::vec3(0, 0, 1)));
}

inline float rampSurfaceY(float x, float degrees) {
  const float a = glm::radians(degrees);
  return (x - (4.0f - 0.1f / std::sin(a))) * std::tan(a);
}

} // namespace test

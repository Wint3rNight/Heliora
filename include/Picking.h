#pragma once

#include <glm/glm.hpp>

#include <optional>

// Clicking on the 3D view: a ray from the camera through the clicked pixel,
// tested against physics bodies (PhysicsWorld::castRay) and, for objects
// without a body, against their bounding spheres.

struct PickRay {
  glm::vec3 origin;
  glm::vec3 direction; // unit length
};

// The ray through a pixel. pixel and viewport are in window coordinates,
// with y counting down from the top. invViewProj undoes the unjittered
// projection times view the frame was drawn with (depth 0 to 1, y flipped
// for Vulkan). The ray starts on the near plane.
PickRay rayFromScreen(glm::vec2 pixel, glm::vec2 viewport,
                      const glm::mat4 &invViewProj);

// How far along the ray it enters the sphere, or nothing if it misses or the
// sphere is behind. From inside, it's the distance to the far side, so a big
// sphere around the camera doesn't beat everything in front of it.
std::optional<float> rayHitsSphere(const PickRay &ray, glm::vec3 centre,
                                   float radius);

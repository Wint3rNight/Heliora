#pragma once

#include <glm/glm.hpp>

#include <vector>

// The geometry of light effects (tracers, muzzle flashes from step 1.6):
// strips built on the CPU each frame and drawn additively in the transparent
// subpass, before bloom, so they glow (EffectsPass). No Vulkan here, so it's
// tested like the rest of heliora_core.

struct EffectVertex {
  glm::vec3 position{0.0f};
  glm::vec4 core{0.0f}; // rgb in HDR, a how far it's faded in
  glm::vec4 edge{0.0f}; // the colour toward its sides
  glm::vec2 uv{0.0f};   // x along it in metres (the flow), y across it, 0 to 1
};

// A strip through the points turned to face the eye, two triangles a
// segment: its width eases from width0 at the first point to width1 at the
// last, its core colour from core0 to core1. Fewer than two points add
// nothing.
void addFacingStrip(std::vector<EffectVertex> &out, const std::vector<glm::vec3> &points,
                    glm::vec3 eye, float width0, float width1, glm::vec4 core0, glm::vec4 core1,
                    glm::vec4 edge);

// The side of a cone from apex along axis (unit), halfAngle (radians) from
// the axis to its side, its rim length from the apex along the side; uv.y
// runs 0 at the apex to 1 at the rim.
void addCone(std::vector<EffectVertex> &out, glm::vec3 apex, glm::vec3 axis, float length,
             float halfAngle, glm::vec4 core, glm::vec4 edge, int segments);

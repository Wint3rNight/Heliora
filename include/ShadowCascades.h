#pragma once

#include <glm/glm.hpp>

#include <cstdint>

constexpr int kShadowCascades = 4;

struct ShadowCascades {
  glm::vec4 splits; // view depth where each cascade ends
  glm::mat4 lightSpace[kShadowCascades];
};

// Fits the sun's shadow cascades to the camera. Each cascade is a sphere
// around the camera, big enough for its slice of the view frustum, snapped
// to shadow-map texels so shadows don't shimmer as the camera moves.
// lightDir is the way the light travels (from the sun into the scene).
ShadowCascades fitShadowCascades(const glm::mat4 &view,
                                 const glm::mat4 &projection,
                                 glm::vec3 lightDir, float csmFar,
                                 float drawDistance, uint32_t shadowMapSize);

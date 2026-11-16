#include "Frustum.h"

void extractFrustumPlanes(const glm::mat4 &viewProj, glm::vec4 planes[6]) {
  // Gribb-Hartmann: GLM is column-major, so transpose to get rows as columns.
  const glm::mat4 t = glm::transpose(viewProj);
  planes[0] = t[3] + t[0]; // left
  planes[1] = t[3] - t[0]; // right
  planes[2] = t[3] + t[1]; // bottom
  planes[3] = t[3] - t[1]; // top
  planes[4] = t[3] + t[2]; // near
  planes[5] = t[3] - t[2]; // far
  for (int i = 0; i < 6; i++)
    planes[i] /= glm::length(glm::vec3(planes[i]));
}

bool sphereInFrustum(const glm::vec4 planes[6], glm::vec3 center,
                     float radius) {
  for (int i = 0; i < 6; i++)
    if (glm::dot(glm::vec3(planes[i]), center) + planes[i].w < -radius)
      return false;
  return true;
}

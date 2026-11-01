#include "ShadowCascades.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

ShadowCascades fitShadowCascades(const glm::mat4 &view,
                                 const glm::mat4 &projection,
                                 glm::vec3 lightDirIn, float csmFarValue,
                                 float drawDistance, uint32_t shadowMapSize) {
  ShadowCascades out;
  const float nearPlane = 0.1f;
  const float csmFar = std::min(csmFarValue, drawDistance);
  const float farPlane = csmFar;
  const float lambda = 0.75f;

  float splits[kShadowCascades];
  for (int i = 0; i < kShadowCascades; i++) {
    float p = (i + 1) / float(kShadowCascades);
    float logSplit = nearPlane * std::pow(farPlane / nearPlane, p);
    float uniformSplit = nearPlane + (farPlane - nearPlane) * p;
    splits[i] = lambda * logSplit + (1.0f - lambda) * uniformSplit;
  }
  out.splits = glm::vec4(splits[0], splits[1], splits[2], splits[3]);

  glm::vec3 lightDir = glm::normalize(lightDirIn);
  glm::vec3 lightUp = (std::abs(lightDir.y) > 0.99f)
                          ? glm::vec3(0.0f, 0.0f, 1.0f)
                          : glm::vec3(0.0f, 1.0f, 0.0f);
  glm::mat4 lightView =
      glm::lookAt(-lightDir * 100.0f, glm::vec3(0.0f), lightUp);
  glm::mat4 invCam = glm::inverse(projection * view);

  auto viewDepthToNdcZ = [&](float depth) -> float {
    glm::vec4 clip =
        projection * glm::vec4(0.0f, 0.0f, -depth, 1.0f);
    return clip.z / clip.w;
  };

  float prevSplit = nearPlane;
  for (int i = 0; i < kShadowCascades; i++) {
    float nearNdc = viewDepthToNdcZ(prevSplit);
    float farNdc = viewDepthToNdcZ(splits[i]);

    glm::vec3 corners[8];
    int k = 0;
    for (float nx : {-1.0f, 1.0f}) {
      for (float ny : {-1.0f, 1.0f}) {
        for (float nz : {nearNdc, farNdc}) {
          glm::vec4 world = invCam * glm::vec4(nx, ny, nz, 1.0f);
          corners[k++] = glm::vec3(world) / world.w;
        }
      }
    }

    glm::vec3 frustumCenter = glm::vec3(glm::inverse(view)[3]);

    float sphereRadius = 0.0f;
    for (const glm::vec3 &corner : corners)
      sphereRadius =
          glm::max(sphereRadius, glm::length(corner - frustumCenter));

    glm::vec3 lsCenter =
        glm::vec3(lightView * glm::vec4(frustumCenter, 1.0f));
    float texelSize = 2.0f * sphereRadius / float(shadowMapSize);
    if (texelSize > 0.0f) {
      lsCenter.x = std::floor(lsCenter.x / texelSize) * texelSize;
      lsCenter.y = std::floor(lsCenter.y / texelSize) * texelSize;
    }

    glm::vec3 minBounds(lsCenter.x - sphereRadius, lsCenter.y - sphereRadius,
                        lsCenter.z - sphereRadius - csmFar);
    glm::vec3 maxBounds(lsCenter.x + sphereRadius, lsCenter.y + sphereRadius,
                        lsCenter.z + sphereRadius + csmFar);

    // glm::ortho takes near and far as distances in front of the light
    // (along -Z), not as light-space z. Passing the z values straight in
    // mirrored the depth window to behind the light, so cascades 0 to 2 never
    // held any geometry and cascade 3 only did when it was big enough to
    // reach back across (a wide FOV, a high camera).
    glm::mat4 lightProj =
        glm::ortho(minBounds.x, maxBounds.x, minBounds.y, maxBounds.y,
                   -maxBounds.z, -minBounds.z);
    // Vulkan's Y flip. This ortho is off-centre, so the y offset has to flip
    // with the scale. Flipping only [1][1] shifted every cascade by twice the
    // camera's light-space height, which only stayed hidden while the camera
    // was near the light-space origin.
    lightProj[1][1] *= -1.0f;
    lightProj[3][1] *= -1.0f;
    out.lightSpace[i] = lightProj * lightView;

    prevSplit = splits[i];
  }
  return out;
}

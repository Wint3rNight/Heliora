#include "Thumbnails.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <format>

namespace {
constexpr float kFov = 0.5235988f; // 30 degrees
// Bump when the way pictures are drawn changes, so old cached ones are
// left behind instead of reused.
constexpr int kVersion = 2;
} // namespace

glm::mat4 thumbnailViewProj(glm::vec3 boundsMin, glm::vec3 boundsMax) {
  // Around the box's bounding sphere, so any box fits: back far enough that
  // the sphere fills about 85% of the picture's height.
  const glm::vec3 centre = 0.5f * (boundsMin + boundsMax);
  const float radius = std::max(0.5f * glm::length(boundsMax - boundsMin), 0.01f);
  const float distance = radius / std::sin(0.5f * kFov) * 1.15f;
  const glm::vec3 from = glm::normalize(glm::vec3(1.0f, 0.75f, 1.4f));
  const glm::mat4 view =
      glm::lookAt(centre + from * distance, centre, glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 proj = glm::perspectiveRH_ZO(kFov, 1.0f, std::max(distance - 2.0f * radius, 0.001f),
                                         distance + 2.0f * radius);
  proj[1][1] *= -1.0f; // Vulkan's y points down
  return proj * view;
}

glm::vec3 thumbnailLightDirection() {
  return glm::normalize(glm::vec3(0.5f, 0.9f, 0.7f));
}

std::string thumbnailKey(std::string_view source, uint64_t size, int64_t modified) {
  // FNV-1a over everything that changes the picture.
  uint64_t hash = 14695981039346656037ull;
  auto mix = [&](std::string_view bytes) {
    for (const char c : bytes) {
      hash ^= static_cast<unsigned char>(c);
      hash *= 1099511628211ull;
    }
  };
  mix(source);
  mix(std::format("|{}|{}|v{}", size, modified, kVersion));
  return std::format("{:016x}", hash);
}

glm::vec4 thumbnailAtlasRect(int index) {
  const float cell = 1.0f / kThumbnailsPerRow;
  const int col = index % kThumbnailsPerRow;
  const int row = index / kThumbnailsPerRow;
  return glm::vec4(col * cell, row * cell, (col + 1) * cell, (row + 1) * cell);
}

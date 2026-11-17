#include "Spawning.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <system_error>

std::vector<std::string> findModelFiles(const std::string &folder) {
  return findFiles(folder, {".gltf", ".glb"});
}

std::vector<std::string> findFiles(const std::string &folder,
                                   std::initializer_list<std::string_view> extensions) {
  namespace fs = std::filesystem;
  // The paths keep the folder's name as given, so they read
  // "Resources/Models/..." like the scene files, wherever it was found.
  fs::path searched(folder);
  if (!fs::is_directory(searched))
    searched = fs::path("..") / folder;
  std::vector<std::string> files;
  std::error_code error;
  if (!fs::is_directory(searched, error))
    return files;
  // The error_code forms throughout: a folder it can't read is skipped, and
  // any other error ends the scan with what it found, instead of throwing
  // out of the frame.
  fs::recursive_directory_iterator it(
      searched, fs::directory_options::skip_permission_denied, error);
  for (; !error && it != fs::recursive_directory_iterator();
       it.increment(error)) {
    const fs::path &path = it->path();
    std::error_code typeError;
    if (it->is_regular_file(typeError) &&
        std::ranges::find(extensions, path.extension().string()) != extensions.end())
      files.push_back(
          (fs::path(folder) / fs::relative(path, searched, typeError))
              .string());
  }
  if (error)
    spdlog::warn("File scan of {} stopped early: {}", searched.string(),
                 error.message());
  std::sort(files.begin(), files.end());
  return files;
}

SceneModelEntry newShapeEntry(PrimitiveShape shape) {
  SceneModelEntry entry;
  entry.shape = shape;
  entry.body = shape == PrimitiveShape::Plane ? PhysicsMotion::Static
                                              : PhysicsMotion::Dynamic;
  return entry;
}

SceneModelEntry newModelFileEntry(const std::string &path) {
  SceneModelEntry entry;
  entry.path = path;
  entry.body = PhysicsMotion::Dynamic;
  return entry;
}

glm::vec3 spawnPosition(const PickRay &ray,
                        const std::optional<PhysicsRayHit> &hit,
                        glm::vec3 boundsMin, glm::vec3 boundsMax) {
  if (!hit)
    return ray.origin + ray.direction * kSpawnFallback;
  // How far the box reaches from its origin into the surface: the furthest
  // of its corners along -normal. Per axis that's whichever of min and max
  // goes further that way, so on a floor it's -min.y, the height of the
  // origin above the bottom.
  const glm::vec3 into = -hit->normal;
  float reach = 0.0f;
  for (int i = 0; i < 3; ++i)
    reach += std::max(into[i] * boundsMin[i], into[i] * boundsMax[i]);
  return ray.origin + ray.direction * hit->distance + hit->normal * reach;
}

glm::vec3 duplicateOffset(glm::vec3 lo, glm::vec3 hi, glm::vec3 cameraRight) {
  const glm::vec3 size = hi - lo;
  if (std::abs(cameraRight.x) >= std::abs(cameraRight.z))
    return glm::vec3(std::copysign(size.x, cameraRight.x), 0.0f, 0.0f);
  return glm::vec3(0.0f, 0.0f, std::copysign(size.z, cameraRight.z));
}

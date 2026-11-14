#include "Spawning.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

std::vector<std::string> findModelFiles(const std::string &folder) {
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
  for (const auto &entry :
       fs::recursive_directory_iterator(searched, error)) {
    const fs::path &path = entry.path();
    if (entry.is_regular_file() &&
        (path.extension() == ".gltf" || path.extension() == ".glb"))
      files.push_back(
          (fs::path(folder) / fs::relative(path, searched)).string());
  }
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

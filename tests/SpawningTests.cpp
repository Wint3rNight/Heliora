#include "Spawning.h"
#include "TestRunner.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

const glm::vec3 kBoxMin(-0.5f); // a 1 m box around its origin
const glm::vec3 kBoxMax(0.5f);

PhysicsRayHit surface(float distance, glm::vec3 normal) {
  PhysicsRayHit hit{};
  hit.distance = distance;
  hit.normal = normal;
  return hit;
}

} // namespace

TEST(spawn_lands_on_the_surface_it_looks_at) {
  // Looking down at 45 degrees from 4 m up: the floor is hit 4 * sqrt(2) m
  // along the ray, 4 m ahead. A 1 m box's origin ends up half a metre above
  // the floor, and a model whose origin is at its bottom sits right on it.
  const PickRay ray{glm::vec3(1.0f, 4.0f, 2.0f),
                    glm::normalize(glm::vec3(0.0f, -1.0f, -1.0f))};
  const PhysicsRayHit floor =
      surface(4.0f * std::sqrt(2.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  CHECK_NEAR(glm::distance(spawnPosition(ray, floor, kBoxMin, kBoxMax),
                           glm::vec3(1.0f, 0.5f, -2.0f)),
             0.0, 1e-4);
  CHECK_NEAR(glm::distance(spawnPosition(ray, floor, glm::vec3(-1, 0, -1),
                                         glm::vec3(1, 3, 1)),
                           glm::vec3(1.0f, 0.0f, -2.0f)),
             0.0, 1e-4);
}

TEST(spawn_against_a_wall_stands_off_it) {
  // Looking straight at a wall 3 m ahead that faces back at the camera: the
  // box's back face touches the wall instead of sitting half inside it.
  const PickRay ray{glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
  const glm::vec3 at = spawnPosition(
      ray, surface(3.0f, glm::vec3(0.0f, 0.0f, 1.0f)), kBoxMin, kBoxMax);
  CHECK_NEAR(glm::distance(at, glm::vec3(0.0f, 1.0f, -2.5f)), 0.0, 1e-5);
  // Looking along +x at a wall facing -x, with a box longer on that side:
  // its far end (max.x = 2) touches the wall.
  const PickRay east{glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)};
  const glm::vec3 off = spawnPosition(
      east, surface(6.0f, glm::vec3(-1.0f, 0.0f, 0.0f)),
      glm::vec3(-0.5f), glm::vec3(2.0f, 0.5f, 0.5f));
  CHECK_NEAR(glm::distance(off, glm::vec3(4.0f, 1.0f, 0.0f)), 0.0, 1e-5);
}

TEST(spawn_with_nothing_in_reach_goes_5_m_ahead) {
  const PickRay ray{glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
  const glm::vec3 at = spawnPosition(ray, std::nullopt, kBoxMin, kBoxMax);
  CHECK_NEAR(glm::distance(at, glm::vec3(0.0f, 2.0f, -kSpawnFallback)), 0.0,
             1e-5);
}

TEST(new_objects_get_a_body_that_suits_them) {
  for (PrimitiveShape shape : {PrimitiveShape::Box, PrimitiveShape::Sphere,
                               PrimitiveShape::Capsule, PrimitiveShape::Plane}) {
    const SceneModelEntry entry = newShapeEntry(shape);
    CHECK(entry.shape == shape);
    CHECK(entry.body == (shape == PrimitiveShape::Plane
                             ? PhysicsMotion::Static
                             : PhysicsMotion::Dynamic));
    // Whatever is spawned can be saved and loaded again.
    CHECK(!problemWith(entry));
  }
  const SceneModelEntry rock =
      newModelFileEntry("Resources/Models/boulder_01/boulder_01_1k.gltf");
  CHECK(rock.path == "Resources/Models/boulder_01/boulder_01_1k.gltf");
  CHECK(rock.body == PhysicsMotion::Dynamic);
  CHECK(!problemWith(rock));
}

TEST(model_files_are_found_in_subfolders) {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "heliora_spawn_test";
  fs::remove_all(root);
  fs::create_directories(root / "rock" / "textures");
  for (const char *file : {"crate.glb", "rock/rock.gltf", "rock/rock.bin",
                           "rock/textures/rock.png", "notes.txt"})
    std::ofstream(root / file) << "x";
  const std::vector<std::string> found = findModelFiles(root.string());
  CHECK(found.size() == 2);
  if (found.size() == 2) {
    CHECK(found[0] == (root / "crate.glb").string());
    CHECK(found[1] == (root / "rock" / "rock.gltf").string());
  }
  CHECK(findModelFiles((root / "missing").string()).empty());
  fs::remove_all(root);
}

TEST(unreadable_folders_are_skipped) {
  // A folder the scan can't read is left out, not a crash.
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "heliora_spawn_locked";
  if (fs::exists(root / "locked"))
    fs::permissions(root / "locked", fs::perms::owner_all);
  fs::remove_all(root);
  fs::create_directories(root / "locked");
  std::ofstream(root / "crate.glb") << "x";
  std::ofstream(root / "locked" / "hidden.glb") << "x";
  fs::permissions(root / "locked", fs::perms::none);
  std::vector<std::string> found;
  bool threw = false;
  try {
    found = findModelFiles(root.string());
  } catch (const std::exception &) {
    threw = true;
  }
  fs::permissions(root / "locked", fs::perms::owner_all);
  fs::remove_all(root);
  CHECK(!threw);
  CHECK(found.size() == 1);
}

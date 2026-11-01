#include "Scene.h"
#include "TestRunner.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

SceneDescription parseText(const std::string &text) {
  std::istringstream in(text);
  return SceneIO::parse(in, "test.scene");
}

// The parse error message, or "" if the text parsed.
std::string errorOf(const std::string &text) {
  try {
    parseText(text);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

} // namespace

TEST(scene_parses_a_generated_box) {
  const SceneDescription scene = parseText("[model]\n"
                                           "shape = box\n"
                                           "size = 2 1 3\n"
                                           "color = 0.5 0.6 0.7\n"
                                           "position = 0 1 0\n");
  CHECK(scene.models.size() == 1);
  const SceneModelEntry &box = scene.models.front();
  CHECK(box.shape == PrimitiveShape::Box);
  CHECK(box.path.empty());
  CHECK(box.size == glm::vec3(2.0f, 1.0f, 3.0f));
  CHECK(box.color == glm::vec3(0.5f, 0.6f, 0.7f));
  CHECK(box.transform.position == glm::vec3(0.0f, 1.0f, 0.0f));
}

TEST(scene_parses_sphere_capsule_and_plane) {
  const SceneDescription scene = parseText("[model]\nshape = sphere\nradius = 0.3\n"
                                           "[model]\nshape = capsule\nradius = 0.4\nheight = 1.8\n"
                                           "[model]\nshape = plane\nsize = 40 1 40\n");
  CHECK(scene.models.size() == 3);
  CHECK(scene.models[0].shape == PrimitiveShape::Sphere);
  CHECK_NEAR(scene.models[0].radius, 0.3, 1e-6);
  CHECK(scene.models[1].shape == PrimitiveShape::Capsule);
  CHECK_NEAR(scene.models[1].radius, 0.4, 1e-6);
  CHECK_NEAR(scene.models[1].height, 1.8, 1e-6);
  CHECK(scene.models[2].shape == PrimitiveShape::Plane);
}

TEST(scene_model_needs_a_path_or_a_shape) {
  CHECK(errorOf("[model]\nposition = 0 0 0\n").find("path") != std::string::npos);
}

TEST(scene_model_cannot_have_both_path_and_shape) {
  CHECK(errorOf("[model]\npath = a.gltf\nshape = box\n").find("both") !=
        std::string::npos);
}

TEST(scene_unknown_shape_names_its_line) {
  const std::string error = errorOf("[model]\nshape = cube\n");
  CHECK(error.find("line 2") != std::string::npos);
  CHECK(error.find("cube") != std::string::npos);
}

TEST(scene_rejects_impossible_sizes) {
  auto mentions = [](const std::string &error, const char *word) {
    return error.find(word) != std::string::npos;
  };
  CHECK(mentions(errorOf("[model]\nshape = box\nsize = 1 0 1\n"), "size"));
  CHECK(mentions(errorOf("[model]\nshape = sphere\nradius = -1\n"), "radius"));
  // A capsule has to be taller than its two round ends.
  CHECK(mentions(
      errorOf("[model]\nshape = capsule\nradius = 1\nheight = 1.5\n"),
      "height"));
}

TEST(scene_shipped_files_still_parse) {
  // CTest runs from the build folder, so the scenes are one level up.
  std::filesystem::path dir = "Resources/Scenes";
  if (!std::filesystem::is_directory(dir))
    dir = "../Resources/Scenes";
  CHECK(std::filesystem::is_directory(dir));
  if (!std::filesystem::is_directory(dir))
    return;

  int files = 0;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() != ".scene")
      continue;
    ++files;
    try {
      SceneIO::loadFromFile(entry.path().string());
    } catch (const std::exception &e) {
      std::printf("    %s\n", e.what());
      CHECK(false);
    }
  }
  CHECK(files >= 3); // the sandbox and the two stress scenes
}

TEST(scene_parses_a_body) {
  const SceneDescription scene =
      parseText("[model]\nshape = box\nbody = dynamic\n"
                "[model]\nshape = plane\nsize = 10 1 10\nbody = static\n");
  CHECK(scene.models[0].body == PhysicsMotion::Dynamic);
  CHECK(scene.models[1].body == PhysicsMotion::Static);
}

TEST(scene_unknown_body_names_its_line) {
  const std::string error = errorOf("[model]\nshape = box\nbody = floaty\n");
  CHECK(error.find("line 3") != std::string::npos);
  CHECK(error.find("floaty") != std::string::npos);
}

TEST(scene_body_needs_a_shape) {
  // No collider for model files yet (mesh colliders aren't in step 1.1).
  CHECK(errorOf("[model]\npath = a.gltf\nbody = static\n").find("shape") !=
        std::string::npos);
}

TEST(scene_plane_body_must_be_static) {
  CHECK(errorOf("[model]\nshape = plane\nbody = dynamic\n").find("static") !=
        std::string::npos);
}

TEST(scene_body_cannot_be_animated) {
  // Physics and spin/orbit/bob would both write the node's transform.
  CHECK(errorOf("[model]\nshape = box\nbody = dynamic\nspin = 0 45 0\n")
            .find("anim") != std::string::npos);
}

TEST(scene_parses_mass_friction_and_restitution) {
  const SceneDescription scene = parseText("[model]\nshape = sphere\nbody = dynamic\n"
                                           "mass = 5\nfriction = 0.6\nrestitution = 0.4\n");
  const SceneModelEntry &ball = scene.models.front();
  CHECK(ball.mass == 5.0f);
  CHECK(ball.friction == 0.6f);
  CHECK(ball.restitution == 0.4f);
}

TEST(scene_physics_keys_need_a_body) {
  CHECK(errorOf("[model]\nshape = box\nmass = 2\n").find("body") !=
        std::string::npos);
}

TEST(scene_rejects_impossible_physics_values) {
  auto mentions = [](const std::string &error, const char *word) {
    return error.find(word) != std::string::npos;
  };
  const std::string box = "[model]\nshape = box\nbody = dynamic\n";
  CHECK(mentions(errorOf(box + "mass = 0\n"), "'mass' must"));
  CHECK(mentions(errorOf(box + "friction = -1\n"), "'friction' must"));
  CHECK(mentions(errorOf(box + "restitution = 1.5\n"), "'restitution' must"));
}

TEST(scene_parses_a_grid) {
  const SceneDescription scene =
      parseText("[grid]\nshape = box\nbody = dynamic\ncolor = 0.5\n"
                "count = 3 2 4\nspacing = 1.5 1 2\nposition = 10 0.5 0\n");
  CHECK(scene.models.empty());
  CHECK(scene.grids.size() == 1);
  const SceneGridEntry &grid = scene.grids.front();
  CHECK(grid.count == glm::ivec3(3, 2, 4));
  CHECK(grid.spacing == glm::vec3(1.5f, 1.0f, 2.0f));
  CHECK(grid.model.shape == PrimitiveShape::Box);
  CHECK(grid.model.body == PhysicsMotion::Dynamic);
}

TEST(grid_copies_are_centred_on_x_and_z_and_stack_up) {
  SceneGridEntry grid;
  grid.model.shape = PrimitiveShape::Box;
  grid.model.transform.position = glm::vec3(10.0f, 0.5f, 0.0f);
  grid.count = glm::ivec3(3, 2, 4);
  grid.spacing = glm::vec3(1.5f, 1.0f, 2.0f);

  const std::vector<SceneModelEntry> copies = expandGrid(grid);
  CHECK(copies.size() == 24);
  glm::vec3 lo(1e9f), hi(-1e9f);
  for (const SceneModelEntry &copy : copies) {
    lo = glm::min(lo, copy.transform.position);
    hi = glm::max(hi, copy.transform.position);
  }
  // x: 3 copies 1.5 apart around 10; y: from 0.5 up; z: 4 copies 2 apart around 0.
  CHECK(lo == glm::vec3(8.5f, 0.5f, -3.0f));
  CHECK(hi == glm::vec3(11.5f, 1.5f, 3.0f));
}

TEST(scene_grid_rejects_bad_counts_and_needs_a_model) {
  CHECK(errorOf("[grid]\nshape = box\ncount = 0 1 1\n").find("count") !=
        std::string::npos);
  CHECK(errorOf("[grid]\ncount = 2 2 2\n").find("shape") != std::string::npos);
  CHECK(errorOf("[grid]\nshape = box\nspacing = 1 0 1\n").find("spacing") !=
        std::string::npos);
}

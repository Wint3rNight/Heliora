#include "Scene.h"
#include "TestRunner.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
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

TEST(model_file_can_have_a_body) {
  // Its collider is built from its own triangles (step 1.1b).
  CHECK(errorOf("[model]\npath = a.gltf\nbody = static\n").empty());
  CHECK(errorOf("[model]\npath = a.gltf\nbody = dynamic\n").empty());
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

// Writing scenes back out (editor saves). A written scene has to parse back
// to exactly what was written, keep the comments a person put above each
// section, and keep the sections in the order they had.

namespace {

SceneDescription reparse(const SceneDescription &scene) {
  std::istringstream in(SceneIO::write(scene));
  SceneDescription again = SceneIO::parse(in, "test.scene");
  again.filePath = scene.filePath;
  return again;
}

// A fresh empty directory for file tests, removed by the caller.
std::filesystem::path scratchDirectory(const std::string &name) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("heliora_tests_" + name);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream file(path);
  std::stringstream text;
  text << file.rdbuf();
  return text.str();
}

} // namespace

TEST(every_scene_file_survives_a_round_trip) {
  const std::vector<SceneDescription> scenes =
      SceneIO::discover("Resources/Scenes");
  CHECK(scenes.size() >= 3);
  for (const SceneDescription &scene : scenes) {
    const bool same = reparse(scene) == scene;
    if (!same)
      std::printf("    changed by a round trip: %s\n", scene.filePath.c_str());
    CHECK(same);
  }
}

TEST(comments_above_sections_survive) {
  const SceneDescription scene = parseText("# A test scene\n"
                                           "# with two header lines\n"
                                           "name = Test\n"
                                           "\n"
                                           "# The floor\n"
                                           "[model]\n"
                                           "shape = plane\n"
                                           "\n"
                                           "# A wall of bricks\n"
                                           "# six wide\n"
                                           "[grid]\n"
                                           "shape = box\n"
                                           "count = 6 1 1\n");
  const std::string written = SceneIO::write(scene);
  CHECK(written.starts_with("# A test scene\n# with two header lines\n"));
  CHECK(written.find("# The floor\n[model]\n") != std::string::npos);
  CHECK(written.find("# A wall of bricks\n# six wide\n[grid]\n") !=
        std::string::npos);
  CHECK(reparse(scene) == scene);
}

TEST(end_of_line_comments_are_dropped) {
  const SceneDescription scene = parseText("[model]\n"
                                           "shape = box\n"
                                           "body = dynamic\n"
                                           "mass = 400   # a heavy one\n");
  const std::string written = SceneIO::write(scene);
  CHECK(written.find("mass = 400\n") != std::string::npos);
  CHECK(written.find("heavy") == std::string::npos);
}

TEST(model_and_grid_order_is_kept) {
  const SceneDescription scene = parseText("[model]\nshape = box\n"
                                           "position = 1 0 0\n"
                                           "[grid]\nshape = box\n"
                                           "position = 2 0 0\n"
                                           "[model]\nshape = box\n"
                                           "position = 3 0 0\n");
  const std::string written = SceneIO::write(scene);
  const size_t first = written.find("position = 1 0 0");
  const size_t second = written.find("position = 2 0 0");
  const size_t third = written.find("position = 3 0 0");
  CHECK(first != std::string::npos && second != std::string::npos &&
        third != std::string::npos);
  CHECK(first < second && second < third);
  CHECK(written.find("[grid]") < third);
}

TEST(rings_and_animations_survive) {
  const SceneDescription scene =
      parseText("[ring]\npath = Resources/Models/rock.glb\ncount = 12\n"
                "radius = 7.5\nheight = 2\nscale = 0.5\n"
                "[model]\npath = Resources/Models/rock.glb\n"
                "spin = 0 45 0\norbit_speed = 30\norbit_radius = 2\n"
                "bob_amplitude = 0.25\nbob_frequency = 0.5\n");
  const std::string written = SceneIO::write(scene);
  CHECK(written.find("[ring]") != std::string::npos);
  CHECK(written.find("spin = 0 45 0") != std::string::npos);
  CHECK(reparse(scene) == scene);
}

TEST(default_values_are_not_written) {
  const SceneDescription scene = parseText("name = Plain\n"
                                           "[model]\nshape = box\n");
  CHECK(SceneIO::write(scene) == "name = Plain\n\n[model]\nshape = box\n");
}

TEST(saved_file_loads_back) {
  const std::filesystem::path dir = scratchDirectory("saved_file_loads_back");
  const std::filesystem::path path = dir / "saved.scene";
  SceneDescription scene = parseText("name = Saved\n[model]\nshape = sphere\n"
                                     "radius = 0.25\nbody = dynamic\n");
  SceneIO::saveToFile(scene, path.string());
  scene.filePath = path.string();
  CHECK(SceneIO::loadFromFile(path.string()) == scene);
  CHECK(!std::filesystem::exists(path.string() + ".tmp"));
  std::filesystem::remove_all(dir);
}

TEST(failed_save_keeps_the_old_file) {
  const std::filesystem::path dir =
      scratchDirectory("failed_save_keeps_the_old_file");
  const std::filesystem::path path = dir / "old.scene";
  {
    std::ofstream old(path);
    old << "name = Old\n";
  }
  // A directory where the temporary file would go makes the write fail.
  std::filesystem::create_directory(path.string() + ".tmp");
  bool threw = false;
  try {
    SceneIO::saveToFile(parseText("name = New\n"), path.string());
  } catch (const std::exception &) {
    threw = true;
  }
  CHECK(threw);
  CHECK(readFile(path) == "name = Old\n");
  std::filesystem::remove_all(dir);
}

TEST(unknown_material_is_an_error_with_its_line) {
  const std::string error = errorOf("[model]\nshape = box\nbody = dynamic\n"
                                    "material = cheese\n");
  CHECK(error.find("line 4") != std::string::npos);
  CHECK(error.find("cheese") != std::string::npos);
}

TEST(density_must_be_positive) {
  CHECK(errorOf("[model]\nshape = box\nbody = dynamic\ndensity = 0\n")
            .find("density") != std::string::npos);
  CHECK(errorOf("[model]\nshape = box\ndensity = 500\n")
            .find("need a 'body'") != std::string::npos);
  CHECK(errorOf("[model]\nshape = box\nmaterial = wood\n")
            .find("need a 'body'") != std::string::npos);
}

TEST(material_and_density_survive_a_round_trip) {
  const SceneDescription scene = parseText("[model]\nshape = box\n"
                                           "body = dynamic\nmaterial = wood\n"
                                           "density = 500\n");
  CHECK(scene.models.front().material == "wood");
  CHECK(scene.models.front().density == 500.0f);
  CHECK(SceneIO::write(scene).find("material = wood\n") != std::string::npos);
  CHECK(reparse(scene) == scene);
}

TEST(moving_mesh_collider_is_an_error) {
  CHECK(errorOf("[model]\npath = rock.glb\nbody = dynamic\ncollider = mesh\n")
            .find("mesh") != std::string::npos);
  CHECK(errorOf("[model]\npath = rock.glb\nbody = static\ncollider = mesh\n")
            .empty());
  const std::string unknown =
      errorOf("[model]\npath = rock.glb\nbody = static\ncollider = blob\n");
  CHECK(unknown.find("line 4") != std::string::npos);
  CHECK(unknown.find("blob") != std::string::npos);
  // Generated shapes collide as themselves.
  CHECK(errorOf("[model]\nshape = box\nbody = dynamic\ncollider = hull\n")
            .find("collider") != std::string::npos);
  CHECK(errorOf("[model]\npath = rock.glb\ncollider = hull\n")
            .find("need a 'body'") != std::string::npos);
}

TEST(collider_survives_a_round_trip) {
  const SceneDescription scene = parseText("[model]\npath = rock.glb\n"
                                           "body = dynamic\ncollider = box\n");
  CHECK(scene.models.front().collider == SceneCollider::Box);
  CHECK(SceneIO::write(scene).find("collider = box\n") != std::string::npos);
  CHECK(reparse(scene) == scene);
}

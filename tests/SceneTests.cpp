#include "Scene.h"
#include "TestRunner.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
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

TEST(missing_sun_and_fog_give_the_defaults) {
  const SceneDescription scene = parseText("[model]\nshape = box\n");
  const glm::vec3 down = glm::normalize(glm::vec3(-0.3f, -1.0f, 0.2f));
  CHECK_NEAR(glm::length(scene.sun.direction - down), 0.0, 1e-6);
  CHECK(scene.sun.color == glm::vec3(1.0f, 0.93f, 0.78f));
  CHECK_NEAR(scene.sun.intensity, 2.5, 1e-6);
  CHECK_NEAR(scene.fog.density, 0.0, 1e-6);
  CHECK_NEAR(scene.fog.falloff, 0.25, 1e-6);
  CHECK_NEAR(scene.fog.max, 1.0, 1e-6);
  const std::string text = SceneIO::write(scene);
  CHECK(text.find("[sun]") == std::string::npos);
  CHECK(text.find("[fog]") == std::string::npos);
}

TEST(sun_and_fog_survive_a_round_trip) {
  const SceneDescription scene = parseText("[sun]\n"
                                           "direction = 0.1 -0.9 0.3\n"
                                           "color = 0.9 0.8 0.7\n"
                                           "intensity = 3.25\n"
                                           "[fog]\n"
                                           "density = 0.015\n"
                                           "falloff = 0.5\n"
                                           "max = 0.8\n");
  CHECK(scene.sun.direction == glm::vec3(0.1f, -0.9f, 0.3f));
  CHECK(scene.sun.color == glm::vec3(0.9f, 0.8f, 0.7f));
  CHECK_NEAR(scene.sun.intensity, 3.25, 1e-6);
  CHECK_NEAR(scene.fog.density, 0.015, 1e-6);
  CHECK_NEAR(scene.fog.falloff, 0.5, 1e-6);
  CHECK_NEAR(scene.fog.max, 0.8, 1e-6);
  CHECK(reparse(scene) == scene);
}

TEST(only_changed_sun_keys_are_written) {
  const SceneDescription scene = parseText("[sun]\nintensity = 4\n");
  const std::string text = SceneIO::write(scene);
  CHECK(text.find("[sun]") != std::string::npos);
  CHECK(text.find("intensity = 4") != std::string::npos);
  CHECK(text.find("direction") == std::string::npos);
  CHECK(text.find("[fog]") == std::string::npos);
}

TEST(comments_above_sun_and_fog_survive) {
  const SceneDescription scene = parseText("name = x\n"
                                           "# Late evening\n"
                                           "[sun]\n"
                                           "# Keep this thin\n"
                                           "[fog]\n");
  CHECK(scene.sun.info.comment == "# Late evening");
  CHECK(scene.fog.info.comment == "# Keep this thin");
  CHECK(reparse(scene) == scene);
}

TEST(bad_sun_and_fog_values_are_errors) {
  CHECK(errorOf("[sun]\ndirection = 0 0 0\n").find("direction") !=
        std::string::npos);
  CHECK(errorOf("[sun]\nintensity = -1\n").find("intensity") !=
        std::string::npos);
  CHECK(errorOf("[sun]\ncolor = 1 -1 1\n").find("color") != std::string::npos);
  CHECK(errorOf("[fog]\ndensity = -0.1\n").find("density") !=
        std::string::npos);
  CHECK(errorOf("[fog]\nfalloff = -1\n").find("falloff") != std::string::npos);
  CHECK(errorOf("[fog]\nmax = 1.5\n").find("max") != std::string::npos);
  CHECK(errorOf("[sun]\nheight = 3\n").find("unknown sun key") !=
        std::string::npos);
  CHECK(errorOf("[fog]\ncolor = 1\n").find("unknown fog key") !=
        std::string::npos);
}

TEST(grid_layout_rotation_turns_the_arrangement) {
  SceneGridEntry grid;
  grid.model.shape = PrimitiveShape::Box;
  grid.model.transform.rotationDeg = glm::vec3(10.0f, 5.0f, 0.0f);
  grid.count = glm::ivec3(3, 1, 1);
  grid.spacing = glm::vec3(2.0f);
  grid.layoutRotation = 90.0f;
  const std::vector<SceneModelEntry> copies = expandGrid(grid);
  CHECK(copies.size() == 3);
  // A quarter turn about y takes the row along x onto z.
  CHECK_NEAR(glm::distance(copies[0].transform.position, glm::vec3(0, 0, 2)),
             0.0, 1e-5);
  CHECK_NEAR(glm::distance(copies[2].transform.position, glm::vec3(0, 0, -2)),
             0.0, 1e-5);
  // Each copy turns with the arrangement, on top of its own rotation.
  CHECK_NEAR(copies[0].transform.rotationDeg.y, 95.0, 1e-4);
  CHECK_NEAR(copies[0].transform.rotationDeg.x, 10.0, 1e-6);
}

TEST(degrees_from_rotation_undoes_rotation_from_degrees) {
  const glm::vec3 cases[] = {{0.0f, 0.0f, 0.0f},
                             {30.0f, 45.0f, -60.0f},
                             {-80.0f, 170.0f, 10.0f},
                             {10.0f, -120.0f, 135.0f}};
  for (const glm::vec3 &degrees : cases) {
    const glm::vec3 back = degreesFromRotation(rotationFromDegrees(degrees));
    CHECK_NEAR(back.x, degrees.x, 1e-2);
    CHECK_NEAR(back.y, degrees.y, 1e-2);
    CHECK_NEAR(back.z, degrees.z, 1e-2);
  }
  // Straight up and down, only the rotation itself has to come back.
  for (float pitch : {90.0f, -90.0f}) {
    const glm::quat steep = rotationFromDegrees(glm::vec3(pitch, 30.0f, 20.0f));
    const glm::quat back = rotationFromDegrees(degreesFromRotation(steep));
    CHECK_NEAR(std::abs(glm::dot(steep, back)), 1.0, 1e-5);
  }
}

TEST(added_model_goes_last_in_the_file) {
  SceneDescription scene = parseText("[model]\nshape = box\n"
                                     "[grid]\nshape = sphere\n"
                                     "[model]\nshape = capsule\n");
  SceneModelEntry added;
  added.shape = PrimitiveShape::Plane;
  added.info.order = -5; // wherever it came from, it goes last
  const SceneEntryRef ref = addModel(scene, added);
  CHECK(ref.kind == SceneEntryRef::Kind::Model);
  CHECK(ref.index == 2);
  CHECK(scene.models.size() == 3);
  CHECK(scene.models.back().shape == PrimitiveShape::Plane);
  CHECK(scene.models.back().info.order == 3);
  CHECK(reparse(scene) == scene);
}

TEST(scene_name_becomes_a_safe_file_name) {
  CHECK(sceneFileNameFor("level1") == "level1.scene");
  CHECK(sceneFileNameFor("My first level") == "My_first_level.scene");
  CHECK(sceneFileNameFor("  stress-test_2  ") == "stress-test_2.scene");
}

TEST(bad_scene_names_are_refused) {
  for (const char *name : {"", "   ", "..", "../sandbox", "a/b", "a\\b",
                           "level.v2", "what?", "na\xc3\xafve", "tab\there"})
    CHECK(!sceneFileNameFor(name));
}

TEST(layout_rotation_survives_a_round_trip) {
  const SceneDescription scene =
      parseText("[grid]\nshape = box\ncount = 2 1 2\nlayout_rotation = 30\n");
  CHECK_NEAR(scene.grids.front().layoutRotation, 30.0, 1e-6);
  CHECK(reparse(scene) == scene);
  CHECK(SceneIO::write(parseText("[grid]\nshape = box\n"))
            .find("layout_rotation") == std::string::npos);
}

TEST(problem_with_uses_the_parsers_rules) {
  SceneModelEntry plane;
  plane.shape = PrimitiveShape::Plane;
  plane.size = glm::vec3(10.0f, 1.0f, 10.0f);
  plane.body = PhysicsMotion::Dynamic;
  CHECK(problemWith(plane).value_or("").find("static") != std::string::npos);
  plane.body = PhysicsMotion::Static;
  CHECK(!problemWith(plane));

  SceneGridEntry grid;
  grid.model.shape = PrimitiveShape::Box;
  grid.spacing = glm::vec3(0.0f);
  CHECK(problemWith(grid).value_or("").find("spacing") != std::string::npos);
}

TEST(problem_with_sun_and_fog_uses_the_parsers_rules) {
  auto mentions = [](const std::optional<std::string> &problem,
                     const char *word) {
    return problem.value_or("").find(word) != std::string::npos;
  };
  SceneSun sun;
  CHECK(!problemWith(sun));
  sun.intensity = -1.0f;
  CHECK(mentions(problemWith(sun), "intensity"));
  sun = SceneSun{};
  sun.color.g = -0.1f;
  CHECK(mentions(problemWith(sun), "color"));
  sun = SceneSun{};
  sun.direction = glm::vec3(0.0f);
  CHECK(mentions(problemWith(sun), "direction"));

  SceneFog fog;
  CHECK(!problemWith(fog));
  fog.max = 1.5f;
  CHECK(mentions(problemWith(fog), "max"));
  fog = SceneFog{};
  fog.density = -0.01f;
  CHECK(mentions(problemWith(fog), "density"));
  fog = SceneFog{};
  fog.falloff = -1.0f;
  CHECK(mentions(problemWith(fog), "falloff"));

  // The parser refuses the same thing.
  bool threw = false;
  try {
    parseText("[fog]\nmax = 1.5\n");
  } catch (const std::runtime_error &) {
    threw = true;
  }
  CHECK(threw);
}

TEST(problem_with_refuses_numbers_that_arent_finite) {
  // A file can't hold inf or nan (the parser can't read them back), so an
  // edit mustn't put one in.
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  SceneModelEntry box;
  box.shape = PrimitiveShape::Box;
  CHECK(!problemWith(box));
  SceneModelEntry far = box;
  far.transform.position.x = inf;
  CHECK(problemWith(far));
  SceneModelEntry turned = box;
  turned.transform.rotationDeg.z = nan;
  CHECK(problemWith(turned));
  SceneModelEntry huge = box;
  huge.transform.scale = inf;
  CHECK(problemWith(huge));

  SceneSun sun;
  sun.intensity = inf;
  CHECK(problemWith(sun));
  SceneFog fog;
  fog.density = nan;
  CHECK(problemWith(fog));
}

TEST(removing_an_entry_keeps_the_rest_in_order) {
  SceneDescription scene = parseText("[model]\nshape = box\nposition = 1 0 0\n"
                                     "[grid]\nshape = sphere\n"
                                     "[model]\nshape = box\nposition = 3 0 0\n");
  removeEntry(scene, {SceneEntryRef::Kind::Model, 0});
  CHECK(scene.models.size() == 1);
  CHECK(scene.models.front().transform.position.x == 3.0f);
  const std::string text = SceneIO::write(scene);
  CHECK(text.find("[grid]") < text.find("[model]"));
  CHECK(reparse(scene) == scene);
}

TEST(split_grid_keeps_its_place_and_its_copies) {
  SceneDescription scene = parseText("[model]\nshape = box\n"
                                     "# The wall\n"
                                     "[grid]\nshape = box\ncount = 3 1 1\n"
                                     "spacing = 2\nbody = dynamic\n"
                                     "[model]\nshape = sphere\n");
  const std::vector<SceneEntryRef> copies = splitGrid(scene, 0);
  CHECK(copies.size() == 3);
  CHECK(scene.grids.empty());
  CHECK(scene.models.size() == 5);
  if (copies.size() == 3) {
    const SceneModelEntry &first = scene.models[copies[0].index];
    CHECK(first.body == PhysicsMotion::Dynamic);
    CHECK(first.info.comment == "# The wall");
    CHECK(scene.models[copies[2].index].transform.position.x == 2.0f);
  }
  // The copies sit where the grid was: after the first box, before the
  // sphere.
  const std::string text = SceneIO::write(scene);
  CHECK(text.find("# The wall") < text.find("shape = sphere"));
  CHECK(reparse(scene) == scene);
}

TEST(nan_and_inf_in_a_scene_file_are_errors) {
  // std::stof reads both, and NaN then fails every range check.
  for (const char *value : {"nan", "inf", "-inf"}) {
    std::string message;
    try {
      parseText(std::string("[model]\nshape = sphere\nradius = ") + value +
                "\n");
    } catch (const std::runtime_error &e) {
      message = e.what();
    }
    CHECK(message.find("line 3") != std::string::npos);
  }
}

TEST(problem_with_refuses_bad_scale_and_numbers_that_arent_finite) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  SceneModelEntry box;
  box.shape = PrimitiveShape::Box;
  box.body = PhysicsMotion::Dynamic;
  CHECK(!problemWith(box));
  for (float scale : {0.0f, -1.0f}) {
    SceneModelEntry scaled = box;
    scaled.transform.scale = scale;
    CHECK(problemWith(scaled).value_or("").find("scale") != std::string::npos);
  }
  SceneModelEntry sized = box;
  sized.size.x = nan;
  CHECK(problemWith(sized));
  SceneModelEntry heavy = box;
  heavy.mass = nan;
  CHECK(problemWith(heavy));
  SceneModelEntry tinted = box;
  tinted.color.g = nan;
  CHECK(problemWith(tinted));
  SceneModelEntry spun = box;
  spun.body.reset();
  spun.animation.spinDegPerSec.y = std::numeric_limits<float>::infinity();
  CHECK(problemWith(spun));
  SceneModelEntry sphere;
  sphere.shape = PrimitiveShape::Sphere;
  sphere.radius = nan;
  CHECK(problemWith(sphere));
  SceneGridEntry grid;
  grid.model = box;
  grid.spacing.z = nan;
  CHECK(problemWith(grid));
}

TEST(paths_with_a_hash_are_refused) {
  // The parser reads # as the start of a comment, so the path wouldn't
  // come back.
  SceneModelEntry rock;
  rock.path = "Resources/Models/Rock #2.glb";
  CHECK(problemWith(rock).value_or("").find("#") != std::string::npos);
  rock.path = "Resources/Models/Rock 2.glb";
  CHECK(!problemWith(rock));
}

TEST(counts_out_of_range_are_errors) {
  auto refused = [](const std::string &text) {
    try {
      parseText(text);
    } catch (const std::runtime_error &) {
      return true;
    }
    return false;
  };
  CHECK(refused("[grid]\nshape = box\ncount = 20000 1 1\n"));
  CHECK(refused("[grid]\nshape = box\ncount = 1e20 1 1\n"));
  CHECK(refused("[grid]\nshape = box\ncount = 1000 1000 2\n")); // 2 million
  CHECK(!refused("[grid]\nshape = box\ncount = 25 8 25\n"));
  CHECK(refused("[ring]\npath = a.gltf\ncount = 1e10\n"));
  CHECK(refused("[ring]\npath = a.gltf\ncount = 2.5\n"));
  CHECK(refused("[ring]\npath = a.gltf\ncount = 0\n"));
  CHECK(!refused("[ring]\npath = a.gltf\ncount = 12\n"));
}

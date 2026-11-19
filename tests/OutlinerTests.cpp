#include "Outliner.h"
#include "TestRunner.h"

#include <algorithm>

namespace {

using Kind = SceneEntryRef::Kind;

// A player start, a commented breakable wooden box, a crate model, a grid
// of boxes and a character, in that file order.
SceneDescription yard() {
  SceneDescription s;
  s.player = ScenePlayerStart{glm::vec3(0.0f, 0.0f, 18.0f), 0.0f, "", {}};
  SceneModelEntry fence;
  fence.shape = PrimitiveShape::Box;
  fence.size = glm::vec3(5.0f, 1.4f, 0.12f);
  fence.body = PhysicsMotion::Static;
  fence.material = "wood";
  fence.breakPieces = 6;
  fence.transform.position = glm::vec3(7.0f, 0.7f, 10.0f);
  fence.info = {1, "# A wooden fence: lighter, it splinters\n"};
  SceneModelEntry crate;
  crate.path = "Resources/Models/crate/crate.gltf";
  crate.body = PhysicsMotion::Dynamic;
  crate.mass = 20.0f;
  crate.transform.position = glm::vec3(1.0f, 0.5f, 2.0f);
  crate.info = {2, ""};
  s.models = {fence, crate};
  SceneGridEntry bricks;
  bricks.model.shape = PrimitiveShape::Box;
  bricks.model.size = glm::vec3(1.0f, 0.5f, 0.5f);
  bricks.count = glm::ivec3(2, 1, 3);
  bricks.info = {3, ""};
  s.grids = {bricks};
  SceneCharacterEntry remy;
  remy.file = "Resources/Characters/remy.character";
  remy.position = glm::vec3(-2.0f, 0.0f, 4.0f);
  remy.info = {4, ""};
  s.characters = {remy};
  return s;
}

std::vector<std::string> names(const std::vector<OutlinerRow> &rows) {
  std::vector<std::string> out;
  for (const OutlinerRow &r : rows)
    out.push_back(r.name);
  return out;
}

} // namespace

TEST(the_outliner_lists_everything_in_the_scene_in_file_order) {
  const std::vector<OutlinerRow> rows = outlinerRows(yard());
  CHECK(rows.size() == 5);
  if (rows.size() != 5)
    return;
  CHECK(rows[0].ref == (SceneEntryRef{Kind::Player, 0}));
  CHECK(rows[0].name == "Player start");
  CHECK(rows[1].ref == (SceneEntryRef{Kind::Model, 0}));
  CHECK(rows[1].name == "A wooden fence");
  CHECK(rows[1].kind == "Box");
  CHECK(rows[1].material == "wood");
  CHECK(rows[1].body == "static");
  CHECK(rows[1].breakPieces == 6);
  CHECK(glm::distance(rows[1].position, glm::vec3(7.0f, 0.7f, 10.0f)) < 1e-6f);
  CHECK(rows[2].name == "crate");
  CHECK(rows[2].kind == "Model");
  CHECK(rows[2].body == "dynamic");
  CHECK(rows[2].mass.has_value() && *rows[2].mass == 20.0f);
  CHECK(rows[3].ref == (SceneEntryRef{Kind::Grid, 0}));
  CHECK(rows[3].name == "Grid 2 x 1 x 3 of Box");
  CHECK(rows[3].kind == "Grid");
  CHECK(rows[4].ref == (SceneEntryRef{Kind::Character, 0}));
  CHECK(rows[4].name == "remy");
  CHECK(rows[4].kind == "Character");
}

TEST(a_name_comes_from_the_comment_or_from_what_it_is) {
  CHECK(entryName("# The hut: four walls with a doorway\n", "Box") == "The hut");
  CHECK(entryName("# Crates\n# stacked three high\n", "Box") == "Crates");
  CHECK(entryName("# Spheres of every size. They roll.\n", "Box") == "Spheres of every size");
  CHECK(entryName("# a very long comment that goes on and on without ever stopping\n", "Box") ==
        "a very long comment that goes on and...");
  CHECK(entryName("", "Box 1 x 2 x 3") == "Box 1 x 2 x 3");
  CHECK(entryName("#\n#   \n", "Sphere") == "Sphere");
  SceneModelEntry box;
  box.shape = PrimitiveShape::Box;
  box.size = glm::vec3(6.0f, 3.0f, 0.3f);
  CHECK(describedName(box) == "Box 6 x 3 x 0.3");
  SceneModelEntry ball;
  ball.shape = PrimitiveShape::Sphere;
  ball.radius = 0.25f;
  CHECK(describedName(ball) == "Sphere r 0.25");
}

TEST(the_search_looks_at_name_kind_and_material_in_any_case) {
  const std::vector<OutlinerRow> rows = outlinerRows(yard());
  auto matching = [&](const char *text) {
    std::vector<OutlinerRow> kept;
    for (const OutlinerRow &r : rows)
      if (rowMatches(r, text))
        kept.push_back(r);
    return names(kept);
  };
  CHECK(matching("").size() == 5);
  CHECK(matching("FENCE") == std::vector<std::string>{"A wooden fence"});
  CHECK(matching("wood") == std::vector<std::string>{"A wooden fence"});
  CHECK(matching("character") == std::vector<std::string>{"remy"});
  CHECK(matching("nothing like it").empty());
}

TEST(the_filters_keep_the_kinds_ticked_and_breakables_only_if_asked) {
  const std::vector<OutlinerRow> rows = outlinerRows(yard());
  auto kept = [&](OutlinerFilter f) {
    std::vector<OutlinerRow> out;
    for (const OutlinerRow &r : rows)
      if (rowPasses(r, f))
        out.push_back(r);
    return names(out);
  };
  OutlinerFilter all;
  CHECK(kept(all).size() == 5);
  OutlinerFilter noShapes;
  noShapes.shapes = false;
  CHECK(kept(noShapes) ==
        (std::vector<std::string>{"Player start", "crate", "Grid 2 x 1 x 3 of Box", "remy"}));
  OutlinerFilter breakable;
  breakable.breakableOnly = true;
  CHECK(kept(breakable) == std::vector<std::string>{"A wooden fence"});
}

TEST(rows_sort_by_any_column_ties_in_file_order) {
  std::vector<OutlinerRow> rows = outlinerRows(yard());
  sortRows(rows, OutlinerColumn::Name, true);
  CHECK(names(rows) == (std::vector<std::string>{"A wooden fence", "crate",
                                                 "Grid 2 x 1 x 3 of Box", "Player start",
                                                 "remy"}));
  sortRows(rows, OutlinerColumn::Body, false);
  CHECK(rows[0].body == "static");
  sortRows(rows, OutlinerColumn::Breaks, false);
  CHECK(rows[0].breakPieces == 6);
  sortRows(rows, OutlinerColumn::Order, true);
  CHECK(rows[0].name == "Player start");
  CHECK(rows[4].name == "remy");
  // Equal bodies (none) keep their order.
  sortRows(rows, OutlinerColumn::Material, true);
  CHECK(rows.back().material == "wood");
}

TEST(a_shapes_mass_is_its_volume_times_its_density) {
  SceneModelEntry slab;
  slab.shape = PrimitiveShape::Box;
  slab.size = glm::vec3(1.0f, 2.0f, 0.5f);
  slab.body = PhysicsMotion::Static;
  slab.material = "stone";
  CHECK(entryMass(slab).has_value());
  CHECK_NEAR(*entryMass(slab), 2400.0, 1e-2);
  slab.density = 500.0f; // its own density beats its material's
  CHECK_NEAR(*entryMass(slab), 500.0, 1e-3);
  slab.mass = 12.0f; // and a mass beats both
  CHECK_NEAR(*entryMass(slab), 12.0, 1e-6);
  SceneModelEntry plain;
  plain.shape = PrimitiveShape::Sphere;
  plain.radius = 0.5f;
  plain.body = PhysicsMotion::Dynamic;
  CHECK_NEAR(*entryMass(plain), 1000.0 * 4.0 / 3.0 * 3.14159265 * 0.125, 0.1);
  SceneModelEntry file;
  file.path = "a.gltf";
  file.body = PhysicsMotion::Dynamic;
  CHECK(!entryMass(file)); // its volume needs its mesh
  SceneModelEntry noBody = slab;
  noBody.body.reset();
  CHECK(!entryMass(noBody));
}

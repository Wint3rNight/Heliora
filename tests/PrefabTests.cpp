#include "Prefab.h"
#include "TestRunner.h"

#include <algorithm>
#include <sstream>

namespace {

SceneDescription parseScene(const std::string &text) {
  std::istringstream in(text);
  return SceneIO::parse(in, "test.scene");
}

ObjectAsset parseObject(const std::string &text) {
  std::istringstream in(text);
  return ObjectIO::parse(in, "crate.object");
}

std::string objectError(const std::string &text) {
  try {
    parseObject(text);
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "";
}

bool has(const std::vector<std::string> &keys, const std::string &key) {
  return std::ranges::find(keys, key) != keys.end();
}

const char *kCrate = "# A wooden crate\nshape = box\nsize = 0.8\ncolor = 0.6 0.4 0.2\n"
                     "body = dynamic\nmaterial = wood\n";

} // namespace

TEST(an_object_file_is_what_the_object_is) {
  const ObjectAsset crate = parseObject(kCrate);
  CHECK(crate.comment == "# A wooden crate\n");
  CHECK(crate.model.shape == PrimitiveShape::Box);
  CHECK(crate.model.size == glm::vec3(0.8f));
  CHECK(crate.model.body == PhysicsMotion::Dynamic);
  CHECK(crate.model.material == std::optional<std::string>("wood"));
  std::istringstream again(ObjectIO::write(crate));
  CHECK(ObjectIO::parse(again, "again.object") == crate);
}

TEST(an_object_file_has_no_place_of_its_own) {
  // Where it stands belongs to each copy in a scene, not to the object.
  CHECK(objectError("shape = box\nposition = 1 2 3\n").find("line 2") != std::string::npos);
  CHECK(objectError("shape = box\nspin = 0 90 0\n").find("line 2") != std::string::npos);
  CHECK(objectError("size = 1\n").find("path") != std::string::npos); // nothing to draw
  CHECK(objectError("shape = box\nwobble = 3\n").find("wobble") != std::string::npos);
}

TEST(a_scene_places_copies_of_an_object) {
  // A copy names its object and keeps its own place and what it overrides.
  const SceneDescription scene =
      parseScene("[model]\nobject = Resources/Objects/crate.object\nposition = 1 0 2\n"
                 "color = 1 0 0\n");
  CHECK(scene.models.size() == 1);
  if (scene.models.size() != 1)
    return;
  const SceneModelEntry &copy = scene.models[0];
  CHECK(copy.object == "Resources/Objects/crate.object");
  CHECK(copy.transform.position == glm::vec3(1.0f, 0.0f, 2.0f));
  CHECK(has(copy.overrides, "color"));
  CHECK(!has(copy.overrides, "position")); // its place is always its own
  // Written back the same way: the object, the place, the override only.
  const std::string text = SceneIO::write(scene);
  CHECK(text.find("object = Resources/Objects/crate.object") != std::string::npos);
  CHECK(text.find("color = 1 0 0") != std::string::npos);
  CHECK(text.find("shape") == std::string::npos);
}

TEST(a_grid_cant_be_made_of_prefab_copies_yet) {
  try {
    parseScene("[grid]\nobject = Resources/Objects/crate.object\ncount = 2 1 1\n");
    CHECK(false);
  } catch (const std::runtime_error &e) {
    CHECK(std::string(e.what()).find("line 2") != std::string::npos);
  }
}

TEST(a_copy_takes_its_objects_values_except_its_overrides) {
  const ObjectAsset crate = parseObject(kCrate);
  SceneModelEntry copy = parseScene("[model]\nobject = Resources/Objects/crate.object\n"
                                    "position = 1 0 2\ncolor = 1 0 0\n")
                             .models[0];
  const SceneModelEntry resolved = resolvePrefab(copy, crate.model);
  CHECK(resolved.shape == PrimitiveShape::Box);
  CHECK(resolved.size == glm::vec3(0.8f));                    // the object's
  CHECK(resolved.color == glm::vec3(1.0f, 0.0f, 0.0f));       // the copy's own
  CHECK(resolved.transform.position == glm::vec3(1.0f, 0.0f, 2.0f));
  CHECK(resolved.object == copy.object);
  CHECK(!problemWith(resolved));
  // Saving a resolved copy still writes only the object, the place and the
  // override.
  SceneDescription scene;
  scene.models.push_back(resolved);
  const std::string text = SceneIO::write(scene);
  CHECK(text.find("size") == std::string::npos);
  CHECK(text.find("color = 1 0 0") != std::string::npos);
}

TEST(editing_a_copy_records_what_it_overrides) {
  const ObjectAsset crate = parseObject(kCrate);
  SceneModelEntry copy;
  copy.object = "Resources/Objects/crate.object";
  copy = resolvePrefab(copy, crate.model);
  SceneModelEntry edited = copy;
  edited.size = glm::vec3(2.0f);
  edited.transform.position = glm::vec3(5.0f); // not an override
  noteOverrides(copy, edited);
  CHECK(has(edited.overrides, "size"));
  CHECK(!has(edited.overrides, "position"));
  CHECK(edited.overrides.size() == 1);
}

TEST(a_copy_can_be_reverted_unpacked_or_made_from_an_object) {
  const ObjectAsset crate = parseObject(kCrate);
  SceneModelEntry copy;
  copy.object = "Resources/Objects/crate.object";
  copy.color = glm::vec3(1.0f, 0.0f, 0.0f);
  copy.overrides = {"color"};
  copy.transform.position = glm::vec3(3.0f);
  copy = resolvePrefab(copy, crate.model);
  // Revert: the object's values again, its place kept.
  const SceneModelEntry reverted = revertOverrides(copy, crate.model);
  CHECK(reverted.color == crate.model.color);
  CHECK(reverted.overrides.empty());
  CHECK(reverted.transform.position == glm::vec3(3.0f));
  // Unpack: an ordinary object with the same values.
  const SceneModelEntry unpacked = unpackPrefab(copy);
  CHECK(unpacked.object.empty() && unpacked.overrides.empty());
  CHECK(unpacked.color == glm::vec3(1.0f, 0.0f, 0.0f));
  // Make prefab: the object without its place; the copy linked to it.
  SceneModelEntry plain = unpacked;
  const ObjectAsset made = prefabFrom(plain, "Resources/Objects/red_crate.object");
  CHECK(made.model.transform == NodeTransform{});
  CHECK(made.model.object.empty());
  CHECK(made.model.color == glm::vec3(1.0f, 0.0f, 0.0f));
  CHECK(plain.object == "Resources/Objects/red_crate.object");
  CHECK(plain.overrides.empty());
}

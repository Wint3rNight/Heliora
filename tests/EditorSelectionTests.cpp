#include "EditorSelection.h"
#include "TestRunner.h"

namespace {

using Kind = SceneEntryRef::Kind;
SceneEntryRef model(int i) { return {Kind::Model, i}; }
SceneEntryRef grid(int i) { return {Kind::Grid, i}; }

} // namespace

TEST(a_click_selects_one_and_ctrl_adds_or_takes_away) {
  EditorSelection s;
  CHECK(s.empty());
  CHECK(!s.primary());
  s.select(model(2));
  s.toggle(model(5));
  s.toggle(grid(0));
  CHECK(s.size() == 3);
  CHECK(*s.primary() == grid(0)); // the last one picked
  CHECK(s.contains(model(5)));
  s.toggle(grid(0)); // off again: the one before is the primary
  CHECK(s.size() == 2);
  CHECK(*s.primary() == model(5));
  s.select(model(7)); // a plain click: just that one
  CHECK(s.size() == 1);
  CHECK(*s.primary() == model(7));
  s.clear();
  CHECK(s.empty());
}

TEST(shift_selects_everything_between_the_last_click_and_this_one) {
  const std::vector<SceneEntryRef> rows = {model(0), model(1), grid(0), model(2), model(3)};
  EditorSelection s;
  s.select(model(1));
  s.selectRange(rows, model(3));
  CHECK(s.size() == 4);
  CHECK(s.contains(grid(0)) && s.contains(model(2)) && !s.contains(model(0)));
  CHECK(*s.primary() == model(3));
  // Upwards from the same place the last click was.
  s.selectRange(rows, model(0));
  CHECK(s.size() == 2);
  CHECK(s.contains(model(0)) && s.contains(model(1)));
  // Nothing clicked yet: just that one.
  EditorSelection fresh;
  fresh.selectRange(rows, grid(0));
  CHECK(fresh.size() == 1);
}

TEST(the_selection_follows_entries_taken_out_or_put_in) {
  EditorSelection s;
  s.select(model(1));
  s.toggle(model(4));
  s.toggle(grid(2));
  s.removed(model(1)); // gone; model 4 is model 3 now
  CHECK(s.size() == 2);
  CHECK(s.contains(model(3)) && !s.contains(model(4)));
  CHECK(s.contains(grid(2)));
  s.shifted(Kind::Model, 2, 5); // five put in at 2
  CHECK(s.contains(model(8)));
  s.removed(grid(2));
  CHECK(*s.primary() == model(8));
}

TEST(several_entries_are_taken_out_highest_first) {
  const std::vector<SceneEntryRef> order =
      deletionOrder({model(1), grid(0), model(7), model(3), grid(2)});
  CHECK(order.size() == 5);
  // Each kind from the highest index down, so taking one out doesn't move
  // the ones still to go.
  int lastModel = 1000, lastGrid = 1000;
  for (const SceneEntryRef &r : order) {
    if (r.kind == Kind::Model) {
      CHECK(r.index < lastModel);
      lastModel = r.index;
    } else {
      CHECK(r.index < lastGrid);
      lastGrid = r.index;
    }
  }
}

TEST(a_change_to_one_object_is_made_on_another_field_by_field) {
  SceneModelEntry before;
  before.shape = PrimitiveShape::Box;
  before.material = "stone";
  before.transform.position = glm::vec3(1.0f, 2.0f, 3.0f);
  SceneModelEntry after = before;
  after.material = "wood";
  after.transform.position.x = 9.0f;
  SceneModelEntry other;
  other.path = "a.gltf";
  other.material = "ice";
  other.transform.position = glm::vec3(-5.0f, -6.0f, -7.0f);
  other.breakPieces = 3;
  copyChanges(before, after, other);
  CHECK(other.material == std::optional<std::string>("wood"));
  CHECK_NEAR(other.transform.position.x, 9.0, 1e-6);   // the part that changed
  CHECK_NEAR(other.transform.position.y, -6.0, 1e-6);  // the rest its own
  CHECK(other.path == "a.gltf");                       // what didn't change stays
  CHECK(!other.shape);
  CHECK(other.breakPieces == std::optional<int>(3));
  // Characters too.
  SceneCharacterEntry a, b, c;
  b = a;
  b.variant = "rage";
  c.position = glm::vec3(4.0f, 0.0f, 0.0f);
  copyChanges(a, b, c);
  CHECK(c.variant == "rage");
  CHECK_NEAR(c.position.x, 4.0, 1e-6);
}

TEST(a_new_character_file_brings_its_variant_along) {
  // The primary had none and still has none after its file changed; the
  // others mustn't keep a variant meant for the old file.
  SceneCharacterEntry before, after, other;
  before.file = after.file = other.file = "a.character";
  other.variant = "rage";
  after.file = "b.character";
  copyChanges(before, after, other);
  CHECK(other.file == "b.character");
  CHECK(other.variant.empty());
}

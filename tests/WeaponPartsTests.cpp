#include "WeaponParts.h"
#include "TestRunner.h"

#include <sstream>

namespace {

WeaponParts parsed(const std::string &text) {
  std::istringstream in(text);
  return weaponPartsFrom(SettingsFile::parse(in, "test.parts"));
}

const char *kParts = R"([weapon]
name = akm
length = 0.88

[part frame]
file = frame.gltf
pivot = 0 0 0

[part magazine]
file = magazine.gltf
pivot = 0 -0.06 0.095

[socket muzzle]
position = 0 0.033 0.535
x = 1 0 0
y = 0 1 0
z = 0 0 1

[socket ejection]
position = -0.023 0.053 0.098
x = 0 0 -1
y = 0 1 0
z = 1 0 0
)";

} // namespace

TEST(a_parts_file_gives_its_parts_in_order_and_its_sockets) {
  const WeaponParts w = parsed(kParts);
  CHECK(w.name == "akm");
  CHECK_NEAR(w.length, 0.88, 1e-6);
  CHECK(w.parts.size() == 2);
  if (w.parts.size() == 2) {
    CHECK(w.parts[0].name == "frame");
    CHECK(w.parts[1].name == "magazine");
    CHECK(w.parts[1].file == "magazine.gltf");
    CHECK(glm::distance(w.parts[1].pivot, glm::vec3(0.0f, -0.06f, 0.095f)) < 1e-6f);
  }
  CHECK(w.part("magazine") != nullptr);
  CHECK(w.part("bolt") == nullptr);
  const std::optional<WeaponSocket> muzzle = w.socket("muzzle");
  CHECK(muzzle.has_value());
  if (muzzle)
    CHECK(glm::distance(muzzle->position, glm::vec3(0.0f, 0.033f, 0.535f)) < 1e-6f);
  CHECK(!w.socket("scope"));
}

TEST(a_sockets_matrix_puts_its_axes_and_place_in_gun_space) {
  const WeaponParts w = parsed(kParts);
  const glm::mat4 m = socketMatrix(*w.socket("ejection"));
  // Its x points back along the gun, its z to the gun's left.
  CHECK(glm::distance(glm::vec3(m * glm::vec4(1, 0, 0, 0)), glm::vec3(0, 0, -1)) < 1e-6f);
  CHECK(glm::distance(glm::vec3(m * glm::vec4(0, 0, 0, 1)), glm::vec3(-0.023f, 0.053f, 0.098f)) <
        1e-6f);
}

TEST(a_parts_file_without_parts_is_empty) {
  const WeaponParts w = parsed("[weapon]\nname = x\n");
  CHECK(w.parts.empty());
  CHECK(!w.socket("muzzle"));
}

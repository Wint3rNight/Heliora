#include "AssetConvert.h"
#include "TestRunner.h"

#include <string>

TEST(a_weapon_is_moved_into_gun_space_barrel_ahead_and_up_up) {
  // A pistol pointing along +x with its trigger at (0.05, 0.03, 0), in
  // metres, made 2% smaller: its trigger goes to the origin, its muzzle
  // ahead along +z, its top up along +y.
  const glm::mat4 m = weaponTransform(glm::vec3(0.05f, 0.03f, 0.0f), glm::vec3(1, 0, 0),
                                      glm::vec3(0, 1, 0), 0.98f);
  auto at = [&](glm::vec3 p) { return glm::vec3(m * glm::vec4(p, 1.0f)); };
  CHECK(glm::distance(at({0.05f, 0.03f, 0.0f}), glm::vec3(0.0f)) < 1e-6f);
  CHECK(glm::distance(at({0.15f, 0.03f, 0.0f}), glm::vec3(0.0f, 0.0f, 0.098f)) < 1e-6f);
  CHECK(glm::distance(at({0.05f, 0.13f, 0.0f}), glm::vec3(0.0f, 0.098f, 0.0f)) < 1e-6f);
  // Its right (+z for a gun along +x with y up, forward cross up) is the
  // gun's right, -x, as on a character facing +z.
  CHECK(glm::distance(at({0.05f, 0.03f, 0.1f}), glm::vec3(-0.098f, 0.0f, 0.0f)) < 1e-6f);
}

TEST(axes_are_named_with_a_sign) {
  CHECK(parseAxis("+x") == glm::vec3(1, 0, 0));
  CHECK(parseAxis("-z") == glm::vec3(0, 0, -1));
  CHECK(parseAxis("y") == glm::vec3(0, 1, 0));
  CHECK(!parseAxis("w"));
  CHECK(!parseAxis(""));
}

TEST(a_packed_map_is_put_in_gltfs_order) {
  // Unreal's packing (roughness, metal, AO) into glTF's (AO, roughness,
  // metal); an image already in glTF's order stays.
  Image rma{1, 1, 4, {10, 20, 30, 255}};
  const Image orm = packOrm(rma, PackedOrder::RoughnessMetalAo);
  CHECK(orm.channels == 3);
  CHECK(orm.pixels == (std::vector<uint8_t>{30, 10, 20}));
  Image same{1, 1, 3, {1, 2, 3}};
  CHECK(packOrm(same, PackedOrder::AoRoughnessMetal).pixels == (std::vector<uint8_t>{1, 2, 3}));
}

TEST(a_normal_maps_green_is_flipped_for_the_other_convention) {
  Image n{2, 1, 3, {128, 0, 255, 10, 255, 200}};
  flipGreen(n);
  CHECK(n.pixels == (std::vector<uint8_t>{128, 255, 255, 10, 0, 200}));
}

TEST(halving_an_image_averages_each_square_of_four) {
  Image big{2, 2, 1, {0, 100, 200, 100}};
  const Image small = halve(big);
  CHECK(small.width == 1 && small.height == 1);
  CHECK(small.pixels == std::vector<uint8_t>{100});
  // An odd edge keeps its last row and column.
  Image odd{3, 1, 1, {10, 30, 90}};
  const Image half = halve(odd);
  CHECK(half.width == 2 && half.height == 1);
  CHECK(half.pixels == (std::vector<uint8_t>{20, 90}));
}

TEST(a_part_is_written_as_gltf_with_its_bounds_and_buffer) {
  GltfPart part;
  part.name = "magazine";
  part.positions = {{0, 0, 0}, {1, 0, 0}, {0, 2, -1}};
  part.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  part.uvs = {{0, 0}, {1, 0}, {0, 1}};
  part.indices = {0, 1, 2};
  const GltfFiles files = writeGltf(part, {"albedo.png", "normal.png", "orm.png"}, "magazine.bin");
  // Positions, normals and UVs (32 floats) and 3 indices.
  CHECK(files.bin.size() == (9 + 9 + 6) * 4 + 3 * 4);
  CHECK(files.json.find("\"uri\":\"magazine.bin\"") != std::string::npos);
  CHECK(files.json.find("\"byteLength\":" + std::to_string(files.bin.size())) != std::string::npos);
  CHECK(files.json.find("\"min\":[0,0,-1]") != std::string::npos);
  CHECK(files.json.find("\"max\":[1,2,0]") != std::string::npos);
  CHECK(files.json.find("\"metallicRoughnessTexture\"") != std::string::npos);
  CHECK(files.json.find("\"occlusionTexture\"") != std::string::npos);
  CHECK(files.json.find("\"uri\":\"orm.png\"") != std::string::npos);
  CHECK(files.json.find("\"name\":\"magazine\"") != std::string::npos);
}

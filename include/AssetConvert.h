#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// What tools/weapon_convert needs that can be tested without files or
// Assimp (step 1.6): moving a gun into gun space, the texture repairs an
// Unreal-made pack needs for glTF, and writing a part as a .gltf.

// Gun space: the origin at a chosen point (the trigger), +z along the
// barrel, +y up, +x to the gun's left (so its right is -x, as on a
// character facing +z), metres. The transform from a model's own space:
// p' = scale * R * (p - origin), R taking forward to +z and up to +y.
glm::mat4 weaponTransform(glm::vec3 origin, glm::vec3 forward, glm::vec3 up, float scale);

// "+x", "-y", "z" and so on.
std::optional<glm::vec3> parseAxis(std::string_view text);

// An 8-bit image, rows top to bottom, channels interleaved.
struct Image {
  int width = 0, height = 0, channels = 0;
  std::vector<uint8_t> pixels;
};

// How a packed map holds its three channels: Unreal's (roughness, metal,
// AO) or glTF's own (AO, roughness, metal).
enum class PackedOrder { RoughnessMetalAo, AoRoughnessMetal };
// glTF's order, three channels.
Image packOrm(const Image &packed, PackedOrder order);
// Roughness and metal as grey maps of their own (Substance's export; the
// first channel of each) into glTF's order, with no occlusion (white).
// Empty pixels if their sizes differ.
Image ormFromMaps(const Image &roughness, const Image &metal);
// DirectX-style normal maps (Unreal's) point green the other way from
// OpenGL's (glTF's).
void flipGreen(Image &normal);
// Half the size, each pixel the average of four (an odd last row or column
// is kept as it is).
Image halve(const Image &image);

// One part of a gun as a mesh, in gun space.
struct GltfPart {
  std::string name;
  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec2> uvs;
  std::vector<uint32_t> indices;
};
// Its textures, file names next to the .gltf (empty for none).
struct GltfMaterial {
  std::string albedo, normal, orm;
};
struct GltfFiles {
  std::string json;         // the .gltf
  std::vector<uint8_t> bin; // the buffer it names as binName
};
GltfFiles writeGltf(const GltfPart &part, const GltfMaterial &material, const std::string &binName);

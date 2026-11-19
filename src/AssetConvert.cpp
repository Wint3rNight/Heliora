#include "AssetConvert.h"

#include <algorithm>
#include <cstring>
#include <format>

glm::mat4 weaponTransform(glm::vec3 origin, glm::vec3 forward, glm::vec3 up, float scale) {
  const glm::vec3 f = glm::normalize(forward);
  // Up made square to the barrel, in case they weren't quite.
  const glm::vec3 u = glm::normalize(up - f * glm::dot(up, f));
  const glm::vec3 left = glm::cross(u, f);
  // Rows: what lands on x, y and z.
  glm::mat4 r(1.0f);
  for (int c = 0; c < 3; ++c) {
    r[c][0] = left[c];
    r[c][1] = u[c];
    r[c][2] = f[c];
  }
  glm::mat4 m = glm::mat4(scale) * r;
  m[3][3] = 1.0f;
  return m * glm::mat4(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0),
                       glm::vec4(-origin, 1.0f));
}

std::optional<glm::vec3> parseAxis(std::string_view text) {
  float sign = 1.0f;
  if (!text.empty() && (text.front() == '+' || text.front() == '-')) {
    sign = text.front() == '-' ? -1.0f : 1.0f;
    text.remove_prefix(1);
  }
  if (text == "x")
    return glm::vec3(sign, 0.0f, 0.0f);
  if (text == "y")
    return glm::vec3(0.0f, sign, 0.0f);
  if (text == "z")
    return glm::vec3(0.0f, 0.0f, sign);
  return std::nullopt;
}

Image packOrm(const Image &packed, PackedOrder order) {
  Image out{packed.width, packed.height, 3, {}};
  const size_t count = static_cast<size_t>(packed.width) * packed.height;
  out.pixels.resize(count * 3);
  const int c = std::max(packed.channels, 1);
  auto at = [&](size_t i, int channel) {
    return packed.pixels[i * c + std::min(channel, c - 1)];
  };
  for (size_t i = 0; i < count; ++i) {
    if (order == PackedOrder::RoughnessMetalAo) {
      out.pixels[i * 3 + 0] = at(i, 2); // AO
      out.pixels[i * 3 + 1] = at(i, 0); // roughness
      out.pixels[i * 3 + 2] = at(i, 1); // metal
    } else {
      for (int k = 0; k < 3; ++k)
        out.pixels[i * 3 + k] = at(i, k);
    }
  }
  return out;
}

void flipGreen(Image &normal) {
  if (normal.channels < 2)
    return;
  const size_t count = static_cast<size_t>(normal.width) * normal.height;
  for (size_t i = 0; i < count; ++i) {
    uint8_t &g = normal.pixels[i * normal.channels + 1];
    g = static_cast<uint8_t>(255 - g);
  }
}

Image halve(const Image &image) {
  Image out{std::max(1, (image.width + 1) / 2), std::max(1, (image.height + 1) / 2),
            image.channels, {}};
  out.pixels.resize(static_cast<size_t>(out.width) * out.height * out.channels);
  for (int y = 0; y < out.height; ++y)
    for (int x = 0; x < out.width; ++x)
      for (int c = 0; c < image.channels; ++c) {
        int sum = 0, n = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            const int sx = 2 * x + dx, sy = 2 * y + dy;
            if (sx >= image.width || sy >= image.height)
              continue;
            sum += image.pixels[(static_cast<size_t>(sy) * image.width + sx) * image.channels + c];
            ++n;
          }
        out.pixels[(static_cast<size_t>(y) * out.width + x) * out.channels + c] =
            static_cast<uint8_t>((sum + n / 2) / std::max(n, 1));
      }
  return out;
}

namespace {

// A float as glTF's JSON wants it: no trailing zeros, no exponent for
// ordinary sizes.
std::string number(float v) {
  if (v == 0.0f)
    return "0";
  std::string s = std::format("{}", v);
  return s;
}

void append(std::vector<uint8_t> &bin, const void *data, size_t bytes) {
  const auto *p = static_cast<const uint8_t *>(data);
  bin.insert(bin.end(), p, p + bytes);
}

} // namespace

GltfFiles writeGltf(const GltfPart &part, const GltfMaterial &material, const std::string &binName) {
  GltfFiles files;
  const size_t n = part.positions.size();
  glm::vec3 lo(0.0f), hi(0.0f);
  if (n > 0) {
    lo = hi = part.positions[0];
    for (const glm::vec3 &p : part.positions) {
      lo = glm::min(lo, p);
      hi = glm::max(hi, p);
    }
  }
  // Positions, normals, UVs, then indices, each four-byte aligned.
  const size_t positionBytes = n * 12, normalBytes = part.normals.size() * 12,
               uvBytes = part.uvs.size() * 8, indexBytes = part.indices.size() * 4;
  append(files.bin, part.positions.data(), positionBytes);
  append(files.bin, part.normals.data(), normalBytes);
  append(files.bin, part.uvs.data(), uvBytes);
  append(files.bin, part.indices.data(), indexBytes);

  std::string views, accessors, attributes;
  size_t offset = 0;
  int index = 0;
  auto view = [&](size_t bytes, int target) {
    views += std::format("{}{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{},\"target\":{}}}",
                         views.empty() ? "" : ",", offset, bytes, target);
    offset += bytes;
    return index++;
  };
  int accessorCount = 0;
  auto accessor = [&](int bufferView, int componentType, size_t count, const char *type,
                      const std::string &extra) {
    accessors += std::format(
        "{}{{\"bufferView\":{},\"componentType\":{},\"count\":{},\"type\":\"{}\"{}}}",
        accessors.empty() ? "" : ",", bufferView, componentType, count, type, extra);
    return accessorCount++;
  };
  const int pos = accessor(view(positionBytes, 34962), 5126, n, "VEC3",
                           std::format(",\"min\":[{},{},{}],\"max\":[{},{},{}]", number(lo.x),
                                       number(lo.y), number(lo.z), number(hi.x), number(hi.y),
                                       number(hi.z)));
  attributes = std::format("\"POSITION\":{}", pos);
  if (normalBytes > 0)
    attributes += std::format(",\"NORMAL\":{}",
                              accessor(view(normalBytes, 34962), 5126, part.normals.size(), "VEC3", ""));
  if (uvBytes > 0)
    attributes += std::format(",\"TEXCOORD_0\":{}",
                              accessor(view(uvBytes, 34962), 5126, part.uvs.size(), "VEC2", ""));
  const int indices = accessor(view(indexBytes, 34963), 5125, part.indices.size(), "SCALAR", "");

  // The material: base colour, normal, and the packed map as both glTF's
  // metal-roughness and its occlusion.
  std::string images, textures, pbr = "\"metallicFactor\":1,\"roughnessFactor\":1", extra;
  int image = 0;
  auto texture = [&](const std::string &file) {
    images += std::format("{}{{\"uri\":\"{}\"}}", images.empty() ? "" : ",", file);
    textures += std::format("{}{{\"source\":{}}}", textures.empty() ? "" : ",", image);
    return image++;
  };
  if (!material.albedo.empty())
    pbr += std::format(",\"baseColorTexture\":{{\"index\":{}}}", texture(material.albedo));
  if (!material.orm.empty()) {
    const int orm = texture(material.orm);
    pbr += std::format(",\"metallicRoughnessTexture\":{{\"index\":{}}}", orm);
    extra += std::format(",\"occlusionTexture\":{{\"index\":{}}}", orm);
  }
  if (!material.normal.empty())
    extra += std::format(",\"normalTexture\":{{\"index\":{}}}", texture(material.normal));

  files.json = std::format(
      "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"Heliora weapon_convert\"}},"
      "\"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],"
      "\"nodes\":[{{\"name\":\"{}\",\"mesh\":0}}],"
      "\"meshes\":[{{\"name\":\"{}\",\"primitives\":[{{\"attributes\":{{{}}},\"indices\":{},"
      "\"material\":0}}]}}],"
      "\"materials\":[{{\"name\":\"{}\",\"pbrMetallicRoughness\":{{{}}}{}}}],"
      "{}{}"
      "\"accessors\":[{}],\"bufferViews\":[{}],"
      "\"buffers\":[{{\"uri\":\"{}\",\"byteLength\":{}}}]}}\n",
      part.name, part.name, attributes, indices, part.name, pbr, extra,
      images.empty() ? "" : std::format("\"images\":[{}],", images),
      textures.empty() ? "" : std::format("\"textures\":[{}],", textures), accessors, views,
      binName, files.bin.size());
  return files;
}

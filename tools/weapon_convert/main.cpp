// weapon_convert (step 1.6): a gun model (FBX, glTF, anything Assimp reads)
// into the game's gun space, split into its moving parts, one .gltf each,
// plus a .parts file with each part's pivot and the sockets (muzzle,
// ejection port, magazine, sights). Built on its own:
//   cmake --build build --target weapon_convert
//
//   weapon_convert <model> <out folder> <name> [options]
//     --length M            scale so it's M metres long along the barrel
//     --length-of P         measure that length on part P alone (a rocket
//                           sticking out of its launcher left out)
//     --forward AXIS        the barrel's way in the model (+z)
//     --up AXIS             the model's up (+y)
//     --origin NODE         the node at gun space's origin (the trigger)
//     --part P=K1,K2,...    part P is made of the vertices of these bones
//                           (a skinned model) or meshes' nodes (otherwise)
//     --default P           the part for vertices nobody named (frame);
//                           "none" drops them
//     --drop K1,K2,...      leave these out
//     --move K=dx,dy,dz     move a node's mesh, in the model's units (parts
//                           laid out beside the gun for show)
//     --socket S=NODE       a socket at a node
//     --socket-at S=x,y,z   a socket at a point, in the model's units, facing
//                           the barrel's way (for models without socket nodes)
//     --socket-gun S=x,y,z  the same at a point in gun space (metres), for
//                           places found by measuring the converted parts
//     --albedo F --normal F [--flip-green] --packed F --packed-order rma|orm
//     --roughness F --metal F  grey maps of their own instead of --packed
//     --for P               the texture options after it are part P's own
//                           (a model with more than one texture set);
//                           before any --for they're every other part's
//     --max-texture N       halve textures until at most N pixels across
//
//   weapon_convert <model> --list
//     each mesh: its node, material, vertex count, and the bones carrying
//     its vertices (how many each), to choose the parts from
//
// Gun space: the origin at --origin, +z along the barrel, +y up, +x to the
// gun's left, metres (AssetConvert.h).

#include "AssetConvert.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

// One texture set.
struct Textures {
  std::string albedo, normal, packed, roughness, metal;
  bool flipGreen = false;
  PackedOrder packedOrder = PackedOrder::AoRoughnessMetal;
};

struct Options {
  std::string model, out, name;
  float length = 0.0f;
  std::string lengthOf; // the part measured, or every part
  glm::vec3 forward{0, 0, 1}, up{0, 1, 0};
  std::string origin;
  std::map<std::string, std::string> partOf; // bone or node -> part
  std::vector<std::string> partOrder;
  std::map<std::string, std::vector<std::string>> keysOf; // part -> its bones or nodes, as listed
  std::string defaultPart = "frame";
  std::set<std::string> dropped;
  std::map<std::string, glm::vec3> moved;
  std::vector<std::pair<std::string, std::string>> sockets; // socket, node
  std::vector<std::pair<std::string, glm::vec3>> socketsAt;  // socket, point
  std::vector<std::pair<std::string, glm::vec3>> socketsInGun; // socket, gun-space point
  int maxTexture = 0;
  // Texture sets: "" for every part without its own, else a part's.
  std::map<std::string, Textures> textures;
};

std::vector<std::string> split(const std::string &text, char by) {
  std::vector<std::string> parts;
  std::stringstream in(text);
  for (std::string part; std::getline(in, part, by);)
    if (!part.empty())
      parts.push_back(part);
  return parts;
}

glm::vec3 vec3Of(const std::string &text) {
  const std::vector<std::string> p = split(text, ',');
  if (p.size() != 3)
    throw std::runtime_error("expected x,y,z: " + text);
  return {std::stof(p[0]), std::stof(p[1]), std::stof(p[2])};
}

Options parse(int argc, char **argv) {
  if (argc < 4)
    throw std::runtime_error("usage: weapon_convert <model> <out folder> <name> [options]");
  Options o;
  o.model = argv[1];
  o.out = argv[2];
  o.name = argv[3];
  std::string forPart; // the texture set the texture options go to
  for (int i = 4; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(flag + " needs a value");
      return argv[++i];
    };
    auto pair = [&](const std::string &text) {
      const size_t eq = text.find('=');
      if (eq == std::string::npos)
        throw std::runtime_error(flag + " wants NAME=VALUE: " + text);
      return std::make_pair(text.substr(0, eq), text.substr(eq + 1));
    };
    if (flag == "--length") {
      o.length = std::stof(value());
    } else if (flag == "--length-of") {
      o.lengthOf = value();
    } else if (flag == "--forward" || flag == "--up") {
      const std::string text = value();
      const std::optional<glm::vec3> axis = parseAxis(text);
      if (!axis)
        throw std::runtime_error("not an axis: " + text);
      (flag == "--forward" ? o.forward : o.up) = *axis;
    } else if (flag == "--origin") {
      o.origin = value();
    } else if (flag == "--part") {
      const auto [part, keys] = pair(value());
      o.partOrder.push_back(part);
      for (const std::string &k : split(keys, ',')) {
        o.partOf[k] = part;
        o.keysOf[part].push_back(k);
      }
    } else if (flag == "--default") {
      o.defaultPart = value();
    } else if (flag == "--drop") {
      for (const std::string &k : split(value(), ','))
        o.dropped.insert(k);
    } else if (flag == "--move") {
      const auto [key, by] = pair(value());
      o.moved[key] = vec3Of(by);
    } else if (flag == "--socket") {
      o.sockets.push_back(pair(value()));
    } else if (flag == "--socket-at") {
      const auto [socket, at] = pair(value());
      o.socketsAt.push_back({socket, vec3Of(at)});
    } else if (flag == "--socket-gun") {
      const auto [socket, at] = pair(value());
      o.socketsInGun.push_back({socket, vec3Of(at)});
    } else if (flag == "--for") {
      forPart = value();
    } else if (flag == "--albedo") {
      o.textures[forPart].albedo = value();
    } else if (flag == "--normal") {
      o.textures[forPart].normal = value();
    } else if (flag == "--flip-green") {
      o.textures[forPart].flipGreen = true;
    } else if (flag == "--packed") {
      o.textures[forPart].packed = value();
    } else if (flag == "--roughness") {
      o.textures[forPart].roughness = value();
    } else if (flag == "--metal") {
      o.textures[forPart].metal = value();
    } else if (flag == "--packed-order") {
      const std::string order = value();
      if (order == "rma")
        o.textures[forPart].packedOrder = PackedOrder::RoughnessMetalAo;
      else if (order == "orm")
        o.textures[forPart].packedOrder = PackedOrder::AoRoughnessMetal;
      else
        throw std::runtime_error("--packed-order is rma or orm");
    } else if (flag == "--max-texture") {
      o.maxTexture = std::stoi(value());
    } else {
      throw std::runtime_error("unknown option " + flag);
    }
  }
  return o;
}

glm::mat4 toGlm(const aiMatrix4x4 &m) {
  // Assimp's rows are glm's columns.
  return glm::transpose(glm::make_mat4(&m.a1));
}

// Every node's transform in the model's space, by name.
void collectNodes(const aiNode *node, const glm::mat4 &parent,
                  std::map<std::string, glm::mat4> &world,
                  std::vector<std::pair<const aiNode *, glm::mat4>> &meshNodes) {
  const glm::mat4 here = parent * toGlm(node->mTransformation);
  world[node->mName.C_Str()] = here;
  if (node->mNumMeshes > 0)
    meshNodes.push_back({node, here});
  for (unsigned i = 0; i < node->mNumChildren; ++i)
    collectNodes(node->mChildren[i], here, world, meshNodes);
}

// One vertex as read, before it's sorted into a part.
struct Vertex {
  glm::vec3 position, normal;
  glm::vec2 uv;
  std::string key; // its bone or its mesh's node
};

std::optional<Image> loadImage(const std::string &path) {
  Image image;
  unsigned char *data = stbi_load(path.c_str(), &image.width, &image.height, &image.channels, 0);
  if (!data) {
    std::cerr << "can't read " << path << "\n";
    return std::nullopt;
  }
  image.pixels.assign(data, data + static_cast<size_t>(image.width) * image.height * image.channels);
  stbi_image_free(data);
  return image;
}

Image fitted(Image image, int most) {
  while (most > 0 && (image.width > most || image.height > most))
    image = halve(image);
  return image;
}

void writePng(const std::filesystem::path &path, const Image &image) {
  if (!stbi_write_png(path.string().c_str(), image.width, image.height, image.channels,
                      image.pixels.data(), image.width * image.channels))
    throw std::runtime_error("can't write " + path.string());
}

std::string vec(glm::vec3 v) { return std::format("{:.5f} {:.5f} {:.5f}", v.x, v.y, v.z); }

// weapon_convert <model> --list: what's in it, to choose the parts from.
int list(const std::string &model) {
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(model, aiProcess_Triangulate | aiProcess_LimitBoneWeights);
  if (!scene || !scene->mRootNode)
    throw std::runtime_error("can't read " + model + ": " + importer.GetErrorString());
  std::map<std::string, glm::mat4> world;
  std::vector<std::pair<const aiNode *, glm::mat4>> meshNodes;
  collectNodes(scene->mRootNode, glm::mat4(1.0f), world, meshNodes);
  for (const auto &[node, transform] : meshNodes)
    for (unsigned m = 0; m < node->mNumMeshes; ++m) {
      const aiMesh *mesh = scene->mMeshes[node->mMeshes[m]];
      const aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
      std::cout << std::format("mesh {} on node {}: material {}, {} vertices\n", mesh->mName.C_Str(),
                               node->mName.C_Str(), material->GetName().C_Str(), mesh->mNumVertices);
      // Each vertex counted for the bone with the most weight on it.
      std::vector<float> best(mesh->mNumVertices, 0.0f);
      std::vector<int> owner(mesh->mNumVertices, -1);
      for (unsigned b = 0; b < mesh->mNumBones; ++b)
        for (unsigned w = 0; w < mesh->mBones[b]->mNumWeights; ++w) {
          const aiVertexWeight &vw = mesh->mBones[b]->mWeights[w];
          if (vw.mWeight > best[vw.mVertexId]) {
            best[vw.mVertexId] = vw.mWeight;
            owner[vw.mVertexId] = static_cast<int>(b);
          }
        }
      std::map<std::string, int> counts;
      for (int b : owner)
        ++counts[b < 0 ? "(no bone)" : mesh->mBones[b]->mName.C_Str()];
      for (const auto &[bone, n] : counts)
        std::cout << std::format("  {}: {} vertices\n", bone, n);
    }
  return 0;
}

int run(const Options &o) {
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(
      o.model, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_LimitBoneWeights);
  if (!scene || !scene->mRootNode)
    throw std::runtime_error("can't read " + o.model + ": " + importer.GetErrorString());
  std::map<std::string, glm::mat4> world;
  std::vector<std::pair<const aiNode *, glm::mat4>> meshNodes;
  collectNodes(scene->mRootNode, glm::mat4(1.0f), world, meshNodes);
  // A bone where the skinned mesh was bound to it, not where the file's
  // pose has it (a first-person model's bones hold the gun up at the
  // camera while its mesh rests elsewhere): its mesh's node times the
  // inverse of its offset matrix.
  for (const auto &[node, transform] : meshNodes)
    for (unsigned m = 0; m < node->mNumMeshes; ++m) {
      const aiMesh *mesh = scene->mMeshes[node->mMeshes[m]];
      for (unsigned b = 0; b < mesh->mNumBones; ++b)
        world[mesh->mBones[b]->mName.C_Str()] =
            transform * glm::inverse(toGlm(mesh->mBones[b]->mOffsetMatrix));
    }

  // Every vertex in the model's space, with the bone (or node) it belongs to.
  std::vector<Vertex> vertices;
  std::vector<uint32_t> triangles;
  for (const auto &[node, transform] : meshNodes) {
    const glm::mat4 moved =
        o.moved.count(node->mName.C_Str())
            ? glm::translate(glm::mat4(1.0f), o.moved.at(node->mName.C_Str())) * transform
            : transform;
    const glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(moved));
    for (unsigned m = 0; m < node->mNumMeshes; ++m) {
      const aiMesh *mesh = scene->mMeshes[node->mMeshes[m]];
      // The bone with the most weight on each vertex.
      std::vector<float> best(mesh->mNumVertices, 0.0f);
      std::vector<std::string> keys(mesh->mNumVertices, node->mName.C_Str());
      for (unsigned b = 0; b < mesh->mNumBones; ++b) {
        const aiBone *bone = mesh->mBones[b];
        for (unsigned w = 0; w < bone->mNumWeights; ++w) {
          const aiVertexWeight &vw = bone->mWeights[w];
          if (vw.mWeight > best[vw.mVertexId]) {
            best[vw.mVertexId] = vw.mWeight;
            keys[vw.mVertexId] = bone->mName.C_Str();
          }
        }
      }
      const uint32_t first = static_cast<uint32_t>(vertices.size());
      for (unsigned v = 0; v < mesh->mNumVertices; ++v) {
        Vertex out;
        const aiVector3D &p = mesh->mVertices[v];
        out.position = glm::vec3(moved * glm::vec4(p.x, p.y, p.z, 1.0f));
        const aiVector3D n = mesh->HasNormals() ? mesh->mNormals[v] : aiVector3D(0, 1, 0);
        out.normal = glm::normalize(normalMatrix * glm::vec3(n.x, n.y, n.z));
        out.uv = mesh->HasTextureCoords(0)
                     ? glm::vec2(mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y)
                     : glm::vec2(0.0f);
        out.key = keys[v];
        vertices.push_back(out);
      }
      for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
        const aiFace &face = mesh->mFaces[f];
        if (face.mNumIndices != 3)
          continue;
        for (int k = 0; k < 3; ++k)
          triangles.push_back(first + face.mIndices[k]);
      }
    }
  }
  auto partOf = [&](const std::string &key) -> std::string {
    if (o.dropped.count(key))
      return "";
    if (const auto it = o.partOf.find(key); it != o.partOf.end())
      return it->second;
    return o.defaultPart == "none" ? "" : o.defaultPart;
  };

  // Scale and gun space, from what's kept.
  if (!o.origin.empty() && !world.count(o.origin))
    throw std::runtime_error("no node called " + o.origin);
  const glm::vec3 origin = o.origin.empty() ? glm::vec3(0.0f) : glm::vec3(world.at(o.origin)[3]);
  float lo = 1e30f, hi = -1e30f;
  for (const Vertex &v : vertices)
    if (!partOf(v.key).empty() && (o.lengthOf.empty() || partOf(v.key) == o.lengthOf)) {
      const float along = glm::dot(v.position, glm::normalize(o.forward));
      lo = std::min(lo, along);
      hi = std::max(hi, along);
    }
  if (hi <= lo)
    throw std::runtime_error("nothing left to convert");
  const float scale = o.length > 0.0f ? o.length / (hi - lo) : 1.0f;
  const glm::mat4 toGun = weaponTransform(origin, o.forward, o.up, scale);
  const glm::mat3 turn = glm::mat3(weaponTransform(glm::vec3(0.0f), o.forward, o.up, 1.0f));
  std::cout << std::format("{}: {:.4f} units long, scaled by {:.6f}\n", o.name, hi - lo, scale);

  // Each part's triangles: a triangle goes where two or more of its corners do.
  std::map<std::string, std::vector<uint32_t>> partTriangles;
  for (size_t t = 0; t + 2 < triangles.size(); t += 3) {
    std::map<std::string, int> votes;
    for (int k = 0; k < 3; ++k)
      ++votes[partOf(vertices[triangles[t + k]].key)];
    std::string part;
    int most = 0;
    for (const auto &[p, n] : votes)
      if (n > most) {
        most = n;
        part = p;
      }
    if (most < 2)
      part = partOf(vertices[triangles[t]].key);
    if (part.empty())
      continue;
    for (int k = 0; k < 3; ++k)
      partTriangles[part].push_back(triangles[t + k]);
  }

  std::filesystem::create_directories(o.out);
  const std::filesystem::path out(o.out);
  // The textures, repaired for glTF: each set written once, named for the
  // gun (every part's) or the part with its own.
  std::map<std::string, GltfMaterial> materials;
  for (const auto &[set, t] : o.textures) {
    GltfMaterial &material = materials[set];
    const std::string stem = set.empty() ? o.name : o.name + "_" + set;
    if (!t.albedo.empty())
      if (std::optional<Image> image = loadImage(t.albedo)) {
        material.albedo = stem + "_albedo.png";
        writePng(out / material.albedo, fitted(*image, o.maxTexture));
      }
    if (!t.normal.empty())
      if (std::optional<Image> image = loadImage(t.normal)) {
        if (t.flipGreen)
          flipGreen(*image);
        material.normal = stem + "_normal.png";
        writePng(out / material.normal, fitted(*image, o.maxTexture));
      }
    if (!t.packed.empty()) {
      if (std::optional<Image> image = loadImage(t.packed)) {
        material.orm = stem + "_orm.png";
        writePng(out / material.orm, fitted(packOrm(*image, t.packedOrder), o.maxTexture));
      }
    } else if (!t.roughness.empty() && !t.metal.empty()) {
      const std::optional<Image> rough = loadImage(t.roughness), metal = loadImage(t.metal);
      if (rough && metal) {
        const Image orm = ormFromMaps(*rough, *metal);
        if (orm.pixels.empty())
          throw std::runtime_error("the roughness and metal maps aren't the same size");
        material.orm = stem + "_orm.png";
        writePng(out / material.orm, fitted(orm, o.maxTexture));
      }
    }
  }
  auto materialOf = [&](const std::string &part) {
    if (const auto it = materials.find(part); it != materials.end())
      return it->second;
    const auto shared = materials.find("");
    return shared != materials.end() ? shared->second : GltfMaterial{};
  };

  // The parts, each about its pivot: the first bone or node named for it.
  std::ostringstream parts;
  parts << std::format("# {} in gun space, made by tools/weapon_convert from {}.\n"
                       "# Gun space: the origin at {}, +z along the barrel, +y up, +x to the\n"
                       "# gun's left, metres. Each part's mesh is about its pivot.\n",
                       o.name, std::filesystem::path(o.model).filename().string(),
                       o.origin.empty() ? "the model's origin" : o.origin);
  parts << std::format("[weapon]\nname = {}\nlength = {:.4f}\n", o.name, (hi - lo) * scale);
  std::vector<std::string> order = o.partOrder;
  if (std::find(order.begin(), order.end(), o.defaultPart) == order.end() && o.defaultPart != "none")
    order.insert(order.begin(), o.defaultPart);
  for (const std::string &name : order) {
    const auto found = partTriangles.find(name);
    if (found == partTriangles.end()) {
      std::cout << "  part " << name << ": no triangles\n";
      continue;
    }
    glm::vec3 pivot(0.0f);
    const auto listed = o.keysOf.find(name);
    for (const std::string &key :
         listed != o.keysOf.end() ? listed->second : std::vector<std::string>{})
      if (world.count(key)) {
        glm::vec3 at = glm::vec3(world.at(key)[3]);
        if (o.moved.count(key))
          at += o.moved.at(key);
        pivot = glm::vec3(toGun * glm::vec4(at, 1.0f));
        break;
      }
    GltfPart mesh;
    mesh.name = name;
    std::map<uint32_t, uint32_t> remap;
    for (uint32_t index : found->second) {
      auto [it, added] = remap.try_emplace(index, static_cast<uint32_t>(mesh.positions.size()));
      if (added) {
        const Vertex &v = vertices[index];
        mesh.positions.push_back(glm::vec3(toGun * glm::vec4(v.position, 1.0f)) - pivot);
        mesh.normals.push_back(glm::normalize(turn * v.normal));
        mesh.uvs.push_back(glm::vec2(v.uv.x, 1.0f - v.uv.y)); // glTF's v runs down
      }
      mesh.indices.push_back(it->second);
    }
    const GltfFiles files = writeGltf(mesh, materialOf(name), name + ".bin");
    std::ofstream(out / (name + ".gltf")) << files.json;
    std::ofstream(out / (name + ".bin"), std::ios::binary)
        .write(reinterpret_cast<const char *>(files.bin.data()), files.bin.size());
    parts << std::format("\n[part {}]\nfile = {}.gltf\npivot = {}\n", name, name, vec(pivot));
    std::cout << std::format("  part {}: {} triangles, pivot {}\n", name,
                             mesh.indices.size() / 3, vec(pivot));
  }
  for (const auto &[socket, node] : o.sockets) {
    if (!world.count(node)) {
      std::cout << "  socket " << socket << ": no node called " << node << "\n";
      continue;
    }
    const glm::mat4 &m = world.at(node);
    glm::vec3 source = glm::vec3(m[3]);
    if (o.moved.count(node))
      source += o.moved.at(node);
    const glm::vec3 at = glm::vec3(toGun * glm::vec4(source, 1.0f));
    parts << std::format("\n[socket {}]\nposition = {}\nx = {}\ny = {}\nz = {}\n", socket, vec(at),
                         vec(glm::normalize(turn * glm::vec3(m[0]))),
                         vec(glm::normalize(turn * glm::vec3(m[1]))),
                         vec(glm::normalize(turn * glm::vec3(m[2]))));
    std::cout << std::format("  socket {} at {}\n", socket, vec(at));
  }
  for (const auto &[socket, point] : o.socketsAt) {
    const glm::vec3 at = glm::vec3(toGun * glm::vec4(point, 1.0f));
    parts << std::format("\n[socket {}]\nposition = {}\nx = 1 0 0\ny = 0 1 0\nz = 0 0 1\n", socket,
                         vec(at));
    std::cout << std::format("  socket {} at {}\n", socket, vec(at));
  }
  for (const auto &[socket, at] : o.socketsInGun) {
    parts << std::format("\n[socket {}]\nposition = {}\nx = 1 0 0\ny = 0 1 0\nz = 0 0 1\n", socket, vec(at));
    std::cout << std::format("  socket {} at {}\n", socket, vec(at));
  }
  std::ofstream(out / (o.name + ".parts")) << parts.str();
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[2]) == "--list")
      return list(argv[1]);
    return run(parse(argc, argv));
  } catch (const std::exception &e) {
    std::cerr << "weapon_convert: " << e.what() << "\n";
    return 1;
  }
}

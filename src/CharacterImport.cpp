#include "CharacterImport.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>

namespace {

glm::mat4 toGlm(const aiMatrix4x4 &m) {
  // Assimp's matrices are row-major, glm's column-major.
  glm::mat4 out;
  std::memcpy(&out, &m, sizeof(out));
  return glm::transpose(out);
}

glm::vec3 toGlm(const aiVector3D &v) { return {v.x, v.y, v.z}; }
glm::quat toGlm(const aiQuaternion &q) { return {q.w, q.x, q.y, q.z}; }

BoneTransform decompose(const aiMatrix4x4 &m) {
  aiVector3D scale, position;
  aiQuaternion rotation;
  m.Decompose(scale, rotation, position);
  return {toGlm(position), glm::normalize(toGlm(rotation)), toGlm(scale)};
}

// FBX pivots would come in as extra "$AssimpFbx$" nodes between the bones,
// and the clips' channels would land on those instead of the bones.
const aiScene *readFile(Assimp::Importer &importer, const std::string &path,
                        unsigned flags) {
  importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
  return importer.ReadFile(path, flags);
}

void addNodes(const aiNode *node, int parent, Skeleton &skeleton) {
  Bone bone;
  bone.name = node->mName.C_Str();
  bone.parent = parent;
  bone.bind = decompose(node->mTransformation);
  skeleton.bones.push_back(bone);
  const int self = static_cast<int>(skeleton.bones.size()) - 1;
  for (unsigned i = 0; i < node->mNumChildren; ++i)
    addNodes(node->mChildren[i], self, skeleton);
}

const aiNode *findNode(const aiNode *node, const std::string &name) {
  if (name == node->mName.C_Str())
    return node;
  for (unsigned i = 0; i < node->mNumChildren; ++i)
    if (const aiNode *found = findNode(node->mChildren[i], name))
      return found;
  return nullptr;
}

// The hips: Mixamo's are "mixamorig:Hips", but some files number the rig.
int findHips(const Skeleton &skeleton) {
  for (size_t i = 0; i < skeleton.bones.size(); ++i) {
    const std::string &name = skeleton.bones[i].name;
    if (name.size() >= 4 && name.compare(name.size() - 4, 4, "Hips") == 0)
      return static_cast<int>(i);
  }
  return -1;
}

// Appends one skinned mesh: its vertices as they are (skinning takes them
// from the mesh's space), its triangles, and up to four bones a vertex.
void addMesh(const aiMesh *mesh, Skeleton &skeleton, CharacterAsset &asset) {
  const uint32_t first = static_cast<uint32_t>(asset.mesh.vertices.size());
  for (unsigned i = 0; i < mesh->mNumVertices; ++i) {
    Vertex v{};
    v.pos = toGlm(mesh->mVertices[i]);
    v.col = glm::vec3(1.0f);
    if (mesh->mTextureCoords[0])
      v.tex = {mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y};
    v.normal = mesh->mNormals ? toGlm(mesh->mNormals[i]) : glm::vec3(0, 1, 0);
    v.tangent = mesh->mTangents ? toGlm(mesh->mTangents[i]) : glm::vec3(1, 0, 0);
    v.bitangent =
        mesh->mBitangents ? toGlm(mesh->mBitangents[i]) : glm::vec3(0, 0, 1);
    asset.mesh.vertices.push_back(v);
  }
  for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
    const aiFace &face = mesh->mFaces[f];
    if (face.mNumIndices != 3)
      continue;
    for (int k = 0; k < 3; ++k)
      asset.mesh.indices.push_back(first + face.mIndices[k]);
  }
  asset.skin.resize(asset.mesh.vertices.size());
  for (unsigned b = 0; b < mesh->mNumBones; ++b) {
    const aiBone *bone = mesh->mBones[b];
    const int index = skeleton.find(bone->mName.C_Str());
    if (index < 0)
      continue;
    skeleton.bones[index].inverseBind = toGlm(bone->mOffsetMatrix);
    for (unsigned w = 0; w < bone->mNumWeights; ++w) {
      VertexSkin &skin = asset.skin[first + bone->mWeights[w].mVertexId];
      // Into the slot with the smallest weight, so a vertex keeps its four
      // strongest bones.
      int slot = 0;
      for (int k = 1; k < 4; ++k)
        if (skin.weights[k] < skin.weights[slot])
          slot = k;
      if (bone->mWeights[w].mWeight > skin.weights[slot]) {
        skin.bones[slot] = static_cast<uint32_t>(index);
        skin.weights[slot] = bone->mWeights[w].mWeight;
      }
    }
  }
  for (uint32_t i = first; i < asset.skin.size(); ++i) {
    VertexSkin &skin = asset.skin[i];
    const float total = skin.weights.x + skin.weights.y + skin.weights.z +
                        skin.weights.w;
    skin.weights = total > 0.0f ? skin.weights / total : glm::vec4(0.0f);
  }
}

// The embedded albedo texture's bytes, when the material points at one.
void takeAlbedo(const aiScene *scene, CharacterAsset &asset) {
  for (unsigned m = 0; m < scene->mNumMaterials; ++m) {
    aiString path;
    if (scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, 0, &path) !=
        AI_SUCCESS)
      continue;
    const aiTexture *texture = scene->GetEmbeddedTexture(path.C_Str());
    if (!texture || texture->mHeight != 0)
      continue; // only compressed files (PNG, JPEG) are embedded this way
    const auto *bytes = reinterpret_cast<const unsigned char *>(texture->pcData);
    asset.albedo.assign(bytes, bytes + texture->mWidth);
    asset.albedoName = path.C_Str();
    return;
  }
}

// The bind-pose bounds: each vertex moved by its bones' skinning matrices,
// which in the bind pose take it from the mesh's space to the skeleton's.
void computeBounds(CharacterAsset &asset) {
  const std::vector<glm::mat4> skinning = skinningMatrices(
      asset.skeleton, modelSpace(asset.skeleton, bindPose(asset.skeleton)));
  glm::vec3 lo(std::numeric_limits<float>::max());
  glm::vec3 hi(std::numeric_limits<float>::lowest());
  for (size_t i = 0; i < asset.mesh.vertices.size(); ++i) {
    const glm::vec3 p = skinVertex(asset.mesh.vertices[i], asset.skin[i], skinning).pos;
    lo = glm::min(lo, p);
    hi = glm::max(hi, p);
  }
  asset.boundsMin = lo;
  asset.boundsMax = hi;
}

// One clip file onto the character's skeleton. Only rotations come from the
// clip, plus the hips' movement scaled by hip height, so a clip made on
// another Mixamo character (X Bot) keeps this one's proportions.
std::optional<CharacterClipData> importClip(const CharacterClip &entry,
                                            const CharacterAsset &asset) {
  const std::string path = resolveCharacterPath(entry.file);
  if (path.empty()) {
    spdlog::warn("Character clip {}: no file {}", clipSlotName(entry.slot),
                 entry.file);
    return std::nullopt;
  }
  Assimp::Importer importer;
  const aiScene *scene = readFile(importer, path, 0);
  if (!scene || scene->mNumAnimations == 0) {
    spdlog::warn("Character clip {}: no animation in {} ({})",
                 clipSlotName(entry.slot), entry.file, importer.GetErrorString());
    return std::nullopt;
  }
  const aiAnimation *animation = scene->mAnimations[0];
  const double ticks = animation->mTicksPerSecond > 0.0 ? animation->mTicksPerSecond
                                                        : 25.0;
  const Skeleton &skeleton = asset.skeleton;
  float hipsScale = 1.0f;
  if (asset.hips >= 0)
    if (const aiNode *hips = findNode(scene->mRootNode, skeleton.bones[asset.hips].name)) {
      const float theirs = glm::length(decompose(hips->mTransformation).translation);
      const float ours = glm::length(skeleton.bones[asset.hips].bind.translation);
      if (theirs > 1e-6f)
        hipsScale = ours / theirs;
    }

  CharacterClipData data;
  data.clip.name = clipSlotName(entry.slot);
  data.clip.duration = static_cast<float>(animation->mDuration / ticks);
  data.clip.tracks.resize(skeleton.bones.size());
  data.start = entry.start;
  data.end = entry.end;
  data.loop = entry.loop;
  int matched = 0;
  for (unsigned c = 0; c < animation->mNumChannels; ++c) {
    const aiNodeAnim *channel = animation->mChannels[c];
    const int bone = skeleton.find(channel->mNodeName.C_Str());
    if (bone < 0)
      continue;
    ++matched;
    BoneTrack &track = data.clip.tracks[bone];
    for (unsigned k = 0; k < channel->mNumRotationKeys; ++k)
      track.rotations.push_back(
          {static_cast<float>(channel->mRotationKeys[k].mTime / ticks),
           glm::normalize(toGlm(channel->mRotationKeys[k].mValue))});
    if (bone == asset.hips)
      for (unsigned k = 0; k < channel->mNumPositionKeys; ++k)
        track.positions.push_back(
            {static_cast<float>(channel->mPositionKeys[k].mTime / ticks),
             toGlm(channel->mPositionKeys[k].mValue) * hipsScale});
  }
  data.rootMotion = removeRootMotion(data.clip, skeleton, asset.hips,
                                     entry.loop ? RootMotionMode::Progression
                                                : RootMotionMode::Path);
  // The capsule does the jumping: the clip's own rise would add to it.
  if (entry.slot == ClipSlot::Jump)
    limitRootHeight(data.clip, skeleton, asset.hips);
  spdlog::info("Character clip {}: {:.2f} s, {} of {} channels on the "
               "skeleton, hips moving {:.2f} units/s",
               clipSlotName(entry.slot), data.clip.duration, matched,
               animation->mNumChannels, data.rootMotion.speed());
  return data;
}

} // namespace

std::string resolveCharacterPath(const std::string &path) {
  namespace fs = std::filesystem;
  for (const fs::path &candidate : {fs::path(path), fs::path("..") / path})
    if (std::error_code ec; fs::is_regular_file(candidate, ec))
      return candidate.string();
  return "";
}

std::optional<CharacterAsset> importCharacter(const CharacterDescription &description,
                                              std::string &error) {
  const auto start = std::chrono::steady_clock::now();
  const std::string modelPath = resolveCharacterPath(description.model);
  if (modelPath.empty()) {
    error = "no file " + description.model;
    return std::nullopt;
  }
  Assimp::Importer importer;
  const aiScene *scene = readFile(
      importer, modelPath,
      aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_SortByPType |
          aiProcess_JoinIdenticalVertices | aiProcess_GenSmoothNormals |
          aiProcess_CalcTangentSpace | aiProcess_LimitBoneWeights);
  if (!scene || !scene->mRootNode) {
    error = std::string("Assimp couldn't read it: ") + importer.GetErrorString();
    return std::nullopt;
  }
  CharacterAsset asset;
  addNodes(scene->mRootNode, -1, asset.skeleton);
  asset.hips = findHips(asset.skeleton);
  int skipped = 0;
  for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
    const aiMesh *mesh = scene->mMeshes[m];
    // Lines and points (the file has a 2-vertex helper) and meshes no bone
    // moves aren't part of the body.
    if (!(mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) ||
        mesh->mNumBones == 0) {
      ++skipped;
      continue;
    }
    addMesh(mesh, asset.skeleton, asset);
  }
  if (asset.mesh.vertices.empty()) {
    error = "no skinned triangle mesh in it";
    return std::nullopt;
  }
  takeAlbedo(scene, asset);
  computeBounds(asset);
  for (const CharacterClip &entry : description.clips)
    asset.library.clips[static_cast<size_t>(entry.slot)] = importClip(entry, asset);
  const size_t loaded = std::count_if(asset.library.clips.begin(), asset.library.clips.end(),
                                      [](const auto &c) { return c.has_value(); });
  spdlog::info("Character: {} nodes, hips at {}, {} vertices, {} triangles ({} "
               "meshes skipped), albedo {}, {:.1f} units tall, {} of {} clips, "
               "{:.0f} ms",
               asset.skeleton.bones.size(), asset.hips, asset.mesh.vertices.size(),
               asset.mesh.indices.size() / 3, skipped,
               asset.albedo.empty() ? "none" : asset.albedoName,
               asset.boundsMax.y - asset.boundsMin.y, loaded,
               description.clips.size(),
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count());
  return asset;
}

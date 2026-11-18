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

// The node for this bone, by sameBoneName: a clip's rig prefix can differ.
const aiNode *findNode(const aiNode *node, const std::string &name) {
  if (sameBoneName(name, node->mName.C_Str()))
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

// Appends one skinned mesh to its material's section: its vertices as they
// are (skinning takes them from the mesh's space), its triangles, and up to
// four bones a vertex.
void addMesh(const aiMesh *mesh, Skeleton &skeleton, CharacterSection &asset) {
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

// A material's embedded albedo texture's bytes, when it points at one.
void takeAlbedo(const aiScene *scene, unsigned material, CharacterSection &section) {
  aiString path;
  if (material >= scene->mNumMaterials ||
      scene->mMaterials[material]->GetTexture(aiTextureType_DIFFUSE, 0, &path) != AI_SUCCESS)
    return;
  const aiTexture *texture = scene->GetEmbeddedTexture(path.C_Str());
  if (!texture || texture->mHeight != 0)
    return; // only compressed files (PNG, JPEG) are embedded this way
  const auto *bytes = reinterpret_cast<const unsigned char *>(texture->pcData);
  section.albedo.assign(bytes, bytes + texture->mWidth);
  section.albedoName = path.C_Str();
}

// The bind-pose bounds: each vertex moved by its bones' skinning matrices,
// which in the bind pose take it from the mesh's space to the skeleton's.
void computeBounds(CharacterAsset &asset) {
  const std::vector<glm::mat4> skinning = skinningMatrices(
      asset.skeleton, modelSpace(asset.skeleton, bindPose(asset.skeleton)));
  glm::vec3 lo(std::numeric_limits<float>::max());
  glm::vec3 hi(std::numeric_limits<float>::lowest());
  for (const CharacterSection &section : asset.sections)
    for (size_t i = 0; i < section.mesh.vertices.size(); ++i) {
      const glm::vec3 p = skinVertex(section.mesh.vertices[i], section.skin[i], skinning).pos;
      lo = glm::min(lo, p);
      hi = glm::max(hi, p);
    }
  asset.boundsMin = lo;
  asset.boundsMax = hi;
}

// One clip file onto the character's skeleton. Only rotations come from the
// clip, retargeted from its rig's rest pose to this one's, plus the hips'
// movement scaled by hip height, so a clip made on another Mixamo character
// (X Bot) keeps this one's proportions.
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
  // The clip file's own rig at rest: Mixamo bakes a clip for the character
  // it was downloaded with, whose rest pose can differ from this one's by
  // 30 degrees at the shoulders (Ganfaul's against Remy's), so keys are
  // retargeted rest to rest instead of copied.
  Skeleton source;
  addNodes(scene->mRootNode, -1, source);
  const std::vector<glm::quat> sourceRest = bindModelRotations(source);
  const std::vector<glm::quat> targetRest = bindModelRotations(skeleton);
  const glm::quat identity(1.0f, 0.0f, 0.0f, 0.0f);
  int matched = 0;
  for (unsigned c = 0; c < animation->mNumChannels; ++c) {
    const aiNodeAnim *channel = animation->mChannels[c];
    const int bone = skeleton.find(channel->mNodeName.C_Str());
    if (bone < 0)
      continue;
    ++matched;
    const int from = source.find(channel->mNodeName.C_Str());
    const int fromParent = from >= 0 ? source.bones[from].parent : -1;
    const int toParent = skeleton.bones[bone].parent;
    const glm::quat sourceBind = from >= 0 ? source.bones[from].bind.rotation : identity;
    const glm::quat targetBind = skeleton.bones[bone].bind.rotation;
    const glm::quat sourceParent = fromParent >= 0 ? sourceRest[fromParent] : identity;
    const glm::quat targetParent = toParent >= 0 ? targetRest[toParent] : identity;
    BoneTrack &track = data.clip.tracks[bone];
    for (unsigned k = 0; k < channel->mNumRotationKeys; ++k) {
      const glm::quat key = glm::normalize(toGlm(channel->mRotationKeys[k].mValue));
      track.rotations.push_back(
          {static_cast<float>(channel->mRotationKeys[k].mTime / ticks),
           from >= 0 ? retargetRotation(key, sourceBind, targetBind, sourceParent, targetParent)
                     : key});
    }
    if (bone == asset.hips)
      for (unsigned k = 0; k < channel->mNumPositionKeys; ++k)
        track.positions.push_back(
            {static_cast<float>(channel->mPositionKeys[k].mTime / ticks),
             retargetPosition(toGlm(channel->mPositionKeys[k].mValue), sourceParent,
                              targetParent) *
                 hipsScale});
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

std::optional<CharacterClipData> importCharacterClip(const CharacterClip &clip,
                                                     const CharacterAsset &asset) {
  return importClip(clip, asset);
}

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
  std::vector<unsigned> materials; // the material of each section
  for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
    const aiMesh *mesh = scene->mMeshes[m];
    // Lines and points (the file has a 2-vertex helper) and meshes no bone
    // moves aren't part of the body.
    if (!(mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) ||
        mesh->mNumBones == 0) {
      ++skipped;
      continue;
    }
    // Meshes sharing a material share a section (Remy's eyes and lashes).
    const auto it = std::find(materials.begin(), materials.end(), mesh->mMaterialIndex);
    const size_t section = static_cast<size_t>(it - materials.begin());
    if (it == materials.end()) {
      materials.push_back(mesh->mMaterialIndex);
      asset.sections.emplace_back();
    }
    addMesh(mesh, asset.skeleton, asset.sections[section]);
  }
  for (size_t k = 0; k < asset.sections.size(); ++k)
    takeAlbedo(scene, materials[k], asset.sections[k]);
  std::erase_if(asset.sections, [](const CharacterSection &s) { return s.mesh.indices.empty(); });
  if (asset.sections.empty()) {
    error = "no skinned triangle mesh in it";
    return std::nullopt;
  }
  computeBounds(asset);
  for (const CharacterClip &entry : description.clips)
    asset.library.clips[static_cast<size_t>(entry.slot)] = importClip(entry, asset);
  const size_t loaded = std::count_if(asset.library.clips.begin(), asset.library.clips.end(),
                                      [](const auto &c) { return c.has_value(); });
  size_t vertices = 0, triangles = 0, textured = 0;
  for (const CharacterSection &section : asset.sections) {
    vertices += section.mesh.vertices.size();
    triangles += section.mesh.indices.size() / 3;
    textured += !section.albedo.empty();
  }
  spdlog::info("Character: {} nodes, hips at {}, {} vertices, {} triangles ({} "
               "meshes skipped), {} materials ({} textured), {:.1f} units tall, {} of {} "
               "clips, {:.0f} ms",
               asset.skeleton.bones.size(), asset.hips, vertices, triangles, skipped,
               asset.sections.size(), textured,
               asset.boundsMax.y - asset.boundsMin.y, loaded,
               description.clips.size(),
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count());
  return asset;
}

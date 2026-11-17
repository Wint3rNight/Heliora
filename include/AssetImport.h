#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// Bringing model files into the project's Resources/ (like importing into
// Unreal's Content folder), and listing what a folder holds for the Assets
// panel's tree. No Vulkan, no Assimp: only files.

struct FileCopy {
  std::filesystem::path from;
  std::filesystem::path to;
};

struct ImportPlan {
  std::vector<FileCopy> copies;
  std::string modelPath; // the model file as scene files name it
};

// What importing source (a .gltf or .glb) copies: into its own folder under
// modelsFolder, named after it ("rock", or "rock-2" if that's taken), a
// .gltf with the files its relative uris name, in their folders. Nothing,
// with the reason in error, for anything else, a missing file, or a uri
// that leaves the model's folder. modelsAsWritten is modelsFolder as
// scene files write it ("Resources/Models").
std::optional<ImportPlan> planModelImport(const std::filesystem::path &source,
                                          const std::filesystem::path &modelsFolder,
                                          const std::string &modelsAsWritten,
                                          std::string &error);
// Copies the files; false, with the reason, if one fails.
bool runImport(const ImportPlan &plan, std::string &error);

// A folder's subfolders (not hidden ones like .thumbnails) and model files,
// names only, sorted.
struct FolderListing {
  std::vector<std::string> folders;
  std::vector<std::string> models;
};
FolderListing listFolder(const std::filesystem::path &folder);

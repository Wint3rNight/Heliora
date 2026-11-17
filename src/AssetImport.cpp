#include "AssetImport.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;

namespace {

bool isModel(const fs::path &path) {
  const std::string ext = path.extension().string();
  return ext == ".gltf" || ext == ".glb";
}

// %20 and the like back to their characters, as a uri writes them.
std::string percentDecoded(const std::string &uri) {
  std::string out;
  for (size_t i = 0; i < uri.size(); ++i) {
    if (uri[i] == '%' && i + 2 < uri.size() && std::isxdigit(uri[i + 1]) &&
        std::isxdigit(uri[i + 2])) {
      out += static_cast<char>(std::stoi(uri.substr(i + 1, 2), nullptr, 16));
      i += 2;
    } else {
      out += uri[i];
    }
  }
  return out;
}

// The files a .gltf's uris name, relative to it. Embedded data and web
// addresses aren't files.
std::vector<std::string> referencedFiles(const fs::path &gltf) {
  std::ifstream in(gltf);
  std::stringstream text;
  text << in.rdbuf();
  const std::string json = text.str();
  static const std::regex uri(R"re("uri"\s*:\s*"((?:[^"\\]|\\.)*)")re");
  std::vector<std::string> files;
  for (auto it = std::sregex_iterator(json.begin(), json.end(), uri);
       it != std::sregex_iterator(); ++it) {
    const std::string value = (*it)[1].str();
    if (value.starts_with("data:") || value.find("://") != std::string::npos)
      continue;
    const std::string file = percentDecoded(value);
    if (std::ranges::find(files, file) == files.end())
      files.push_back(file);
  }
  return files;
}

} // namespace

std::optional<ImportPlan> planModelImport(const fs::path &source, const fs::path &modelsFolder,
                                          const std::string &modelsAsWritten,
                                          std::string &error) {
  if (!isModel(source)) {
    error = std::format("{} isn't a .gltf or .glb model", source.filename().string());
    return std::nullopt;
  }
  std::error_code ec;
  if (!fs::is_regular_file(source, ec)) {
    error = std::format("There's no file {}", source.string());
    return std::nullopt;
  }
  // Its own folder, named after it, never over one already there.
  const std::string stem = source.stem().string();
  std::string folderName = stem;
  for (int n = 2; fs::exists(modelsFolder / folderName, ec); ++n)
    folderName = std::format("{}-{}", stem, n);
  const fs::path to = modelsFolder / folderName;

  ImportPlan plan;
  plan.copies.push_back({source, to / source.filename()});
  if (source.extension() == ".gltf")
    for (const std::string &file : referencedFiles(source)) {
      const fs::path relative = fs::path(file).lexically_normal();
      if (relative.is_absolute() || relative.empty() || *relative.begin() == "..") {
        error = std::format("{} uses {}, outside its folder, which can't come along",
                            source.filename().string(), file);
        return std::nullopt;
      }
      const fs::path from = source.parent_path() / relative;
      if (!fs::is_regular_file(from, ec)) {
        error = std::format("{} needs {}, which isn't there", source.filename().string(), file);
        return std::nullopt;
      }
      plan.copies.push_back({from, to / relative});
    }
  plan.modelPath =
      (fs::path(modelsAsWritten) / folderName / source.filename()).generic_string();
  return plan;
}

bool runImport(const ImportPlan &plan, std::string &error) {
  for (const FileCopy &copy : plan.copies) {
    std::error_code ec;
    fs::create_directories(copy.to.parent_path(), ec);
    if (!ec)
      fs::copy_file(copy.from, copy.to, fs::copy_options::none, ec);
    if (ec) {
      error = std::format("Couldn't copy {}: {}", copy.from.filename().string(), ec.message());
      return false;
    }
  }
  return true;
}

FolderListing listFolder(const fs::path &folder) {
  FolderListing listing;
  std::error_code ec;
  for (fs::directory_iterator it(folder, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::directory_iterator(); it.increment(ec)) {
    const std::string name = it->path().filename().string();
    std::error_code typeError;
    if (it->is_directory(typeError)) {
      if (!name.starts_with("."))
        listing.folders.push_back(name);
    } else if (it->is_regular_file(typeError) && isModel(it->path())) {
      listing.models.push_back(name);
    } else if (it->is_regular_file(typeError) && it->path().extension() == ".fbx") {
      listing.clips.push_back(name);
    }
  }
  std::ranges::sort(listing.folders);
  std::ranges::sort(listing.models);
  std::ranges::sort(listing.clips);
  return listing;
}

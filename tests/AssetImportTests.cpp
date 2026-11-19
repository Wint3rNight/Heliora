#include "AssetImport.h"
#include "TestRunner.h"

#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

namespace {

// A fresh folder under the temp folder.
fs::path freshFolder(const char *name) {
  const fs::path root = fs::temp_directory_path() / name;
  fs::remove_all(root);
  fs::create_directories(root);
  return root;
}

void write(const fs::path &file, const std::string &text) {
  fs::create_directories(file.parent_path());
  std::ofstream(file) << text;
}

bool copies(const ImportPlan &plan, const fs::path &from, const fs::path &to) {
  return std::ranges::any_of(plan.copies, [&](const FileCopy &c) {
    return c.from == from && c.to == to;
  });
}

} // namespace

TEST(a_glb_is_copied_on_its_own) {
  const fs::path root = freshFolder("heliora_import_glb");
  write(root / "src" / "rock.glb", "glb");
  std::string error;
  const std::optional<ImportPlan> plan =
      planModelImport(root / "src" / "rock.glb", root / "Models", "Resources/Models", error);
  CHECK(plan.has_value());
  if (!plan)
    return;
  CHECK(plan->copies.size() == 1);
  CHECK(copies(*plan, root / "src" / "rock.glb", root / "Models" / "rock" / "rock.glb"));
  CHECK(plan->modelPath == "Resources/Models/rock/rock.glb");
  CHECK(runImport(*plan, error));
  CHECK(fs::exists(root / "Models" / "rock" / "rock.glb"));
  fs::remove_all(root);
}

TEST(a_gltf_brings_its_buffers_and_textures) {
  // The files its uris name come along, in their folders; a data: uri is
  // inside the file already, and %20 is a space.
  const fs::path root = freshFolder("heliora_import_gltf");
  write(root / "src" / "crate.gltf",
        R"({"buffers":[{"uri":"crate.bin"},{"uri":"data:application/octet-stream;base64,AAAA"}],)"
        R"("images":[{"uri" : "textures/crate%20albedo.png"}]})");
  write(root / "src" / "crate.bin", "bin");
  write(root / "src" / "textures" / "crate albedo.png", "png");
  std::string error;
  const std::optional<ImportPlan> plan =
      planModelImport(root / "src" / "crate.gltf", root / "Models", "Resources/Models", error);
  CHECK(plan.has_value());
  if (!plan)
    return;
  const fs::path to = root / "Models" / "crate";
  CHECK(plan->copies.size() == 3);
  CHECK(copies(*plan, root / "src" / "crate.gltf", to / "crate.gltf"));
  CHECK(copies(*plan, root / "src" / "crate.bin", to / "crate.bin"));
  CHECK(copies(*plan, root / "src" / "textures" / "crate albedo.png",
               to / "textures" / "crate albedo.png"));
  CHECK(runImport(*plan, error));
  CHECK(fs::exists(to / "textures" / "crate albedo.png"));
  fs::remove_all(root);
}

TEST(an_import_never_overwrites_one_already_there) {
  const fs::path root = freshFolder("heliora_import_clash");
  write(root / "src" / "rock.glb", "new");
  write(root / "Models" / "rock" / "rock.glb", "old");
  write(root / "Models" / "rock-2" / "rock.glb", "older");
  std::string error;
  const std::optional<ImportPlan> plan =
      planModelImport(root / "src" / "rock.glb", root / "Models", "Resources/Models", error);
  CHECK(plan.has_value());
  if (plan) {
    CHECK(copies(*plan, root / "src" / "rock.glb", root / "Models" / "rock-3" / "rock.glb"));
    CHECK(plan->modelPath == "Resources/Models/rock-3/rock.glb");
  }
  fs::remove_all(root);
}

TEST(what_cant_be_imported_says_why) {
  const fs::path root = freshFolder("heliora_import_refused");
  write(root / "notes.txt", "x");
  std::string error;
  CHECK(!planModelImport(root / "notes.txt", root / "Models", "Resources/Models", error));
  CHECK(error.find(".gltf or .glb") != std::string::npos);
  write(root / "lonely.gltf", R"({"buffers":[{"uri":"gone.bin"}]})");
  CHECK(!planModelImport(root / "lonely.gltf", root / "Models", "Resources/Models", error));
  CHECK(error.find("gone.bin") != std::string::npos);
  write(root / "src" / "climber.gltf", R"({"images":[{"uri":"../shared/wall.png"}]})");
  write(root / "shared" / "wall.png", "png");
  CHECK(!planModelImport(root / "src" / "climber.gltf", root / "Models", "Resources/Models",
                         error));
  CHECK(error.find("outside") != std::string::npos);
  CHECK(!planModelImport(root / "missing.glb", root / "Models", "Resources/Models", error));
  fs::remove_all(root);
}

TEST(a_folder_lists_its_subfolders_and_models) {
  const fs::path root = freshFolder("heliora_import_listing");
  write(root / "rock" / "rock.gltf", "x");
  write(root / "crate.glb", "x");
  write(root / "barrel.gltf", "x");
  write(root / "notes.txt", "x");
  write(root / "walking.fbx", "x"); // a clip
  write(root / "idle.fbx", "x");
  write(root / "crate.object", "x"); // a prefab
  write(root / "remy.character", "x");
  write(root / "goblin.character", "x");
  write(root / ".thumbnails" / "a.png", "x");
  fs::create_directories(root / "empty");
  const FolderListing listing = listFolder(root);
  CHECK((listing.folders == std::vector<std::string>{"empty", "rock"}));
  CHECK((listing.models == std::vector<std::string>{"barrel.gltf", "crate.glb"}));
  CHECK((listing.clips == std::vector<std::string>{"idle.fbx", "walking.fbx"}));
  CHECK((listing.objects == std::vector<std::string>{"crate.object"}));
  CHECK((listing.characters ==
         std::vector<std::string>{"goblin.character", "remy.character"}));
  CHECK(listFolder(root / "missing").folders.empty());
  fs::remove_all(root);
}

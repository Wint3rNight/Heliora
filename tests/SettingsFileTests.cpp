#include "SettingsFile.h"
#include "TestRunner.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

SettingsFile parseText(const std::string &text) {
  std::istringstream in(text);
  return SettingsFile::parse(in, "test.cfg");
}

} // namespace

TEST(settings_survive_a_round_trip) {
  SettingsFile file;
  file.setFloat("render", "fov", 62.5f);
  file.setInt("physics", "steps_per_second", 90);
  file.setBool("render", "taa", false);
  file.setVec3("physics", "gravity", glm::vec3(0.0f, -3.7f, 0.25f));

  const SettingsFile back = parseText(file.write());
  CHECK(back.getFloat("render", "fov") == 62.5f);
  CHECK(back.getInt("physics", "steps_per_second") == 90);
  CHECK(back.getBool("render", "taa") == false);
  CHECK(back.getVec3("physics", "gravity") == glm::vec3(0.0f, -3.7f, 0.25f));
  CHECK(back.write() == file.write());
}

TEST(missing_settings_file_gives_defaults) {
  const SettingsFile file =
      SettingsFile::load("/nonexistent/heliora/settings.cfg");
  CHECK(file.keys("render").empty());

  float fov = 45.0f;
  const SettingBinding bindings[] = {{"render", "fov", &fov, 30.0, 120.0}};
  readSettings(file, bindings);
  CHECK(fov == 45.0f);
}

TEST(broken_line_is_skipped) {
  const SettingsFile file = parseText("orphan = 1\n"
                                      "[render]\n"
                                      "fov = 70\n"
                                      "this line has no equals sign\n"
                                      "= 3\n"
                                      "taa = false  # end-of-line comment\n"
                                      "[physics]\n"
                                      "steps_per_second = 120\n");
  CHECK(file.getFloat("render", "fov") == 70.0f);
  CHECK(file.getBool("render", "taa") == false);
  CHECK(file.getInt("physics", "steps_per_second") == 120);
  CHECK(file.keys("render").size() == 2);
  CHECK(file.keys("").empty());
}

TEST(broken_section_header_skips_its_keys) {
  const SettingsFile file = parseText("[render\n"
                                      "fov = 70\n"
                                      "[physics]\n"
                                      "steps_per_second = 120\n");
  CHECK(!file.text("render", "fov"));
  CHECK(file.getInt("physics", "steps_per_second") == 120);
}

TEST(wrong_type_keeps_the_current_value) {
  const SettingsFile file = parseText("[render]\n"
                                      "fov = wide\n"
                                      "taa = maybe\n"
                                      "ssgi_samples = 8.5\n"
                                      "exposure = nan\n");
  float fov = 45.0f;
  bool taa = true;
  int samples = 8;
  float exposure = 0.0f;
  const SettingBinding bindings[] = {
      {"render", "fov", &fov, 30.0, 120.0},
      {"render", "taa", &taa},
      {"render", "ssgi_samples", &samples, 4.0, 12.0},
      {"render", "exposure", &exposure, -6.0, 6.0},
  };
  readSettings(file, bindings);
  CHECK(fov == 45.0f);
  CHECK(taa);
  CHECK(samples == 8);
  CHECK(exposure == 0.0f);
}

TEST(out_of_range_setting_is_clamped) {
  const SettingsFile file = parseText("[render]\n"
                                      "fov = 500\n"
                                      "[physics]\n"
                                      "max_bodies = -5\n"
                                      "gravity = 0 -500 0\n");
  float fov = 45.0f;
  uint32_t maxBodies = 16384;
  glm::vec3 gravity(0.0f, -9.81f, 0.0f);
  const SettingBinding bindings[] = {
      {"render", "fov", &fov, 30.0, 120.0},
      {"physics", "max_bodies", &maxBodies, 64.0, 1048576.0},
      {"physics", "gravity", &gravity, -100.0, 100.0},
  };
  readSettings(file, bindings);
  CHECK(fov == 120.0f);
  CHECK(maxBodies == 64u);
  CHECK(gravity == glm::vec3(0.0f, -100.0f, 0.0f));
}

TEST(only_tuned_settings_are_written) {
  float fov = 45.0f;
  bool taa = true;
  const SettingBinding bindings[] = {{"render", "fov", &fov, 30.0, 120.0},
                                     {"render", "taa", &taa}};
  SettingsFile defaults;
  writeSettings(defaults, bindings);

  fov = 70.0f;
  SettingsFile file;
  writeSettings(file, bindings, "render", defaults);
  CHECK(file.getFloat("render", "fov") == 70.0f);
  CHECK(!file.text("render", "taa"));

  // Tuned back to the default, the key goes again.
  fov = 45.0f;
  writeSettings(file, bindings, "render", defaults);
  CHECK(!file.text("render", "fov"));
}

TEST(one_section_at_a_time) {
  SettingsFile file = parseText("[render]\n"
                                "fov = 90\n"
                                "[physics]\n"
                                "steps_per_second = 120\n");
  float fov = 45.0f;
  int steps = 60;
  const SettingBinding bindings[] = {
      {"render", "fov", &fov, 30.0, 120.0},
      {"physics", "steps_per_second", &steps, 10.0, 480.0},
  };
  readSettings(file, bindings, "render");
  CHECK(fov == 90.0f);
  CHECK(steps == 60);

  fov = 70.0f;
  writeSettings(file, bindings, "render");
  CHECK(file.getFloat("render", "fov") == 70.0f);
  CHECK(file.getInt("physics", "steps_per_second") == 120);
}

TEST(saved_settings_load_back) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "heliora_settings_test.cfg";
  SettingsFile file;
  file.setFloat("render", "fov", 75.0f);
  file.setBool("physics", "unknown_key_survives", true);
  file.save(path.string());

  const SettingsFile back = SettingsFile::load(path.string());
  CHECK(back.getFloat("render", "fov") == 75.0f);
  CHECK(back.getBool("physics", "unknown_key_survives") == true);
  CHECK(!std::filesystem::exists(path.string() + ".tmp"));
  std::filesystem::remove(path);
}

TEST(save_to_a_missing_folder_throws) {
  SettingsFile file;
  file.setFloat("render", "fov", 75.0f);
  bool threw = false;
  try {
    file.save("/nonexistent/heliora/settings.cfg");
  } catch (const std::exception &) {
    threw = true;
  }
  CHECK(threw);
}

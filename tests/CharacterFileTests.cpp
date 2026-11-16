#include "CharacterFile.h"
#include "TestRunner.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace {

CharacterDescription parseText(const std::string &text) {
  std::istringstream in(text);
  return CharacterIO::parse(in, "test.character");
}

// The error a broken file gives, or "" if it parses.
std::string errorFor(const std::string &text) {
  try {
    parseText(text);
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "";
}

bool contains(const std::string &text, const std::string &part) {
  return text.find(part) != std::string::npos;
}

const std::string kGanfaul =
    "# Ganfaul, from Mixamo\n"
    "model = Resources/Characters/mixamo/Ganfaul M Aure.fbx\n"
    "\n"
    "[clip]\n"
    "slot = idle\n"
    "file = Resources/Characters/mixamo/idle.fbx\n"
    "\n"
    "[clip]\n"
    "slot = jump\n"
    "file = Resources/Characters/mixamo/Standing Jump.fbx\n"
    "start = 0.35\n";

} // namespace

TEST(character_file_names_the_model_and_its_clips) {
  const CharacterDescription c = parseText(kGanfaul);
  CHECK(c.model == "Resources/Characters/mixamo/Ganfaul M Aure.fbx");
  CHECK(c.clips.size() == 2);
  if (c.clips.size() != 2)
    return;
  CHECK(c.clips[0].slot == ClipSlot::Idle);
  CHECK(c.clips[0].file == "Resources/Characters/mixamo/idle.fbx");
  CHECK(c.clips[1].slot == ClipSlot::Jump);
  CHECK_NEAR(c.clips[1].start, 0.35, 1e-6);
  CHECK(c.clip(ClipSlot::Jump) == &c.clips[1]);
  CHECK(c.clip(ClipSlot::Roll) == nullptr);
}

TEST(moves_that_repeat_loop_unless_told_otherwise) {
  const CharacterDescription c = parseText(kGanfaul);
  CHECK(c.clips.size() == 2);
  if (c.clips.size() != 2)
    return;
  CHECK(c.clips[0].loop);  // idle repeats
  CHECK(!c.clips[1].loop); // a jump happens once
  CHECK(defaultLoop(ClipSlot::RunForward));
  CHECK(defaultLoop(ClipSlot::Fall));
  CHECK(!defaultLoop(ClipSlot::Land));
  CHECK(!defaultLoop(ClipSlot::Roll));
  CHECK(!defaultLoop(ClipSlot::DodgeLeft));
  const CharacterDescription told = parseText(
      "model = a.fbx\n[clip]\nslot = idle\nfile = b.fbx\nloop = false\n");
  CHECK(told.clips.size() == 1 && !told.clips[0].loop);
}

TEST(character_file_survives_a_save_and_load) {
  const CharacterDescription c = parseText(kGanfaul);
  std::istringstream again(CharacterIO::write(c));
  CHECK(CharacterIO::parse(again, "again.character") == c);
}

TEST(every_slot_has_a_name_that_reads_back) {
  for (int i = 0; i < static_cast<int>(ClipSlot::Count); ++i) {
    const ClipSlot slot = static_cast<ClipSlot>(i);
    CHECK(clipSlotFromName(clipSlotName(slot)) == slot);
  }
  CHECK(!clipSlotFromName("backflip"));
}

TEST(broken_character_files_say_what_and_where) {
  CHECK(errorFor(kGanfaul).empty());
  const std::string unknownSlot =
      errorFor("model = a.fbx\n[clip]\nslot = backflip\nfile = b.fbx\n");
  CHECK(contains(unknownSlot, "line 3"));
  CHECK(contains(unknownSlot, "backflip"));
  CHECK(contains(errorFor("model = a.fbx\n[clip]\nslot = idle\n"), "file"));
  CHECK(contains(errorFor("model = a.fbx\n[clip]\nfile = b.fbx\n"), "slot"));
  CHECK(contains(errorFor("[clip]\nslot = idle\nfile = b.fbx\n"), "model"));
  CHECK(contains(errorFor("model = a.fbx\nspeed = 3\n"), "line 2"));
  CHECK(contains(errorFor("model = a.fbx\n[clip]\nslot = idle\nfile = b.fbx\n"
                          "start = soon\n"),
                 "line 5"));
  CHECK(contains(errorFor("model = a.fbx\n[clip]\nslot = idle\nfile = b.fbx\n"
                          "[clip]\nslot = idle\nfile = c.fbx\n"),
                 "idle"));
}

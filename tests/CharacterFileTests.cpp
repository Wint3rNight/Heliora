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

const std::string kRemy =
    "# Remy, from Mixamo\n"
    "model = Resources/Characters/mixamo/characters/Remy.fbx\n"
    "\n"
    "[clip]\n"
    "slot = idle\n"
    "file = Resources/Characters/mixamo/idle.fbx\n"
    "\n"
    "[clip]\n"
    "slot = jump\n"
    "file = Resources/Characters/mixamo/Standing Jump.fbx\n"
    "start = 0.35\n"
    "end = 1.25\n";

} // namespace

TEST(character_file_names_the_model_and_its_clips) {
  const CharacterDescription c = parseText(kRemy);
  CHECK(c.model == "Resources/Characters/mixamo/characters/Remy.fbx");
  CHECK(c.clips.size() == 2);
  if (c.clips.size() != 2)
    return;
  CHECK(c.clips[0].slot == ClipSlot::Idle);
  CHECK(c.clips[0].file == "Resources/Characters/mixamo/idle.fbx");
  CHECK(c.clips[1].slot == ClipSlot::Jump);
  CHECK_NEAR(c.clips[1].start, 0.35, 1e-6);
  CHECK_NEAR(c.clips[1].end, 1.25, 1e-6);
  CHECK(c.clips[0].end == 0.0f); // not given: the file's end
  CHECK(c.clip(ClipSlot::Jump) == &c.clips[1]);
  CHECK(c.clip(ClipSlot::Roll) == nullptr);
}

TEST(moves_that_repeat_loop_unless_told_otherwise) {
  const CharacterDescription c = parseText(kRemy);
  CHECK(c.clips.size() == 2);
  if (c.clips.size() != 2)
    return;
  CHECK(c.clips[0].loop);  // idle repeats
  CHECK(!c.clips[1].loop); // a jump happens once
  CHECK(defaultLoop(ClipSlot::RunForward));
  CHECK(defaultLoop(ClipSlot::Fall));
  CHECK(!defaultLoop(ClipSlot::Land));
  CHECK(!defaultLoop(ClipSlot::LandHard));
  CHECK(!defaultLoop(ClipSlot::Roll));
  CHECK(!defaultLoop(ClipSlot::DodgeLeft));
  const CharacterDescription told = parseText(
      "model = a.fbx\n[clip]\nslot = idle\nfile = b.fbx\nloop = false\n");
  CHECK(told.clips.size() == 1 && !told.clips[0].loop);
}

TEST(character_file_survives_a_save_and_load) {
  const CharacterDescription c = parseText(kRemy);
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
  CHECK(errorFor(kRemy).empty());
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
                          "end = later\n"),
                 "line 5"));
  const std::string endBeforeStart =
      errorFor("model = a.fbx\n[clip]\nslot = jump\nfile = b.fbx\n"
               "start = 0.5\nend = 0.4\n");
  CHECK(contains(endBeforeStart, "line 2"));
  CHECK(contains(endBeforeStart, "end"));
  CHECK(contains(errorFor("model = a.fbx\n[clip]\nslot = idle\nfile = b.fbx\n"
                          "[clip]\nslot = idle\nfile = c.fbx\n"),
                 "idle"));
}

TEST(comments_survive_a_save_and_load) {
  // The notes above the model line and above each clip stay with them.
  const std::string text = "# Who this is\n# and where from\nmodel = a.fbx\n\n"
                           "# Standing still\n[clip]\nslot = idle\nfile = b.fbx\n\n"
                           "[clip]\nslot = jump\nfile = c.fbx\n";
  const CharacterDescription c = parseText(text);
  CHECK(c.comment == "# Who this is\n# and where from\n");
  CHECK(c.clips.size() == 2);
  if (c.clips.size() != 2)
    return;
  CHECK(c.clips[0].comment == "# Standing still\n");
  CHECK(c.clips[1].comment.empty());
  const std::string written = CharacterIO::write(c);
  CHECK(contains(written, "# Who this is\n# and where from\nmodel = a.fbx\n"));
  CHECK(contains(written, "# Standing still\n[clip]\nslot = idle\n"));
  std::istringstream again(written);
  CHECK(CharacterIO::parse(again, "again.character") == c);
}

TEST(clips_are_set_and_cleared_by_slot) {
  CharacterDescription c = parseText("model = a.fbx\n\n# The jump\n[clip]\n"
                                     "slot = jump\nfile = j.fbx\n");
  CharacterClip roll;
  roll.slot = ClipSlot::Roll;
  roll.file = "r.fbx";
  roll.start = 0.45f;
  roll.loop = false;
  c.setClip(roll);
  CHECK(c.clip(ClipSlot::Roll) && c.clip(ClipSlot::Roll)->file == "r.fbx");
  CharacterClip jump = *c.clip(ClipSlot::Jump);
  jump.file = "j2.fbx";
  jump.comment.clear();
  c.setClip(jump); // replaces it where it was, keeping its notes
  CHECK(c.clips.size() == 2);
  CHECK(c.clips[0].file == "j2.fbx");
  CHECK(c.clips[0].comment == "# The jump\n");
  c.removeClip(ClipSlot::Jump);
  CHECK(!c.clip(ClipSlot::Jump));
  CHECK(c.clips.size() == 1);
}

TEST(tuning_and_variants_are_read_and_written) {
  const std::string text = "model = a.fbx\n\n# How he moves\n[tuning]\nrun_speed = 6\n"
                           "air_speed_limit = true\n\n[clip]\nslot = idle\nfile = b.fbx\n\n"
                           "# For the climb\n[variant arcade]\nair_speed_limit = false\n";
  const CharacterDescription c = parseText(text);
  CHECK(c.tuning.size() == 2);
  if (c.tuning.size() == 2) {
    CHECK(c.tuning[0].key == "run_speed" && c.tuning[0].value == "6");
    CHECK(c.tuning[1].key == "air_speed_limit" && c.tuning[1].value == "true");
  }
  CHECK(c.tuningComment == "# How he moves\n");
  const CharacterVariant *arcade = c.variant("arcade");
  CHECK(arcade != nullptr);
  if (arcade) {
    CHECK(arcade->comment == "# For the climb\n");
    CHECK(arcade->values.size() == 1);
    CHECK(arcade->values.size() == 1 && arcade->values[0].value == "false");
  }
  CHECK(!c.variant("racing"));
  std::istringstream again(CharacterIO::write(c));
  CHECK(CharacterIO::parse(again, "again.character") == c);
}

TEST(broken_tuning_and_variants_say_what_and_where) {
  CHECK(contains(errorFor("model = a.fbx\n[variant]\nx = 1\n"), "line 2"));
  CHECK(contains(errorFor("model = a.fbx\n[variant a]\nx = 1\n[variant a]\ny = 2\n"),
                 "twice"));
  CHECK(contains(errorFor("model = a.fbx\n[tuning]\nx = 1\n[tuning]\ny = 2\n"), "twice"));
  CHECK(contains(errorFor("model = a.fbx\n[tuning]\njust words\n"), "line 3"));
}

TEST(a_clip_can_belong_to_a_gun_stance) {
  const CharacterDescription c = parseText(
      "model = m.fbx\n[clip]\nslot = idle\nfile = idle.fbx\n"
      "[clip]\nslot = idle\nstance = rifle\nfile = rifle idle.fbx\n"
      "[clip]\nslot = aim_idle\nstance = rifle\nfile = rifle aiming.fbx\n"
      "[clip]\nslot = reload\nstance = rifle\nfile = reload.fbx\n");
  CHECK(c.clip(ClipSlot::Idle) && c.clip(ClipSlot::Idle)->file == "idle.fbx");
  const CharacterClip *rifle = c.clip(ClipSlot::Idle, Stance::Rifle);
  CHECK(rifle && rifle->file == "rifle idle.fbx");
  CHECK(!c.clip(ClipSlot::Idle, Stance::Pistol));
  CHECK(c.clip(ClipSlot::AimIdle, Stance::Rifle) != nullptr);
  const CharacterClip *reload = c.clip(ClipSlot::Reload, Stance::Rifle);
  CHECK(reload && !reload->loop); // a reload happens once
  // Written and read back the same.
  CHECK(parseText(CharacterIO::write(c)) == c);
}

TEST(a_slot_twice_in_one_stance_or_an_unknown_stance_is_an_error) {
  CHECK(contains(errorFor("model = m.fbx\n[clip]\nslot = idle\nstance = pistol\nfile = a.fbx\n"
                          "[clip]\nslot = idle\nstance = pistol\nfile = b.fbx\n"),
                 "twice"));
  CHECK(contains(errorFor("model = m.fbx\n[clip]\nslot = idle\nstance = bow\nfile = a.fbx\n"),
                 "stance"));
  CHECK(std::string(stanceName(Stance::Pistol)) == "pistol");
  CHECK(stanceFromName("rifle") == Stance::Rifle);
  // Carried relaxed, not aiming: their own clips.
  CHECK(stanceFromName("rifle_carry") == Stance::RifleCarry);
  CHECK(stanceFromName("pistol_carry") == Stance::PistolCarry);
  CHECK(std::string(stanceName(Stance::RifleCarry)) == "rifle_carry");
  CHECK(!stanceFromName("bow"));
}

TEST(setting_a_stances_clip_leaves_the_unarmed_one_alone) {
  CharacterDescription c = parseText("model = m.fbx\n[clip]\nslot = idle\nfile = idle.fbx\n");
  CharacterClip pistolIdle;
  pistolIdle.slot = ClipSlot::Idle;
  pistolIdle.stance = Stance::Pistol;
  pistolIdle.file = "pistol idle.fbx";
  c.setClip(pistolIdle);
  CHECK(c.clips.size() == 2);
  CHECK(c.clip(ClipSlot::Idle)->file == "idle.fbx");
  c.removeClip(ClipSlot::Idle, Stance::Pistol);
  CHECK(c.clips.size() == 1);
  CHECK(c.clip(ClipSlot::Idle) != nullptr);
}

TEST(death_and_hit_slots_read_as_one_shots) {
  const CharacterDescription d = parseText(
      "model = m.fbx\n[clip]\nslot = death_front\nfile = a.fbx\n[clip]\nslot = hit\nfile = b.fbx\n");
  CHECK(d.clips.size() == 2);
  if (d.clips.size() == 2) {
    CHECK(d.clips[0].slot == ClipSlot::DeathFront && !d.clips[0].loop);
    CHECK(d.clips[1].slot == ClipSlot::Hit && !d.clips[1].loop);
  }
}

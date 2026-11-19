#include "CharacterSettings.h"
#include "TestRunner.h"

#include <sstream>

namespace {

CharacterDescription parseText(const std::string &text) {
  std::istringstream in(text);
  return CharacterIO::parse(in, "test.character");
}

const char *kFile = "model = a.fbx\n[tuning]\nrun_speed = 6\nturn_grip = 20\n"
                    "anim_fall_speed = 9\n[variant arcade]\nair_speed_limit = false\n"
                    "turn_grip = 30\n";

} // namespace

TEST(a_character_files_tuning_is_applied) {
  const CharacterDescription c = parseText(kFile);
  CharacterTuning movement;
  AnimatorTuning animation;
  applyCharacterTuning(c, "", movement, animation);
  CHECK_NEAR(movement.runSpeed, 6.0, 1e-6);
  CHECK_NEAR(movement.turnGrip, 20.0, 1e-6);
  CHECK_NEAR(animation.fallSpeed, 9.0, 1e-6);
  CHECK(movement.airSpeedLimit);                       // not in the file: the default
  CHECK_NEAR(movement.sprintSpeed, CharacterTuning{}.sprintSpeed, 1e-6);
}

TEST(a_variant_overrides_only_what_it_names) {
  const CharacterDescription c = parseText(kFile);
  CharacterTuning movement;
  AnimatorTuning animation;
  applyCharacterTuning(c, "arcade", movement, animation);
  CHECK(!movement.airSpeedLimit);
  CHECK_NEAR(movement.turnGrip, 30.0, 1e-6);
  CHECK_NEAR(movement.runSpeed, 6.0, 1e-6); // the base's
  // An unknown variant is the base.
  CharacterTuning plain;
  applyCharacterTuning(c, "racing", plain, animation);
  CHECK(plain.airSpeedLimit);
  CHECK_NEAR(plain.turnGrip, 20.0, 1e-6);
}

TEST(tuning_stored_in_a_character_reads_back) {
  CharacterDescription c = parseText(kFile);
  CharacterTuning movement;
  AnimatorTuning animation;
  movement.jumpHeight = 1.5f;
  movement.airSpeedLimit = false;
  animation.playbackRate = 1.25f;
  storeCharacterTuning(c, movement, animation);
  CharacterTuning back;
  AnimatorTuning backAnimation;
  applyCharacterTuning(c, "", back, backAnimation);
  CHECK_NEAR(back.jumpHeight, 1.5, 1e-6);
  CHECK(!back.airSpeedLimit);
  CHECK_NEAR(backAnimation.playbackRate, 1.25, 1e-6);
  // Every key is written, and the variants are left alone.
  CHECK(c.tuning.size() == characterTuningBindings(movement, animation).size());
  CHECK(c.variant("arcade") && c.variant("arcade")->values.size() == 2);
}

TEST(the_aim_animation_timing_is_saved_with_the_character) {
  CharacterDescription c = parseText("model = a.fbx\n[tuning]\nanim_fade_aim = 0.3\n");
  CharacterTuning movement;
  AnimatorTuning animation;
  applyCharacterTuning(c, "", movement, animation);
  CHECK_NEAR(animation.aimFade, 0.3, 1e-4);
  animation.aimLean = 0.5f;
  storeCharacterTuning(c, movement, animation);
  AnimatorTuning back;
  applyCharacterTuning(c, "", movement, back);
  CHECK_NEAR(back.aimLean, 0.5, 1e-4);
  CHECK(back == animation);
}

TEST(the_foot_ik_tuning_is_saved_with_the_character) {
  CharacterDescription c =
      parseText("model = a.fbx\n[tuning]\nik_enabled = false\nik_foot_reach = 0.3\n"
                "ik_stride_warp = false\n");
  CharacterTuning movement;
  AnimatorTuning animation;
  applyCharacterTuning(c, "", movement, animation);
  CHECK(!animation.footIk.enabled);
  CHECK(!animation.footIk.strideWarp);
  CHECK_NEAR(animation.footIk.reach, 0.3, 1e-4);
  animation.footIk.tilt = 0.4f;
  animation.footIk.halfLife = 0.2f;
  storeCharacterTuning(c, movement, animation);
  AnimatorTuning back;
  applyCharacterTuning(c, "", movement, back);
  CHECK(back.footIk == animation.footIk);
}

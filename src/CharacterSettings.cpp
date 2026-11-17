#include "CharacterSettings.h"

#include <spdlog/spdlog.h>

namespace {
constexpr std::string_view kSection = "tuning";

SettingsFile asSettings(const std::vector<CharacterSetting> &values) {
  SettingsFile file;
  for (const CharacterSetting &s : values)
    file.setString(kSection, s.key, s.value);
  return file;
}
} // namespace

std::vector<SettingBinding> characterTuningBindings(CharacterTuning &c, AnimatorTuning &a) {
  // Angles in radians, as the code keeps them; the tab shows degrees.
  return {
      {kSection, "run_speed", &c.runSpeed, 0.5, 20},
      {kSection, "sprint_speed", &c.sprintSpeed, 0.5, 30},
      {kSection, "crouch_speed", &c.crouchSpeed, 0.2, 10},
      {kSection, "ground_acceleration", &c.groundAcceleration, 1, 200},
      {kSection, "ground_braking", &c.groundBraking, 1, 200},
      {kSection, "turn_grip", &c.turnGrip, 1, 100},
      {kSection, "turn_rate", &c.turnRate, 0.5, 63},
      {kSection, "aim_acceleration", &c.aimAcceleration, 1, 200},
      {kSection, "air_acceleration", &c.airAcceleration, 0, 100},
      {kSection, "air_speed_limit", &c.airSpeedLimit},
      {kSection, "jump_height", &c.jumpHeight, 0.1, 5},
      {kSection, "coyote_time", &c.coyoteTime, 0, 0.5},
      {kSection, "jump_buffer", &c.jumpBuffer, 0, 0.5},
      {kSection, "dodge_scale", &c.dodgeScale, 0.1, 5},
      {kSection, "dodge_distance", &c.dodgeDistance, 0, 20},
      {kSection, "dodge_time", &c.dodgeTime, 0.05, 1},
      {kSection, "dodge_cooldown", &c.dodgeCooldown, 0, 5},
      {kSection, "max_slope", &c.maxSlope, 0.08, 1.4},
      {kSection, "step_up", &c.stepUp, 0, 1},
      {kSection, "stick_down", &c.stickDown, 0, 2},
      {kSection, "push_strength", &c.pushStrength, 0, 5000},
      {kSection, "radius", &c.radius, 0.1, 1},
      {kSection, "height", &c.height, 0.5, 3},
      {kSection, "crouch_height", &c.crouchHeight, 0.3, 3},
      {kSection, "mass", &c.mass, 1, 1000},
      {kSection, "anim_playback_rate", &a.playbackRate, 0.25, 3},
      {kSection, "anim_fade_moving", &a.locomotionFade, 0, 1},
      {kSection, "anim_fade_crouch", &a.crouchFade, 0, 1},
      {kSection, "anim_fade_jump", &a.jumpFade, 0, 1},
      {kSection, "anim_fade_fall", &a.fallFade, 0, 1},
      {kSection, "anim_fade_land", &a.landFade, 0, 1},
      {kSection, "anim_fade_dodge", &a.dodgeFade, 0, 1},
      {kSection, "anim_fall_speed", &a.fallSpeed, 1, 30},
      {kSection, "anim_air_pose_delay", &a.fallDelay, 0, 1},
      {kSection, "anim_landing_after", &a.landAirTime, 0, 2},
      {kSection, "anim_shortest_landing", &a.landMinTime, 0, 1},
  };
}

void applyCharacterTuning(const CharacterDescription &character, std::string_view variant,
                          CharacterTuning &movement, AnimatorTuning &animation) {
  const std::vector<SettingBinding> bindings = characterTuningBindings(movement, animation);
  readSettings(asSettings(character.tuning), bindings, kSection);
  applyCharacterVariant(character, variant, movement, animation);
}

void applyCharacterVariant(const CharacterDescription &character, std::string_view variant,
                           CharacterTuning &movement, AnimatorTuning &animation) {
  if (variant.empty())
    return;
  if (const CharacterVariant *v = character.variant(variant))
    readSettings(asSettings(v->values), characterTuningBindings(movement, animation), kSection);
  else
    spdlog::warn("The character has no variant '{}', using its own tuning", variant);
}

void storeCharacterTuning(CharacterDescription &character, const CharacterTuning &movement,
                          const AnimatorTuning &animation) {
  CharacterTuning m = movement;
  AnimatorTuning a = animation;
  const std::vector<SettingBinding> bindings = characterTuningBindings(m, a);
  SettingsFile file;
  writeSettings(file, bindings, kSection);
  character.tuning.clear();
  for (const SettingBinding &b : bindings)
    if (const std::optional<std::string> value = file.text(kSection, b.key))
      character.tuning.push_back({std::string(b.key), *value});
}

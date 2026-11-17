#pragma once

#include "CharacterAnimator.h"
#include "CharacterController.h"
#include "CharacterFile.h"
#include "SettingsFile.h"

#include <string_view>
#include <vector>

// A character's tuning in its .character file: the [tuning] keys bound to
// the movement and animation structs, read with a variant's overrides on
// top, and written back. The same table the settings file uses, so values
// are typed, clamped and logged the same way.

// Every [tuning] key, in the order they're written.
std::vector<SettingBinding> characterTuningBindings(CharacterTuning &movement,
                                                    AnimatorTuning &animation);

// Sets the structs from the file's [tuning], then from the named variant's
// overrides. Keys the file doesn't give keep the structs' values; an
// unknown variant is logged and left out.
void applyCharacterTuning(const CharacterDescription &character, std::string_view variant,
                          CharacterTuning &movement, AnimatorTuning &animation);

// Only the named variant's overrides, over whatever the structs hold (the
// Character tab's unsaved values, say). Empty applies nothing.
void applyCharacterVariant(const CharacterDescription &character, std::string_view variant,
                           CharacterTuning &movement, AnimatorTuning &animation);

// Writes every key's value into the file's [tuning], keeping its comment
// and the variants.
void storeCharacterTuning(CharacterDescription &character, const CharacterTuning &movement,
                          const AnimatorTuning &animation);

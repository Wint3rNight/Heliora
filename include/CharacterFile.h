#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// A .character file: which model a character is, and which clip plays each
// of its moves. The scene files' format: `key = value` lines, `[clip]`
// sections, `#` comments. The model and clips are files on disk, read when
// the character loads.

// The moves a clip can play. The directional ones are relative to where the
// character faces.
enum class ClipSlot : uint8_t {
  Idle,
  WalkForward,
  WalkBack,
  WalkLeft,
  WalkRight,
  RunForward,
  RunBack,
  RunLeft,
  RunRight,
  Sprint,
  Jump,
  Fall,
  Land,
  LandHard, // after a real fall; the plain landing without it
  CrouchIdle,
  CrouchForward,
  CrouchBack,
  CrouchLeft,
  CrouchRight,
  Roll,
  DodgeForward,
  DodgeBack,
  DodgeLeft,
  DodgeRight,
  AimIdle, // standing still while aiming, in a gun stance ("idle aiming")
  Reload,  // on the upper body, over whatever the legs are doing
  // Hit: a flinch on the upper body. Deaths, by where the shot came from
  // (front, back, the right; a head shot; crouched), played once and held.
  Hit,
  DeathFront,
  DeathBack,
  DeathRight,
  DeathHead,
  DeathCrouched,
  Throw, // something thrown overhand, on the upper body
  // The powers, on the upper body: the lift runs into the hold.
  PowerLift,
  PowerHold,
  PowerThrow,
  PowerBurst,
  Count
};

// Which clips a character moves with: its own, or holding a pistol or a
// rifle (step 1.6). A gun stance plays its own clip for a move, else the
// unarmed one.
// A gun's stance is for aiming; its carry stance for when it's held
// relaxed (moves it has no clip for are the unarmed ones).
enum class Stance : uint8_t { Unarmed, Pistol, Rifle, PistolCarry, RifleCarry, Count };
const char *stanceName(Stance stance);
std::optional<Stance> stanceFromName(std::string_view name);

// The name a file uses, like "run_forward", and back.
const char *clipSlotName(ClipSlot slot);
std::optional<ClipSlot> clipSlotFromName(std::string_view name);
// Whether a move repeats: idle, walking, running, crouching and falling do;
// a jump, a landing, a roll and the dodges happen once.
bool defaultLoop(ClipSlot slot);

struct CharacterClip {
  ClipSlot slot = ClipSlot::Idle;
  Stance stance = Stance::Unarmed;
  std::string file;
  float start = 0.0f; // seconds into the file to start at (past a wind-up)
  // Seconds into the file to stop at, after start; 0 is the file's end. A
  // jump stops before its feet come down, a landing begins at that moment.
  float end = 0.0f;
  bool loop = true;
  std::string comment; // the `#` lines above its [clip], each ending in \n
  bool operator==(const CharacterClip &) const = default;
};

// One `key = value` of a [tuning] or [variant] section, as written; what
// the keys mean is CharacterSettings.h's business.
struct CharacterSetting {
  std::string key;
  std::string value;
  bool operator==(const CharacterSetting &) const = default;
};

// A named set of overrides for [tuning] ("rage"), which a scene's player
// start can ask for.
struct CharacterVariant {
  std::string name;
  std::string comment;
  std::vector<CharacterSetting> values;
  bool operator==(const CharacterVariant &) const = default;
};

struct CharacterDescription {
  std::string comment; // the `#` lines above the model line
  std::string model;
  std::string tuningComment;
  std::vector<CharacterSetting> tuning; // [tuning], in file order
  std::vector<CharacterVariant> variants;
  const CharacterVariant *variant(std::string_view name) const;
  std::vector<CharacterClip> clips; // one per slot and stance at most
  // The clip for a slot in a stance, or nothing.
  const CharacterClip *clip(ClipSlot slot, Stance stance = Stance::Unarmed) const;
  // Gives clip's slot (in its stance) this clip: in the old one's place,
  // keeping its notes if the new one has none, or at the end.
  void setClip(const CharacterClip &clip);
  void removeClip(ClipSlot slot, Stance stance = Stance::Unarmed);
  bool operator==(const CharacterDescription &) const = default;
};

namespace CharacterIO {
// Throws std::runtime_error naming the source and line for anything it
// can't read: an unknown key or slot, a clip without a slot or file, a slot
// given twice, no model.
CharacterDescription parse(std::istream &in, const std::string &sourceName);
CharacterDescription loadFromFile(const std::string &path);
// Text that parses back to the same description; keys at their default
// (loop matching the slot's, start and end 0) are left out.
std::string write(const CharacterDescription &character);
} // namespace CharacterIO

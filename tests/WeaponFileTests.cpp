#include "WeaponFile.h"
#include "TestRunner.h"

#include <sstream>

namespace {

SettingsFile parsed(const std::string &text) {
  std::istringstream in(text);
  return SettingsFile::parse(in, "test.weapon");
}

const char *kAkmText = R"([gun]
name = AKM
model = Resources/Models/Weapons/akm/akm.parts
[round]
bullet_mass = 0.0079
muzzle_speed = 715
drag = G7
coefficient = 0.15
zero_range = 100
tracer_every = 5
[penetration]
wood_depth = 0.35
wood_speed = 580
wood_depth2 = 0.25
wood_speed2 = 326
stone_depth = 0.135
stone_speed = 623
[action]
magazine = 30
rate = 600
automatic = true
[spread]
first_up = 0.59
first_side = 0.44
follow_up = 1.19
follow_side = 1.63
[reload]
partial = 2.5
empty = 3.2
[carry]
spare_magazines = 4
[sounds]
shot = Resources/Sounds/guns/akm_shot_near.ogg
shot_far = Resources/Sounds/guns/akm_shot_mid.ogg
loudness = 160
)";

} // namespace

TEST(a_weapon_file_gives_every_number_it_holds) {
  const WeaponSpec akm = weaponFromSettings(parsed(kAkmText));
  CHECK(akm.name == "AKM");
  CHECK(akm.model == "Resources/Models/Weapons/akm/akm.parts");
  CHECK_NEAR(akm.bulletMass, 0.0079, 1e-6);
  CHECK_NEAR(akm.muzzleSpeed, 715.0, 1e-4);
  CHECK(akm.drag == DragModel::G7);
  CHECK_NEAR(akm.coefficient, 0.15, 1e-6);
  CHECK(akm.tracerEvery == 5);
  CHECK(akm.magazine == 30);
  CHECK(akm.automatic);
  CHECK_NEAR(akm.firstSpread.y, 0.59, 1e-6);
  CHECK_NEAR(akm.followSpread.x, 1.63, 1e-6);
  CHECK_NEAR(akm.emptyReload, 3.2, 1e-6);
  CHECK(akm.spareMagazines == 4);
  CHECK(akm.shotSound == "Resources/Sounds/guns/akm_shot_near.ogg");
  CHECK(akm.farShotSound == "Resources/Sounds/guns/akm_shot_mid.ogg");
  CHECK_NEAR(akm.loudness, 160.0, 1e-4);
  const MatterPenetration &wood = akm.penetration[size_t(Matter::Wood)];
  CHECK_NEAR(wood.depth2, 0.25, 1e-6);
  CHECK_NEAR(wood.speed2, 326.0, 1e-4);
}

TEST(a_weapon_file_turns_into_its_round) {
  const WeaponSpec akm = weaponFromSettings(parsed(kAkmText));
  const Round round = roundOf(akm);
  CHECK(round.projectile.model == DragModel::G7);
  CHECK_NEAR(round.mass, 0.0079, 1e-6);
  // Pine through both of NSD's depths; brick through its one, with pine's
  // b / a; nothing given for rubber, so it doesn't hold the bullet back.
  const Resistance &pine = round.resistance[size_t(Matter::Wood)];
  CHECK_NEAR(stoppingDepth(pine, round.mass, 580.0f), 0.35, 1e-3);
  CHECK_NEAR(stoppingDepth(pine, round.mass, 326.0f), 0.25, 1e-3);
  const Resistance &brick = round.resistance[size_t(Matter::Stone)];
  CHECK_NEAR(stoppingDepth(brick, round.mass, 623.0f), 0.135, 1e-3);
  CHECK_NEAR(ratioOf(brick), ratioOf(pine), 1e-9);
  CHECK(round.resistance[size_t(Matter::Rubber)] == Resistance{});
}

TEST(a_weapon_written_and_read_back_is_the_same) {
  WeaponSpec spec = weaponFromSettings(parsed(kAkmText));
  spec.locksOpen = true;
  spec.hipSpread = 12.0f;
  spec.model = "Resources/Models/Weapons/x/x.parts";
  spec.shotSound = "a.ogg";
  spec.farShotSound = "b.ogg";
  spec.loudness = 150.0f;
  spec.drag = DragModel::G1;
  SettingsFile file;
  weaponToSettings(spec, file);
  const WeaponSpec back = weaponFromSettings(parsed(file.write()));
  CHECK(back == spec);
}

TEST(a_value_missing_or_unknown_keeps_the_default) {
  const WeaponSpec spec = weaponFromSettings(parsed("[gun]\nname = X\n[round]\ndrag = G9\n"));
  CHECK(spec.name == "X");
  CHECK(spec.drag == WeaponSpec{}.drag);
  CHECK(spec.magazine == WeaponSpec{}.magazine);
}

TEST(the_shipped_weapons_read_as_their_sources_say) {
  const std::string folder = std::string(HELIORA_SOURCE_DIR) + "/Resources/Weapons/";
  const WeaponSpec akm = weaponFromSettings(SettingsFile::load(folder + "akm.weapon"));
  CHECK(akm.name == "AKM");
  CHECK(akm.magazine == 30);
  CHECK(akm.automatic && !akm.locksOpen);
  CHECK_NEAR(akm.muzzleSpeed, 715.0, 1e-3);
  CHECK_NEAR(akm.followSpread.x, 1.63, 1e-5);
  const Round round = roundOf(akm);
  CHECK_NEAR(stoppingDepth(round.resistance[size_t(Matter::Wood)], round.mass, 580.0f), 0.35, 1e-3);
  CHECK_NEAR(stoppingDepth(round.resistance[size_t(Matter::Stone)], round.mass, 623.0f), 0.135,
             1e-3);
  for (const Resistance &r : round.resistance)
    CHECK(r.a > 0.0f); // every material holds a bullet back
  const WeaponSpec p38 = weaponFromSettings(SettingsFile::load(folder + "p38.weapon"));
  CHECK(p38.name == "Walther P38");
  CHECK(p38.magazine == 8);
  CHECK(!p38.automatic && p38.locksOpen);
  CHECK(p38.drag == DragModel::G1);
  CHECK_NEAR(p38.muzzleSpeed, 340.0, 1e-3);
  for (const Resistance &r : roundOf(p38).resistance)
    CHECK(r.a > 0.0f);
  // The kick you see: a pistol flips higher than a shouldered rifle.
  CHECK(akm.kickBack > 0.0f && akm.kickUp > 0.0f && akm.kickPeak > 0.0f);
  CHECK(p38.kickUp > akm.kickUp);
}

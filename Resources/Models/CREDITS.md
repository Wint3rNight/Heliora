# Models: where they come from

The folders below are gitignored (about 700 MB); get them again from these
pages. Everything here is free to use in a game. CC0 needs no credit; CC BY
4.0 needs the credit line given, in the game's credits.

## Weapons (`Weapons/`)

| Folder | What | Source | Licence |
| --- | --- | --- | --- |
| `service_pistol` | Cold War service pistol, two variants, separate slide, hammer, trigger, mags, and a bullet; 4k | https://polyhaven.com/a/service_pistol | CC0 |
| `bolt_action_rifle_7_62` | Scoped bolt-action rifle, separate bolt and scope, and a 7.62x54 bullet; 4k | https://polyhaven.com/a/bolt_action_rifle_7_62 | CC0 |
| `ammo_box` | Military ammo can; 2k | https://polyhaven.com/a/ammo_box | CC0 |
| `stein_classic/` | Free Classic Weapons Pack by Stein Games: AK47, 1911, MP5, M14, M700, Mac10, Tec9, lever action; skinned FBX with bones for the magazine, slide or bolt and trigger; 2k, Unreal-style packed maps | https://stein-indie.itch.io/classic-weapons-pack | CC0 (`license.txt`) |
| `akm/` | The game's AKM: `stein_classic/AK47` through `tools/weapon_convert/convert_guns.sh` (metres, 880 mm, split into frame, magazine, bolt, selector and trigger, sockets, the maps repaired for glTF) | made here from the Stein pack | CC0 |
| `p38/` | The game's P38: `service_pistol` variant A through `tools/weapon_convert/convert_guns.sh` (215 mm, frame, slide, hammer, trigger and magazine) | made here from Poly Haven's | CC0 |
| `rpg7/javadbayat/` | RPG-7 with a rocket, two meshes; six PBR texture sets (wood, metals, plastic, the rocket), but the FBX has one material, so which faces use which set is lost | "RPG 7 ( Free Model )" by javadbayat, https://sketchfab.com/3d-models/rpg-7-free-model-99c0af8a1803490a86edc48cd3bfc700 | CC BY 4.0: credit the author in the game's credits |
| `rpg7/theparazit/` | RPG-7 with rockets (two texture sets, 4k), skinned, plus first-person arm animations on Unreal's mannequin arms (show, hide, idle, shoot, reload, walk); no full-body clips | "RPG-7 Animations" by TheParaziT, https://sketchfab.com/3d-models/rpg-7-animations-f2bef085e0164a34a4b090c73dcdc3f5 | CC BY 4.0: credit the author in the game's credits |
| `rpg7/` (its files, not the folders) | The game's RPG-7: `rpg7/theparazit` through `tools/weapon_convert/convert_guns.sh` (950 mm, the launcher and its rocket as two parts with their own texture sets, sockets measured on the parts; the arms left out) | made here from TheParaziT's | CC BY 4.0: credit "RPG-7 Animations" by TheParaziT in the game's credits |

## Vehicles (`Vehicles/`)

| Folder | What | Source | Licence |
| --- | --- | --- | --- |
| `CarConcept` | Concept car with interior, separate wheels, doors, hood and hatch | https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CarConcept | CC BY 4.0 |
| `covered_car` | Parked car under a tarp, separate wheels | https://polyhaven.com/a/covered_car | CC0 |

Credit for `CarConcept`: "Car Concept" model and textures by Eric Chadwick,
© 2024 Darmstadt Graphics Group GmbH, licensed CC BY 4.0. Based on a CC0
model by Unity Fan. Its Khronos and 3D Commerce logos are trademarks outside
the licence: replace those textures before shipping.

## Props (`Props/`)

From Poly Haven (https://polyhaven.com/a/<folder>), CC0, glTF with 2k
textures: `street_lamp_01`, `street_lamp_02`, `concrete_road_barrier`,
`concrete_road_barrier_02`, `fire_hydrant`, `metal_trash_can`, `Barrel_01`,
`Barrel_02`, `barrel_03`, `wooden_crate_01`, `wooden_crate_02`,
`cardboard_box_01`, `plastic_crate_01`, `modular_street_seating`,
`painted_wooden_bench`, `old_tyre`, `trashbag`, `metal_jerrycan`,
`propane_tank`, `utility_box_01`, `water_manhole_cover`,
`modular_electricity_poles`, `modular_chainlink_fence`.

| Folder | What | Source | Licence |
| --- | --- | --- | --- |
| `pallet_1` | Wooden pallet, FBX, 2k | https://www.sharetextures.com/models/equipments/pallet-1 | CC0 |
| `bin_1` | Outdoor litter bin, FBX, 2k | https://www.sharetextures.com/models/equipments/3d-bin-1 | CC0 |
| `gazebo_Dumpster`, `gazebo_Construction_Cone`, `gazebo_Stop_Sign` | Low-poly placeholders, Collada, not PBR | https://app.gazebosim.org/OpenRobotics/fuel/models | CC0 |

## Buildings (`Buildings/`)

From Poly Haven (https://polyhaven.com/a/<folder>), CC0, glTF with 2k
textures: `modular_urban_apartments_facade`, `modular_factory_facade`,
`rollershutter_door`, `modular_fire_escape`, `modular_metal_gutter`,
`exterior_aircon_unit`.

## Already in git

`boulder_01`: https://polyhaven.com/a/boulder_01, CC0.

#!/bin/sh
# Makes the game's guns from the downloaded models (step 1.6). The results,
# like the downloads, are in Resources/Models/Weapons/, which git ignores,
# so run this after downloading them (Resources/Models/CREDITS.md):
#   cmake --build build --target weapon_convert && tools/weapon_convert/convert_guns.sh
set -e
cd "$(dirname "$0")/../.."
W=Resources/Models/Weapons
TOOL=build/weapon_convert

# The AKM: Stein Games' AK47, a skinned FBX in centimetres. Its bones are the
# moving parts; MAGAZINE_EXTRA is a spare magazine for the hand, left out.
S=$W/stein_classic/AK47
$TOOL $S/SKM_AK47.fbx $W/akm akm --length 0.88 --origin TRIGGER \
  --part magazine=MAGAZINE,BULLET --part bolt=SLIDE --part selector=SELECTOR \
  --part trigger=TRIGGER --drop MAGAZINE_EXTRA \
  --socket muzzle=FX_FIRE --socket ejection=FX_CASING --socket magazine=MAGAZINE \
  --socket bolt=SLIDE --socket rear_sight=ATTACH_SCOPE --socket front_sight=IRONSIGHT \
  --albedo $S/T_AK47_C.png --normal $S/T_AK47_N.png --flip-green \
  --packed $S/T_AK47_RMAO.png --packed-order rma --max-texture 2048

# The P38: Poly Haven's service pistol, in metres along +x, two variants and
# its magazines laid out beside it. Variant A, the loaded magazine moved back
# into the grip; the muzzle at the barrel's tip.
S=$W/service_pistol
$TOOL $S/service_pistol_4k.gltf $W/p38 p38 --length 0.215 --forward +x --up +y \
  --origin service_pistol_trigger_a --default none \
  --part frame=service_pistol_pistol_a --part slide=service_pistol_slide_a \
  --part hammer=service_pistol_hammer_a --part trigger=service_pistol_trigger_a \
  --part magazine=service_pistol_magazine_loaded \
  --move service_pistol_magazine_loaded=0.1,0,0 \
  --socket magazine=service_pistol_magazine_loaded --socket-at muzzle=0.174,0.058,0 \
  --albedo $S/textures/service_pistol_diff_4k.jpg --normal $S/textures/service_pistol_nor_gl_4k.jpg \
  --packed $S/textures/service_pistol_arm_4k.jpg --packed-order orm --max-texture 2048

# The RPG-7: TheParaziT's (Sketchfab, CC BY 4.0), a skinned FBX with
# first-person arms and two texture sets (the launcher's and the rocket's).
# Its bones are posed holding it up to the camera, so the parts and sockets
# come from where the mesh was bound. 950 mm long without the rocket's
# warhead out of the muzzle. The sockets were measured on the converted
# parts: the muzzle and the venturi's back end on the tube's axis, the
# tube's underside where it sits on the shoulder, the iron sights' top, the
# two grips (the trigger's and the one behind it) and the loaded rocket's
# nose.
S=$W/rpg7/theparazit
$TOOL $S/RPG_7.fbx $W/rpg7 rpg7 --length 0.95 --length-of frame --origin trigger --default none \
  --part frame=body,hammer,trigger,missile_release --part rocket=missile \
  --socket-gun muzzle=0,0.037,0.165 --socket-gun venturi=0,0.037,-0.79 --socket-gun nose=0,0.037,0.513 \
  --socket-gun shoulder=0,0.0,-0.40 --socket-gun sight=0.013,0.132,-0.18 \
  --socket-gun grip=0,-0.03,-0.04 --socket-gun rear_grip=0,-0.03,-0.21 \
  --albedo $S/textures/RPG-7_low_rpg-7_BaseColor.png --normal $S/textures/RPG-7_low_rpg-7_Normal.png \
  --flip-green --roughness $S/textures/RPG-7_low_rpg-7_Roughness.png \
  --metal $S/textures/RPG-7_low_rpg-7_Metallic.png \
  --for rocket --albedo $S/textures/RPG-7_low_rockets_BaseColor.png \
  --normal $S/textures/RPG-7_low_rockets_Normal.png --flip-green \
  --roughness $S/textures/RPG-7_low_rockets_Roughness.png \
  --metal $S/textures/RPG-7_low_rockets_Metallic.png --max-texture 2048

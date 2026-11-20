#pragma once

// What a character carries and has in its hands (step 1.6): a pistol in a
// holster on the right hip and one long gun on its back (the user: "One
// long gun"), drawn with 1 and 2 and put away with H, and a rocket
// launcher on the back too (step 1.7), with 3. Drawing takes time,
// and swapping puts the first gun away before the next comes out.

enum class Holdable { None, Pistol, Long, Launcher };

// Seconds to draw and to put away (est.: a pistol from the holster to the
// first aimed shot takes 1.5 to 1.8 s, of which getting it into the hand is
// about the first 0.6; a rifle off the back takes longer).
struct LoadoutTiming {
  float pistolDraw = 0.6f;
  float pistolHolster = 0.7f;
  float longDraw = 1.2f;
  float longHolster = 1.3f;
  // A rocket launcher off the back, heavier and longer still (est.).
  float launcherDraw = 1.5f;
  float launcherHolster = 1.5f;
};

class Loadout {
public:
  explicit Loadout(LoadoutTiming timing = {});
  // What it carries; anything in its hands that it no longer carries goes
  // at once.
  void give(bool pistol, bool longGun, bool launcher = false);
  bool has(Holdable which) const;
  // 1, 2 or H: the gun wanted in the hands (None to put it away). One it
  // doesn't carry is ignored.
  void select(Holdable which);
  void update(float dt);

  // The gun in the hands, coming out or going away; None with both put away.
  Holdable inHand() const { return current; }
  // 0 in its holster (or on the back) to 1 in the hand.
  float drawn() const { return amount; }
  bool ready() const { return current != Holdable::None && amount >= 1.0f && wanted == current; }
  bool busy() const { return current != Holdable::None && !ready(); }

private:
  float drawTime(Holdable which) const;
  float holsterTime(Holdable which) const;

  LoadoutTiming timing;
  bool pistol = false, longGun = false, launcher = false;
  Holdable current = Holdable::None; // in the hands
  Holdable wanted = Holdable::None;
  float amount = 0.0f;
};

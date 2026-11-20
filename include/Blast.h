#pragma once

#include <glm/glm.hpp>

// Explosions (step 1.7): the blast and the fragments, for the game to push
// bodies, break pieces and hurt people with. No physics here, so it's
// tested on its own.

enum class Explosive { Tnt, CompB, C4 };

// Its mass times this is the TNT that makes the same blast. Comp B 1.32
// (the most cited value; sources run 1.10 to 1.36). C-4 1.20 below
// 1.38 MPa and 1.37 above, 1.19 for impulse (UFC 3-340-02, via Bogosian
// 2016). pressure is the peak overpressure it's asked for, Pa.
float tntEquivalent(Explosive explosive, float pressure, bool impulse);

// Hopkinson-Cranz scaled distance, m/kg^(1/3): range over the cube root of
// the TNT mass.
float scaledDistance(float range, float tntKg);

// Peak side-on overpressure, Pa, of a free-air TNT burst at scaled distance
// z (Kinney and Graham, Explosive Shocks in Air, 1985), at sea level.
float peakOverpressure(float z);
// Its positive phase: how long it lasts (s) and its impulse (Pa s) at
// scaled distance z for tntKg of TNT. Kinney and Graham's forms as
// remembered; still to check against a published table (they agree with
// the overpressure: the impulse is near the triangle under the curve).
float positiveDuration(float z, float tntKg);
float positiveImpulse(float z, float tntKg);
// What the blast reflected off a wall facing it gives, over the side-on:
// Rankine-Hugoniot's ratio for a normal reflection in air (gamma 1.4).
float reflectionFactor(float overpressure);

// How fast a blast's impulse (Pa s) throws a body: dv = Cd A I / m, the
// drag form (Glasstone and Dolan), for area A (m^2) facing it and mass m
// (kg); 0 for a body that doesn't move (m 0).
float blastPush(float dragCoefficient, float area, float impulse, float mass);

// What blasts do to people (IOM 2014): eardrums burst, lungs are damaged,
// and it can kill. Pa.
constexpr float kEardrumRupture = 34e3f;
constexpr float kLungDamage = 207e3f;
constexpr float kBlastLethal = 414e3f;

// A cased charge: its explosive and case (Comp B, for Gurney's constant).
struct CasedCharge {
  Explosive explosive = Explosive::CompB;
  float charge = 0.184f;   // kg of explosive
  float caseMass = 0.14f;  // kg of steel that turns into fragments
  float casualtyRadius = 15.0f; // m, documented: half the people standing there hit
};
// The M67 grenade: 184 g of Comp B; 397 g in all with its fuze; its steel
// body about 140 g (est.: the total less the filler and about 73 g of fuze).
// Its casualty radius 15 m (FM 3-23.30).
constexpr CasedCharge kM67{Explosive::CompB, 0.184f, 0.14f, 15.0f};
// The M112 demolition block: 0.57 kg of C-4, 2 by 1 by 11 in, no case so
// no fragments.
constexpr CasedCharge kM112{Explosive::C4, 0.57f, 0.0f, 0.0f};

// Gurney's sqrt(2E) for Comp B, m/s (est. 2.70 km/s, the usual figure).
constexpr float kGurneyCompB = 2700.0f;
// A sphere's case thrown by its charge: v = sqrt(2E) / sqrt(M/C + 3/5).
float gurneySphere(float charge, float caseMass, float gurney);

// A fragment's speed after flying range m from start speed v0: slowed by
// air drag on a tumbling steel chip (its average presented area a cube's,
// 1.5 (m / 7850)^(2/3); drag coefficient 1, est.).
float fragmentSpeed(float v0, float mass, float range);
// The classic lethal fragment: 79 J (58 ft lbf).
constexpr float kCasualtyEnergy = 79.0f;

// How a case breaks: this many fragments of this mass each, leaving at
// this speed.
struct FragmentSpread {
  int count = 0;
  float mass = 0.0f;  // kg
  float speed = 0.0f; // m/s at the start
};
// A standing person's presented area to fragments, m^2 (est.).
constexpr float kPersonArea = 0.5f;
// Fitted so a person standing at the charge's casualty radius is hit half
// the time (each fragment flying straight, spread evenly over the sphere).
// With the M67's real case mass that means about 3900 fragments of 36 mg,
// which carry less than 79 J by 15 m: no single size gives both, so every
// hit counts toward the fit, and the energy each carries says how bad it is.
FragmentSpread fitFragments(const CasedCharge &charge);
// The chance a person standing range m away is hit at least once.
float casualtyChance(const FragmentSpread &spread, const CasedCharge &charge, float range);

// How far a thrown sphere (the M67: 397 g, 64 mm, drag coefficient 0.47)
// flies, m, launched at speed (m/s) and angle (radians up) from height (m)
// above flat ground, stepped with air drag.
float throwRange(float speed, float angle, float height);
// A fuse that burns 4 to 5 s (FM 3-23.30), spread evenly: dice from 0 to 1.
float fuseTime(float dice);

#include "Blast.h"
#include "TestRunner.h"

#include <cmath>

TEST(tnt_equivalents_are_the_documented_ones) {
  CHECK_NEAR(tntEquivalent(Explosive::CompB, 1e5f, false), 1.32, 1e-6);
  // C-4: 1.20 under 1.38 MPa, 1.37 over it, 1.19 for impulse (UFC 3-340-02).
  CHECK_NEAR(tntEquivalent(Explosive::C4, 1.0e6f, false), 1.20, 1e-6);
  CHECK_NEAR(tntEquivalent(Explosive::C4, 2.0e6f, false), 1.37, 1e-6);
  CHECK_NEAR(tntEquivalent(Explosive::C4, 1.0e6f, true), 1.19, 1e-6);
  CHECK_NEAR(tntEquivalent(Explosive::Tnt, 1.0e6f, false), 1.0, 1e-6);
}

TEST(overpressure_follows_kinney_and_graham) {
  // Their formula at a few scaled distances, worked by hand:
  // Ps/Pa = 808 (1 + (Z/4.5)^2) / sqrt((1 + (Z/0.048)^2)(1 + (Z/0.32)^2)(1 + (Z/1.35)^2)).
  auto byHand = [](double z) {
    return 101325.0 * 808.0 * (1.0 + std::pow(z / 4.5, 2)) /
           std::sqrt((1.0 + std::pow(z / 0.048, 2)) * (1.0 + std::pow(z / 0.32, 2)) *
                     (1.0 + std::pow(z / 1.35, 2)));
  };
  for (const float z : {0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 30.0f})
    CHECK_NEAR(peakOverpressure(z), byHand(z), 1e-3 * byHand(z));
  // About a megapascal at 1 m/kg^(1/3), about 10 kPa at 10 (free-air TNT).
  CHECK(peakOverpressure(1.0f) > 0.9e6f && peakOverpressure(1.0f) < 1.1e6f);
  CHECK(peakOverpressure(10.0f) > 9e3f && peakOverpressure(10.0f) < 11e3f);
  CHECK_NEAR(scaledDistance(8.0f, 8.0f), 4.0, 1e-5); // 8 m from 8 kg: Z = 8 / 2
}

TEST(a_grenades_blast_hurts_people_only_close) {
  // The M67's 184 g of Comp B: 0.243 kg of TNT. Lungs (207 kPa) only
  // within about a metre and a half, eardrums (34 kPa) within about 3 m.
  const float w = 0.184f * tntEquivalent(Explosive::CompB, 1e5f, false);
  auto at = [&](float metres) { return peakOverpressure(scaledDistance(metres, w)); };
  CHECK(at(1.0f) > kLungDamage);
  CHECK(at(2.0f) < kLungDamage);
  CHECK(at(2.5f) > kEardrumRupture);
  CHECK(at(3.5f) < kEardrumRupture);
}

TEST(the_push_falls_off_with_mass_and_facing_area) {
  // dv = Cd A I / m (Glasstone and Dolan's drag form).
  CHECK_NEAR(blastPush(1.2f, 0.5f, 100.0f, 30.0f), 1.2 * 0.5 * 100.0 / 30.0, 1e-5);
  CHECK(blastPush(1.2f, 0.5f, 100.0f, 60.0f) < blastPush(1.2f, 0.5f, 100.0f, 30.0f));
  CHECK_NEAR(blastPush(1.2f, 0.5f, 100.0f, 0.0f), 0.0, 1e-6); // static: nothing
}

TEST(gurney_gives_the_fragments_their_start_speed) {
  // A sphere: v = sqrt(2E) / sqrt(M/C + 3/5).
  CHECK_NEAR(gurneySphere(0.184f, 0.14f, kGurneyCompB),
             kGurneyCompB / std::sqrt(0.14 / 0.184 + 0.6), 1e-3);
  CHECK(gurneySphere(0.184f, 0.28f, kGurneyCompB) < gurneySphere(0.184f, 0.14f, kGurneyCompB));
}

TEST(the_m67s_fragments_make_a_casualty_half_the_time_at_15_m) {
  // The count is fitted to the M67's documented casualty radius (half the
  // people standing at 15 m hit); its 5 m kill radius then comes out by
  // itself as nearly certain hits.
  const FragmentSpread m67 = fitFragments(kM67);
  CHECK_NEAR(casualtyChance(m67, kM67, 15.0f), 0.5, 0.01);
  CHECK(casualtyChance(m67, kM67, 5.0f) > 0.99f);
  CHECK(casualtyChance(m67, kM67, 30.0f) < 0.2f);
  // The fragments add up to the case.
  CHECK_NEAR(m67.count * m67.mass, kM67.caseMass, 1e-3 * kM67.caseMass);
}

TEST(a_fragment_slows_with_distance_and_a_heavier_one_slows_less) {
  const float v0 = gurneySphere(kM67.charge, kM67.caseMass, kGurneyCompB);
  CHECK_NEAR(fragmentSpeed(v0, 0.001f, 0.0f), v0, 1e-3);
  CHECK(fragmentSpeed(v0, 0.001f, 20.0f) < fragmentSpeed(v0, 0.001f, 10.0f));
  CHECK(fragmentSpeed(v0, 0.004f, 20.0f) > fragmentSpeed(v0, 0.001f, 20.0f));
}

TEST(impulse_and_duration_agree_with_the_overpressure) {
  // Kinney and Graham's impulse and duration (from memory, to check
  // against a published table): together with the overpressure, the
  // impulse should be near the triangle under the pressure-time curve.
  for (const float z : {2.0f, 5.0f, 10.0f, 20.0f}) {
    const float triangle = 0.5f * peakOverpressure(z) * positiveDuration(z, 1.0f);
    const float impulse = positiveImpulse(z, 1.0f);
    CHECK(impulse > 0.7f * triangle && impulse < 1.4f * triangle);
  }
  // Scaled by the cube root of the charge: 8 kg at twice the distance has
  // twice the impulse and duration of 1 kg.
  CHECK_NEAR(positiveImpulse(5.0f, 8.0f), 2.0 * positiveImpulse(5.0f, 1.0f), 1e-3 * positiveImpulse(5.0f, 1.0f));
  CHECK_NEAR(positiveDuration(5.0f, 8.0f), 2.0 * positiveDuration(5.0f, 1.0f), 1e-6);
}

TEST(a_grenade_thrown_at_45_degrees_reaches_the_documented_distance) {
  // 18.5 m/s (Fu Jen 2008) at 45 degrees with air drag reaches about 35 m
  // (FM 3-23.30), less than the 34.9 m of no drag.
  const float range = throwRange(18.5f, 0.785398f, 1.5f);
  CHECK(range > 30.0f && range < 36.5f);
  // The fuse: 4 to 5 s whatever the dice say.
  CHECK_NEAR(fuseTime(0.0f), 4.0, 1e-6);
  CHECK_NEAR(fuseTime(1.0f), 5.0, 1e-6);
}

TEST(a_reflected_blast_is_two_to_eight_times_the_side_on) {
  // Rankine-Hugoniot: a weak wave doubles off a wall, a strong one up to
  // eight times.
  CHECK_NEAR(reflectionFactor(100.0f), 2.0, 0.01);
  CHECK(reflectionFactor(1e8f) > 7.9f && reflectionFactor(1e8f) < 8.0f);
  CHECK(reflectionFactor(2e5f) > reflectionFactor(2e4f));
}

TEST(a_block_of_c4_is_more_than_a_grenades_blast) {
  // The M112: 0.57 kg of C-4; a contact charge, no case.
  CHECK_NEAR(kM112.charge, 0.57, 1e-6);
  CHECK_NEAR(kM112.caseMass, 0.0, 1e-9);
  CHECK(kM112.explosive == Explosive::C4);
  const float block = kM112.charge * tntEquivalent(Explosive::C4, 1e5f, false);
  const float grenade = kM67.charge * tntEquivalent(Explosive::CompB, 1e5f, false);
  CHECK(peakOverpressure(scaledDistance(2.0f, block)) > peakOverpressure(scaledDistance(2.0f, grenade)));
}

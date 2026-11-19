#include "WeaponHold.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {

glm::vec3 point(const glm::mat4 &m, glm::vec3 p) { return glm::vec3(m * glm::vec4(p, 1.0f)); }

} // namespace

TEST(a_scaled_bone_matrix_becomes_a_rigid_frame) {
  const glm::mat4 bone = glm::translate(glm::mat4(1.0f), {1, 2, 3}) *
                         glm::rotate(glm::mat4(1.0f), 0.7f, glm::vec3(0, 1, 0)) *
                         glm::scale(glm::mat4(1.0f), glm::vec3(0.01f));
  const glm::mat4 r = rigidFrame(bone);
  CHECK_NEAR(glm::length(glm::vec3(r[0])), 1.0, 1e-5);
  CHECK_NEAR(glm::length(glm::vec3(r[2])), 1.0, 1e-5);
  CHECK(glm::distance(glm::vec3(r[3]), glm::vec3(1, 2, 3)) < 1e-6f);
  CHECK_NEAR(glm::dot(glm::vec3(r[0]), glm::vec3(r[1])), 0.0, 1e-5);
}

TEST(the_gun_points_where_the_aiming_clip_aims_with_its_trigger_at_the_finger) {
  // An aiming clip aims along the character's facing (+z): the gun lies
  // that way, upright, its trigger where the finger is, whatever the hands'
  // line (the finger is on the gun's right, the palm under it).
  const glm::mat4 hand = glm::translate(glm::mat4(1.0f), {-0.2f, 1.23f, 0.02f}) *
                         glm::rotate(glm::mat4(1.0f), 1.1f, glm::vec3(1, 0, 0));
  const glm::vec3 trigger(-0.2f, 1.3f, 0.17f);
  const WeaponHold hold = holdFrom(hand, trigger, glm::vec3(0, 0, 1));
  const glm::mat4 gun = hand * hold.gripInHand;
  CHECK(glm::distance(point(gun, glm::vec3(0.0f)), trigger) < 1e-5f);
  CHECK(glm::distance(glm::vec3(gun[2]), glm::vec3(0, 0, 1)) < 1e-5f);
  CHECK(glm::distance(glm::vec3(gun[1]), glm::vec3(0, 1, 0)) < 1e-5f);
  CHECK(!hold.twoHanded);
}

TEST(the_off_hand_goes_on_the_guns_centre_line_under_it) {
  // The left palm off to the side and below: put on the centre line at the
  // height asked for, its wrist moved with it.
  const glm::mat4 hand(1.0f);
  const glm::vec3 trigger(0.0f, 0.0f, 0.15f);
  OffHand off;
  off.palm = glm::vec3(0.05f, -0.03f, 0.40f);
  off.wrist = glm::vec3(0.06f, -0.06f, 0.33f);
  off.height = -0.01f;
  const WeaponHold hold = holdFrom(hand, trigger, glm::vec3(0, 0, 1), off);
  CHECK(hold.twoHanded);
  // The wrist, in gun space, moved by what puts the palm at (0, -0.01, z).
  CHECK(glm::distance(hold.offHand, glm::vec3(0.01f, -0.04f, 0.18f)) < 1e-5f);
  CHECK(glm::distance(hold.offPalm, glm::vec3(0.0f, -0.01f, 0.25f)) < 1e-5f);
}

TEST(the_grip_moves_with_the_hand) {
  const glm::mat4 hand(1.0f);
  const WeaponHold hold = holdFrom(hand, glm::vec3(0, 0, 0.05f), glm::vec3(0, 0, 1));
  const glm::mat4 moved = glm::translate(glm::mat4(1.0f), {3, 0, 0}) *
                          glm::rotate(glm::mat4(1.0f), 0.5f, glm::vec3(0, 1, 0));
  const glm::mat4 gun = moved * hold.gripInHand;
  CHECK(glm::distance(point(gun, glm::vec3(0.0f)), point(moved, glm::vec3(0, 0, 0.05f))) < 1e-5f);
}

TEST(a_pistol_cupped_in_both_hands_is_held_in_one) {
  // The off hand cupping the right: closer than kTwoHandedReach, one grip.
  OffHand off;
  off.palm = glm::vec3(0.03f, -0.02f, 0.10f);
  off.wrist = glm::vec3(0.04f, -0.04f, 0.02f);
  const WeaponHold hold = holdFrom(glm::mat4(1.0f), glm::vec3(0, 0, 0.04f), glm::vec3(0, 0, 1), off);
  CHECK(!hold.twoHanded);
}

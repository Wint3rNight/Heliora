#include "SpringKick.h"
#include "TestRunner.h"

#include <algorithm>
#include <cmath>

TEST(a_kick_peaks_at_its_size_at_its_peak_time_and_comes_back_without_bouncing) {
  // 2.5 cm back and 4 degrees up, peaking at 30 ms.
  SpringKick k;
  const glm::vec3 amount(0.025f, glm::radians(4.0f), 0.0f);
  k.kick(amount, 0.03f);
  float peak = 0.0f, lowest = 1.0f, at = 0.0f;
  for (int i = 1; i <= 2400; ++i) {
    k.update(1.0f / 2400.0f, 0.03f);
    if (k.offset.x > peak) {
      peak = k.offset.x;
      at = i / 2400.0f;
    }
    lowest = std::min(lowest, k.offset.x);
  }
  CHECK_NEAR(peak, 0.025, 1e-4);
  CHECK_NEAR(at, 0.03, 1e-3);
  CHECK(lowest > -1e-6f);          // no bounce past where it was
  CHECK(std::abs(k.offset.x) < 1e-4f); // back after a second
  CHECK(std::abs(k.offset.y) < 1e-4f);
}

TEST(kicks_add_up_and_the_spring_is_the_same_at_any_frame_rate) {
  SpringKick fast, slow;
  fast.kick({0.02f, 0.05f, 0.01f}, 0.03f);
  slow.kick({0.02f, 0.05f, 0.01f}, 0.03f);
  for (int i = 0; i < 240; ++i)
    fast.update(1.0f / 2400.0f, 0.03f);
  for (int i = 0; i < 3; ++i)
    slow.update(1.0f / 30.0f, 0.03f); // 0.1 s either way
  CHECK_NEAR(fast.offset.x, slow.offset.x, 1e-5);
  CHECK_NEAR(fast.velocity.y, slow.velocity.y, 1e-4);
  // A second jolt before the first has settled goes higher than one.
  SpringKick one, two;
  one.kick({0.02f, 0.0f, 0.0f}, 0.03f);
  two.kick({0.02f, 0.0f, 0.0f}, 0.03f);
  for (int i = 0; i < 10; ++i) {
    one.update(0.01f, 0.03f);
    two.update(0.01f, 0.03f);
  }
  two.kick({0.02f, 0.0f, 0.0f}, 0.03f);
  float highestOne = 0.0f, highestTwo = 0.0f;
  for (int i = 0; i < 20; ++i) {
    one.update(0.005f, 0.03f);
    two.update(0.005f, 0.03f);
    highestOne = std::max(highestOne, one.offset.x);
    highestTwo = std::max(highestTwo, two.offset.x);
  }
  CHECK(highestTwo > highestOne + 0.01f);
}

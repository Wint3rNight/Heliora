#include "AudioMixer.h"
#include "TestRunner.h"

#include <cmath>
#include <thread>

namespace {

constexpr int kRate = 48000;

// A clip that holds one value, so what comes out is the gain it got.
SoundClip steady(float seconds, float value = 1.0f) {
  SoundClip c;
  c.samples.assign(static_cast<size_t>(seconds * kRate), value);
  return c;
}

// Mixes this many seconds in 10 ms blocks; returns the left channel.
std::vector<float> run(AudioMixer &m, float seconds) {
  std::vector<float> left;
  std::vector<float> block(2 * 480);
  for (int b = 0; b < int(std::lround(seconds * 100.0f)); ++b) {
    m.mix(block.data(), 480);
    for (int i = 0; i < 480; ++i)
      left.push_back(block[2 * i]);
  }
  return left;
}

int firstSound(const std::vector<float> &samples) {
  for (size_t i = 0; i < samples.size(); ++i)
    if (std::abs(samples[i]) > 1e-6f)
      return static_cast<int>(i);
  return -1;
}

// A window pinned at 200 dB with nothing above it, so gains are just each
// sound's level under it; these made-up levels would ring the ears, so that
// is off.
AudioMixerTuning pinned() {
  AudioMixerTuning t;
  t.windowFloor = 200.0f;
  t.windowHeight = 300.0f;
  t.headroom = 0.0f;
  t.ringFrom = 1e9f;
  return t;
}

SoundParams at(glm::vec3 position, float loudness) {
  SoundParams p;
  p.position = position;
  p.loudness = loudness;
  return p;
}

} // namespace

TEST(sound_levels_come_from_pressure) {
  CHECK_NEAR(audio::soundLevel(2e-5f), 0.0, 1e-3);
  CHECK_NEAR(audio::soundLevel(20.0f), 120.0, 1e-3);
  CHECK_NEAR(audio::soundLevel(1000.0f), 154.0, 0.1);
}

TEST(a_sound_is_six_db_quieter_for_each_doubling_of_distance) {
  CHECK_NEAR(audio::levelAt(100.0f, 1.0f), 100.0, 1e-4);
  CHECK_NEAR(audio::levelAt(100.0f, 2.0f), 93.98, 0.01);
  CHECK_NEAR(audio::levelAt(100.0f, 4.0f), 87.96, 0.01);
  CHECK_NEAR(audio::levelAt(100.0f, 0.25f), 100.0, 1e-4); // no louder inside 1 m
}

TEST(sound_takes_a_second_to_cross_343_m) {
  CHECK_NEAR(audio::travelTime(343.0f), 1.0, 1e-5);
  CHECK_NEAR(audio::travelTime(0.0f), 0.0, 1e-9);
}

TEST(the_air_takes_the_highs_away_with_distance) {
  // 3 dB lost at 8 kHz by 39 m, at 4 kHz by 131 m, at 2 kHz by 333 m and
  // at 1 kHz by 600 m (76.6, 22.9, 9.0 and 5.0 dB/km).
  CHECK(audio::airCutoff(1.0f) >= 16000.0f);
  CHECK_NEAR(audio::airCutoff(39.0f), 8000.0, 1.0);
  CHECK_NEAR(audio::airCutoff(131.0f), 4000.0, 1.0);
  CHECK_NEAR(audio::airCutoff(333.0f), 2000.0, 1.0);
  CHECK_NEAR(audio::airCutoff(600.0f), 1000.0, 1.0);
  float last = 1e9f;
  for (float d = 1.0f; d < 3000.0f; d *= 1.5f) {
    CHECK(audio::airCutoff(d) <= last);
    last = audio::airCutoff(d);
  }
}

TEST(panning_keeps_the_power_the_same) {
  for (float side = -1.0f; side <= 1.0f; side += 0.1f) {
    const auto g = audio::equalPowerPan(side);
    CHECK_NEAR(g[0] * g[0] + g[1] * g[1], 1.0, 1e-5);
  }
  CHECK_NEAR(audio::equalPowerPan(0.0f)[0], std::sqrt(0.5), 1e-5);
  CHECK_NEAR(audio::equalPowerPan(1.0f)[0], 0.0, 1e-5);
  CHECK_NEAR(audio::equalPowerPan(-1.0f)[1], 0.0, 1e-5);
}

TEST(a_source_coming_closer_sounds_higher) {
  const glm::vec3 o(0.0f);
  const float c = audio::kSpeedOfSound;
  CHECK_NEAR(audio::dopplerPitch({100, 0, 0}, {-34.3f, 0, 0}, o, o), c / (c - 34.3f), 1e-4);
  CHECK_NEAR(audio::dopplerPitch({100, 0, 0}, {34.3f, 0, 0}, o, o), c / (c + 34.3f), 1e-4);
  CHECK_NEAR(audio::dopplerPitch({100, 0, 0}, o, o, {10, 0, 0}), (c + 10.0f) / c, 1e-4);
  CHECK_NEAR(audio::dopplerPitch({100, 0, 0}, {0, 30, 0}, o, o), 1.0, 1e-5); // crossing
}

TEST(only_a_very_loud_sound_rings_the_ears) {
  CHECK_NEAR(audio::ringSeconds(169.0f), 0.0, 1e-6);
  CHECK_NEAR(audio::ringSeconds(170.0f), 1.0, 1e-4);
  CHECK_NEAR(audio::ringSeconds(177.5f), 5.5, 1e-3);
  CHECK_NEAR(audio::ringSeconds(185.0f), 10.0, 1e-4);
  CHECK_NEAR(audio::ringSeconds(200.0f), 10.0, 1e-4);
}

TEST(a_sound_343_m_away_is_heard_a_second_late) {
  const SoundClip click = steady(0.01f);
  AudioMixer m(kRate);
  m.setTuning(pinned());
  // Straight ahead (the listener faces -z), so both channels hear it.
  CHECK(m.start(1, &click, at({0.0f, 0.0f, -343.0f}, 250.0f)));
  const int first = firstSound(run(m, 1.2f));
  CHECK(std::abs(first - kRate) <= 2);
}

TEST(a_nearer_listener_hears_it_sooner) {
  const SoundClip click = steady(0.01f);
  AudioMixer m(kRate);
  m.setTuning(pinned());
  CHECK(m.start(1, &click, at({0.0f, 0.0f, -343.0f}, 250.0f)));
  run(m, 0.5f);
  // Half a second in, the wavefront is 171.5 m out: walk to 200 m and it
  // has just passed, so it plays at once.
  AudioListener moved;
  moved.position = glm::vec3(0.0f, 0.0f, -200.0f);
  m.setListener(moved);
  const int first = firstSound(run(m, 0.2f));
  CHECK(first >= 0 && first < 480);
}

TEST(a_sound_twice_as_far_away_plays_at_half_the_amplitude) {
  const SoundClip hum = steady(1.0f);
  AudioMixer near(kRate), far(kRate);
  near.setTuning(pinned());
  far.setTuning(pinned());
  // Ahead of the listener, so both channels get the same.
  CHECK(near.start(1, &hum, at({0.0f, 0.0f, -1.0f}, 200.0f)));
  CHECK(far.start(1, &hum, at({0.0f, 0.0f, -2.0f}, 200.0f)));
  const std::vector<float> a = run(near, 0.5f), b = run(far, 0.5f);
  const float nearOut = a[20000], farOut = b[20000];
  CHECK_NEAR(nearOut, std::sqrt(0.5), 1e-3);
  CHECK_NEAR(farOut / nearOut, std::pow(10.0, -6.0206 / 20.0), 1e-3);
}

TEST(a_sound_on_the_right_is_louder_on_the_right) {
  const SoundClip hum = steady(0.5f);
  AudioMixer m(kRate);
  m.setTuning(pinned());
  CHECK(m.start(1, &hum, at({5.0f, 0.0f, 0.0f}, 200.0f))); // listener faces -z: +x is right
  std::vector<float> block(2 * 480);
  m.mix(block.data(), 480); // the 5 m trip takes 700 frames
  m.mix(block.data(), 480);
  m.mix(block.data(), 480);
  CHECK(block[2 * 400 + 1] > 10.0f * std::abs(block[2 * 400]));
}

TEST(a_loud_sound_covers_quiet_ones_then_lets_them_back) {
  // A 70 dB hum, then a 150 dB bang a metre away: the window's top jumps to
  // 150, the hum is 80 dB under it and goes silent; afterwards the top falls
  // 10 dB a second until the hum plays again.
  const SoundClip hum = steady(20.0f, 0.5f);
  const SoundClip bang = steady(0.05f);
  AudioMixer m(kRate);
  AudioMixerTuning t;
  t.windowFloor = 85.0f;
  t.windowHeight = 60.0f;
  t.windowRelease = 10.0f;
  m.setTuning(t);
  CHECK(m.start(1, &hum, at({0.0f, 0.0f, -1.0f}, 70.0f)));
  const std::vector<float> before = run(m, 0.5f);
  CHECK(std::abs(before.back()) > 1e-4f);
  CHECK_NEAR(m.windowTop(), 85.0, 1e-3);
  CHECK(m.start(2, &bang, at({0.0f, 0.0f, -1.0f}, 150.0f)));
  run(m, 0.06f);
  CHECK(m.windowTop() > 149.0f);
  const std::vector<float> covered = run(m, 0.5f);
  CHECK(std::abs(covered.back()) < 1e-6f);
  run(m, 2.5f); // about 3 s after the bang
  CHECK_NEAR(m.windowTop(), 120.0, 1.0);
  run(m, 3.0f); // under 130 dB the hum is back in the window
  CHECK(std::abs(run(m, 0.1f).back()) > 1e-6f);
}

TEST(when_every_voice_is_busy_the_quietest_gives_way) {
  const SoundClip hum = steady(2.0f);
  AudioMixer m(kRate, 2);
  CHECK(m.start(1, &hum, at({0, 0, -1}, 60.0f)));
  CHECK(m.start(2, &hum, at({0, 0, -1}, 80.0f)));
  CHECK(m.start(3, &hum, at({0, 0, -1}, 70.0f))); // the 60 dB one goes
  CHECK(m.voices() == 2);
  CHECK(!m.start(4, &hum, at({0, 0, -1}, 50.0f))); // quieter than both
  CHECK(m.voices() == 2);
  m.stop(2);
  CHECK(m.voices() == 1);
  m.stopAll();
  CHECK(m.voices() == 0);
}

TEST(a_blast_loud_enough_rings_the_ears) {
  const SoundClip bang = steady(0.05f);
  AudioMixer m(kRate);
  CHECK(m.start(1, &bang, at({0.0f, 0.0f, -1.0f}, 180.0f)));
  run(m, 0.05f);
  CHECK_NEAR(m.ringing(), audio::ringSeconds(180.0f), 0.1);
  AudioMixer quiet(kRate);
  CHECK(quiet.start(1, &bang, at({0.0f, 0.0f, -1.0f}, 160.0f)));
  run(quiet, 0.05f);
  CHECK_NEAR(quiet.ringing(), 0.0, 1e-6);
}

TEST(a_finished_sound_frees_its_voice) {
  const SoundClip click = steady(0.01f);
  AudioMixer m(kRate);
  CHECK(m.start(1, &click, at({0, 0, -1}, 80.0f)));
  run(m, 0.1f);
  CHECK(m.voices() == 0);
}

TEST(commands_cross_the_ring_in_order) {
  SpscRing<int, 8> ring;
  for (int i = 0; i < 7; ++i)
    CHECK(ring.push(i));
  CHECK(!ring.push(99)); // full: one slot is always free
  int v = -1;
  for (int i = 0; i < 7; ++i) {
    CHECK(ring.pop(v));
    CHECK(v == i);
  }
  CHECK(!ring.pop(v));

  // One thread writing, one reading, nothing lost or reordered.
  SpscRing<int, 64> shared;
  constexpr int kCount = 200000;
  std::thread writer([&] {
    for (int i = 0; i < kCount;)
      if (shared.push(i))
        ++i;
  });
  int expected = 0;
  bool inOrder = true;
  while (expected < kCount) {
    int got;
    if (shared.pop(got)) {
      inOrder = inOrder && got == expected;
      ++expected;
    }
  }
  writer.join();
  CHECK(inOrder);
}

TEST(a_wild_pitch_is_held_to_four_times) {
  // A camera snapping across the map for one frame reads as thousands of
  // m/s: the sound should still play, at most four times as fast.
  const SoundClip clip = steady(0.1f);
  AudioMixer m(kRate);
  m.setTuning(pinned());
  SoundParams p = at({0.0f, 0.0f, -1.0f}, 200.0f);
  p.pitch = 100.0f;
  CHECK(m.start(1, &clip, p));
  const std::vector<float> out = run(m, 0.2f);
  int heard = 0;
  for (float x : out)
    heard += std::abs(x) > 1e-6f ? 1 : 0;
  CHECK(heard >= static_cast<int>(0.1f * kRate / 4.0f) - 2);
}

TEST(numbers_that_arent_numbers_are_turned_away) {
  const SoundClip clip = steady(0.1f);
  AudioMixer m(kRate);
  const float nan = std::nanf("");
  CHECK(!m.start(1, &clip, at({nan, 0.0f, 0.0f}, 80.0f)));
  SoundParams p = at({0.0f, 0.0f, -1.0f}, 80.0f);
  p.pitch = nan;
  CHECK(!m.start(2, &clip, p));
  CHECK(m.start(3, &clip, at({0.0f, 0.0f, -1.0f}, 80.0f)));
  m.move(3, {nan, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}); // ignored
  AudioListener bad;
  bad.position.x = nan;
  m.setListener(bad); // ignored
  const std::vector<float> out = run(m, 0.05f);
  for (float x : out)
    CHECK(std::isfinite(x));
}

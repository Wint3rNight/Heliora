#pragma once

#include <glm/glm.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

// The game's sound (step 1.5c): our own mixer, no device here, so it's
// tested like the rest of heliora_core. AudioDevice (the game) feeds it
// commands from the game thread and calls mix() on the audio thread.
//
// What makes it sound like the real world:
// - a sound starts when its wavefront gets to the listener, at 343 m/s;
// - it's 6 dB quieter for every doubling of distance (a point source);
// - air takes the highs away with distance (a low-pass, from ISO 9613);
// - loudness is real (dB SPL at 1 m), and a window that follows the loudest
//   sound decides what plays (HDR audio, as in DICE's Frostbite and Wwise):
//   a gunshot covers the footsteps for a moment, the way ears do;
// - a blast loud enough rings the ears and muffles everything for a while.

// A sound decoded to floats at the mixer's rate: one channel for sounds in
// the world, two for sounds that play as they are.
struct SoundClip {
  std::vector<float> samples; // interleaved
  int channels = 1;
  size_t frames() const { return channels > 0 ? samples.size() / channels : 0; }
};

namespace audio {

constexpr float kSpeedOfSound = 343.0f;   // m/s in air at 20 °C (340 to 343)
constexpr float kReferencePressure = 2e-5f; // Pa: 0 dB SPL

// dB SPL of a sound pressure, Pa.
float soundLevel(float pascals);
// The level at a distance, m, of a sound this loud at 1 m: 6 dB down per
// doubling; never louder than at 1 m.
float levelAt(float levelAt1m, float distance);
// Seconds a sound takes to travel the distance.
float travelTime(float distance);
// The low-pass cutoff, Hz, standing in for the air's absorption over the
// distance: the frequency that has lost about 3 dB by then.
float airCutoff(float distance);
// Left and right gains for a sound from one side (-1 left, 0 ahead or
// behind, 1 right), their squares adding up to 1.
std::array<float, 2> equalPowerPan(float side);
// The pitch a moving source and listener hear: c plus the listener's speed
// toward the source, over c minus the source's speed toward the listener.
float dopplerPitch(glm::vec3 source, glm::vec3 sourceVelocity, glm::vec3 listener,
                   glm::vec3 listenerVelocity);
// Seconds the ears ring after a sound this loud reaches them: none under
// 170 dB, 1 s at 170 to 10 s at 185, no more past it (est.).
float ringSeconds(float level);

} // namespace audio

struct SoundParams {
  glm::vec3 position{0.0f};
  glm::vec3 velocity{0.0f};
  float loudness = 80.0f; // dB SPL at 1 m (or as heard, for a sound that isn't in the world)
  bool inWorld = true;    // false: no delay, distance or direction (the interface)
  bool loop = false;
  float pitch = 1.0f;
};

struct AudioListener {
  glm::vec3 position{0.0f};
  glm::vec3 forward{0.0f, 0.0f, -1.0f};
  glm::vec3 up{0.0f, 1.0f, 0.0f};
  glm::vec3 velocity{0.0f};
};

struct AudioMixerTuning {
  float windowHeight = 60.0f;  // dB under the window's top that still play (est.)
  float windowRelease = 10.0f; // dB a second the top falls back after a loud sound (est.)
  float windowFloor = 65.0f;   // dB: the top never goes lower, so quiet scenes stay quiet (est.;
                               // footsteps a few metres away play about 20 dB under full scale)
  float headroom = -6.0f;      // dB below full scale the window's top plays at
  float volume = 1.0f;         // the player's master volume
  float ringFrom = 170.0f;     // dB at the listener that rings the ears (est.)
  bool operator==(const AudioMixerTuning &) const = default;
};

// One audio thread's mixer. Voices are named by ids the caller picks (so
// the game knows a sound's id before the audio thread has seen it).
class AudioMixer {
public:
  explicit AudioMixer(int sampleRate, int maxVoices = 64);

  // False if every voice is busy with something louder. The clip must
  // outlive the voice.
  bool start(int id, const SoundClip *clip, const SoundParams &params);
  void stop(int id);
  void move(int id, glm::vec3 position, glm::vec3 velocity);
  void stopAll() { active.clear(); }
  void setListener(const AudioListener &listener);
  void setTuning(const AudioMixerTuning &tuning);

  // Writes frames of stereo, interleaved.
  void mix(float *out, int frames);

  double time() const { return now; }     // seconds mixed so far
  float windowTop() const { return top; } // dB
  float ringing() const { return ring; }  // seconds of ringing left
  int voices() const;                     // playing or on their way

private:
  struct Voice {
    int id = 0;
    const SoundClip *clip = nullptr;
    SoundParams params;
    glm::vec3 emitted{0.0f}; // where it was when it was made
    double emitTime = 0.0;
    bool arrived = false;
    double cursor = 0.0;     // frames into the clip
    float level = 0.0f;      // dB at the listener this block
    std::array<float, 2> gain{0.0f, 0.0f};
    std::array<float, 2> filtered{0.0f, 0.0f}; // the low-pass's memory
    bool fresh = true;       // no block mixed yet: start at its gain, no ramp
    int startFrame = 0;      // where in this block it starts (the frame it arrives)
  };
  float levelOf(const Voice &v) const;
  Voice *find(int id);

  int rate;
  int maxVoices;
  std::vector<Voice> active;
  AudioListener listener;
  AudioMixerTuning tune;
  uint64_t framesMixed = 0; // counted, so time adds up exactly
  double now = 0.0;
  float top = 0.0f;
  float ring = 0.0f;
  float ringPhase = 0.0f;
  std::array<float, 2> muffled{0.0f, 0.0f};
  float limiter = 1.0f;
};

// A fixed queue from one thread to one other, no locks: the game pushes
// commands, the audio callback pops them.
template <typename T, size_t N> class SpscRing {
public:
  bool push(const T &value) {
    const size_t head = writeAt.load(std::memory_order_relaxed);
    const size_t next = (head + 1) % N;
    if (next == readAt.load(std::memory_order_acquire))
      return false; // full
    slots[head] = value;
    writeAt.store(next, std::memory_order_release);
    return true;
  }
  bool pop(T &value) {
    const size_t tail = readAt.load(std::memory_order_relaxed);
    if (tail == writeAt.load(std::memory_order_acquire))
      return false; // empty
    value = slots[tail];
    readAt.store((tail + 1) % N, std::memory_order_release);
    return true;
  }

private:
  std::array<T, N> slots{};
  std::atomic<size_t> writeAt{0};
  std::atomic<size_t> readAt{0};
};

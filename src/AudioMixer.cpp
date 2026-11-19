#include "AudioMixer.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kTwoPi = 6.2831853f;

// The ringing tone: noise-induced hearing loss is worst around 4 kHz, which
// is also where tinnitus after loud noise usually sits.
constexpr float kRingHz = 4000.0f;
constexpr float kMuffledHz = 700.0f; // the cutoff while fully deafened (est.)

// A one-pole low-pass's share of the gap closed each sample.
float lowPassShare(float cutoff, int rate) {
  return 1.0f - std::exp(-kTwoPi * cutoff / static_cast<float>(rate));
}

float amplitude(float decibels) { return std::pow(10.0f, decibels / 20.0f); }

bool finite(glm::vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }

// Pitch held to two octaves either way: a camera snapping across the map
// reads as thousands of m/s for a frame, and a NaN must never reach the
// cursor.
float safePitch(float pitch) {
  return std::isfinite(pitch) ? std::clamp(pitch, 0.25f, 4.0f) : 1.0f;
}

} // namespace

namespace audio {

float soundLevel(float pascals) {
  return 20.0f * std::log10(std::max(pascals, 1e-12f) / kReferencePressure);
}

float levelAt(float levelAt1m, float distance) {
  return levelAt1m - 20.0f * std::log10(std::max(distance, 1.0f));
}

float travelTime(float distance) { return std::max(distance, 0.0f) / kSpeedOfSound; }

float airCutoff(float distance) {
  // The distance by which each frequency has lost about 3 dB, from ISO
  // 9613's absorption at 20 °C and 70% humidity (8 kHz 76.6, 4 kHz 22.9,
  // 2 kHz 9.0, 1 kHz 5.0 dB/km; from memory, to check against the
  // standard), joined in log-log steps and carried on past the last one.
  struct Point {
    float distance, hertz;
  };
  static constexpr Point kPoints[] = {
      {1.0f, 20000.0f}, {39.0f, 8000.0f}, {131.0f, 4000.0f}, {333.0f, 2000.0f}, {600.0f, 1000.0f}};
  if (distance <= kPoints[0].distance)
    return kPoints[0].hertz;
  constexpr int n = sizeof(kPoints) / sizeof(kPoints[0]);
  int i = 0;
  while (i + 2 < n && distance > kPoints[i + 1].distance)
    ++i;
  const Point a = kPoints[i], b = kPoints[i + 1];
  const float along = std::log(distance / a.distance) / std::log(b.distance / a.distance);
  const float hertz = a.hertz * std::pow(b.hertz / a.hertz, along);
  return std::clamp(hertz, 100.0f, 20000.0f);
}

std::array<float, 2> equalPowerPan(float side) {
  const float angle = (std::clamp(side, -1.0f, 1.0f) + 1.0f) * 0.25f * 3.14159265f;
  return {std::cos(angle), std::sin(angle)};
}

float dopplerPitch(glm::vec3 source, glm::vec3 sourceVelocity, glm::vec3 listener,
                   glm::vec3 listenerVelocity) {
  const glm::vec3 between = listener - source;
  const float distance = glm::length(between);
  if (distance < 1e-4f)
    return 1.0f;
  const glm::vec3 toListener = between / distance;
  // Speeds toward each other, kept under the speed of sound.
  const float c = kSpeedOfSound;
  const float sourceToward = std::min(glm::dot(sourceVelocity, toListener), 0.9f * c);
  const float listenerToward = std::max(glm::dot(listenerVelocity, -toListener), -0.9f * c);
  return (c + listenerToward) / (c - sourceToward);
}

float ringSeconds(float level) {
  if (level < 170.0f)
    return 0.0f;
  return 1.0f + 9.0f * std::min(1.0f, (level - 170.0f) / 15.0f);
}

} // namespace audio

AudioMixer::AudioMixer(int sampleRate, int voices)
    : rate(std::max(sampleRate, 1)), maxVoices(std::max(voices, 1)) {
  active.reserve(static_cast<size_t>(maxVoices)); // start() never allocates
  top = tune.windowFloor;
}

float AudioMixer::levelOf(const Voice &v) const {
  if (!v.params.inWorld)
    return v.params.loudness;
  return audio::levelAt(v.params.loudness, glm::distance(v.params.position, listener.position));
}

AudioMixer::Voice *AudioMixer::find(int id) {
  for (Voice &v : active)
    if (v.id == id)
      return &v;
  return nullptr;
}

bool AudioMixer::start(int id, const SoundClip *clip, const SoundParams &params) {
  if (!clip || clip->frames() == 0 || clip->channels < 1)
    return false;
  if (!finite(params.position) || !finite(params.velocity) || !std::isfinite(params.loudness) ||
      !std::isfinite(params.pitch))
    return false;
  Voice v;
  v.id = id;
  v.clip = clip;
  v.params = params;
  v.emitted = params.position;
  v.emitTime = now;
  v.arrived = !params.inWorld;
  v.level = levelOf(v);
  if (static_cast<int>(active.size()) >= maxVoices) {
    auto quietest = std::min_element(active.begin(), active.end(),
                                     [](const Voice &a, const Voice &b) { return a.level < b.level; });
    if (quietest->level >= v.level)
      return false;
    *quietest = active.back();
    active.pop_back();
  }
  active.push_back(v);
  return true;
}

void AudioMixer::stop(int id) {
  for (size_t i = 0; i < active.size(); ++i)
    if (active[i].id == id) {
      active[i] = active.back();
      active.pop_back();
      return;
    }
}

void AudioMixer::move(int id, glm::vec3 position, glm::vec3 velocity) {
  if (!finite(position) || !finite(velocity))
    return;
  if (Voice *v = find(id)) {
    v->params.position = position;
    v->params.velocity = velocity;
  }
}

void AudioMixer::setListener(const AudioListener &l) {
  if (finite(l.position) && finite(l.forward) && finite(l.up) && finite(l.velocity))
    listener = l;
}

void AudioMixer::setTuning(const AudioMixerTuning &t) {
  tune = t;
  top = std::max(top, tune.windowFloor);
}

int AudioMixer::voices() const { return static_cast<int>(active.size()); }

void AudioMixer::mix(float *out, int frames) {
  if (frames <= 0)
    return;
  std::fill(out, out + 2 * frames, 0.0f);
  const double blockStart = static_cast<double>(framesMixed) / rate;
  const double blockSeconds = static_cast<double>(frames) / rate;
  glm::vec3 right = glm::cross(listener.forward, listener.up);
  right = glm::length(right) > 1e-6f ? glm::normalize(right) : glm::vec3(1.0f, 0.0f, 0.0f);

  // --- What has reached the listener, and the window's top.
  float loudest = tune.windowFloor;
  for (Voice &v : active) {
    v.level = levelOf(v);
    v.startFrame = 0;
    if (!v.arrived) {
      const double arrival =
          v.emitTime + audio::travelTime(glm::distance(v.emitted, listener.position));
      if (arrival >= blockStart + blockSeconds)
        continue;
      v.arrived = true;
      v.startFrame =
          std::clamp(static_cast<int>(std::ceil((arrival - blockStart) * rate - 1e-6)), 0, frames - 1);
      if (v.level >= tune.ringFrom)
        ring = std::max(ring, audio::ringSeconds(v.level + 170.0f - tune.ringFrom));
    }
    loudest = std::max(loudest, v.level);
  }
  top = std::max(loudest, top - tune.windowRelease * static_cast<float>(blockSeconds));

  // --- Each voice that has arrived, into the block.
  for (size_t i = 0; i < active.size();) {
    Voice &v = active[i];
    if (!v.arrived) {
      ++i;
      continue;
    }
    const float inWindow = v.level >= top - tune.windowHeight
                               ? amplitude(v.level - top + tune.headroom) * tune.volume
                               : 0.0f;
    std::array<float, 2> target{inWindow, inWindow};
    float share = 1.0f; // no air filter
    float pitch = v.params.pitch;
    if (v.params.inWorld) {
      const glm::vec3 between = v.params.position - listener.position;
      const float distance = glm::length(between);
      const float side = distance > 1e-4f ? glm::dot(between / distance, right) : 0.0f;
      const std::array<float, 2> pan = audio::equalPowerPan(side);
      target = {inWindow * pan[0], inWindow * pan[1]};
      share = lowPassShare(audio::airCutoff(distance), rate);
      pitch *= audio::dopplerPitch(v.params.position, v.params.velocity, listener.position,
                                   listener.velocity);
    }
    pitch = safePitch(pitch);
    const std::array<float, 2> from = v.fresh ? target : v.gain;
    const SoundClip &clip = *v.clip;
    const size_t length = clip.frames();
    const int channels = clip.channels;
    const bool stereo = channels >= 2 && !v.params.inWorld;
    const int span = frames - v.startFrame;
    bool finished = false;
    for (int f = v.startFrame; f < frames; ++f) {
      const float t = static_cast<float>(f - v.startFrame + 1) / static_cast<float>(span);
      const float gl = from[0] + (target[0] - from[0]) * t;
      const float gr = from[1] + (target[1] - from[1]) * t;
      // The clip at the cursor, between its two nearest frames.
      const size_t at = static_cast<size_t>(v.cursor);
      const float frac = static_cast<float>(v.cursor - static_cast<double>(at));
      const size_t next = at + 1 < length ? at + 1 : (v.params.loop ? 0 : at);
      auto sample = [&](int channel) {
        const float a = clip.samples[at * channels + channel];
        const float b = clip.samples[next * channels + channel];
        return a + (b - a) * frac;
      };
      if (stereo) {
        out[2 * f] += gl * sample(0);
        out[2 * f + 1] += gr * sample(1);
      } else {
        float x = sample(0);
        for (int c = 1; c < channels; ++c)
          x += sample(c);
        x /= static_cast<float>(channels);
        v.filtered[0] += share * (x - v.filtered[0]);
        out[2 * f] += gl * v.filtered[0];
        out[2 * f + 1] += gr * v.filtered[0];
      }
      v.cursor += pitch;
      if (v.cursor >= static_cast<double>(length)) {
        if (!v.params.loop) {
          finished = true;
          break;
        }
        v.cursor = std::fmod(v.cursor, static_cast<double>(length));
      }
    }
    v.gain = target;
    v.fresh = false;
    if (finished) {
      active[i] = active.back();
      active.pop_back();
    } else {
      ++i;
    }
  }

  // --- Ringing ears: everything muffled and a tone, fading in the last 2 s.
  if (ring > 0.0f) {
    const float strength = std::min(1.0f, ring / 2.0f);
    const float cutoff = 20000.0f + (kMuffledHz - 20000.0f) * strength;
    const float share = lowPassShare(cutoff, rate);
    const float tone = 0.03f * strength * tune.volume;
    const float step = kTwoPi * kRingHz / static_cast<float>(rate);
    for (int f = 0; f < frames; ++f) {
      for (int c = 0; c < 2; ++c) {
        muffled[c] += share * (out[2 * f + c] - muffled[c]);
        out[2 * f + c] = muffled[c] + tone * std::sin(ringPhase);
      }
      ringPhase = std::fmod(ringPhase + step, kTwoPi);
    }
    ring = std::max(0.0f, ring - static_cast<float>(blockSeconds));
  } else {
    muffled = {0.0f, 0.0f};
  }

  // --- A peak limiter: instant down, back up over 50 ms.
  const float recover = 1.0f / (0.05f * static_cast<float>(rate));
  for (int f = 0; f < frames; ++f) {
    const float peak = std::max(std::abs(out[2 * f]), std::abs(out[2 * f + 1]));
    if (peak * limiter > 0.98f)
      limiter = 0.98f / peak;
    else
      limiter = std::min(1.0f, limiter + recover);
    out[2 * f] *= limiter;
    out[2 * f + 1] *= limiter;
  }

  framesMixed += static_cast<uint64_t>(frames);
  now = static_cast<double>(framesMixed) / rate;
}

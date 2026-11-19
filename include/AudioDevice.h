#pragma once

#include "AudioMixer.h"

#include <memory>
#include <string>

// The sound card (step 1.5c): miniaudio's playback device, whose callback
// runs our AudioMixer on the audio thread. The game talks to it through a
// lock-free ring of commands, so neither thread waits on the other. It also
// decodes sound files (WAV, FLAC, MP3, Ogg Vorbis) into clips at its rate.
//
// Three ways to run: the speakers; offline, mixing as the game's clock goes
// and writing a WAV file (captures, to check what was heard); or silent.
class AudioDevice {
public:
  enum class Output { Speakers, File, Silent };
  // filePath is only for Output::File. Speakers that fail to open fall back
  // to silent, with a log line.
  explicit AudioDevice(Output output, const std::string &filePath = {});
  ~AudioDevice();
  AudioDevice(const AudioDevice &) = delete;
  AudioDevice &operator=(const AudioDevice &) = delete;

  int sampleRate() const;
  // A clip from a file, decoded once and kept: one channel for sounds in the
  // world, two for the rest. nullptr if it can't be read (logged once).
  const SoundClip *load(const std::string &path, bool inWorld = true);

  // The sound's id, or 0 if it couldn't be queued.
  int play(const SoundClip *clip, const SoundParams &params);
  void stop(int id);
  void move(int id, glm::vec3 position, glm::vec3 velocity);
  void stopAll();
  void setListener(const AudioListener &listener);
  void setTuning(const AudioMixerTuning &tuning);

  // Output::File: mixes the game's frame, seconds long, into the file.
  void advance(float seconds);

  // What the audio thread last reported, for the editor.
  float windowTop() const;
  float ringing() const;
  int voices() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

#include "AudioDevice.h"

#include <spdlog/spdlog.h>

// miniaudio decodes Ogg Vorbis through stb_vorbis, included this way (its
// documented pattern): the header part before miniaudio, the rest after.
#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING_MP3
#include "miniaudio.h"
#undef STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kRate = 48000;

struct Command {
  enum class Kind { Start, Stop, Move, StopAll, Listener, Tuning } kind = Kind::Stop;
  int id = 0;
  const SoundClip *clip = nullptr;
  SoundParams params;
  AudioListener listener;
  AudioMixerTuning tuning;
};

} // namespace

struct AudioDevice::Impl {
  // Declared first, so they go last: the device stops before its clips do.
  std::unordered_map<std::string, std::unique_ptr<SoundClip>> clips;
  std::unordered_map<std::string, bool> failed;
  Output output = Output::Silent;
  AudioMixer mixer{kRate};
  SpscRing<Command, 1024> ring;
  std::atomic<int> nextId{1};
  std::atomic<float> top{0.0f};
  std::atomic<float> ring_{0.0f};
  std::atomic<int> voiceCount{0};
  ma_device device{};
  bool deviceOpen = false;
  ma_encoder encoder{};
  bool encoderOpen = false;
  double owed = 0.0; // frames the file is behind the game's clock
  std::vector<float> scratch;
  int dropped = 0;   // commands lost to a full queue, for the log

  void apply(const Command &c) {
    switch (c.kind) {
    case Command::Kind::Start: mixer.start(c.id, c.clip, c.params); break;
    case Command::Kind::Stop: mixer.stop(c.id); break;
    case Command::Kind::Move: mixer.move(c.id, c.params.position, c.params.velocity); break;
    case Command::Kind::StopAll: mixer.stopAll(); break;
    case Command::Kind::Listener: mixer.setListener(c.listener); break;
    case Command::Kind::Tuning: mixer.setTuning(c.tuning); break;
    }
  }
  // On the audio thread (or the game's, offline): the queued commands, then
  // the block.
  void render(float *out, int frames) {
    Command c;
    while (ring.pop(c))
      apply(c);
    mixer.mix(out, frames);
    top.store(mixer.windowTop(), std::memory_order_relaxed);
    ring_.store(mixer.ringing(), std::memory_order_relaxed);
    voiceCount.store(mixer.voices(), std::memory_order_relaxed);
  }
  // False if it was dropped: silent, or the queue is full (the audio
  // thread has stalled; said once, then every 500th).
  bool send(const Command &c) {
    if (output == Output::Silent)
      return false;
    if (ring.push(c))
      return true;
    if (dropped++ % 500 == 0)
      spdlog::warn("Sound: the command queue is full, {} commands dropped", dropped);
    return false;
  }
  // File output, on the game's thread: the commands, even with no time to
  // mix (paused), so the queue never fills.
  void drain() {
    Command c;
    while (ring.pop(c))
      apply(c);
  }
  static void callback(ma_device *device, void *out, const void *, ma_uint32 frames) {
    static_cast<Impl *>(device->pUserData)->render(static_cast<float *>(out),
                                                   static_cast<int>(frames));
  }
};

AudioDevice::AudioDevice(Output output, const std::string &filePath)
    : impl(std::make_unique<Impl>()) {
  impl->output = output;
  if (output == Output::Speakers) {
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 2;
    config.sampleRate = kRate;
    config.dataCallback = &Impl::callback;
    config.pUserData = impl.get();
    const bool made = ma_device_init(nullptr, &config, &impl->device) == MA_SUCCESS;
    if (made && ma_device_start(&impl->device) == MA_SUCCESS) {
      impl->deviceOpen = true;
      spdlog::info("Sound: {} at {} Hz", impl->device.playback.name, kRate);
    } else {
      if (made)
        ma_device_uninit(&impl->device); // made but wouldn't start
      spdlog::warn("Sound: no playback device, the game is silent");
      impl->output = Output::Silent;
    }
  } else if (output == Output::File) {
    ma_encoder_config config =
        ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 2, kRate);
    if (ma_encoder_init_file(filePath.c_str(), &config, &impl->encoder) == MA_SUCCESS) {
      impl->encoderOpen = true;
      spdlog::info("Sound: mixing into {}", filePath);
    } else {
      spdlog::warn("Sound: can't write {}, the game is silent", filePath);
      impl->output = Output::Silent;
    }
  }
}

AudioDevice::~AudioDevice() {
  if (impl->deviceOpen)
    ma_device_uninit(&impl->device);
  if (impl->encoderOpen)
    ma_encoder_uninit(&impl->encoder);
}

int AudioDevice::sampleRate() const { return kRate; }

const SoundClip *AudioDevice::load(const std::string &path, bool inWorld) {
  const std::string key = path + (inWorld ? "#1" : "#2");
  if (auto it = impl->clips.find(key); it != impl->clips.end())
    return it->second.get();
  if (impl->failed.count(key))
    return nullptr;
  // As written, or one folder up: the game usually runs from a build folder.
  std::string file = path;
  if (!std::filesystem::exists(file) && std::filesystem::exists("../" + path))
    file = "../" + path;
  const ma_uint32 channels = inWorld ? 1 : 2;
  ma_decoder_config config = ma_decoder_config_init(ma_format_f32, channels, kRate);
  ma_decoder decoder;
  if (ma_decoder_init_file(file.c_str(), &config, &decoder) != MA_SUCCESS) {
    spdlog::warn("Sound: can't read {}", path);
    impl->failed[key] = true;
    return nullptr;
  }
  auto clip = std::make_unique<SoundClip>();
  clip->channels = static_cast<int>(channels);
  std::vector<float> chunk(4096 * channels);
  for (;;) {
    ma_uint64 got = 0;
    ma_decoder_read_pcm_frames(&decoder, chunk.data(), 4096, &got);
    if (got == 0)
      break;
    clip->samples.insert(clip->samples.end(), chunk.begin(), chunk.begin() + got * channels);
  }
  ma_decoder_uninit(&decoder);
  if (clip->samples.empty()) {
    spdlog::warn("Sound: {} has no sound in it", path);
    impl->failed[key] = true;
    return nullptr;
  }
  const SoundClip *kept = clip.get();
  impl->clips.emplace(key, std::move(clip));
  return kept;
}

int AudioDevice::play(const SoundClip *clip, const SoundParams &params) {
  if (!clip || impl->output == Output::Silent)
    return 0;
  Command c;
  c.kind = Command::Kind::Start;
  c.id = impl->nextId.fetch_add(1);
  c.clip = clip;
  c.params = params;
  return impl->send(c) ? c.id : 0;
}

void AudioDevice::stop(int id) {
  Command c;
  c.kind = Command::Kind::Stop;
  c.id = id;
  impl->send(c);
}

void AudioDevice::move(int id, glm::vec3 position, glm::vec3 velocity) {
  Command c;
  c.kind = Command::Kind::Move;
  c.id = id;
  c.params.position = position;
  c.params.velocity = velocity;
  impl->send(c);
}

void AudioDevice::stopAll() {
  Command c;
  c.kind = Command::Kind::StopAll;
  impl->send(c);
}

void AudioDevice::setListener(const AudioListener &listener) {
  Command c;
  c.kind = Command::Kind::Listener;
  c.listener = listener;
  impl->send(c);
}

void AudioDevice::setTuning(const AudioMixerTuning &tuning) {
  Command c;
  c.kind = Command::Kind::Tuning;
  c.tuning = tuning;
  impl->send(c);
}

void AudioDevice::advance(float seconds) {
  if (!impl->encoderOpen)
    return;
  impl->drain();
  if (seconds <= 0.0f)
    return;
  impl->owed += static_cast<double>(seconds) * kRate;
  int frames = static_cast<int>(impl->owed);
  impl->owed -= frames;
  // In blocks like a sound card's, so a long frame (the first one takes
  // seconds) doesn't become one block with one set of levels.
  constexpr int kBlock = 1024;
  impl->scratch.resize(2 * kBlock);
  while (frames > 0) {
    const int n = std::min(frames, kBlock);
    impl->render(impl->scratch.data(), n);
    ma_encoder_write_pcm_frames(&impl->encoder, impl->scratch.data(), n, nullptr);
    frames -= n;
  }
}

float AudioDevice::windowTop() const { return impl->top.load(std::memory_order_relaxed); }
float AudioDevice::ringing() const { return impl->ring_.load(std::memory_order_relaxed); }
int AudioDevice::voices() const { return impl->voiceCount.load(std::memory_order_relaxed); }

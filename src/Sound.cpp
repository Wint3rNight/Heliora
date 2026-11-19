// VulkanRenderer's sound (step 1.5c): the device, the clips, and what in the
// game makes a sound. The mixing itself is AudioMixer's (heliora_core).

#include "VulkanRenderer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>

namespace {
const char *kImpactFolder = "Resources/Sounds/kenney_impact/";

// The pack's files come in fives: footstep_wood_000.ogg to _004.
std::vector<std::string> fiveOf(const std::string &name) {
  std::vector<std::string> paths;
  for (int i = 0; i < 5; ++i)
    paths.push_back(std::format("{}{}_{:03}.ogg", kImpactFolder, name, i));
  return paths;
}

// Loudness, dB SPL at 1 m (est.): a footstep a little louder the faster
// it's taken, a hit by how hard it was.
float footstepLoudness(float speed) { return 58.0f + 10.0f * std::log10(1.0f + speed); }
float impactLoudness(float impulse) {
  return std::clamp(80.0f + 10.0f * std::log10(std::max(impulse, 1.0f) / 10.0f), 65.0f, 115.0f);
}
constexpr float kHeavyImpact = 100.0f;  // N s: past this, the heavy recordings
constexpr int kImpactSoundsPerFrame = 6; // the loudest few; a collapse is hundreds of hits

} // namespace

VulkanRenderer::SoundSurface VulkanRenderer::surfaceOf(const std::optional<std::string> &material) {
  if (!material)
    return SoundSurface::Stone;
  if (*material == "wood")
    return SoundSurface::Wood;
  if (*material == "metal")
    return SoundSurface::Metal;
  if (*material == "rubber")
    return SoundSurface::Rubber;
  if (*material == "ice")
    return SoundSurface::Ice;
  return SoundSurface::Stone;
}

VulkanRenderer::SoundSurface VulkanRenderer::surfaceUnder(uint64_t userData) const {
  if (userData & kRubbleBodyFlag)
    return SoundSurface::Rubble;
  if (userData & kBreakBodyFlag) {
    const uint64_t index = userData & ~kBreakBodyFlag;
    if (index < breakBodies.size() && !breakBodies[index].pieces.empty()) {
      const int object = breakPieces[breakBodies[index].pieces.front()].object;
      if (object >= 0 && object < static_cast<int>(breakables.size()))
        return breakables[object].surface;
    }
    return SoundSurface::Stone;
  }
  if (userData < physicsLinks.size())
    return physicsLinks[userData].surface;
  return SoundSurface::Stone;
}

void VulkanRenderer::startSound() {
  if (!audio) {
    // Captures run unattended: never out loud; into a file if asked.
    const char *capture = std::getenv("HELIORA_CAPTURE");
    const char *file = std::getenv("HELIORA_CAPTURE_AUDIO");
    const AudioDevice::Output output = !capture ? AudioDevice::Output::Speakers
                                       : file   ? AudioDevice::Output::File
                                                : AudioDevice::Output::Silent;
    audio = std::make_unique<AudioDevice>(output, file ? file : "");
    auto load = [&](const std::vector<std::string> &names) {
      SoundSet set;
      for (const std::string &name : names)
        for (const std::string &path : fiveOf(name))
          if (const SoundClip *clip = audio->load(path))
            set.clips.push_back(clip);
      return set;
    };
    using S = SoundSurface;
    auto at = [](S s) { return static_cast<size_t>(s); };
    // Kenney's pack has no metal or gravel footsteps: concrete stands in for
    // metal, and the crunch of snow for rubble.
    footstepSounds[at(S::Stone)] = load({"footstep_concrete"});
    footstepSounds[at(S::Wood)] = load({"footstep_wood"});
    footstepSounds[at(S::Metal)] = load({"footstep_concrete"});
    footstepSounds[at(S::Rubber)] = load({"footstep_carpet"});
    footstepSounds[at(S::Ice)] = load({"footstep_concrete"});
    footstepSounds[at(S::Rubble)] = load({"footstep_snow"});
    knockSounds[at(S::Stone)] = load({"impactMining"});
    knockSounds[at(S::Wood)] = load({"impactWood_light", "impactPlank_medium"});
    knockSounds[at(S::Metal)] = load({"impactMetal_light"});
    knockSounds[at(S::Rubber)] = load({"impactSoft_medium"});
    knockSounds[at(S::Ice)] = load({"impactGlass_light"});
    knockSounds[at(S::Rubble)] = load({"impactMining"});
    breakSounds[at(S::Stone)] = load({"impactMining", "impactPlate_heavy"});
    breakSounds[at(S::Wood)] = load({"impactWood_heavy"});
    breakSounds[at(S::Metal)] = load({"impactMetal_heavy"});
    breakSounds[at(S::Rubber)] = load({"impactSoft_heavy"});
    breakSounds[at(S::Ice)] = load({"impactGlass_heavy"});
    breakSounds[at(S::Rubble)] = load({"impactMining"});
  }
  soundPanel.open = true;
  listenerPlaced = false;
  sentTuning = AudioMixerTuning{};
  audio->setTuning(sentTuning);
}

void VulkanRenderer::stopSound() {
  if (audio)
    audio->stopAll();
  footsteps.clear();
  soundPanel.open = false;
}

int VulkanRenderer::playSound(const SoundSet &set, glm::vec3 where, float loudness) {
  if (!audio || set.clips.empty())
    return 0;
  // xorshift32: which of the recordings this time.
  soundDice ^= soundDice << 13;
  soundDice ^= soundDice >> 17;
  soundDice ^= soundDice << 5;
  SoundParams params;
  params.position = where;
  params.loudness = loudness;
  // A little pitch spread, so the same recording twice doesn't sound copied.
  params.pitch = 0.94f + 0.12f * static_cast<float>(soundDice % 1000) / 1000.0f;
  return audio->play(set.clips[soundDice % set.clips.size()], params);
}

void VulkanRenderer::updateSound(float seconds) {
  // The game starts in play mode without a Play press: start here then.
  if (!soundPanel.open)
    startSound();
  AudioMixerTuning tuning;
  tuning.volume = soundPanel.mute ? 0.0f : soundPanel.volume;
  if (!(tuning == sentTuning)) {
    audio->setTuning(tuning);
    sentTuning = tuning;
  }

  // The listener is the camera: the player's, or the one this frame is
  // drawn from if nobody is played.
  AudioListener listener;
  if (playerController) {
    const ThirdPersonCamera &view = playerController->camera();
    listener.position = view.position();
    listener.forward = view.forward();
  } else {
    const glm::mat4 toCamera = glm::inverse(sceneUbo.view);
    listener.position = glm::vec3(toCamera[3]);
    listener.forward = -glm::vec3(toCamera[2]);
    listener.up = glm::vec3(toCamera[1]);
  }
  // Its velocity for Doppler: none across a jump (a respawn, the camera
  // popping in at a wall), and never faster than a car.
  const glm::vec3 moved = listener.position - lastListener;
  if (listenerPlaced && seconds > 0.0f && glm::length(moved) < 3.0f) {
    listener.velocity = moved / seconds;
    if (const float speed = glm::length(listener.velocity); speed > 50.0f)
      listener.velocity *= 50.0f / speed;
  }
  const bool first = !listenerPlaced;
  lastListener = listener.position;
  listenerPlaced = true;
  if (seconds > 0.0f || first)
    audio->setListener(listener);

  for (const Footstep &step : footsteps)
    playSound(footstepSounds[static_cast<size_t>(surfaceUnder(step.ground))], step.point,
              footstepLoudness(step.speed));
  footsteps.clear();

  audio->advance(seconds); // a capture's file keeps up with the game
  soundPanel.windowTop = audio->windowTop();
  soundPanel.voices = audio->voices();
  soundPanel.ringing = audio->ringing();
}

void VulkanRenderer::soundImpacts(const std::vector<PhysicsImpact> &impacts) {
  if (!audio || impacts.empty())
    return;
  // The hardest few this frame, and not two on the same spot.
  std::vector<const PhysicsImpact *> hardest;
  for (const PhysicsImpact &impact : impacts)
    hardest.push_back(&impact);
  std::sort(hardest.begin(), hardest.end(),
            [](const PhysicsImpact *a, const PhysicsImpact *b) { return a->impulse > b->impulse; });
  std::vector<glm::vec3> played;
  for (const PhysicsImpact *impact : hardest) {
    if (static_cast<int>(played.size()) >= kImpactSoundsPerFrame)
      break;
    const bool near = std::any_of(played.begin(), played.end(), [&](glm::vec3 p) {
      return glm::distance(p, impact->point) < 0.5f;
    });
    if (near)
      continue;
    const size_t surface = static_cast<size_t>(surfaceUnder(impact->userData));
    const SoundSet &set =
        impact->impulse >= kHeavyImpact ? breakSounds[surface] : knockSounds[surface];
    if (playSound(set, impact->point, impactLoudness(impact->impulse)))
      played.push_back(impact->point);
  }
}

// VulkanRenderer's hits (from step 1.6's firing, kept for melee and the
// powers): what a hit throws off (sparks, grit, splinters, blood), damage
// that builds up in a breakable piece until it breaks loose (PieceDamage.h),
// and a character hurt, with the death clip picked by where the blow came
// from. The maths is heliora_core's.

#include "VulkanRenderer.h"

#include "Particles.h"
#include "ScenePhysics.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace {

// What a hit throws off, back out of the surface (est., by eye; the
// particles only glow, so dust and splinters are dim and short): sparks
// off metal and stone, grit off stone and dirt, splinters off wood.
void hitDebris(std::vector<ParticleSpawn> &out, ParticleRandom &random, glm::vec3 point,
               glm::vec3 normal, Matter matter, float energy) {
  const float scale = std::clamp(energy / 1500.0f, 0.3f, 1.5f); // an AK round's 1.5 kJ is 1
  ParticleSpawn kind;
  kind.position = point + normal * 0.01f;
  auto burst = [&](int count, float slow, float fast, float spread) {
    emitInCone(out, kind.position, normal, spread, std::max(1, static_cast<int>(count * scale)), slow,
               fast, kind, random);
  };
  if (matter == Matter::Metal || matter == Matter::Stone) {
    kind.colour = matter == Matter::Metal ? glm::vec3(9.0f, 5.5f, 1.8f) : glm::vec3(5.0f, 3.0f, 1.0f);
    kind.size = 0.012f;
    kind.life = 0.25f;
    kind.drag = 2.0f;
    kind.gravity = 1.0f;
    burst(matter == Matter::Metal ? 18 : 6, 3.0f, 9.0f, glm::radians(55.0f));
  }
  if (matter == Matter::Stone || matter == Matter::Dirt || matter == Matter::Ice) {
    kind.colour = matter == Matter::Ice ? glm::vec3(0.7f, 0.75f, 0.8f) : glm::vec3(0.45f, 0.42f, 0.37f);
    kind.opacity = 0.5f; // a puff laid over, not glowing (est.)
    kind.size = 0.06f;
    kind.life = 0.9f;
    kind.drag = 4.0f;
    kind.swirl = 0.6f;
    kind.gravity = 0.15f;
    burst(14, 0.8f, 3.0f, glm::radians(40.0f));
    kind.size = 0.015f; // grit
    kind.opacity = 0.9f;
    kind.life = 0.6f;
    kind.drag = 1.0f;
    kind.swirl = 0.0f;
    kind.gravity = 1.0f;
    burst(10, 2.0f, 6.0f, glm::radians(35.0f));
  }
  if (matter == Matter::Wood) {
    kind.colour = glm::vec3(0.45f, 0.32f, 0.18f);
    kind.opacity = 0.9f; // splinters
    kind.size = 0.025f;
    kind.life = 0.8f;
    kind.drag = 1.5f;
    kind.gravity = 1.0f;
    burst(14, 2.0f, 6.0f, glm::radians(45.0f));
    kind.size = 0.05f; // a puff of wood dust
    kind.opacity = 0.4f;
    kind.colour = glm::vec3(0.5f, 0.42f, 0.3f);
    kind.life = 0.6f;
    kind.drag = 5.0f;
    kind.swirl = 0.4f;
    kind.gravity = 0.1f;
    burst(6, 0.5f, 1.5f, glm::radians(40.0f));
  }
  if (matter == Matter::Flesh) {
    kind.colour = glm::vec3(0.35f, 0.02f, 0.02f);
    kind.opacity = 0.9f;
    kind.size = 0.03f;
    kind.life = 0.4f;
    kind.drag = 3.0f;
    kind.gravity = 1.0f;
    burst(10, 1.0f, 3.0f, glm::radians(35.0f));
  }
}

} // namespace

void VulkanRenderer::hitEffects(glm::vec3 point, glm::vec3 normal, Matter matter, float energy) {
  static ParticleRandom random(0x51ed2701u);
  hitDebris(particleSpawns, random, point, normal, matter, energy);
}

bool VulkanRenderer::damagePiece(int p, float energy, glm::vec3 point, glm::vec3 dir, float impulse,
                                  std::vector<BreakHit> &broke) {
  BreakPiece &piece = breakPieces[p];
  if (!(piece.damage.toughness > 0.0f)) {
    const SceneModelEntry &entry = currentScene.models[breakables[piece.object].entry.index];
    const std::optional<PhysicsMaterial> material = findMaterial(entry.material.value_or("stone"));
    piece.damage.toughness = pieceToughness(piece.shape->volume, material ? material->damageEnergy : 0.0f);
  }
  if (!(piece.damage.toughness > 0.0f))
    return false; // damage never breaks it
  if (energy < 0.0f) // all it has left
    energy = std::max(piece.damage.toughness - piece.damage.energy, 0.0f) + 1e-3f;
  stressDirty = true; // its links are weaker now
  if (!piece.damage.add(energy))
    return false;
  spdlog::info("A piece ({:.3f} m^3) broke after {:.0f} J", piece.shape->volume, piece.damage.energy);
  for (int link : pieceLinks[p])
    breakLinks[link].broken = true;
  for (int link : pieceGroundLinks[p])
    groundLinks[link].broken = true;
  piece.anchor = false; // broken loose: nothing holds it in place now
  queueBulletSplit(p, point, dir);
  broke.push_back({point, dir, impulse, PhysicsBodyId{}, glm::vec3(0.0f)});
  return true;
}

bool VulkanRenderer::hurtCharacter(int index, HitZone zone, float points, glm::vec3 direction) {
  if (index < 0 || index >= static_cast<int>(characters.size()))
    return false;
  LiveCharacter &c = characters[index];
  if (c.health.dead())
    return false;
  c.hitThisFrame = true;
  if (!c.health.wound(points)) {
    spdlog::info("Character {} hit: {:.0f} points, {:.0f} left", index, points, c.health.points);
    return false;
  }
  // Which way he goes down: by where the blow came from against how he
  // faces, a blow to the head from the front falling back, crouched
  // crumpling where he is (the right's clip for either side).
  const glm::vec3 left = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), c.facing));
  const float along = glm::dot(direction, c.facing);
  c.deathClip = c.crouched                             ? ClipSlot::DeathCrouched
                : along > 0.5f                         ? ClipSlot::DeathBack
                : zone == HitZone::Head                ? ClipSlot::DeathHead
                : along < -0.5f                        ? ClipSlot::DeathFront
                : glm::dot(direction, left) > 0.0f     ? ClipSlot::DeathRight
                                                       : ClipSlot::DeathFront;
  spdlog::info("Character {} killed ({})", index, clipSlotName(c.deathClip));
  return true;
}

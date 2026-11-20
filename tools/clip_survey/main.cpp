// tools/clip_survey: every animation clip measured and drawn, to pick the
// one for each move. Run by hand (cmake --build build --target clip_survey):
//
//   clip_survey <model.fbx> <out folder> <clip.fbx>...
//
// For each clip, onto the model's skeleton as the game reads it: a line in
// <out>/survey.tsv (length; how fast and which way the hips travel; where
// the hands are against the hips; how far apart; how far the spine leans;
// how the chest and hips face; how low the hips sit) and a picture,
// <out>/<n>.ppm, of the skeleton from the front and the side at four
// moments (right arm red, left green, legs blue, spine white).

#include "CharacterImport.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

int boneNamed(const Skeleton &skeleton, std::string_view name) {
  for (int i = 0; i < static_cast<int>(skeleton.bones.size()); ++i) {
    const std::string_view bone = skeleton.bones[i].name;
    if (bone.size() >= name.size() && bone.substr(bone.size() - name.size()) == name &&
        (bone.size() == name.size() || bone[bone.size() - name.size() - 1] == ':'))
      return i;
  }
  return -1;
}

struct Picture {
  int width, height;
  std::vector<unsigned char> rgb;
  Picture(int w, int h) : width(w), height(h), rgb(static_cast<size_t>(w * h * 3), 24) {}
  void dot(int x, int y, glm::u8vec3 c) {
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        const int px = x + dx, py = y + dy;
        if (px < 0 || py < 0 || px >= width || py >= height)
          continue;
        unsigned char *p = &rgb[static_cast<size_t>((py * width + px) * 3)];
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
      }
  }
  void line(glm::vec2 a, glm::vec2 b, glm::u8vec3 c) {
    const int steps = std::max(1, static_cast<int>(glm::length(b - a)));
    for (int i = 0; i <= steps; ++i) {
      const glm::vec2 p = glm::mix(a, b, static_cast<float>(i) / steps);
      dot(static_cast<int>(p.x), static_cast<int>(p.y), c);
    }
  }
  void save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  }
};

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: clip_survey <model.fbx> <out folder> <clip.fbx>...\n");
    return 1;
  }
  CharacterDescription description;
  description.model = argv[1];
  std::string error;
  const std::optional<CharacterAsset> asset = importCharacter(description, error);
  if (!asset) {
    std::fprintf(stderr, "can't read the model: %s\n", error.c_str());
    return 1;
  }
  const Skeleton &skeleton = asset->skeleton;
  // Model units to metres: the body 1.78 m tall (Remy, est.).
  const float toMetres = 1.78f / std::max(asset->boundsMax.y - asset->boundsMin.y, 1e-6f);
  const int hips = boneNamed(skeleton, "Hips"), neck = boneNamed(skeleton, "Neck");
  const int rightHand = boneNamed(skeleton, "RightHand"), leftHand = boneNamed(skeleton, "LeftHand");
  const int rightArm = boneNamed(skeleton, "RightArm"), leftArm = boneNamed(skeleton, "LeftArm");
  const int rightLeg = boneNamed(skeleton, "RightUpLeg"), leftLeg = boneNamed(skeleton, "LeftUpLeg");
  if (hips < 0 || neck < 0 || rightHand < 0 || leftHand < 0) {
    std::fprintf(stderr, "the model isn't a Mixamo rig\n");
    return 1;
  }
  // Each bone's colour by the side and limb it's on.
  std::vector<glm::u8vec3> colour(skeleton.bones.size(), glm::u8vec3(230, 230, 230));
  for (size_t i = 0; i < skeleton.bones.size(); ++i) {
    const std::string &n = skeleton.bones[i].name;
    if (n.find("RightShoulder") != std::string::npos || n.find("RightArm") != std::string::npos ||
        n.find("RightForeArm") != std::string::npos || n.find("RightHand") != std::string::npos)
      colour[i] = {240, 70, 60};
    else if (n.find("LeftShoulder") != std::string::npos || n.find("LeftArm") != std::string::npos ||
             n.find("LeftForeArm") != std::string::npos || n.find("LeftHand") != std::string::npos)
      colour[i] = {70, 220, 90};
    else if (n.find("Leg") != std::string::npos || n.find("Foot") != std::string::npos ||
             n.find("Toe") != std::string::npos)
      colour[i] = {80, 140, 250};
  }
  const std::vector<glm::mat4> bind = modelSpace(skeleton, bindPose(skeleton));
  const float standingHips = bind[hips][3].y * toMetres;

  // clip_survey <model> --trace <aiming clip> <clip>...: where a gun gripped
  // as in the aiming clip (lying along its facing) points in each clip, by
  // the right hand's turn, against the line from the right hand to the
  // left; and the hands' heights. Ten moments a clip.
  if (std::string(argv[2]) == "--trace" && argc >= 5) {
    CharacterClip aimEntry;
    aimEntry.file = argv[3];
    const std::optional<CharacterClipData> aim = importCharacterClip(aimEntry, *asset);
    if (!aim)
      return 1;
    Pose aimPose = bindPose(skeleton);
    sampleClip(aim->clip, skeleton, 0.0f, true, aimPose);
    const glm::quat handAim = glm::quat_cast(glm::mat3(modelSpace(skeleton, aimPose)[rightHand]));
    const glm::quat gunInHand = glm::inverse(glm::normalize(handAim)); // the gun along +z there
    auto angles = [](glm::vec3 d) {
      d = glm::normalize(d);
      return glm::vec2(glm::degrees(std::atan2(d.x, d.z)), glm::degrees(std::asin(std::clamp(d.y, -1.0f, 1.0f))));
    };
    for (int a = 4; a < argc; ++a) {
      CharacterClip entry;
      entry.file = argv[a];
      const std::optional<CharacterClipData> data = importCharacterClip(entry, *asset);
      if (!data)
        continue;
      std::printf("%s (%.2f s)\n  time  gun yaw/pitch  hands-line yaw/pitch  rhand up  lhand up\n", argv[a], data->clip.duration);
      for (int k = 0; k < 10; ++k) {
        const float t = data->clip.duration * k / 10.0f;
        Pose pose = bindPose(skeleton);
        sampleClip(data->clip, skeleton, t, true, pose);
        const std::vector<glm::mat4> m = modelSpace(skeleton, pose);
        const glm::quat hand = glm::normalize(glm::quat_cast(glm::mat3(m[rightHand])));
        const glm::vec2 gun = angles(hand * gunInHand * glm::vec3(0, 0, 1));
        const glm::vec2 line = angles(glm::vec3(m[leftHand][3]) - glm::vec3(m[rightHand][3]));
        const float h = m[hips][3].y;
        std::printf("  %4.2f  %6.0f %6.0f    %6.0f %6.0f           %5.2f    %5.2f\n", t, gun.x, gun.y, line.x, line.y,
                    (m[rightHand][3].y - h) * toMetres, (m[leftHand][3].y - h) * toMetres);
      }
    }
    return 0;
  }
  const std::string folder = argv[2];
  std::ofstream table(folder + "/survey.tsv");
  table << "n\tfile\tseconds\tspeed\theading\trhand_up\trhand_ahead\thands_apart\tlean\tchest\thips_turn\thips_low\n";
  for (int a = 3; a < argc; ++a) {
    CharacterClip entry;
    entry.file = argv[a];
    entry.loop = true;
    const std::optional<CharacterClipData> data = importCharacterClip(entry, *asset);
    const int n = a - 3;
    if (!data) {
      table << n << "\t" << argv[a] << "\tunreadable\n";
      continue;
    }
    const float length = data->clip.duration;
    // Travel: the root motion taken out of the hips. Heading 0 is ahead,
    // 90 to the left, -90 to the right, 180 back.
    const glm::vec2 travel = data->rootMotion.total() * toMetres;
    const float speed = length > 0.0f ? glm::length(travel) / length : 0.0f;
    const float heading = glm::length(travel) > 0.05f ? glm::degrees(std::atan2(travel.x, travel.y)) : 0.0f;
    // Averages over the clip, against the hips.
    constexpr int kSamples = 24;
    float up = 0.0f, ahead = 0.0f, apart = 0.0f, lean = 0.0f, chest = 0.0f, hipsTurn = 0.0f, low = 0.0f;
    Picture picture(4 * 2 * 96, 192);
    for (int s = 0; s < kSamples; ++s) {
      const float t = length * static_cast<float>(s) / kSamples;
      Pose pose = bindPose(skeleton);
      sampleClip(data->clip, skeleton, t, true, pose);
      const std::vector<glm::mat4> m = modelSpace(skeleton, pose);
      auto at = [&](int bone) { return glm::vec3(m[bone][3]) * toMetres; };
      const glm::vec3 h = at(hips);
      up += (at(rightHand).y - h.y) / kSamples;
      ahead += (at(rightHand).z - h.z) / kSamples;
      apart += glm::distance(at(rightHand), at(leftHand)) / kSamples;
      const glm::vec3 spine = at(neck) - h;
      lean += glm::degrees(std::atan2(spine.z, spine.y)) / kSamples;
      auto facing = [](glm::vec3 left, glm::vec3 right) {
        const glm::vec3 across = left - right;
        return glm::degrees(std::atan2(-across.z, across.x)); // 0 square, positive turned right
      };
      if (leftArm >= 0 && rightArm >= 0)
        chest += facing(at(leftArm), at(rightArm)) / kSamples;
      if (leftLeg >= 0 && rightLeg >= 0)
        hipsTurn += facing(at(leftLeg), at(rightLeg)) / kSamples;
      low += (standingHips - h.y) / kSamples;
      // Four of the moments drawn: from the front (x left, so mirrored to
      // read as facing us) and from the side (z ahead to the right).
      if (s % (kSamples / 4) == 0) {
        const int column = s / (kSamples / 4);
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
          const int parent = skeleton.bones[i].parent;
          if (parent < 0 || static_cast<int>(i) == hips)
            continue;
          const glm::vec3 p = at(static_cast<int>(i)) - glm::vec3(h.x, 0.0f, h.z);
          const glm::vec3 q = at(parent) - glm::vec3(h.x, 0.0f, h.z);
          if (glm::distance(p, q) > 0.6f)
            continue; // the armature's root to the hips
          auto front = [&](glm::vec3 v) {
            return glm::vec2(column * 192 + 48 - v.x * 90.0f, 182 - v.y * 90.0f);
          };
          auto side = [&](glm::vec3 v) {
            return glm::vec2(column * 192 + 96 + 48 + v.z * 90.0f, 182 - v.y * 90.0f);
          };
          picture.line(front(p), front(q), colour[i]);
          picture.line(side(p), side(q), colour[i]);
        }
      }
    }
    picture.save(folder + "/" + std::to_string(n) + ".ppm");
    char row[512];
    std::snprintf(row, sizeof(row), "%d\t%s\t%.2f\t%.2f\t%.0f\t%.2f\t%.2f\t%.2f\t%.0f\t%.0f\t%.0f\t%.2f\n", n,
                  argv[a], length, speed, heading, up, ahead, apart, lean, chest, hipsTurn, low);
    table << row;
  }
  return 0;
}

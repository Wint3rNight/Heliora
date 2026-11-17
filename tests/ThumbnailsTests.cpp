#include "Thumbnails.h"
#include "TestRunner.h"

#include <algorithm>
#include <array>
#include <cmath>

TEST(a_thumbnail_frames_the_whole_object) {
  // Every corner of the box lands inside the picture, with some margin,
  // and in front of the camera, whatever the box's size and place.
  const std::array<std::pair<glm::vec3, glm::vec3>, 3> boxes = {{
      {glm::vec3(-0.5f), glm::vec3(0.5f)},
      {glm::vec3(-3.0f, 0.0f, -1.0f), glm::vec3(3.0f, 0.2f, 1.0f)}, // a slab
      {glm::vec3(10.0f, 0.0f, 10.0f), glm::vec3(10.4f, 2.0f, 10.4f)}, // tall, off origin
  }};
  for (const auto &[lo, hi] : boxes) {
    const glm::mat4 viewProj = thumbnailViewProj(lo, hi);
    float widest = 0.0f;
    for (int c = 0; c < 8; ++c) {
      const glm::vec3 corner((c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y,
                             (c & 4) ? hi.z : lo.z);
      const glm::vec4 clip = viewProj * glm::vec4(corner, 1.0f);
      CHECK(clip.w > 0.0f);
      const glm::vec3 ndc = glm::vec3(clip) / clip.w;
      CHECK(std::abs(ndc.x) <= 0.95f && std::abs(ndc.y) <= 0.95f);
      CHECK(ndc.z >= 0.0f && ndc.z <= 1.0f);
      widest = std::max({widest, std::abs(ndc.x), std::abs(ndc.y)});
    }
    CHECK(widest > 0.5f); // and fills it, rather than a dot in the middle
  }
}

TEST(a_thumbnails_key_changes_with_its_file) {
  const std::string a = thumbnailKey("Resources/Models/rock.gltf", 1000, 50);
  CHECK(a == thumbnailKey("Resources/Models/rock.gltf", 1000, 50));
  CHECK(a != thumbnailKey("Resources/Models/rock.gltf", 1001, 50)); // edited
  CHECK(a != thumbnailKey("Resources/Models/rock.gltf", 1000, 51));
  CHECK(a != thumbnailKey("Resources/Models/crate.gltf", 1000, 50));
  CHECK(a.size() == 16); // a file name: hex digits only
  CHECK(a.find_first_not_of("0123456789abcdef") == std::string::npos);
}

TEST(atlas_cells_tile_the_atlas) {
  const glm::vec4 first = thumbnailAtlasRect(0);
  CHECK_NEAR(first.x, 0.0, 1e-6);
  CHECK_NEAR(first.y, 0.0, 1e-6);
  CHECK_NEAR(first.z, 1.0 / kThumbnailsPerRow, 1e-6);
  CHECK_NEAR(first.w, 1.0 / kThumbnailsPerRow, 1e-6);
  const glm::vec4 nextRow = thumbnailAtlasRect(kThumbnailsPerRow + 1);
  CHECK_NEAR(nextRow.x, 1.0 / kThumbnailsPerRow, 1e-6);
  CHECK_NEAR(nextRow.y, 1.0 / kThumbnailsPerRow, 1e-6);
  CHECK(kThumbnailsPerRow * kThumbnailSize == kThumbnailAtlasSize);
}

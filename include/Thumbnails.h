#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <string_view>

// Pictures of assets for the Assets panel's tiles: how an object is framed,
// where its picture sits in the shared atlas image, and the name its picture
// is cached under on disk. No Vulkan here.

constexpr int kThumbnailSize = 128;      // pixels, square
constexpr int kThumbnailsPerRow = 16;    // in the atlas
constexpr int kThumbnailAtlasSize = kThumbnailSize * kThumbnailsPerRow;

// A camera looking at the box from the front right and a little above,
// close enough that it fills most of the picture, with Vulkan's depth (0
// to 1, near to far) and its y down.
glm::mat4 thumbnailViewProj(glm::vec3 boundsMin, glm::vec3 boundsMax);
// Where the light comes from, the same for every thumbnail (world space,
// towards the light).
glm::vec3 thumbnailLightDirection();

// A file name for a picture (16 hex digits): from what it shows (a path,
// or "shape:Box") and the file's size and modification time, so an edited
// file gets a new picture.
std::string thumbnailKey(std::string_view source, uint64_t size, int64_t modified);

// Cell i of the atlas as UV corners (u0, v0, u1, v1), row by row.
glm::vec4 thumbnailAtlasRect(int index);

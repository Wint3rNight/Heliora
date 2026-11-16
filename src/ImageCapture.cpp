#include "ImageCapture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>

bool writeCapturePng(const std::string &path, const uint8_t *pixels,
                     uint32_t width, uint32_t height, uint32_t rowPitch,
                     bool bgra) {
  // Alpha dropped: the scene image's alpha is whatever the composite left.
  std::vector<uint8_t> rgb(size_t(width) * height * 3);
  for (uint32_t y = 0; y < height; ++y) {
    const uint8_t *row = pixels + size_t(y) * rowPitch;
    for (uint32_t x = 0; x < width; ++x) {
      const uint8_t *in = row + x * 4;
      uint8_t *out = rgb.data() + (size_t(y) * width + x) * 3;
      out[0] = bgra ? in[2] : in[0];
      out[1] = in[1];
      out[2] = bgra ? in[0] : in[2];
    }
  }
  return stbi_write_png(path.c_str(), int(width), int(height), 3, rgb.data(),
                        int(width) * 3) != 0;
}

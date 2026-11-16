#pragma once

#include <cstdint>
#include <string>

// Writes 8-bit RGBA or BGRA pixels, rowPitch bytes apart, as an RGB PNG.
// For the scene captures (HELIORA_CAPTURE). False if the file can't be
// written.
bool writeCapturePng(const std::string &path, const uint8_t *pixels,
                     uint32_t width, uint32_t height, uint32_t rowPitch,
                     bool bgra);

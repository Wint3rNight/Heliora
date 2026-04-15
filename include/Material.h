#pragma once

// For Phase 1, Material acts as an abstraction layer above raw texture descriptor indices.
// In later phases, this will expand to include shaders, parameters (roughness, metallic), etc.

struct Material {
    int albedoTextureId = 0; // Represents the descriptor set index for the diffuse texture
    // int normalTextureId = 0;
    // int metallicRoughnessTextureId = 0;
    // float roughnessFactor = 1.0f;
    // float metallicFactor = 0.0f;
    // glm::vec4 baseColorFactor = glm::vec4(1.0f);
};

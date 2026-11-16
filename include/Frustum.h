#pragma once

#include <glm/glm.hpp>

// The six planes of a view-projection matrix's clip volume, normals pointing
// inwards and normalised, in the order left, right, bottom, top, near, far.
// The near plane is the depth -1 plane of OpenGL's clip space; with Vulkan's
// 0 to 1 depth that sits beyond the real near plane, so a test against it
// keeps a little extra and never drops what the rasterizer would draw.
void extractFrustumPlanes(const glm::mat4 &viewProj, glm::vec4 planes[6]);

// False only when the sphere is entirely outside one of the planes.
bool sphereInFrustum(const glm::vec4 planes[6], glm::vec3 center, float radius);

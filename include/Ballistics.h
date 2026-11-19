#pragma once

#include <glm/glm.hpp>

// Arcs for what's thrown or launched (grenades in step 1.7). Kept from the
// powers' throw when they went (step 1.5).

// The launch velocity, at this speed, that lands on `to` under gravity: the
// low arc; out of reach, 45 degrees up toward it.
glm::vec3 throwVelocity(glm::vec3 from, glm::vec3 to, float speed, glm::vec3 gravity);

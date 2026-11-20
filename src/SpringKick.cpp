#include "SpringKick.h"

#include <algorithm>
#include <cmath>

void SpringKick::kick(glm::vec3 amount, float peak) {
  // From rest, x = v t e^(-t / peak) is highest at t = peak, at v peak / e.
  velocity += amount * (2.718281828f / std::max(peak, 1e-4f));
}

void SpringKick::update(float dt, float peak) {
  if (!(dt > 0.0f))
    return;
  // x(t) = (x0 + (v0 + w x0) t) e^(-w t), w = 1 / peak.
  const float w = 1.0f / std::max(peak, 1e-4f);
  const float fade = std::exp(-w * dt);
  const glm::vec3 lead = velocity + w * offset;
  const glm::vec3 x = (offset + lead * dt) * fade;
  const glm::vec3 v = (velocity - w * lead * dt) * fade;
  offset = x;
  velocity = v;
}

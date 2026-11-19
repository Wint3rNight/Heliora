#include "Bullets.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace {

// How far past a far face a bullet looks again, so it doesn't find the face
// it just came out of.
constexpr float kNudge = 1e-3f; // m

} // namespace

float ratioOf(const Resistance &r) { return r.a > 0.0f ? r.b / r.a : 0.0f; }

float stoppingDepth(const Resistance &r, float mass, float speed) {
  const float v2 = speed * speed;
  if (r.a <= 0.0f && r.b <= 0.0f)
    return std::numeric_limits<float>::infinity();
  if (r.b <= 0.0f)
    return mass * v2 / (2.0f * r.a);
  if (r.a <= 0.0f)
    return std::numeric_limits<float>::infinity();
  return mass / (2.0f * r.b) * std::log1p(r.b * v2 / r.a);
}

float exitSpeed(const Resistance &r, float mass, float speed, float thickness) {
  if (!(speed > 0.0f))
    return 0.0f;
  if (!(thickness > 0.0f) || (r.a <= 0.0f && r.b <= 0.0f) || !(mass > 0.0f))
    return speed;
  const float v2 = speed * speed;
  float out2 = 0.0f;
  if (r.b <= 0.0f) {
    out2 = v2 - 2.0f * r.a * thickness / mass;
  } else if (r.a <= 0.0f) {
    return speed * std::exp(-r.b * thickness / mass);
  } else {
    // m dv/dx = -(a + b v^2) / v * v, solved for v over the thickness.
    const float k = r.a / r.b;
    out2 = (v2 + k) * std::exp(-2.0f * r.b * thickness / mass) - k;
  }
  return out2 > 0.0f ? std::sqrt(out2) : 0.0f;
}

Resistance ponceletFrom(float mass, float depth, float speed, float ratio) {
  if (!(depth > 0.0f) || !(speed > 0.0f) || !(mass > 0.0f))
    return {};
  if (!(ratio > 0.0f))
    return {mass * speed * speed / (2.0f * depth), 0.0f};
  const float b = mass * std::log1p(ratio * speed * speed) / (2.0f * depth);
  return {b / ratio, b};
}

Resistance ponceletFit(float mass, float depth1, float speed1, float depth2, float speed2) {
  if (speed1 < speed2) {
    std::swap(speed1, speed2);
    std::swap(depth1, depth2);
  }
  if (!(depth1 > 0.0f) || !(depth2 > 0.0f) || !(speed2 > 0.0f) || speed1 <= speed2)
    return ponceletFrom(mass, depth1, speed1, 0.0f);
  // The depths' ratio runs from the speeds' squared ratio (b / a near 0, a
  // constant force) down to 1 (b / a large); b / a by bisection on its log.
  const double wanted = static_cast<double>(depth1) / depth2;
  const double v1 = speed1, v2 = speed2;
  if (wanted >= (v1 * v1) / (v2 * v2))
    return ponceletFrom(mass, depth1, speed1, 0.0f);
  double low = 1e-12, high = 1.0;
  for (int i = 0; i < 100; ++i) {
    const double ratio = std::sqrt(low * high);
    const double got = std::log1p(ratio * v1 * v1) / std::log1p(ratio * v2 * v2);
    (got > wanted ? low : high) = ratio;
  }
  return ponceletFrom(mass, depth1, speed1, static_cast<float>(std::sqrt(low * high)));
}

void stepBullets(std::vector<Bullet> &bullets, float dt, const Air &air, glm::vec3 gravity,
                 const BulletWorld &world, std::vector<BulletImpact> &impacts,
                 const BulletRules &rules) {
  if (!(dt > 0.0f))
    return;
  const int steps = std::max(1, static_cast<int>(std::ceil(dt / std::max(rules.subStep, 1e-5f))));
  const float h = dt / static_cast<float>(steps);
  std::vector<char> spent(bullets.size(), 0);
  for (size_t i = 0; i < bullets.size(); ++i) {
    Bullet &b = bullets[i];
    if (!b.round) {
      spent[i] = 1;
      continue;
    }
    for (int s = 0; s < steps && !spent[i]; ++s) {
      if (glm::length(b.flight.velocity) < rules.slowest || b.age >= rules.longest) {
        spent[i] = 1;
        break;
      }
      const Flight start = b.flight;
      stepFlight(b.flight, b.round->projectile, air, gravity, h);
      b.age += h;
      const float stepLength = std::max(glm::distance(start.position, b.flight.position), 1e-6f);
      const float startSpeed = glm::length(start.velocity);
      const float endSpeed = glm::length(b.flight.velocity);

      // This sub-step's segment, and what's in it, one thing after another.
      glm::vec3 from = start.position, to = b.flight.position;
      float kept = 1.0f; // of its speed, after what it has gone through
      for (int c = 0; c < rules.contactsPerStep; ++c) {
        const glm::vec3 segment = to - from;
        const float length = glm::length(segment);
        if (length < 1e-6f)
          break;
        const std::optional<BulletContact> hit = world(from, to);
        if (!hit)
          break;
        const glm::vec3 dir = segment / length;
        const float along =
            std::clamp(glm::distance(start.position, hit->point) / stepLength, 0.0f, 1.0f);
        const float speedIn = (startSpeed + (endSpeed - startSpeed) * along) * kept;
        const Resistance &r = b.round->resistance[static_cast<size_t>(hit->matter)];
        float out = exitSpeed(r, b.round->mass, speedIn, hit->thickness);
        if (out < rules.slowest)
          out = 0.0f;

        BulletImpact impact;
        impact.point = hit->point;
        impact.normal = hit->normal;
        impact.direction = dir;
        impact.body = hit->body;
        impact.matter = hit->matter;
        impact.speedIn = speedIn;
        impact.speedOut = out;
        impact.mass = b.round->mass;
        impact.thickness =
            out > 0.0f ? hit->thickness
                       : std::min(hit->thickness, stoppingDepth(r, b.round->mass, speedIn));
        impact.stopped = out <= 0.0f;
        impact.owner = b.owner;
        impacts.push_back(impact);
        if (impact.stopped) {
          b.flight.position = hit->point;
          spent[i] = 1;
          break;
        }
        // Out the far side, slower, and on along the rest of the segment as
        // far as the time left takes it at the new speed.
        const glm::vec3 exit = hit->point + dir * hit->thickness;
        const float rest = (length - hit->distance - hit->thickness) * out / speedIn;
        kept *= out / speedIn;
        from = exit + dir * kNudge;
        if (rest <= kNudge) {
          b.flight.position = from;
          break;
        }
        to = exit + dir * rest;
        b.flight.position = to;
      }
      b.flight.velocity *= kept;
    }
  }
  size_t kept = 0;
  for (size_t i = 0; i < bullets.size(); ++i)
    if (!spent[i])
      bullets[kept++] = bullets[i];
  bullets.resize(kept);
}

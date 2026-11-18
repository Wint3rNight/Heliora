#include "BreakGraph.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

int root(std::vector<int> &parent, int i) {
  while (parent[i] != i)
    i = parent[i] = parent[parent[i]];
  return i;
}

// A convex polygon (counter-clockwise in its plane) cut down to the part on
// the inner side of the line from a to b, all in 2D.
std::vector<glm::vec2> clipToEdge(const std::vector<glm::vec2> &polygon, glm::vec2 a,
                                  glm::vec2 b) {
  auto side = [&](glm::vec2 p) {
    const glm::vec2 e = b - a, q = p - a;
    return e.x * q.y - e.y * q.x; // > 0: left of a -> b, inside
  };
  std::vector<glm::vec2> out;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const glm::vec2 p = polygon[i], q = polygon[(i + 1) % polygon.size()];
    const float sp = side(p), sq = side(q);
    if (sp >= 0.0f)
      out.push_back(p);
    if ((sp >= 0.0f) != (sq >= 0.0f))
      out.push_back(p + (q - p) * (sp / (sp - sq)));
  }
  return out;
}

float area2d(const std::vector<glm::vec2> &polygon, glm::vec2 *middle) {
  float area = 0.0f;
  glm::vec2 weighted(0.0f);
  for (size_t i = 1; i + 1 < polygon.size(); ++i) {
    const glm::vec2 a = polygon[0], b = polygon[i], c = polygon[i + 1];
    const float t = 0.5f * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    area += t;
    weighted += t * (a + b + c) / 3.0f;
  }
  if (middle)
    *middle = area != 0.0f ? weighted / area : glm::vec2(0.0f);
  return area;
}

struct Bounds {
  glm::vec3 lo{1e30f}, hi{-1e30f};
};

Bounds boundsOf(const PlacedPiece &piece) {
  Bounds b;
  for (const FractureFace &face : piece.faces)
    for (const glm::vec3 &p : face.points) {
      b.lo = glm::min(b.lo, p);
      b.hi = glm::max(b.hi, p);
    }
  return b;
}

// The overlap of two outside faces lying against each other, or 0, and its
// corners if asked.
float faceOverlap(const FractureFace &a, const FractureFace &b, float tolerance,
                  glm::vec3 &centre, std::vector<glm::vec3> *corners = nullptr) {
  if (glm::dot(a.normal, b.normal) > -0.999f)
    return 0.0f;
  // b's plane within tolerance of a's (b lies on a's outside).
  const float apart = glm::dot(a.normal, b.points[0] - a.points[0]);
  if (apart < -tolerance || apart > tolerance)
    return 0.0f;
  // Both in a's plane: u, v with u x v = a's normal, so a stays
  // counter-clockwise; b faces the other way, so its order is reversed.
  const glm::vec3 helper = std::fabs(a.normal.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
  const glm::vec3 u = glm::normalize(glm::cross(helper, a.normal));
  const glm::vec3 v = glm::cross(a.normal, u);
  const glm::vec3 origin = a.points[0];
  auto flat = [&](const glm::vec3 &p) { return glm::vec2(glm::dot(p - origin, u), glm::dot(p - origin, v)); };
  std::vector<glm::vec2> polygon;
  for (const glm::vec3 &p : a.points)
    polygon.push_back(flat(p));
  std::vector<glm::vec2> other;
  for (auto it = b.points.rbegin(); it != b.points.rend(); ++it)
    other.push_back(flat(*it));
  for (size_t i = 0; i < other.size() && polygon.size() >= 3; ++i)
    polygon = clipToEdge(polygon, other[i], other[(i + 1) % other.size()]);
  if (polygon.size() < 3)
    return 0.0f;
  glm::vec2 middle;
  const float area = area2d(polygon, &middle);
  centre = origin + u * middle.x + v * middle.y + a.normal * (0.5f * apart);
  if (corners) {
    corners->clear();
    for (const glm::vec2 &p : polygon)
      corners->push_back(origin + u * p.x + v * p.y + a.normal * (0.5f * apart));
  }
  return std::max(area, 0.0f);
}

} // namespace

std::vector<BreakLink> cutLinks(const Fracture &fracture, float strengthPerArea, int firstPiece) {
  float mean = 0.0f;
  for (const FractureLink &l : fracture.links)
    mean += l.area;
  mean /= static_cast<float>(std::max<size_t>(fracture.links.size(), 1));
  std::vector<BreakLink> links;
  for (const FractureLink &l : fracture.links)
    links.push_back({firstPiece + l.a, firstPiece + l.b,
                     std::max(l.area, 0.25f * mean) * strengthPerArea});
  return links;
}

std::vector<BreakGroup> splitGroups(int pieceCount, const std::vector<BreakLink> &links,
                                    const std::vector<bool> &anchors) {
  std::vector<int> parent(pieceCount);
  std::iota(parent.begin(), parent.end(), 0);
  for (const BreakLink &l : links)
    if (!l.broken)
      parent[root(parent, l.a)] = root(parent, l.b);
  std::vector<int> groupOf(pieceCount, -1);
  std::vector<BreakGroup> groups;
  for (int i = 0; i < pieceCount; ++i) {
    const int r = root(parent, i);
    if (groupOf[r] < 0) {
      groupOf[r] = static_cast<int>(groups.size());
      groups.emplace_back();
    }
    BreakGroup &g = groups[groupOf[r]];
    g.pieces.push_back(i);
    g.anchored = g.anchored || (i < static_cast<int>(anchors.size()) && anchors[i]);
  }
  return groups;
}

std::vector<int> linksBrokenBy(const std::vector<BreakLink> &links,
                               const std::vector<LinkAt> &candidates, glm::vec3 hit,
                               float impulse) {
  std::vector<int> broken;
  for (const LinkAt &c : candidates) {
    const BreakLink &l = links[c.link];
    if (l.broken)
      continue;
    const float d = glm::distance(c.position, hit) / kBreakSpread;
    if (impulse / (1.0f + d * d) > l.strength)
      broken.push_back(c.link);
  }
  return broken;
}

PlacedPiece placePiece(const FracturePiece &piece, const glm::mat4 &where, int object) {
  PlacedPiece out;
  out.object = object;
  const glm::mat3 normals = glm::transpose(glm::inverse(glm::mat3(where)));
  for (const FractureFace &face : piece.faces) {
    FractureFace placed;
    placed.neighbour = face.neighbour;
    placed.normal = glm::normalize(normals * face.normal);
    for (const glm::vec3 &p : face.points)
      placed.points.push_back(glm::vec3(where * glm::vec4(p, 1.0f)));
    out.faces.push_back(std::move(placed));
  }
  return out;
}

std::vector<PieceTouch> touchingPieces(const std::vector<PlacedPiece> &pieces, float tolerance) {
  std::vector<Bounds> bounds;
  for (const PlacedPiece &p : pieces)
    bounds.push_back(boundsOf(p));
  std::vector<PieceTouch> touches;
  for (int i = 0; i < static_cast<int>(pieces.size()); ++i)
    for (int j = i + 1; j < static_cast<int>(pieces.size()); ++j) {
      if (pieces[i].object == pieces[j].object)
        continue;
      const Bounds &a = bounds[i], &b = bounds[j];
      if (glm::any(glm::greaterThan(a.lo, b.hi + tolerance)) ||
          glm::any(glm::greaterThan(b.lo, a.hi + tolerance)))
        continue;
      float area = 0.0f, biggest = 0.0f;
      glm::vec3 weighted(0.0f);
      LinkFace face;
      std::vector<glm::vec3> corners;
      for (const FractureFace &fa : pieces[i].faces) {
        if (fa.neighbour != kOutsideFace)
          continue;
        for (const FractureFace &fb : pieces[j].faces) {
          if (fb.neighbour != kOutsideFace)
            continue;
          glm::vec3 centre;
          const float overlap = faceOverlap(fa, fb, tolerance, centre, &corners);
          if (overlap > 1e-6f) {
            area += overlap;
            weighted += centre * overlap;
            if (overlap > biggest) {
              biggest = overlap;
              face = linkFace(corners, fa.normal);
            }
          }
        }
      }
      if (area > 1e-6f)
        touches.push_back({i, j, area, weighted / area, face});
    }
  return touches;
}

float touchArea(const PlacedPiece &a, const PlacedPiece &b, float tolerance, glm::vec3 &centre) {
  float area = 0.0f;
  glm::vec3 weighted(0.0f);
  for (const FractureFace &fa : a.faces)
    for (const FractureFace &fb : b.faces) {
      glm::vec3 middle;
      const float overlap = faceOverlap(fa, fb, tolerance, middle);
      if (overlap > 1e-7f) {
        area += overlap;
        weighted += middle * overlap;
      }
    }
  if (area > 0.0f)
    centre = weighted / area;
  return area;
}

bool staysPut(const BreakGroup &group, const std::vector<int> &objectOf,
              const std::vector<float> &volume, const std::vector<float> &objectVolume) {
  if (!group.anchored)
    return false;
  std::vector<float> held(objectVolume.size(), 0.0f);
  for (int p : group.pieces)
    held[objectOf[p]] += volume[p];
  for (size_t o = 0; o < held.size(); ++o)
    if (held[o] > 0.0f && held[o] >= kStructuralShare * objectVolume[o])
      return true;
  return false;
}

std::vector<float> shareHit(float momentum, const std::vector<float> &distances) {
  std::vector<float> shares;
  float total = 0.0f;
  for (float d : distances) {
    const float x = d / kBreakSpread;
    shares.push_back(1.0f / (1.0f + x * x));
    total += shares.back();
  }
  for (float &s : shares)
    s *= momentum / total;
  return shares;
}

void varyStrengths(std::vector<BreakLink> &links, uint32_t seed, float amount) {
  // A small hash per link (splitmix32), so the same seed gives the same
  // strengths whatever order they're read in.
  for (size_t i = 0; i < links.size(); ++i) {
    uint32_t x = seed ^ static_cast<uint32_t>(i * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    const float unit = static_cast<float>(x >> 8) / 16777216.0f; // [0, 1)
    links[i].strength *= 1.0f + amount * (2.0f * unit - 1.0f);
  }
}

float splitImpulse(float volume, float strengthPerArea) {
  return std::pow(std::max(volume, 0.0f), 2.0f / 3.0f) * strengthPerArea;
}

bool restsOn(const PlacedPiece &piece, float height, float tolerance) {
  for (const FractureFace &face : piece.faces) {
    if (face.neighbour != kOutsideFace || face.normal.y > -0.9f)
      continue;
    if (std::all_of(face.points.begin(), face.points.end(),
                    [&](const glm::vec3 &p) { return std::fabs(p.y - height) <= tolerance; }))
      return true;
  }
  return false;
}

LinkFace touchFace(const PlacedPiece &a, const PlacedPiece &b, float tolerance) {
  LinkFace face;
  float biggest = 0.0f;
  std::vector<glm::vec3> corners;
  for (const FractureFace &fa : a.faces)
    for (const FractureFace &fb : b.faces) {
      glm::vec3 middle;
      const float overlap = faceOverlap(fa, fb, tolerance, middle, &corners);
      if (overlap > 1e-7f && overlap > biggest) {
        biggest = overlap;
        face = linkFace(corners, fa.normal);
      }
    }
  return face;
}

std::vector<LinkFace> cutFaces(const Fracture &fracture) {
  std::vector<LinkFace> faces;
  faces.reserve(fracture.links.size());
  for (const FractureLink &link : fracture.links) {
    LinkFace face;
    for (const FractureFace &f : fracture.pieces[link.a].faces)
      if (f.neighbour == link.b) {
        face = linkFace(f.points, f.normal);
        break;
      }
    faces.push_back(face);
  }
  return faces;
}

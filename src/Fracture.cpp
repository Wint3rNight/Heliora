#include "Fracture.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

// PCG32: the same numbers on every machine and standard library, unlike the
// standard distributions.
struct Pcg32 {
  uint64_t state;
  uint64_t inc;
  explicit Pcg32(uint64_t seed) : state(0), inc(0x14057b7ef767814fULL) {
    next();
    state += seed;
    next();
  }
  uint32_t next() {
    const uint64_t old = state;
    state = old * 6364136223846793005ULL + inc;
    const uint32_t shifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
    const uint32_t rot = static_cast<uint32_t>(old >> 59u);
    return (shifted >> rot) | (shifted << ((32u - rot) & 31u));
  }
  float unit() { return static_cast<float>(next() >> 8) / 16777216.0f; } // [0, 1)
};

using Polyhedron = std::vector<FractureFace>;

// The box as six faces, each counter-clockwise seen from outside (u x v is
// the normal).
Polyhedron boxFaces(glm::vec3 half) {
  const glm::vec3 X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);
  const struct {
    glm::vec3 n, u, v;
  } sides[] = {{X, Y, Z}, {-X, Z, Y}, {Y, Z, X}, {-Y, X, Z}, {Z, X, Y}, {-Z, Y, X}};
  Polyhedron box;
  for (const auto &s : sides) {
    const glm::vec3 c = s.n * half;
    const glm::vec3 u = s.u * half, v = s.v * half;
    box.push_back({{c - u - v, c + u - v, c + u + v, c - u + v}, s.n, -1});
  }
  return box;
}

glm::vec3 faceMiddle(const FractureFace &face) {
  glm::vec3 sum(0.0f);
  for (const glm::vec3 &p : face.points)
    sum += p;
  return sum / static_cast<float>(face.points.size());
}

// Drops corners closer than eps to the one before them.
void dropRepeats(std::vector<glm::vec3> &points, float eps) {
  std::vector<glm::vec3> kept;
  for (const glm::vec3 &p : points)
    if (kept.empty() || glm::distance(kept.back(), p) > eps)
      kept.push_back(p);
  while (kept.size() > 1 && glm::distance(kept.front(), kept.back()) <= eps)
    kept.pop_back();
  points = std::move(kept);
}

// Keeps the part of the polyhedron where dot(n, x) <= d. The cut becomes a
// face onto neighbour. Nothing changes if the plane misses it.
void clip(Polyhedron &poly, glm::vec3 n, float d, int neighbour, float eps) {
  bool outside = false, inside = false;
  for (const FractureFace &face : poly)
    for (const glm::vec3 &p : face.points) {
      const float s = glm::dot(n, p) - d;
      outside = outside || s > eps;
      inside = inside || s < -eps;
    }
  if (!outside)
    return;
  if (!inside) {
    poly.clear();
    return;
  }
  Polyhedron kept;
  std::vector<glm::vec3> cap;
  for (const FractureFace &face : poly) {
    std::vector<glm::vec3> points;
    const size_t m = face.points.size();
    for (size_t i = 0; i < m; ++i) {
      const glm::vec3 a = face.points[i], b = face.points[(i + 1) % m];
      const float da = glm::dot(n, a) - d, db = glm::dot(n, b) - d;
      if (da <= eps)
        points.push_back(a);
      if (std::fabs(da) <= eps)
        cap.push_back(a);
      if ((da < -eps && db > eps) || (da > eps && db < -eps)) {
        const glm::vec3 x = a + (b - a) * (da / (da - db));
        points.push_back(x);
        cap.push_back(x);
      }
    }
    dropRepeats(points, eps);
    if (points.size() >= 3) {
      FractureFace cut = face;
      cut.points = std::move(points);
      if (faceArea(cut) > eps * eps)
        kept.push_back(std::move(cut));
    }
  }
  // The cap: the cut's corners around their middle, counter-clockwise about
  // n, which points out of what's kept.
  std::vector<glm::vec3> corners;
  for (const glm::vec3 &p : cap)
    if (std::none_of(corners.begin(), corners.end(),
                     [&](const glm::vec3 &q) { return glm::distance(p, q) <= eps; }))
      corners.push_back(p);
  if (corners.size() >= 3) {
    glm::vec3 middle(0.0f);
    for (const glm::vec3 &p : corners)
      middle += p;
    middle /= static_cast<float>(corners.size());
    const glm::vec3 helper = std::fabs(n.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    const glm::vec3 u = glm::normalize(glm::cross(helper, n));
    const glm::vec3 v = glm::cross(n, u);
    std::sort(corners.begin(), corners.end(), [&](const glm::vec3 &a, const glm::vec3 &b) {
      return std::atan2(glm::dot(a - middle, v), glm::dot(a - middle, u)) <
             std::atan2(glm::dot(b - middle, v), glm::dot(b - middle, u));
    });
    FractureFace face{corners, n, neighbour};
    if (faceArea(face) > eps * eps)
      kept.push_back(std::move(face));
  }
  poly = std::move(kept);
}

// Volume and its middle: tetrahedra from a point inside to each face's fan.
void measure(FracturePiece &piece) {
  glm::vec3 inside(0.0f);
  size_t count = 0;
  for (const FractureFace &face : piece.faces)
    for (const glm::vec3 &p : face.points) {
      inside += p;
      ++count;
    }
  inside /= static_cast<float>(std::max<size_t>(count, 1));
  double volume = 0.0;
  glm::dvec3 weighted(0.0);
  for (const FractureFace &face : piece.faces)
    for (size_t i = 1; i + 1 < face.points.size(); ++i) {
      const glm::dvec3 a(face.points[0] - inside), b(face.points[i] - inside),
          c(face.points[i + 1] - inside);
      const double v = glm::dot(a, glm::cross(b, c)) / 6.0;
      volume += v;
      weighted += v * (a + b + c) / 4.0;
    }
  piece.volume = static_cast<float>(volume);
  piece.centre = inside + glm::vec3(volume > 0.0 ? weighted / volume : glm::dvec3(0.0));
}

} // namespace

float faceArea(const FractureFace &face) {
  glm::vec3 sum(0.0f);
  for (size_t i = 1; i + 1 < face.points.size(); ++i)
    sum += glm::cross(face.points[i] - face.points[0], face.points[i + 1] - face.points[0]);
  return 0.5f * glm::length(sum);
}

Fracture fractureBox(glm::vec3 size, int count, uint32_t seed) {
  Fracture out;
  count = std::max(count, 1);
  const glm::vec3 half = 0.5f * size;
  const float scale = std::max({size.x, size.y, size.z});
  const float eps = 1e-5f * scale;

  // The points, spread out a little so no piece is a sliver: each at least
  // a quarter of the typical piece's size from the others, when possible.
  Pcg32 random(seed);
  const float spacing = 0.25f * std::cbrt(size.x * size.y * size.z / static_cast<float>(count));
  std::vector<glm::vec3> points;
  while (static_cast<int>(points.size()) < count) {
    glm::vec3 p;
    for (int attempt = 0; attempt < 30; ++attempt) {
      p = (glm::vec3(random.unit(), random.unit(), random.unit()) * 2.0f - 1.0f) * half * 0.98f;
      if (std::all_of(points.begin(), points.end(),
                      [&](const glm::vec3 &q) { return glm::distance(p, q) >= spacing; }))
        break;
    }
    points.push_back(p);
  }

  // Each piece: the box cut by the plane half way to every other point,
  // nearest first, stopping once the planes are too far to reach it.
  std::vector<int> order(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return glm::distance(points[i], points[a]) < glm::distance(points[i], points[b]);
    });
    Polyhedron cell = boxFaces(half);
    float reach = glm::length(size);
    for (int j : order) {
      if (j == static_cast<int>(i))
        continue;
      const glm::vec3 toward = points[j] - points[i];
      const float gap = glm::length(toward);
      if (0.5f * gap > reach)
        break;
      const glm::vec3 n = toward / gap;
      clip(cell, n, glm::dot(n, 0.5f * (points[i] + points[j])), j, eps);
      reach = 0.0f;
      for (const FractureFace &face : cell)
        for (const glm::vec3 &p : face.points)
          reach = std::max(reach, glm::distance(p, points[i]));
    }
    FracturePiece piece;
    piece.faces = std::move(cell);
    measure(piece);
    out.pieces.push_back(std::move(piece));
  }

  // Links: a cut face both pieces have.
  for (int a = 0; a < static_cast<int>(out.pieces.size()); ++a) {
    for (const FractureFace &face : out.pieces[a].faces) {
      const int b = face.neighbour;
      if (b <= a)
        continue;
      const bool shared = std::any_of(out.pieces[b].faces.begin(), out.pieces[b].faces.end(),
                                      [&](const FractureFace &f) { return f.neighbour == a; });
      if (!shared)
        continue;
      const float area = faceArea(face);
      auto it = std::find_if(out.links.begin(), out.links.end(), [&](const FractureLink &l) {
        return l.a == a && l.b == b;
      });
      if (it == out.links.end()) {
        out.links.push_back({a, b, area, faceMiddle(face)});
      } else {
        it->centre = (it->centre * it->area + faceMiddle(face) * area) / (it->area + area);
        it->area += area;
      }
    }
  }
  return out;
}

MeshData makePieceMesh(const FracturePiece &piece, glm::vec3 outside, glm::vec3 inside) {
  MeshData mesh;
  for (const FractureFace &face : piece.faces) {
    const glm::vec3 n = face.normal;
    // u and v across the face, v = cross(n, u); the bitangent follows
    // texture v, which runs down, as in the generated shapes.
    const glm::vec3 helper = std::fabs(n.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    const glm::vec3 u = glm::normalize(glm::cross(helper, n));
    const glm::vec3 v = glm::cross(n, u);
    const glm::vec3 colour = face.neighbour < 0 ? outside : inside;
    const uint32_t first = static_cast<uint32_t>(mesh.vertices.size());
    for (const glm::vec3 &p : face.points)
      mesh.vertices.push_back({p, colour, glm::vec2(glm::dot(p, u), -glm::dot(p, v)), n, u, -v});
    for (uint32_t i = 1; i + 1 < face.points.size(); ++i)
      for (uint32_t k : {0u, i, i + 1})
        mesh.indices.push_back(first + k);
  }
  return mesh;
}

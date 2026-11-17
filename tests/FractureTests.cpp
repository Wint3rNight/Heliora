#include "Fracture.h"
#include "TestRunner.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <utility>

namespace {

const glm::vec3 kWall(2.0f, 1.5f, 0.3f);

float totalVolume(const Fracture &f) {
  float sum = 0.0f;
  for (const FracturePiece &p : f.pieces)
    sum += p.volume;
  return sum;
}

} // namespace

TEST(the_pieces_fill_the_box) {
  const Fracture f = fractureBox(kWall, 24, 7);
  CHECK(f.pieces.size() == 24);
  CHECK_NEAR(totalVolume(f), kWall.x * kWall.y * kWall.z, 1e-4);
}

TEST(every_piece_is_convex_and_inside_the_box) {
  const Fracture f = fractureBox(kWall, 24, 7);
  const glm::vec3 half = 0.5f * kWall;
  for (const FracturePiece &p : f.pieces) {
    CHECK(p.volume > 0.0f);
    CHECK(p.faces.size() >= 4);
    for (const FractureFace &face : p.faces) {
      CHECK(face.points.size() >= 3);
      for (const FractureFace &other : p.faces)
        for (const glm::vec3 &q : other.points) {
          // Every point of the piece is behind every face's plane.
          CHECK(glm::dot(face.normal, q - face.points[0]) < 1e-4f);
          CHECK(std::fabs(q.x) <= half.x + 1e-4f);
          CHECK(std::fabs(q.y) <= half.y + 1e-4f);
          CHECK(std::fabs(q.z) <= half.z + 1e-4f);
        }
    }
  }
}

TEST(links_are_faces_both_pieces_share) {
  const Fracture f = fractureBox(kWall, 24, 7);
  CHECK(!f.links.empty());
  std::set<std::pair<int, int>> seen;
  for (const FractureLink &link : f.links) {
    CHECK(link.a < link.b);
    CHECK(link.area > 0.0f);
    CHECK(seen.insert({link.a, link.b}).second); // once each
    // Each side has a face onto the other, of the same area.
    float areaA = 0.0f, areaB = 0.0f;
    for (const FractureFace &face : f.pieces[link.a].faces)
      if (face.neighbour == link.b)
        areaA += faceArea(face);
    for (const FractureFace &face : f.pieces[link.b].faces)
      if (face.neighbour == link.a)
        areaB += faceArea(face);
    CHECK_NEAR(areaA, link.area, 1e-4);
    CHECK_NEAR(areaB, link.area, 1e-4);
  }
}

TEST(the_outside_faces_cover_the_box) {
  const Fracture f = fractureBox(kWall, 24, 7);
  float outside = 0.0f;
  for (const FracturePiece &p : f.pieces)
    for (const FractureFace &face : p.faces)
      if (face.neighbour < 0)
        outside += faceArea(face);
  const glm::vec3 s = kWall;
  CHECK_NEAR(outside, 2.0f * (s.x * s.y + s.y * s.z + s.z * s.x), 1e-3);
}

TEST(the_links_hold_every_piece_together) {
  // Cells of one box all touch through cut faces: one connected graph.
  const Fracture f = fractureBox(kWall, 24, 7);
  std::vector<int> group(f.pieces.size());
  std::iota(group.begin(), group.end(), 0);
  bool merged = true;
  while (merged) {
    merged = false;
    for (const FractureLink &link : f.links) {
      const int g = std::min(group[link.a], group[link.b]);
      if (group[link.a] != g || group[link.b] != g) {
        group[link.a] = group[link.b] = g;
        merged = true;
      }
    }
  }
  for (int g : group)
    CHECK(g == 0);
}

TEST(the_same_seed_cuts_the_same_pieces) {
  const Fracture a = fractureBox(kWall, 12, 3);
  const Fracture b = fractureBox(kWall, 12, 3);
  const Fracture c = fractureBox(kWall, 12, 4);
  CHECK(a.pieces.size() == b.pieces.size());
  bool same = true, differs = false;
  for (size_t i = 0; i < a.pieces.size(); ++i) {
    same = same && a.pieces[i].centre == b.pieces[i].centre;
    differs = differs || a.pieces[i].centre != c.pieces[i].centre;
  }
  CHECK(same);
  CHECK(differs);
}

TEST(one_piece_is_the_whole_box) {
  const Fracture f = fractureBox(kWall, 1, 0);
  CHECK(f.pieces.size() == 1);
  CHECK(f.links.empty());
  CHECK_NEAR(f.pieces[0].volume, kWall.x * kWall.y * kWall.z, 1e-5);
  CHECK(f.pieces[0].faces.size() == 6);
}

TEST(a_piece_knows_its_middle) {
  // The volume-weighted middles of the pieces average to the box's middle.
  const Fracture f = fractureBox(glm::vec3(1.0f), 16, 11);
  glm::vec3 weighted(0.0f);
  for (const FracturePiece &p : f.pieces)
    weighted += p.centre * p.volume;
  CHECK_NEAR(glm::length(weighted), 0.0, 1e-4);
}

TEST(thin_and_large_cuts_stay_whole) {
  // Planks, slabs and cubes, few and many pieces, many seeds: no piece lost,
  // none overlapping (the volumes add up exactly), every link two-sided.
  const glm::vec3 shapes[] = {glm::vec3(4.0f, 0.1f, 4.0f), glm::vec3(0.2f, 3.0f, 0.2f),
                              glm::vec3(1.0f), glm::vec3(6.0f, 3.0f, 0.4f)};
  for (const glm::vec3 &size : shapes)
    for (int count : {2, 5, 17, 60})
      for (uint32_t seed = 0; seed < 12; ++seed) {
        const Fracture f = fractureBox(size, count, seed);
        CHECK(static_cast<int>(f.pieces.size()) == count);
        CHECK_NEAR(totalVolume(f) / (size.x * size.y * size.z), 1.0, 1e-3);
        for (const FracturePiece &p : f.pieces)
          CHECK(p.volume > 0.0f);
        for (const FractureLink &link : f.links)
          CHECK(link.area > 0.0f);
      }
}

TEST(a_piece_mesh_is_its_flat_faces) {
  // Each face a fan of triangles with its own normal; the box's outside in
  // the object's colour, cut faces in the raw colour.
  const Fracture f = fractureBox(kWall, 12, 2);
  const glm::vec3 outside(0.6f, 0.3f, 0.2f), inside(0.3f, 0.15f, 0.1f);
  for (const FracturePiece &piece : f.pieces) {
    const MeshData mesh = makePieceMesh(piece, outside, inside);
    size_t triangles = 0, corners = 0;
    for (const FractureFace &face : piece.faces) {
      triangles += face.points.size() - 2;
      corners += face.points.size();
    }
    CHECK(mesh.indices.size() == 3 * triangles);
    CHECK(mesh.vertices.size() == corners);
    int badWinding = 0, badFrames = 0;
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
      const Vertex &a = mesh.vertices[mesh.indices[i]];
      const Vertex &b = mesh.vertices[mesh.indices[i + 1]];
      const Vertex &c = mesh.vertices[mesh.indices[i + 2]];
      if (glm::dot(glm::cross(b.pos - a.pos, c.pos - a.pos), a.normal) <= 0.0f)
        ++badWinding;
    }
    for (const Vertex &v : mesh.vertices)
      if (std::fabs(glm::length(v.tangent) - 1.0f) > 1e-4f ||
          std::fabs(glm::dot(v.tangent, v.normal)) > 1e-4f)
        ++badFrames;
    CHECK(badWinding == 0);
    CHECK(badFrames == 0);
    // Colours follow the faces, in the order they're written.
    size_t at = 0;
    for (const FractureFace &face : piece.faces) {
      CHECK(mesh.vertices[at].col == (face.neighbour < 0 ? outside : inside));
      CHECK(mesh.vertices[at].normal == face.normal);
      at += face.points.size();
    }
  }
}

namespace {

// Area of a piece's faces of one kind.
float areaOf(const FracturePiece &piece, bool outside) {
  float area = 0.0f;
  for (const FractureFace &face : piece.faces)
    if ((face.neighbour == kOutsideFace) == outside && (outside || face.neighbour < 0))
      area += faceArea(face);
  return area;
}

bool insideOf(const FracturePiece &piece, glm::vec3 p, float margin) {
  for (const FractureFace &face : piece.faces)
    if (glm::dot(face.normal, p - face.points[0]) > margin)
      return false;
  return true;
}

// A corner piece of a wall: some faces outside, some cut.
FracturePiece wallPiece() {
  const Fracture f = fractureBox(kWall, 6, 3);
  return f.pieces.front();
}

} // namespace

TEST(a_piece_cut_again_is_filled_by_its_pieces) {
  const FracturePiece parent = wallPiece();
  const glm::vec3 hit = parent.centre;
  const std::vector<glm::vec3> points =
      pointsAround(parent, hit, glm::vec3(0, 0, -1), 8, FracturePattern::Clustered, 5);
  CHECK(points.size() == 8);
  const Fracture f = fractureConvex(parent, points);
  CHECK(f.pieces.size() == points.size());
  float volume = 0.0f, outside = 0.0f, oldCuts = 0.0f;
  for (const FracturePiece &p : f.pieces) {
    volume += p.volume;
    outside += areaOf(p, true);
    oldCuts += areaOf(p, false);
    for (const FractureFace &face : p.faces)
      for (const glm::vec3 &q : face.points)
        CHECK(insideOf(parent, q, 1e-4f)); // inside the piece it came from
  }
  CHECK_NEAR(volume / parent.volume, 1.0, 1e-3);
  // What was the object's outside is still outside; what was cut stays cut
  // (onto pieces not in this set); the new faces are links between them.
  float parentOutside = 0.0f, parentCuts = 0.0f;
  for (const FractureFace &face : parent.faces)
    (face.neighbour == kOutsideFace ? parentOutside : parentCuts) += faceArea(face);
  CHECK_NEAR(outside, parentOutside, 1e-3);
  CHECK_NEAR(oldCuts, parentCuts, 1e-3);
  CHECK(!f.links.empty());
  for (const FractureLink &link : f.links)
    CHECK(link.area > 0.0f && link.a < link.b);
}

TEST(points_around_a_hit_are_inside_and_repeatable) {
  const FracturePiece parent = wallPiece();
  for (FracturePattern pattern :
       {FracturePattern::Clustered, FracturePattern::Splinters, FracturePattern::Shards}) {
    const auto a = pointsAround(parent, parent.centre, glm::vec3(0, 0, -1), 10, pattern, 9);
    const auto b = pointsAround(parent, parent.centre, glm::vec3(0, 0, -1), 10, pattern, 9);
    const auto c = pointsAround(parent, parent.centre, glm::vec3(0, 0, -1), 10, pattern, 10);
    CHECK(a == b);
    CHECK(a != c);
    CHECK(a.size() >= 2);
    for (const glm::vec3 &p : a)
      CHECK(insideOf(parent, p, 0.0f));
  }
}

TEST(a_hit_breaks_small_near_and_big_far) {
  // Clustered points: the pieces nearest the hit are smaller than the
  // farthest ones, on average.
  const glm::vec3 size(2.0f, 2.0f, 0.3f);
  const Fracture whole = fractureBox(size, 1, 0);
  const glm::vec3 hit(-0.9f, -0.9f, 0.15f);
  float near = 0.0f, far = 0.0f;
  for (uint32_t seed = 0; seed < 8; ++seed) {
    const Fracture f = fractureConvex(
        whole.pieces[0],
        pointsAround(whole.pieces[0], hit, glm::vec3(0, 0, -1), 12, FracturePattern::Clustered, seed));
    std::vector<const FracturePiece *> order;
    for (const FracturePiece &p : f.pieces)
      order.push_back(&p);
    std::sort(order.begin(), order.end(), [&](const FracturePiece *a, const FracturePiece *b) {
      return glm::distance(a->centre, hit) < glm::distance(b->centre, hit);
    });
    for (size_t i = 0; i < 3; ++i) {
      near += order[i]->volume;
      far += order[order.size() - 1 - i]->volume;
    }
  }
  CHECK(near * 2.0f < far);
}

TEST(wood_splinters_along_its_grain) {
  // Cells stretched along x: pieces are longer along x than across.
  const Fracture whole = fractureBox(glm::vec3(1.0f), 1, 0);
  const FracturePiece &cube = whole.pieces[0];
  float along = 0.0f, across = 0.0f;
  for (uint32_t seed = 0; seed < 6; ++seed) {
    const Fracture f = fractureConvex(
        cube, pointsAround(cube, glm::vec3(0.0f), glm::vec3(0, 0, -1), 12, FracturePattern::Splinters, seed),
        glm::vec3(1, 0, 0), 4.0f);
    float volume = 0.0f;
    for (const FracturePiece &p : f.pieces)
      volume += p.volume;
    CHECK_NEAR(volume, 1.0, 1e-3);
    for (const FracturePiece &p : f.pieces) {
      glm::vec3 lo(1e9f), hi(-1e9f);
      for (const FractureFace &face : p.faces)
        for (const glm::vec3 &q : face.points) {
          lo = glm::min(lo, q);
          hi = glm::max(hi, q);
        }
      along += hi.x - lo.x;
      across += 0.5f * ((hi.y - lo.y) + (hi.z - lo.z));
    }
  }
  CHECK(along > 1.5f * across);
}

TEST(shards_spread_across_the_face_that_was_hit) {
  // Glass and ice: points in rings around the hit, close to the face it was
  // hit on, spread across it rather than into the depth.
  const Fracture whole = fractureBox(glm::vec3(2.0f, 2.0f, 0.2f), 1, 0);
  const glm::vec3 hit(0.0f, 0.0f, 0.1f);
  const auto points = pointsAround(whole.pieces[0], hit, glm::vec3(0, 0, -1), 12,
                                   FracturePattern::Shards, 3);
  CHECK(points.size() >= 6);
  float spread = 0.0f, depth = 0.0f;
  for (const glm::vec3 &p : points) {
    spread += glm::length(glm::vec2(p.x, p.y));
    depth += std::fabs(p.z - hit.z);
  }
  CHECK(spread > 4.0f * depth);
}

TEST(a_rubble_mound_is_a_low_rough_dome) {
  const MeshData mound = makeRubbleMound(4, glm::vec3(0.5f));
  CHECK(!mound.vertices.empty());
  CHECK(mound.indices.size() % 3 == 0);
  float top = 0.0f;
  int badWinding = 0;
  for (const Vertex &v : mound.vertices) {
    CHECK(v.pos.y >= -1e-5f);
    CHECK(glm::length(glm::vec2(v.pos.x, v.pos.z)) <= 1.0f + 1e-4f);
    top = std::max(top, v.pos.y);
  }
  for (size_t i = 0; i < mound.indices.size(); i += 3) {
    const Vertex &a = mound.vertices[mound.indices[i]];
    const Vertex &b = mound.vertices[mound.indices[i + 1]];
    const Vertex &c = mound.vertices[mound.indices[i + 2]];
    if (glm::dot(glm::cross(b.pos - a.pos, c.pos - a.pos), a.normal + b.normal + c.normal) <= 0.0f)
      ++badWinding;
  }
  CHECK(badWinding == 0);
  CHECK(top > 0.25f && top < 0.5f); // low: about a third as high as it's wide
}

TEST(rubble_spreads_wider_as_more_settles) {
  // A mound of radius r holds pi r^2 h / 2 with h a third of r or so.
  CHECK(rubbleRadius(0.1f) < rubbleRadius(0.2f));
  const float r = rubbleRadius(1.0f);
  CHECK_NEAR(3.14159265f * r * r * (kRubbleHeight * r) / 2.0f, 1.0, 1e-3);
}

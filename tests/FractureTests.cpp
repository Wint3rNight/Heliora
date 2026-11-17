#include "Fracture.h"
#include "TestRunner.h"

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

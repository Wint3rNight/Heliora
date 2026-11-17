#include "BreakGraph.h"
#include "TestRunner.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace {

glm::mat4 at(glm::vec3 position) { return glm::translate(glm::mat4(1.0f), position); }

std::vector<PlacedPiece> placeAll(const Fracture &f, const glm::mat4 &where, int object) {
  std::vector<PlacedPiece> out;
  for (const FracturePiece &p : f.pieces)
    out.push_back(placePiece(p, where, object));
  return out;
}

// A wall's links from its cut, with a strength per square metre.
std::vector<BreakLink> wallLinks(const Fracture &f, float strength) {
  return cutLinks(f, strength, 0);
}

int brokenCount(const std::vector<BreakLink> &links, float impulse, glm::vec3 hit,
                const Fracture &f) {
  std::vector<LinkAt> candidates;
  for (size_t i = 0; i < f.links.size(); ++i)
    candidates.push_back({static_cast<int>(i), f.links[i].centre});
  return static_cast<int>(linksBrokenBy(links, candidates, hit, impulse).size());
}

} // namespace

TEST(groups_follow_the_unbroken_links) {
  std::vector<BreakLink> chain = {{0, 1, 1.0f}, {1, 2, 1.0f}, {2, 3, 1.0f}};
  std::vector<BreakGroup> groups = splitGroups(4, chain, {true, false, false, false});
  CHECK(groups.size() == 1);
  CHECK(groups[0].anchored);
  chain[1].broken = true;
  groups = splitGroups(4, chain, {true, false, false, false});
  CHECK(groups.size() == 2);
  CHECK((groups[0].pieces == std::vector<int>{0, 1}));
  CHECK(groups[0].anchored);
  CHECK((groups[1].pieces == std::vector<int>{2, 3}));
  CHECK(!groups[1].anchored); // nothing holds it up: it falls
}

TEST(a_piece_with_no_links_is_a_group_of_its_own) {
  const std::vector<BreakGroup> groups = splitGroups(3, {{0, 1, 1.0f, true}}, {false, false, true});
  CHECK(groups.size() == 3);
  CHECK(groups[2].anchored);
}

TEST(touching_objects_hold_onto_each_other) {
  // One box resting on another: the lower one's top pieces touch the upper
  // one's bottom pieces, over the whole 2 x 2 m contact.
  const Fracture f = fractureBox(glm::vec3(2.0f, 1.0f, 2.0f), 8, 5);
  std::vector<PlacedPiece> pieces = placeAll(f, at({0.0f, 0.5f, 0.0f}), 0);
  const std::vector<PlacedPiece> upper = placeAll(f, at({0.0f, 1.5f, 0.0f}), 1);
  pieces.insert(pieces.end(), upper.begin(), upper.end());
  const std::vector<PieceTouch> touches = touchingPieces(pieces, 0.01f);
  CHECK(!touches.empty());
  float area = 0.0f;
  for (const PieceTouch &t : touches) {
    CHECK(pieces[t.a].object != pieces[t.b].object);
    CHECK(t.area > 0.0f);
    CHECK_NEAR(t.centre.y, 1.0, 1e-4);
    area += t.area;
  }
  CHECK_NEAR(area, 4.0, 1e-3);
}

TEST(only_the_overlap_touches) {
  // The upper box moved 1 m along x: half the contact; lifted 5 cm: none.
  const Fracture f = fractureBox(glm::vec3(2.0f, 1.0f, 2.0f), 6, 2);
  auto touchArea = [&](glm::vec3 upperAt) {
    std::vector<PlacedPiece> pieces = placeAll(f, at({0.0f, 0.5f, 0.0f}), 0);
    const std::vector<PlacedPiece> upper = placeAll(f, at(upperAt), 1);
    pieces.insert(pieces.end(), upper.begin(), upper.end());
    float area = 0.0f;
    for (const PieceTouch &t : touchingPieces(pieces, 0.01f))
      area += t.area;
    return area;
  };
  CHECK_NEAR(touchArea({1.0f, 1.5f, 0.0f}), 2.0, 1e-3);
  CHECK_NEAR(touchArea({0.0f, 1.55f, 0.0f}), 0.0, 1e-6);
}

TEST(pieces_of_one_object_are_not_touches) {
  // Their own links come from the cut; touches are between objects.
  const Fracture f = fractureBox(glm::vec3(1.0f), 8, 1);
  CHECK(touchingPieces(placeAll(f, at({0.0f, 0.5f, 0.0f}), 0), 0.01f).empty());
}

TEST(a_roof_falls_when_its_walls_go) {
  // Two posts on the ground and a slab across them. The posts' bottom pieces
  // are anchors; knock out every piece of both posts and the slab's pieces
  // are left with no path to the ground.
  const Fracture post = fractureBox(glm::vec3(0.4f, 2.0f, 0.4f), 4, 3);
  const Fracture slab = fractureBox(glm::vec3(3.0f, 0.3f, 1.0f), 6, 9);
  std::vector<PlacedPiece> pieces = placeAll(post, at({-1.2f, 1.0f, 0.0f}), 0);
  const std::vector<PlacedPiece> right = placeAll(post, at({1.2f, 1.0f, 0.0f}), 1);
  const std::vector<PlacedPiece> roof = placeAll(slab, at({0.0f, 2.15f, 0.0f}), 2);
  pieces.insert(pieces.end(), right.begin(), right.end());
  pieces.insert(pieces.end(), roof.begin(), roof.end());
  const int count = static_cast<int>(pieces.size());

  std::vector<BreakLink> links;
  auto addCut = [&](const Fracture &f, int first) {
    const std::vector<BreakLink> cut = cutLinks(f, 100.0f, first);
    links.insert(links.end(), cut.begin(), cut.end());
  };
  addCut(post, 0);
  addCut(post, 4);
  addCut(slab, 8);
  for (const PieceTouch &t : touchingPieces(pieces, 0.01f))
    links.push_back({t.a, t.b, t.area * 100.0f});
  std::vector<bool> anchors(count, false);
  for (int i = 0; i < count; ++i)
    anchors[i] = restsOn(pieces[i], 0.0f, 0.01f); // on the ground at y = 0

  std::vector<BreakGroup> groups = splitGroups(count, links, anchors);
  CHECK(groups.size() == 1); // one building, held up
  CHECK(groups[0].anchored);

  for (BreakLink &l : links)
    if (l.a < 8 || l.b < 8)
      l.broken = true; // the posts are gone
  groups = splitGroups(count, links, anchors);
  const auto roofGroup = std::find_if(groups.begin(), groups.end(), [](const BreakGroup &g) {
    return std::find(g.pieces.begin(), g.pieces.end(), 8) != g.pieces.end();
  });
  CHECK(roofGroup != groups.end());
  if (roofGroup != groups.end()) {
    CHECK(roofGroup->pieces.size() == 6); // the whole slab, still in one piece
    CHECK(!roofGroup->anchored);          // and it falls
  }
}

TEST(a_hit_breaks_the_links_near_it) {
  const Fracture f = fractureBox(glm::vec3(4.0f, 2.0f, 0.3f), 30, 4);
  const std::vector<BreakLink> links = wallLinks(f, 300.0f);
  const glm::vec3 hit(-1.8f, 0.0f, 0.15f); // near the left end
  std::vector<LinkAt> candidates;
  for (size_t i = 0; i < f.links.size(); ++i)
    candidates.push_back({static_cast<int>(i), f.links[i].centre});
  const std::vector<int> broken = linksBrokenBy(links, candidates, hit, 120.0f);
  CHECK(!broken.empty());
  CHECK(broken.size() < links.size()); // a chunk, not the whole wall
  for (int i : broken)
    CHECK(f.links[i].centre.x < 0.5f); // nothing at the far end
}

TEST(a_harder_hit_breaks_more) {
  const Fracture f = fractureBox(glm::vec3(4.0f, 2.0f, 0.3f), 30, 4);
  const std::vector<BreakLink> links = wallLinks(f, 300.0f);
  const glm::vec3 hit(0.0f, 0.0f, 0.15f);
  const int soft = brokenCount(links, 60.0f, hit, f);
  const int hard = brokenCount(links, 600.0f, hit, f);
  CHECK(hard > soft);
  CHECK(brokenCount(links, 1.0f, hit, f) == 0); // a tap breaks nothing
}

TEST(no_link_is_much_weaker_than_the_rest) {
  const Fracture f = fractureBox(glm::vec3(4.0f, 2.0f, 0.3f), 30, 4);
  const std::vector<BreakLink> links = cutLinks(f, 300.0f, 10);
  float mean = 0.0f;
  for (const FractureLink &l : f.links)
    mean += l.area;
  mean /= static_cast<float>(f.links.size());
  for (size_t i = 0; i < links.size(); ++i) {
    CHECK(links[i].a == f.links[i].a + 10); // numbered from the first piece
    CHECK(links[i].strength >= 0.25f * mean * 300.0f - 1e-3f);
    CHECK(links[i].strength >= f.links[i].area * 300.0f - 1e-3f);
  }
}

TEST(broken_links_stay_broken) {
  // Already broken links aren't reported again.
  const Fracture f = fractureBox(glm::vec3(1.0f), 8, 6);
  std::vector<BreakLink> links = wallLinks(f, 10.0f);
  for (BreakLink &l : links)
    l.broken = true;
  CHECK(brokenCount(links, 1e6f, glm::vec3(0.0f), f) == 0);
}

TEST(a_lone_piece_on_the_ground_moves_and_a_big_remainder_stays) {
  // Object 0 is 10 m^3. An anchored group holding 4 m^3 of it is still the
  // structure; an anchored piece of 0.3 m^3 is just lying there.
  const std::vector<int> objectOf = {0, 0, 0, 1};
  const std::vector<float> volume = {3.0f, 1.0f, 0.3f, 0.5f};
  const std::vector<float> objectVolume = {10.0f, 0.5f};
  CHECK(staysPut({{0, 1}, true}, objectOf, volume, objectVolume));
  CHECK(!staysPut({{2}, true}, objectOf, volume, objectVolume));
  CHECK(!staysPut({{0, 1}, false}, objectOf, volume, objectVolume)); // nothing holds it
  CHECK(staysPut({{3}, true}, objectOf, volume, objectVolume)); // all of a small object
}

TEST(a_hit_is_shared_between_what_it_frees) {
  const std::vector<float> shares = shareHit(100.0f, {0.0f, 0.5f, 3.0f});
  CHECK(shares.size() == 3);
  CHECK_NEAR(shares[0] + shares[1] + shares[2], 100.0, 1e-3);
  CHECK(shares[0] > shares[1]);
  CHECK(shares[1] > shares[2]);
  CHECK(shareHit(50.0f, {}).empty());
}

TEST(link_strengths_vary_within_a_quarter_and_repeat) {
  std::vector<BreakLink> a(50, BreakLink{0, 1, 100.0f});
  std::vector<BreakLink> b = a, c = a;
  varyStrengths(a, 7, 0.25f);
  varyStrengths(b, 7, 0.25f);
  varyStrengths(c, 8, 0.25f);
  bool same = true, differs = false, spread = false;
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].strength >= 75.0f && a[i].strength <= 125.0f);
    same = same && a[i].strength == b[i].strength;
    differs = differs || a[i].strength != c[i].strength;
    spread = spread || std::fabs(a[i].strength - 100.0f) > 10.0f;
  }
  CHECK(same);
  CHECK(differs);
  CHECK(spread);
}

TEST(bigger_pieces_take_more_to_split) {
  // As strong as its cross-section: eight times the volume, four times the hit.
  CHECK_NEAR(splitImpulse(8.0f, 100.0f) / splitImpulse(1.0f, 100.0f), 4.0, 1e-4);
  CHECK_NEAR(splitImpulse(1.0f, 100.0f), 100.0, 1e-4);
}

TEST(two_pieces_touch_through_any_face) {
  // Pieces cut while playing link to their old piece's neighbours through
  // faces that were cut faces, so any face counts here.
  const Fracture f = fractureBox(glm::vec3(1.0f), 6, 2);
  const FractureLink &link = f.links.front();
  const PlacedPiece a = placePiece(f.pieces[link.a], glm::mat4(1.0f), 0);
  const PlacedPiece b = placePiece(f.pieces[link.b], glm::mat4(1.0f), 0);
  glm::vec3 centre(0.0f);
  CHECK_NEAR(touchArea(a, b, 0.01f, centre), link.area, 1e-4);
  CHECK_NEAR(glm::distance(centre, link.centre), 0.0, 1e-3);
  const PlacedPiece far = placePiece(f.pieces[link.b], glm::translate(glm::mat4(1.0f), glm::vec3(0, 3, 0)), 0);
  CHECK(touchArea(a, far, 0.01f, centre) == 0.0f);
}

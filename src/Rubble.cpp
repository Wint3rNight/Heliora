#include "Rubble.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int N = kRubbleSamples;
constexpr int kCells = N - 1;                       // 32 a side
constexpr float kCellArea = kRubbleSpacing * kRubbleSpacing;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kSmallestMove = 1e-4f;  // m: slumping stops below this
constexpr float kEmpty = 1e-4f;         // m: drawn as no rubble below this
constexpr float kCastAgain = 0.05f;     // m: ground differing more is cast sample by sample
constexpr int kCoarse = 4;              // ground is cast on every 4th sample first
constexpr float kOffLedge = 10.0f;      // m: no ground counts as this far down
const glm::vec3 kBareColour(0.5f);

int floorDiv(int a, int b) {
  int q = a / b;
  if (a % b != 0 && ((a < 0) != (b < 0)))
    --q;
  return q;
}

int at(int i, int j) { return j * N + i; }

uint32_t hashSample(int x, int z, uint32_t salt) {
  uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(x)) * 0x9E3779B97F4A7C15ull ^
               static_cast<uint64_t>(static_cast<uint32_t>(z)) * 0xC2B2AE3D27D4EB4Full ^
               static_cast<uint64_t>(salt) * 0x165667B19E3779F9ull;
  h ^= h >> 31;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 27;
  h *= 0x94D049BB133111EBull;
  h ^= h >> 33;
  return static_cast<uint32_t>(h);
}

float unit(uint32_t h) { return static_cast<float>(h >> 8) * (1.0f / 16777216.0f); }

// Value noise on a lattice of every 4th sample (0.5 m), -1 to 1.
float smoothNoise(int gx, int gz) {
  const int cx = floorDiv(gx, 4), cz = floorDiv(gz, 4);
  auto ease = [](float t) { return t * t * (3.0f - 2.0f * t); };
  const float fx = ease(static_cast<float>(gx - 4 * cx) / 4.0f);
  const float fz = ease(static_cast<float>(gz - 4 * cz) / 4.0f);
  auto corner = [&](int x, int z) { return 2.0f * unit(hashSample(x, z, 7u)) - 1.0f; };
  const float a = corner(cx, cz) + (corner(cx + 1, cz) - corner(cx, cz)) * fx;
  const float b = corner(cx, cz + 1) + (corner(cx + 1, cz + 1) - corner(cx, cz + 1)) * fx;
  return a + (b - a) * fz;
}

// A global sample's tiles: its own (local index 0 to 31) and, on a border,
// the one before it that holds it as index 32.
template <typename F> void forHolders(int gx, int gz, F &&f) {
  const int kx = floorDiv(gx, kCells), kz = floorDiv(gz, kCells);
  const int lx = gx - kx * kCells, lz = gz - kz * kCells;
  const int xs[2][2] = {{kx, lx}, {kx - 1, kCells}};
  const int zs[2][2] = {{kz, lz}, {kz - 1, kCells}};
  for (int a = 0; a < (lx == 0 ? 2 : 1); ++a)
    for (int b = 0; b < (lz == 0 ? 2 : 1); ++b)
      f(RubbleKey(xs[a][0], zs[b][0]), at(xs[a][1], zs[b][1]));
}

} // namespace

RubbleKey rubbleKeyAt(glm::vec2 xz) {
  return {static_cast<int>(std::floor(xz.x / kRubbleTile)),
          static_cast<int>(std::floor(xz.y / kRubbleTile))};
}

glm::vec2 rubbleTileCorner(RubbleKey key) {
  return glm::vec2(static_cast<float>(key.first), static_cast<float>(key.second)) * kRubbleTile;
}

RubbleLayer::RubbleLayer(Ground g) : ground(std::move(g)) {}

RubbleLayer::Tile &RubbleLayer::tile(RubbleKey key) {
  if (auto found = tiles_.find(key); found != tiles_.end())
    return found->second;
  Tile t;
  t.ground.assign(N * N, kNaN);
  t.depth.assign(N * N, 0.0f);
  t.colour.assign(N * N, kBareColour);
  const glm::vec2 corner = rubbleTileCorner(key);
  auto cast = [&](int i, int j) {
    const std::optional<float> h = ground(corner + glm::vec2(i, j) * kRubbleSpacing);
    t.ground[at(i, j)] = h ? *h : kNaN;
  };
  auto close = [](float a, float b) {
    return !std::isnan(a) && !std::isnan(b) && std::fabs(a - b) <= kCastAgain;
  };
  // Every 4th sample, then the edges between them (decided by their two
  // ends only, so a border shared with a neighbour comes out the same in
  // both), then the insides of the cells: blended where the ground is flat
  // enough, cast sample by sample where it isn't, so steps stay sharp.
  for (int j = 0; j < N; j += kCoarse)
    for (int i = 0; i < N; i += kCoarse)
      cast(i, j);
  for (int line = 0; line < N; line += kCoarse)
    for (int s = 0; s < kCells; s += kCoarse) {
      const float ax = t.ground[at(s, line)], bx = t.ground[at(s + kCoarse, line)];
      const float az = t.ground[at(line, s)], bz = t.ground[at(line, s + kCoarse)];
      for (int k = 1; k < kCoarse; ++k) {
        const float f = static_cast<float>(k) / kCoarse;
        if (close(ax, bx))
          t.ground[at(s + k, line)] = ax + (bx - ax) * f;
        else
          cast(s + k, line);
        if (close(az, bz))
          t.ground[at(line, s + k)] = az + (bz - az) * f;
        else
          cast(line, s + k);
      }
    }
  for (int j0 = 0; j0 < kCells; j0 += kCoarse)
    for (int i0 = 0; i0 < kCells; i0 += kCoarse) {
      const float a = t.ground[at(i0, j0)], b = t.ground[at(i0 + kCoarse, j0)];
      const float c = t.ground[at(i0, j0 + kCoarse)], d = t.ground[at(i0 + kCoarse, j0 + kCoarse)];
      const bool flat = close(a, b) && close(a, c) && close(a, d) && close(b, c) &&
                        close(b, d) && close(c, d);
      for (int dj = 1; dj < kCoarse; ++dj)
        for (int di = 1; di < kCoarse; ++di) {
          if (!flat) {
            cast(i0 + di, j0 + dj);
            continue;
          }
          const float fx = static_cast<float>(di) / kCoarse, fz = static_cast<float>(dj) / kCoarse;
          const float top = a + (b - a) * fx, bottom = c + (d - c) * fx;
          t.ground[at(i0 + di, j0 + dj)] = top + (bottom - top) * fz;
        }
    }
  // A border it shares with a tile already there takes that tile's ground,
  // so both copies agree even if the ground rays now see something else.
  for (int dz = -1; dz <= 1; ++dz)
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dz == 0)
        continue;
      const Tile *other = findTile({key.first + dx, key.second + dz});
      if (!other)
        continue;
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
          const int oi = i - dx * kCells, oj = j - dz * kCells; // the same sample in the other
          if (oi >= 0 && oi < N && oj >= 0 && oj < N)
            t.ground[at(i, j)] = other->ground[at(oi, oj)];
        }
    }
  return tiles_.emplace(key, std::move(t)).first->second;
}

const RubbleLayer::Tile *RubbleLayer::findTile(RubbleKey key) const {
  const auto found = tiles_.find(key);
  return found == tiles_.end() ? nullptr : &found->second;
}

float RubbleLayer::groundAt(int gx, int gz) {
  const int kx = floorDiv(gx, kCells), kz = floorDiv(gz, kCells);
  return tile({kx, kz}).ground[at(gx - kx * kCells, gz - kz * kCells)];
}

float RubbleLayer::sampleDepth(int gx, int gz) const {
  const int kx = floorDiv(gx, kCells), kz = floorDiv(gz, kCells);
  const Tile *t = findTile({kx, kz});
  return t ? t->depth[at(gx - kx * kCells, gz - kz * kCells)] : 0.0f;
}

float RubbleLayer::sampleGround(int gx, int gz) const {
  const int kx = floorDiv(gx, kCells), kz = floorDiv(gz, kCells);
  if (const Tile *t = findTile({kx, kz}))
    return t->ground[at(gx - kx * kCells, gz - kz * kCells)];
  const std::optional<float> h = ground(glm::vec2(gx, gz) * kRubbleSpacing);
  return h ? *h : kNaN;
}

void RubbleLayer::addDepth(int gx, int gz, float amount, glm::vec3 c) {
  forHolders(gx, gz, [&](RubbleKey key, int index) {
    Tile &t = tile(key);
    const float before = t.depth[index];
    if (amount > 0.0f)
      t.colour[index] = (t.colour[index] * before + c * amount) / (before + amount);
    t.depth[index] = std::max(0.0f, before + amount);
    t.changed = true;
    t.unsettled = true;
  });
}

void RubbleLayer::deposit(glm::vec2 where, float volume, float radius, glm::vec3 c) {
  if (!(volume > 0.0f))
    return;
  const float reach = std::max(radius, kRubbleSpacing);
  struct Share {
    int gx, gz;
    float weight;
  };
  std::vector<Share> shares;
  float sum = 0.0f;
  const int x0 = static_cast<int>(std::floor((where.x - reach) / kRubbleSpacing));
  const int x1 = static_cast<int>(std::ceil((where.x + reach) / kRubbleSpacing));
  const int z0 = static_cast<int>(std::floor((where.y - reach) / kRubbleSpacing));
  const int z1 = static_cast<int>(std::ceil((where.y + reach) / kRubbleSpacing));
  for (int gz = z0; gz <= z1; ++gz)
    for (int gx = x0; gx <= x1; ++gx) {
      const float r = glm::length(glm::vec2(gx, gz) * kRubbleSpacing - where) / reach;
      if (r >= 1.0f || std::isnan(groundAt(gx, gz)))
        continue;
      const float w = (1.0f - r * r) * (1.0f - r * r);
      shares.push_back({gx, gz, w});
      sum += w;
    }
  if (shares.empty())
    return; // over nothing: dust
  for (const Share &s : shares)
    addDepth(s.gx, s.gz, volume * s.weight / sum / kCellArea, c);
}

bool RubbleLayer::settle(int passes) {
  static constexpr int kNeighbours[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1},
                                            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  bool moved = false;
  for (int pass = 0; pass < passes; ++pass) {
    std::vector<RubbleKey> keys;
    for (auto &[key, t] : tiles_)
      if (t.unsettled) {
        keys.push_back(key);
        t.unsettled = false;
      }
    moved = false;
    for (RubbleKey key : keys)
      for (int lz = 0; lz < kCells; ++lz)
        for (int lx = 0; lx < kCells; ++lx) {
          const int gx = key.first * kCells + lx, gz = key.second * kCells + lz;
          Tile &t = tile(key);
          const float d = t.depth[at(lx, lz)];
          const float g = t.ground[at(lx, lz)];
          if (d <= 0.0f || std::isnan(g))
            continue;
          const float s = g + d;
          float excess[8];
          float total = 0.0f, worst = 0.0f;
          for (int n = 0; n < 8; ++n) {
            const int nx = lx + kNeighbours[n][0], nz = lz + kNeighbours[n][1];
            float other;
            if (nx >= 0 && nz >= 0) // in this tile's own samples (0 to 32)
              other = t.ground[at(nx, nz)] + t.depth[at(nx, nz)];
            else
              other = groundAt(gx + kNeighbours[n][0], gz + kNeighbours[n][1]) +
                      sampleDepth(gx + kNeighbours[n][0], gz + kNeighbours[n][1]);
            if (std::isnan(other))
              other = g - kOffLedge;
            const float run = (n < 4 ? 1.0f : 1.41421356f) * kRubbleSpacing;
            excess[n] = std::max(0.0f, s - other - kRubbleSlope * run);
            total += excess[n];
            worst = std::max(worst, excess[n]);
          }
          const float move = std::min(d, 0.5f * worst);
          if (move < kSmallestMove)
            continue;
          const glm::vec3 c = t.colour[at(lx, lz)];
          addDepth(gx, gz, -move, c);
          for (int n = 0; n < 8; ++n) {
            if (excess[n] <= 0.0f)
              continue;
            const int nx = gx + kNeighbours[n][0], nz = gz + kNeighbours[n][1];
            if (std::isnan(groundAt(nx, nz)))
              continue; // off the edge of the ground: lost
            addDepth(nx, nz, move * excess[n] / total, c);
          }
          moved = true;
        }
    if (!moved)
      break;
  }
  return moved;
}

float RubbleLayer::depth(glm::vec2 p) const {
  const float fx = p.x / kRubbleSpacing, fz = p.y / kRubbleSpacing;
  const int gx = static_cast<int>(std::floor(fx)), gz = static_cast<int>(std::floor(fz));
  const float tx = fx - gx, tz = fz - gz;
  const float a = sampleDepth(gx, gz) + (sampleDepth(gx + 1, gz) - sampleDepth(gx, gz)) * tx;
  const float b =
      sampleDepth(gx, gz + 1) + (sampleDepth(gx + 1, gz + 1) - sampleDepth(gx, gz + 1)) * tx;
  return a + (b - a) * tz;
}

float RubbleLayer::surface(glm::vec2 p) const {
  const float fx = p.x / kRubbleSpacing, fz = p.y / kRubbleSpacing;
  const int gx = static_cast<int>(std::floor(fx)), gz = static_cast<int>(std::floor(fz));
  const float tx = fx - gx, tz = fz - gz;
  auto s = [&](int x, int z) { return sampleGround(x, z) + sampleDepth(x, z); };
  const float a = tx == 0.0f ? s(gx, gz) : s(gx, gz) + (s(gx + 1, gz) - s(gx, gz)) * tx;
  if (tz == 0.0f)
    return a;
  const float b =
      tx == 0.0f ? s(gx, gz + 1) : s(gx, gz + 1) + (s(gx + 1, gz + 1) - s(gx, gz + 1)) * tx;
  return a + (b - a) * tz;
}

glm::vec3 RubbleLayer::colour(glm::vec2 p) const {
  const int gx = static_cast<int>(std::lround(p.x / kRubbleSpacing));
  const int gz = static_cast<int>(std::lround(p.y / kRubbleSpacing));
  const int kx = floorDiv(gx, kCells), kz = floorDiv(gz, kCells);
  const Tile *t = findTile({kx, kz});
  return t ? t->colour[at(gx - kx * kCells, gz - kz * kCells)] : kBareColour;
}

float RubbleLayer::volume() const {
  double sum = 0.0;
  for (const auto &[key, t] : tiles_)
    for (int lz = 0; lz < kCells; ++lz)
      for (int lx = 0; lx < kCells; ++lx)
        sum += t.depth[at(lx, lz)];
  return static_cast<float>(sum * kCellArea);
}

std::vector<RubbleKey> RubbleLayer::takeChanged() {
  std::vector<RubbleKey> out;
  for (auto &[key, t] : tiles_)
    if (t.changed) {
      out.push_back(key);
      t.changed = false;
    }
  return out;
}

std::vector<RubbleKey> RubbleLayer::tiles() const {
  std::vector<RubbleKey> out;
  out.reserve(tiles_.size());
  for (const auto &[key, t] : tiles_)
    out.push_back(key);
  return out;
}

float RubbleLayer::drawnHeight(int gx, int gz, float fallbackGround) const {
  const int kx = floorDiv(gx, kCells), kz = floorDiv(gz, kCells);
  const Tile *t = findTile({kx, kz});
  const int index = at(gx - kx * kCells, gz - kz * kCells);
  float g = t ? t->ground[index] : kNaN;
  if (std::isnan(g))
    g = fallbackGround;
  const float d = t ? t->depth[index] : 0.0f;
  if (d < kEmpty) {
    // Bare: under the lowest ground around it, so next to a step (a block's
    // edge) the mesh drops inside the block instead of hanging a slanted
    // sheet out of its side.
    for (int dz = -1; dz <= 1; ++dz)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = gx + dx, nz = gz + dz;
        const int ox = floorDiv(nx, kCells), oz = floorDiv(nz, kCells);
        if (const Tile *other = findTile({ox, oz})) {
          const float og = other->ground[at(nx - ox * kCells, nz - oz * kCells)];
          if (!std::isnan(og))
            g = std::min(g, og);
        }
      }
    return g - kRubbleSunk;
  }
  // Lumps scaled by depth (thin layers stay flat), never below the ground.
  const float amount = std::min(0.25f * d, kRubbleLumps);
  const float lump =
      0.6f * smoothNoise(gx, gz) + 0.4f * (2.0f * unit(hashSample(gx, gz, 1u)) - 1.0f);
  return g + d + amount * lump;
}

MeshData RubbleLayer::tileMesh(RubbleKey key) const {
  const Tile *t = findTile(key);
  if (!t)
    return {};
  float lowest = std::numeric_limits<float>::max();
  for (float g : t->ground)
    if (!std::isnan(g))
      lowest = std::min(lowest, g);
  if (lowest == std::numeric_limits<float>::max())
    lowest = 0.0f;
  // The drawn heights with a ring of the neighbours' around them, read once,
  // for the normals across tile borders.
  constexpr int W = N + 2;
  std::vector<float> drawn(W * W);
  for (int j = -1; j <= N; ++j)
    for (int i = -1; i <= N; ++i)
      drawn[(j + 1) * W + (i + 1)] =
          drawnHeight(key.first * kCells + i, key.second * kCells + j, lowest);
  auto h = [&](int i, int j) { return drawn[(j + 1) * W + (i + 1)]; };
  const glm::vec2 corner = rubbleTileCorner(key);
  MeshData mesh;
  mesh.vertices.reserve(N * N);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) {
      const int gx = key.first * kCells + i, gz = key.second * kCells + j;
      const float slopeX = (h(i + 1, j) - h(i - 1, j)) / (2.0f * kRubbleSpacing);
      const float slopeZ = (h(i, j + 1) - h(i, j - 1)) / (2.0f * kRubbleSpacing);
      const glm::vec3 n = glm::normalize(glm::vec3(-slopeX, 1.0f, -slopeZ));
      // Bare samples take the colour of their deepest neighbour, so a pile's
      // edge doesn't fade to grey.
      glm::vec3 c = t->colour[at(i, j)];
      if (t->depth[at(i, j)] < kEmpty) {
        float deepest = kEmpty;
        for (int dz = -1; dz <= 1; ++dz)
          for (int dx = -1; dx <= 1; ++dx) {
            const int ni = i + dx, nj = j + dz;
            if (ni < 0 || nj < 0 || ni >= N || nj >= N || t->depth[at(ni, nj)] <= deepest)
              continue;
            deepest = t->depth[at(ni, nj)];
            c = t->colour[at(ni, nj)];
          }
      }
      c *= 0.88f + 0.24f * unit(hashSample(gx, gz, 2u));
      const glm::vec3 u = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), n));
      const glm::vec3 v = glm::cross(n, u);
      const glm::vec3 p(i * kRubbleSpacing, h(i, j), j * kRubbleSpacing);
      mesh.vertices.push_back({p, c, corner + glm::vec2(p.x, p.z), n, u, -v});
    }
  mesh.indices.reserve(kCells * kCells * 6);
  for (int j = 0; j < kCells; ++j)
    for (int i = 0; i < kCells; ++i) {
      const uint32_t a = at(i, j), b = at(i + 1, j), c = at(i, j + 1), d = at(i + 1, j + 1);
      for (uint32_t index : {a, c, d, a, d, b})
        mesh.indices.push_back(index);
    }
  return mesh;
}

std::vector<float> RubbleLayer::tileHeights(RubbleKey key) const {
  const Tile *t = findTile(key);
  if (!t)
    return {};
  std::vector<float> heights(N * N);
  for (int k = 0; k < N * N; ++k)
    heights[k] = (std::isnan(t->ground[k]) || t->depth[k] < kRubbleThinnest)
                     ? kNaN
                     : t->ground[k] + t->depth[k];
  return heights;
}

void RubbleLayer::clear() { tiles_.clear(); }

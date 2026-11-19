#include "EditorSelection.h"

#include <algorithm>

namespace {

template <typename T> void copyIfChanged(const T &before, const T &after, T &target) {
  if (!(before == after))
    target = after;
}

template <int N, typename T>
void copyParts(const glm::vec<N, T> &before, const glm::vec<N, T> &after, glm::vec<N, T> &target) {
  for (int i = 0; i < N; ++i)
    if (before[i] != after[i])
      target[i] = after[i];
}

} // namespace

void EditorSelection::clear() {
  picked.clear();
  anchor.reset();
}

void EditorSelection::select(SceneEntryRef ref) {
  picked = {ref};
  anchor = ref;
}

void EditorSelection::toggle(SceneEntryRef ref) {
  const auto at = std::find(picked.begin(), picked.end(), ref);
  if (at != picked.end()) {
    picked.erase(at);
  } else {
    picked.push_back(ref);
    anchor = ref;
  }
}

void EditorSelection::selectRange(const std::vector<SceneEntryRef> &order, SceneEntryRef to) {
  const auto end = std::find(order.begin(), order.end(), to);
  const auto start = anchor ? std::find(order.begin(), order.end(), *anchor) : order.end();
  if (end == order.end() || start == order.end()) {
    select(to);
    return;
  }
  const auto [first, last] = std::minmax(start, end);
  picked.clear();
  for (auto it = first; it <= last; ++it)
    if (*it != to)
      picked.push_back(*it);
  picked.push_back(to); // the primary
}

bool EditorSelection::contains(SceneEntryRef ref) const {
  return std::find(picked.begin(), picked.end(), ref) != picked.end();
}

std::optional<SceneEntryRef> EditorSelection::primary() const {
  if (picked.empty())
    return std::nullopt;
  return picked.back();
}

void EditorSelection::removed(SceneEntryRef ref) {
  picked.erase(std::remove(picked.begin(), picked.end(), ref), picked.end());
  if (anchor == ref)
    anchor.reset();
  auto down = [&](SceneEntryRef &r) {
    if (r.kind == ref.kind && r.kind != SceneEntryRef::Kind::Player && r.index > ref.index)
      --r.index;
  };
  for (SceneEntryRef &r : picked)
    down(r);
  if (anchor)
    down(*anchor);
}

void EditorSelection::shifted(SceneEntryRef::Kind kind, int from, int by) {
  auto move = [&](SceneEntryRef &r) {
    if (r.kind == kind && r.index >= from)
      r.index += by;
  };
  for (SceneEntryRef &r : picked)
    move(r);
  if (anchor)
    move(*anchor);
}

std::vector<SceneEntryRef> deletionOrder(std::vector<SceneEntryRef> refs) {
  std::sort(refs.begin(), refs.end(), [](const SceneEntryRef &a, const SceneEntryRef &b) {
    if (a.kind != b.kind)
      return a.kind < b.kind;
    return a.index > b.index;
  });
  refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
  return refs;
}

void copyChanges(const SceneModelEntry &before, const SceneModelEntry &after,
                 SceneModelEntry &target) {
  copyIfChanged(before.path, after.path, target.path);
  copyIfChanged(before.shape, after.shape, target.shape);
  copyParts(before.size, after.size, target.size);
  copyIfChanged(before.radius, after.radius, target.radius);
  copyIfChanged(before.height, after.height, target.height);
  copyParts(before.color, after.color, target.color);
  copyIfChanged(before.body, after.body, target.body);
  copyIfChanged(before.mass, after.mass, target.mass);
  copyIfChanged(before.density, after.density, target.density);
  copyIfChanged(before.material, after.material, target.material);
  copyIfChanged(before.collider, after.collider, target.collider);
  copyIfChanged(before.friction, after.friction, target.friction);
  copyIfChanged(before.restitution, after.restitution, target.restitution);
  copyIfChanged(before.breakPieces, after.breakPieces, target.breakPieces);
  copyIfChanged(before.breakSeed, after.breakSeed, target.breakSeed);
  copyIfChanged(before.breakStrength, after.breakStrength, target.breakStrength);
  copyIfChanged(before.glued, after.glued, target.glued);
  copyParts(before.transform.position, after.transform.position, target.transform.position);
  copyParts(before.transform.rotationDeg, after.transform.rotationDeg,
            target.transform.rotationDeg);
  copyIfChanged(before.transform.scale, after.transform.scale, target.transform.scale);
  copyIfChanged(before.animation, after.animation, target.animation);
}

void copyChanges(const SceneGridEntry &before, const SceneGridEntry &after,
                 SceneGridEntry &target) {
  copyChanges(before.model, after.model, target.model);
  copyParts(before.count, after.count, target.count);
  copyParts(before.spacing, after.spacing, target.spacing);
  copyIfChanged(before.layoutRotation, after.layoutRotation, target.layoutRotation);
}

void copyChanges(const SceneCharacterEntry &before, const SceneCharacterEntry &after,
                 SceneCharacterEntry &target) {
  copyIfChanged(before.file, after.file, target.file);
  copyParts(before.position, after.position, target.position);
  copyIfChanged(before.yaw, after.yaw, target.yaw);
  // A variant belongs to its file: a new file brings the primary's along.
  if (before.variant != after.variant || before.file != after.file)
    target.variant = after.variant;
}

#pragma once

#include "Scene.h"

#include <optional>
#include <vector>

// What the editor has selected: one entry or several (Ctrl adds or takes
// away, Shift takes a range), the last one picked being the primary, the
// one the Inspector shows and the gizmo sits on. Kept right as entries are
// deleted or added. No ImGui here, so it's tested like the rest of
// heliora_core.
class EditorSelection {
public:
  void clear();
  // Just this one (a plain click).
  void select(SceneEntryRef ref);
  // Added if it wasn't selected (and made the primary), taken away if it was.
  void toggle(SceneEntryRef ref);
  // Everything in `order` from the last plain or Ctrl click to `to`, in
  // place of what was selected; just `to` without such a click.
  void selectRange(const std::vector<SceneEntryRef> &order, SceneEntryRef to);

  bool contains(SceneEntryRef ref) const;
  std::optional<SceneEntryRef> primary() const;
  const std::vector<SceneEntryRef> &all() const { return picked; }
  bool empty() const { return picked.empty(); }
  size_t size() const { return picked.size(); }

  // That entry was taken out: it goes, and later ones of its kind move down.
  void removed(SceneEntryRef ref);
  // `by` entries of that kind were put in at `from`: the ones there and
  // after move up.
  void shifted(SceneEntryRef::Kind kind, int from, int by);

private:
  std::vector<SceneEntryRef> picked; // in the order picked, the primary last
  std::optional<SceneEntryRef> anchor; // where a Shift range starts
};

// The order to take several entries out in, so taking one out never moves
// one still to go: each kind from its highest index down.
std::vector<SceneEntryRef> deletionOrder(std::vector<SceneEntryRef> refs);

// Editing several at once: whatever changed from before to after (field by
// field, and each part of a position, turn, size or colour on its own) is
// set the same on target. Where it came from (its prefab and its comment)
// and the player tick aren't copied. A character's new file brings its
// variant with it.
void copyChanges(const SceneModelEntry &before, const SceneModelEntry &after,
                 SceneModelEntry &target);
void copyChanges(const SceneGridEntry &before, const SceneGridEntry &after,
                 SceneGridEntry &target);
void copyChanges(const SceneCharacterEntry &before, const SceneCharacterEntry &after,
                 SceneCharacterEntry &target);

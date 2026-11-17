#pragma once

#include "CharacterMovement.h"
#include "Controller.h"

#include <memory>
#include <vector>

// The characters in play: each a body, driven by a controller or standing
// still. A body has one controller at most and a controller one body:
// possessing takes the body from the controller it had, and the controller
// from the body it had. Controllers must outlive the roster, or be released
// first.
class CharacterRoster {
public:
  CharacterRoster() = default;
  ~CharacterRoster() { clear(); }
  CharacterRoster(const CharacterRoster &) = delete;
  CharacterRoster &operator=(const CharacterRoster &) = delete;

  // A body with no controller. Its index stays its own until clear().
  size_t add(std::unique_ptr<CharacterMovement> body);
  // Takes every body out of the world, letting go of their controllers.
  void clear();

  size_t size() const { return bodies.size(); }
  CharacterMovement &body(size_t index) { return *bodies[index].body; }
  const CharacterMovement &body(size_t index) const { return *bodies[index].body; }
  Controller *controllerOf(size_t index) const { return bodies[index].controller; }

  // Hands the body to the controller (Unreal's Possess).
  void possess(Controller &controller, size_t index);
  // Its body stands still from now on.
  void release(Controller &controller);

  // One physics step for every body: its controller's commands, or none.
  // Right before PhysicsWorld::step(), with the same dt.
  void step(float dt);

private:
  struct Entry {
    std::unique_ptr<CharacterMovement> body;
    Controller *controller = nullptr;
  };
  std::vector<Entry> bodies;
};

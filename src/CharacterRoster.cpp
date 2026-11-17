#include "CharacterRoster.h"

size_t CharacterRoster::add(std::unique_ptr<CharacterMovement> body) {
  bodies.push_back({std::move(body), nullptr});
  return bodies.size() - 1;
}

void CharacterRoster::clear() {
  for (Entry &entry : bodies)
    if (entry.controller) {
      entry.controller->possessed = nullptr;
      entry.controller->onPossess();
    }
  bodies.clear(); // each body takes itself out of the world
}

void CharacterRoster::possess(Controller &controller, size_t index) {
  Entry &entry = bodies[index];
  if (entry.controller == &controller)
    return;
  release(controller);
  if (entry.controller) {
    entry.controller->possessed = nullptr;
    entry.controller->onPossess();
  }
  entry.controller = &controller;
  controller.possessed = entry.body.get();
  controller.onPossess();
}

void CharacterRoster::release(Controller &controller) {
  for (Entry &entry : bodies)
    if (entry.controller == &controller) {
      entry.controller = nullptr;
      controller.possessed = nullptr;
      controller.onPossess();
    }
}

void CharacterRoster::step(float dt) {
  for (Entry &entry : bodies)
    entry.body->update(entry.controller ? entry.controller->decide() : CharacterInput{},
                       dt);
}

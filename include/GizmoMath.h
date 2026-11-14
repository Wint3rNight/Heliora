#pragma once

#include "Picking.h"

#include <glm/glm.hpp>

#include <optional>

// The maths behind the gizmo, the handles drawn on the selected object.
// While a handle is dragged, the cursor is a ray (rayFromScreen) and these
// turn it into a position, an angle or a scale.

// The point on the line through axisOrigin along axisDir (unit length) that
// comes closest to the ray. Nothing when the ray runs along the axis, or
// when that point is behind the ray's start.
std::optional<glm::vec3> closestPointOnAxis(const PickRay &ray,
                                            glm::vec3 axisOrigin,
                                            glm::vec3 axisDir);

// Where the ray crosses the plane through centre that faces along axis (unit
// length), as an angle in radians around axis, growing the right-hand way.
// Only the change between two calls means anything. Nothing when the ray
// runs along the plane or crosses it behind its start.
std::optional<float> angleAroundAxis(const PickRay &ray, glm::vec3 centre,
                                     glm::vec3 axis);

// value rounded to the nearest multiple of step. A step of 0 or less leaves
// it alone.
float snap(float value, float step);

// The pixel a point lands on, in window coordinates with y counting down,
// or nothing if it's behind the camera. The reverse of rayFromScreen, with
// the same unjittered projection times view.
std::optional<glm::vec2> screenFromWorld(glm::vec3 point, glm::vec2 viewport,
                                         const glm::mat4 &viewProj);

// How far point is from the segment a to b.
float distanceToSegment(glm::vec2 point, glm::vec2 a, glm::vec2 b);

// The scale handle: what to multiply the scale by when the cursor goes from
// grabbed to now, both measured from the gizmo's centre on screen. Twice as
// far out doubles it.
float scaleFromDrag(glm::vec2 centre, glm::vec2 grabbed, glm::vec2 now);

#include "Stress.h"

#include <algorithm>
#include <cmath>

namespace {

// Stiffness: only the faces matter for which link carries what, so one
// material for all of them, with the shear modulus of a Poisson ratio of 0.25.
constexpr double kStiffness = 1.0;
constexpr double kShearStiffness = 0.4;
// A contact only rests: four springs across its face that only push, soft
// across (kContactShear of a glued face's), and a turning spring this weak
// (kContactTurn) only so the system can't go singular while it tips.
constexpr double kContactShear = 0.01;
constexpr double kContactTurn = 1e-4;
constexpr int kContactRounds = 12; // lift-off rounds a build
constexpr double kShortestSpring = 0.01; // m
constexpr float kLeastForce = 50.0f;     // N: a contact carrying less isn't judged (5 kg)

glm::dmat3 skew(const glm::dvec3 &r) {
  return glm::dmat3(glm::dvec3(0.0, r.z, -r.y), glm::dvec3(-r.z, 0.0, r.x),
                    glm::dvec3(r.y, -r.x, 0.0));
}

double dot(const std::vector<glm::dvec3> &a, const std::vector<glm::dvec3> &b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    sum += glm::dot(a[i], b[i]);
  return sum;
}

} // namespace

LinkFace linkFace(const std::vector<glm::vec3> &polygon, glm::vec3 normal) {
  LinkFace face;
  if (polygon.size() < 3 || glm::length(normal) < 1e-6f)
    return face;
  const glm::dvec3 n = glm::normalize(glm::dvec3(normal));
  const glm::dvec3 e1 = glm::normalize(
      glm::cross(n, std::fabs(n.x) < 0.9 ? glm::dvec3(1, 0, 0) : glm::dvec3(0, 1, 0)));
  const glm::dvec3 e2 = glm::cross(n, e1);
  const glm::dvec3 origin(polygon.front());
  std::vector<glm::dvec2> q;
  for (const glm::vec3 &p : polygon)
    q.emplace_back(glm::dot(glm::dvec3(p) - origin, e1), glm::dot(glm::dvec3(p) - origin, e2));
  // Area, middle and second moments of the polygon (sums over its edges),
  // about its corner first, then moved to its middle.
  double twice = 0.0, cx = 0.0, cy = 0.0, ixx = 0.0, iyy = 0.0, ixy = 0.0; // ixx = sum y^2 dA
  for (size_t i = 0; i < q.size(); ++i) {
    const glm::dvec2 a = q[i], b = q[(i + 1) % q.size()];
    const double cross = a.x * b.y - b.x * a.y;
    twice += cross;
    cx += (a.x + b.x) * cross;
    cy += (a.y + b.y) * cross;
    ixx += cross * (a.y * a.y + a.y * b.y + b.y * b.y);
    iyy += cross * (a.x * a.x + a.x * b.x + b.x * b.x);
    ixy += cross * (a.x * b.y + 2.0 * a.x * a.y + 2.0 * b.x * b.y + b.x * a.y);
  }
  if (std::fabs(twice) < 1e-12)
    return face;
  cx /= 3.0 * twice;
  cy /= 3.0 * twice;
  const double sign = twice > 0.0 ? 1.0 : -1.0;
  const double area = 0.5 * twice * sign;
  ixx = sign * ixx / 12.0 - area * cy * cy;
  iyy = sign * iyy / 12.0 - area * cx * cx;
  ixy = sign * ixy / 24.0 - area * cx * cy;
  // The principal axes: about u the second moment is the sum of the squared
  // distance across it, w = -x sin + y cos.
  const double phi = 0.5 * std::atan2(2.0 * ixy, iyy - ixx);
  const double c = std::cos(phi), s = std::sin(phi);
  const double inertiaU = s * s * iyy + c * c * ixx - 2.0 * s * c * ixy;
  const double inertiaV = c * c * iyy + s * s * ixx + 2.0 * s * c * ixy;
  double reachU = 0.0, reachV = 0.0;
  for (const glm::dvec2 &p : q) {
    const glm::dvec2 d = p - glm::dvec2(cx, cy);
    reachU = std::max(reachU, std::fabs(d.x * c + d.y * s));
    reachV = std::max(reachV, std::fabs(-d.x * s + d.y * c));
  }
  face.area = static_cast<float>(area);
  face.centre = glm::vec3(origin + cx * e1 + cy * e2);
  face.normal = glm::vec3(n);
  face.axisU = glm::vec3(c * e1 + s * e2);
  face.inertiaU = static_cast<float>(std::max(inertiaU, 0.0));
  face.inertiaV = static_cast<float>(std::max(inertiaV, 0.0));
  face.reachU = static_cast<float>(reachU);
  face.reachV = static_cast<float>(reachV);
  return face;
}

LinkFace movedFace(const LinkFace &face, const glm::mat4 &where) {
  LinkFace moved = face;
  const glm::mat3 turn(where);
  moved.centre = glm::vec3(where * glm::vec4(face.centre, 1.0f));
  moved.normal = glm::normalize(turn * face.normal);
  moved.axisU = glm::normalize(turn * face.axisU);
  return moved;
}

void StressSolver::build(const std::vector<StressPiece> &pieces, const std::vector<StressLink> &links,
                         glm::vec3 gravity) {
  const int count = static_cast<int>(pieces.size());
  // Only pieces with a path to the ground can be balanced; the rest would
  // drift (and make the system singular).
  std::vector<std::vector<int>> neighbours(count);
  std::vector<int> frontier;
  std::vector<char> reached(count, 0);
  for (const StressLink &link : links) {
    if (link.a < 0 || link.a >= count || link.b >= count || link.face.area <= 0.0f)
      continue; // no face: no spring, so it holds nothing
    if (link.b < 0) {
      if (!reached[link.a])
        frontier.push_back(link.a);
      reached[link.a] = 1;
    } else {
      neighbours[link.a].push_back(link.b);
      neighbours[link.b].push_back(link.a);
    }
  }
  while (!frontier.empty()) {
    const int p = frontier.back();
    frontier.pop_back();
    for (int o : neighbours[p])
      if (!reached[o]) {
        reached[o] = 1;
        frontier.push_back(o);
      }
  }
  solvedOf_.assign(count, -1);
  centres_.clear();
  loadForce_.clear();
  loadTorque_.clear();
  for (int i = 0; i < count; ++i)
    if (reached[i]) {
      solvedOf_[i] = static_cast<int>(centres_.size());
      centres_.emplace_back(pieces[i].centre);
      loadForce_.push_back(glm::dvec3(gravity) * static_cast<double>(pieces[i].mass));
      loadTorque_.emplace_back(0.0);
    }
  const size_t solved = centres_.size();

  springs_.clear();
  points_.clear();
  springOf_.assign(links.size(), -1);
  for (size_t l = 0; l < links.size(); ++l) {
    const StressLink &link = links[l];
    if (link.a < 0 || link.a >= count || solvedOf_[link.a] < 0 || link.face.area <= 0.0f)
      continue;
    if (link.b >= 0 && (link.b >= count || solvedOf_[link.b] < 0))
      continue;
    Spring sp;
    sp.a = solvedOf_[link.a];
    sp.b = link.b < 0 ? -1 : solvedOf_[link.b];
    sp.point = glm::dvec3(link.face.centre);
    sp.normal = glm::normalize(glm::dvec3(link.face.normal));
    sp.axisU = glm::normalize(glm::dvec3(link.face.axisU));
    sp.axisV = glm::cross(sp.normal, sp.axisU);
    sp.area = link.face.area;
    sp.inertiaU = link.face.inertiaU;
    sp.inertiaV = link.face.inertiaV;
    sp.reachU = link.face.reachU;
    sp.reachV = link.face.reachV;
    sp.contact = link.contact;
    const double length = std::max(
        kShortestSpring, sp.b < 0 ? 2.0 * std::fabs(glm::dot(sp.point - centres_[sp.a], sp.normal))
                                  : glm::length(centres_[sp.b] - centres_[sp.a]));
    const double turnShare = sp.contact ? kContactTurn : 1.0;
    sp.kBendU = turnShare * kStiffness * sp.inertiaU / length;
    sp.kBendV = turnShare * kStiffness * sp.inertiaV / length;
    sp.kTwist = turnShare * kShearStiffness * (sp.inertiaU + sp.inertiaV) / length;
    if (!sp.contact) {
      sp.kNormal = kStiffness * sp.area / length;
      sp.kShear = kShearStiffness * sp.area / length;
    } else {
      // Four push-only springs across the face, at its reach along its axes.
      sp.firstPoint = static_cast<int>(points_.size());
      const double side = 0.5 * std::sqrt(sp.area);
      const double ru = sp.reachU > 0.0 ? sp.reachU : side;
      const double rv = sp.reachV > 0.0 ? sp.reachV : side;
      for (const glm::dvec3 &offset : {ru * sp.axisU, -ru * sp.axisU, rv * sp.axisV, -rv * sp.axisV}) {
        Point pt;
        pt.position = sp.point + offset;
        pt.kNormal = kStiffness * 0.25 * sp.area / length;
        pt.kShear = kContactShear * kShearStiffness * 0.25 * sp.area / length;
        points_.push_back(pt);
      }
    }
    springOf_[l] = static_cast<int>(springs_.size());
    springs_.push_back(sp);
  }
  loose_.assign(solved, 0);
  liveForce_.assign(solved, glm::dvec3(0.0));
  liveTorque_.assign(solved, glm::dvec3(0.0));
  rebuildBlocks();
  shift_.assign(solved, glm::dvec3(0.0));
  turn_.assign(solved, glm::dvec3(0.0));
  started_ = false;
  settled_ = false;
  iterations_ = 0;
  rounds_ = 0;
}

// The diagonal blocks for the preconditioner (each piece's shift against
// shift and turn against turn), inverted; again whenever contacts lift.
void StressSolver::rebuildBlocks() {
  const size_t solved = centres_.size();
  shiftBlock_.assign(solved, glm::dmat3(0.0));
  turnBlock_.assign(solved, glm::dmat3(0.0));
  for (const Spring &sp : springs_) {
    const glm::dmat3 nn = glm::outerProduct(sp.normal, sp.normal);
    auto translation = [&](const glm::dvec3 &at, double kNormal, double kShear) {
      const glm::dmat3 s = kNormal * nn + kShear * (glm::dmat3(1.0) - nn);
      for (int end : {sp.a, sp.b}) {
        if (end < 0)
          continue;
        const glm::dmat3 x = skew(at - centres_[end]);
        shiftBlock_[end] += s;
        turnBlock_[end] -= x * s * x;
      }
    };
    if (!sp.contact)
      translation(sp.point, sp.kNormal, sp.kShear);
    else
      for (int k = 0; k < 4; ++k)
        if (points_[sp.firstPoint + k].active)
          translation(points_[sp.firstPoint + k].position, points_[sp.firstPoint + k].kNormal,
                      points_[sp.firstPoint + k].kShear);
    const glm::dmat3 r = sp.kTwist * nn + sp.kBendU * glm::outerProduct(sp.axisU, sp.axisU) +
                         sp.kBendV * glm::outerProduct(sp.axisV, sp.axisV);
    for (int end : {sp.a, sp.b})
      if (end >= 0)
        turnBlock_[end] += r;
  }
  for (size_t i = 0; i < solved; ++i)
    for (glm::dmat3 *block : {&shiftBlock_[i], &turnBlock_[i]}) {
      const double trace = (*block)[0][0] + (*block)[1][1] + (*block)[2][2];
      // Nothing holds it this way (a piece whose contacts all lifted): its
      // load is taken off, so nothing to precondition.
      *block = trace > 1e-30 ? glm::inverse(*block + glm::dmat3(1e-9 * trace)) : glm::dmat3(0.0);
    }
}

void StressSolver::setStart(const std::vector<glm::vec3> &shifts, const std::vector<glm::vec3> &turns) {
  for (size_t i = 0; i < solvedOf_.size(); ++i) {
    const int k = solvedOf_[i];
    if (k < 0)
      continue;
    if (i < shifts.size())
      shift_[k] = glm::dvec3(shifts[i]);
    if (i < turns.size())
      turn_[k] = glm::dvec3(turns[i]);
  }
  // Which contacts had lifted comes with the answer.
  if (liftContacts()) {
    findLoose();
    rebuildBlocks();
  }
  started_ = false;
  settled_ = false;
}

void StressSolver::addLoad(int piece, glm::vec3 force, glm::vec3 torque) {
  if (piece < 0 || piece >= static_cast<int>(solvedOf_.size()) || solvedOf_[piece] < 0)
    return;
  loadForce_[solvedOf_[piece]] += glm::dvec3(force);
  loadTorque_[solvedOf_[piece]] += glm::dvec3(torque);
  started_ = false;
  settled_ = false;
}

// How far the link's second side moved from its first at this point.
glm::dvec3 StressSolver::relative(const Spring &sp, const glm::dvec3 &at,
                                  const std::vector<glm::dvec3> &shift,
                                  const std::vector<glm::dvec3> &turn) const {
  glm::dvec3 d = -(shift[sp.a] + glm::cross(turn[sp.a], at - centres_[sp.a]));
  if (sp.b >= 0)
    d += shift[sp.b] + glm::cross(turn[sp.b], at - centres_[sp.b]);
  return d;
}

// The springs' forces and torques on every piece for these shifts and turns:
// the stiffness matrix times them, without building it.
void StressSolver::multiply(const std::vector<glm::dvec3> &shift, const std::vector<glm::dvec3> &turn,
                            std::vector<glm::dvec3> &outForce,
                            std::vector<glm::dvec3> &outTorque) const {
  outForce.assign(shift.size(), glm::dvec3(0.0));
  outTorque.assign(shift.size(), glm::dvec3(0.0));
  for (const Spring &sp : springs_) {
    auto apply = [&](const glm::dvec3 &at, const glm::dvec3 &force) {
      if (sp.b >= 0) {
        outForce[sp.b] += force;
        outTorque[sp.b] += glm::cross(at - centres_[sp.b], force);
      }
      outForce[sp.a] -= force;
      outTorque[sp.a] -= glm::cross(at - centres_[sp.a], force);
    };
    auto spring = [&](const glm::dvec3 &at, double kNormal, double kShear) {
      const glm::dvec3 d = relative(sp, at, shift, turn);
      const double along = glm::dot(d, sp.normal);
      apply(at, kNormal * along * sp.normal + kShear * (d - along * sp.normal));
    };
    if (!sp.contact)
      spring(sp.point, sp.kNormal, sp.kShear);
    else
      for (int k = 0; k < 4; ++k)
        if (points_[sp.firstPoint + k].active)
          spring(points_[sp.firstPoint + k].position, points_[sp.firstPoint + k].kNormal,
                 points_[sp.firstPoint + k].kShear);
    const glm::dvec3 phi = (sp.b >= 0 ? turn[sp.b] : glm::dvec3(0.0)) - turn[sp.a];
    const glm::dvec3 torque = sp.kTwist * glm::dot(phi, sp.normal) * sp.normal +
                              sp.kBendU * glm::dot(phi, sp.axisU) * sp.axisU +
                              sp.kBendV * glm::dot(phi, sp.axisV) * sp.axisV;
    if (sp.b >= 0)
      outTorque[sp.b] += torque;
    outTorque[sp.a] -= torque;
  }
}

void StressSolver::precondition(const std::vector<glm::dvec3> &force,
                                const std::vector<glm::dvec3> &torque,
                                std::vector<glm::dvec3> &outShift,
                                std::vector<glm::dvec3> &outTurn) const {
  outShift.resize(force.size());
  outTurn.resize(force.size());
  for (size_t i = 0; i < force.size(); ++i) {
    outShift[i] = shiftBlock_[i] * force[i];
    outTurn[i] = turnBlock_[i] * torque[i];
  }
}

// Contacts' springs that would pull lift off; lifted ones pressed back in
// come back. The active set: true if anything changed.
bool StressSolver::liftContacts() {
  double scale = 0.0;
  for (const glm::dvec3 &s : shift_)
    scale = std::max(scale, glm::length(s));
  const double slack = 1e-6 * scale;
  bool changed = false;
  for (const Spring &sp : springs_) {
    if (!sp.contact)
      continue;
    for (int k = 0; k < 4; ++k) {
      Point &pt = points_[sp.firstPoint + k];
      const double stretch = glm::dot(relative(sp, pt.position, shift_, turn_), sp.normal);
      if (pt.active && stretch > slack) {
        pt.active = false;
        changed = true;
      } else if (!pt.active && stretch < -slack) {
        pt.active = true;
        changed = true;
      }
    }
  }
  return changed;
}

// Pieces no longer held by anything (every contact under them lifted): their
// load comes off, so the rest can still be balanced; they're falling.
void StressSolver::findLoose() {
  const size_t solved = centres_.size();
  std::vector<std::vector<int>> neighbours(solved);
  std::vector<char> held(solved, 0);
  std::vector<int> frontier;
  for (const Spring &sp : springs_) {
    bool holds = !sp.contact;
    for (int k = 0; holds == false && k < 4; ++k)
      holds = points_[sp.firstPoint + k].active;
    if (!holds)
      continue;
    if (sp.b < 0) {
      if (!held[sp.a])
        frontier.push_back(sp.a);
      held[sp.a] = 1;
    } else {
      neighbours[sp.a].push_back(sp.b);
      neighbours[sp.b].push_back(sp.a);
    }
  }
  while (!frontier.empty()) {
    const int p = frontier.back();
    frontier.pop_back();
    for (int o : neighbours[p])
      if (!held[o]) {
        held[o] = 1;
        frontier.push_back(o);
      }
  }
  for (size_t i = 0; i < solved; ++i)
    loose_[i] = !held[i];
}

void StressSolver::restart() {
  std::vector<glm::dvec3> kf, kt;
  multiply(shift_, turn_, kf, kt);
  rForce_.resize(kf.size());
  rTorque_.resize(kt.size());
  double norm = 0.0;
  for (size_t i = 0; i < kf.size(); ++i) {
    const glm::dvec3 f = loose_[i] ? glm::dvec3(0.0) : loadForce_[i] + liveForce_[i];
    const glm::dvec3 t = loose_[i] ? glm::dvec3(0.0) : loadTorque_[i] + liveTorque_[i];
    norm += glm::dot(f, f) + glm::dot(t, t);
    rForce_[i] = f - kf[i];
    rTorque_[i] = t - kt[i];
    if (loose_[i] && glm::dot(shiftBlock_[i][0], shiftBlock_[i][0]) == 0.0)
      rForce_[i] = glm::dvec3(0.0); // nothing holds it: nothing to balance
  }
  precondition(rForce_, rTorque_, zShift_, zTurn_);
  pShift_ = zShift_;
  pTurn_ = zTurn_;
  rz_ = dot(rForce_, zShift_) + dot(rTorque_, zTurn_);
  loadNorm_ = std::sqrt(norm);
  started_ = true;
}

bool StressSolver::solve(int maxIterations, float tolerance) {
  if (centres_.empty()) {
    settled_ = true;
    return true;
  }
  if (!started_)
    restart();
  auto residual = [&] { return std::sqrt(dot(rForce_, rForce_) + dot(rTorque_, rTorque_)); };
  std::vector<glm::dvec3> apForce, apTorque;
  for (int k = 0; k < maxIterations; ++k) {
    const double target = static_cast<double>(tolerance) * std::max(loadNorm_, 1e-12);
    if (residual() <= target) {
      // Balanced for these contacts: lift those that pull, then again.
      if (rounds_ < kContactRounds && liftContacts()) {
        ++rounds_;
        findLoose();
        rebuildBlocks();
        restart();
        continue;
      }
      settled_ = true;
      return true;
    }
    multiply(pShift_, pTurn_, apForce, apTorque);
    const double pap = dot(pShift_, apForce) + dot(pTurn_, apTorque);
    if (!(pap > 0.0)) {
      settled_ = residual() <= target; // nothing left it can improve
      return settled_;
    }
    const double alpha = rz_ / pap;
    for (size_t i = 0; i < shift_.size(); ++i) {
      shift_[i] += alpha * pShift_[i];
      turn_[i] += alpha * pTurn_[i];
      rForce_[i] -= alpha * apForce[i];
      rTorque_[i] -= alpha * apTorque[i];
    }
    precondition(rForce_, rTorque_, zShift_, zTurn_);
    const double rz = dot(rForce_, zShift_) + dot(rTorque_, zTurn_);
    const double beta = rz / rz_;
    rz_ = rz;
    for (size_t i = 0; i < shift_.size(); ++i) {
      pShift_[i] = zShift_[i] + beta * pShift_[i];
      pTurn_[i] = zTurn_[i] + beta * pTurn_[i];
    }
    ++iterations_;
  }
  settled_ = false;
  return false;
}

std::vector<glm::vec3> StressSolver::shifts() const {
  std::vector<glm::vec3> out(solvedOf_.size(), glm::vec3(0.0f));
  for (size_t i = 0; i < solvedOf_.size(); ++i)
    if (solvedOf_[i] >= 0)
      out[i] = glm::vec3(shift_[solvedOf_[i]]);
  return out;
}

std::vector<glm::vec3> StressSolver::turns() const {
  std::vector<glm::vec3> out(solvedOf_.size(), glm::vec3(0.0f));
  for (size_t i = 0; i < solvedOf_.size(); ++i)
    if (solvedOf_[i] >= 0)
      out[i] = glm::vec3(turn_[solvedOf_[i]]);
  return out;
}

LinkStress StressSolver::stress(int link) const {
  LinkStress out;
  if (link < 0 || link >= static_cast<int>(springOf_.size()) || springOf_[link] < 0)
    return out;
  const Spring &sp = springs_[springOf_[link]];
  const glm::dvec3 phi = (sp.b >= 0 ? turn_[sp.b] : glm::dvec3(0.0)) - turn_[sp.a];
  const double twist = sp.kTwist * glm::dot(phi, sp.normal);
  const double bendU = sp.kBendU * glm::dot(phi, sp.axisU);
  const double bendV = sp.kBendV * glm::dot(phi, sp.axisV);
  out.area = static_cast<float>(sp.area);
  out.twist = static_cast<float>(std::fabs(twist));
  out.bendU = static_cast<float>(bendU);
  out.bendV = static_cast<float>(bendV);
  out.reachU = static_cast<float>(sp.reachU);
  out.reachV = static_cast<float>(sp.reachV);
  if (sp.contact) {
    // Its springs' push and slide; it can't pull. What turning force it
    // carries is only what its pressure can't: tipping.
    double normal = 0.0, push = 0.0;
    glm::dvec3 across(0.0);
    int lifted = 0;
    for (int k = 0; k < 4; ++k) {
      const Point &pt = points_[sp.firstPoint + k];
      if (!pt.active) {
        ++lifted;
        continue;
      }
      const glm::dvec3 d = relative(sp, pt.position, shift_, turn_);
      const double along = glm::dot(d, sp.normal);
      normal += pt.kNormal * along;
      push = std::max(push, -pt.kNormal * along / (0.25 * sp.area));
      across += pt.kShear * (d - along * sp.normal);
    }
    out.normalForce = static_cast<float>(normal);
    out.shearForce = static_cast<float>(glm::length(across));
    out.bending = static_cast<float>(std::sqrt(bendU * bendU + bendV * bendV));
    out.push = static_cast<float>(push);
    out.slide = static_cast<float>(glm::length(across) / sp.area);
    out.liftedShare = 0.25f * static_cast<float>(lifted);
    return out;
  }
  const glm::dvec3 d = relative(sp, sp.point, shift_, turn_);
  const double along = glm::dot(d, sp.normal);
  const double normal = sp.kNormal * along;
  const double shear = sp.kShear * glm::length(d - along * sp.normal);
  // Bending about u stretches and squeezes the face across it, along v.
  const double edge = std::fabs(bendU) * sp.reachV / std::max(sp.inertiaU, 1e-12) +
                      std::fabs(bendV) * sp.reachU / std::max(sp.inertiaV, 1e-12);
  const double axial = normal / sp.area;
  out.normalForce = static_cast<float>(normal);
  out.shearForce = static_cast<float>(shear);
  out.bending = static_cast<float>(std::sqrt(bendU * bendU + bendV * bendV));
  out.pull = static_cast<float>(axial + edge);
  out.push = static_cast<float>(-axial + edge);
  out.slide = static_cast<float>(shear / sp.area + std::fabs(twist) * std::max(sp.reachU, sp.reachV) /
                                                       std::max(sp.inertiaU + sp.inertiaV, 1e-12));
  return out;
}

float overload(const LinkStress &stress, const LinkLimits &limits) {
  float worst = stress.push / std::max(limits.push, 1e-6f);
  if (!limits.contact) {
    worst = std::max({worst, stress.pull / std::max(limits.pull, 1e-6f),
                      stress.slide / std::max(limits.slide, 1e-6f)});
    return std::max(worst, 0.0f);
  }
  // Resting: open, it holds nothing (whoever asked opens it); pressed, it
  // slides past friction times the push, and tips if it has to carry a
  // turning force its pressure can't.
  if (stress.liftedShare >= 1.0f)
    return std::max(worst, 0.0f);
  const float pressing = std::max(0.0f, -stress.normalForce);
  if (stress.shearForce > kLeastForce) {
    const float grip = limits.friction * pressing;
    worst = std::max(worst, grip > 0.0f ? stress.shearForce / grip : 10.0f);
  }
  return std::max({worst, tipping(stress, limits), 0.0f});
}

float tipping(const LinkStress &stress, const LinkLimits &limits) {
  // Only once half its springs lifted (a line or a point left); pressed wider
  // than that, its pressure carries the turning itself.
  if (!limits.contact || stress.liftedShare < 0.5f || stress.liftedShare >= 1.0f)
    return 0.0f;
  const float pressing = std::max(0.0f, -stress.normalForce);
  float worst = 0.0f;
  // Turning about the face's u axis tips it over an edge along u, with the
  // reach along v to lean on; and the other way round.
  auto axis = [&](float bend, float across) {
    if (std::fabs(bend) <= kLeastForce * std::max(across, 0.01f))
      return;
    const float holds = kTipShare * pressing * across;
    worst = std::max(worst, holds > 0.0f ? std::fabs(bend) / holds : 10.0f);
  };
  axis(stress.bendU, stress.reachV);
  axis(stress.bendV, stress.reachU);
  return worst;
}

void StressSolver::setLiveLoads(const std::vector<LiveLoad> &loads) {
  // Settled, this is a fresh solve; mid-solve it keeps counting, so one
  // that crawls still reaches the cap.
  if (settled_) {
    iterations_ = 0;
    rounds_ = 0;
  }
  std::fill(liveForce_.begin(), liveForce_.end(), glm::dvec3(0.0));
  std::fill(liveTorque_.begin(), liveTorque_.end(), glm::dvec3(0.0));
  for (const LiveLoad &load : loads) {
    if (load.piece < 0 || load.piece >= static_cast<int>(solvedOf_.size()) ||
        solvedOf_[load.piece] < 0)
      continue;
    liveForce_[solvedOf_[load.piece]] += glm::dvec3(load.force);
    liveTorque_[solvedOf_[load.piece]] += glm::dvec3(load.torque);
  }
  started_ = false;
  settled_ = false;
}

float overloadTime(float before, float overload, float seconds) {
  return overload > 1.0f ? before + seconds : 0.0f;
}

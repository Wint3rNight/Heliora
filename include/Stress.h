#pragma once

#include <glm/glm.hpp>

#include <vector>

// Weight: a stress solver over the breakable pieces (step 1.3d). The pieces
// are rigid and the links between them are springs at their faces (the
// rigid-body-spring model used for concrete fracture): stiff along the face's
// normal and across it in proportion to its area, and against bending and
// twisting in proportion to its second moments of area. Each piece gets a
// small shift and turn that balance gravity (and whatever else is put on
// it); from them every link gets its force and twist, and from those the
// stress on its stretched and squeezed edges and across it. No Vulkan and no
// physics here, so it's tested like the rest of heliora_core.

// A link's face, what its spring needs: its area, middle and normal (from the
// link's first piece to its second, or into the ground), a principal axis in
// it (axisV = cross(normal, axisU)), its second moments of area about those
// axes, and how far it reaches from its middle along each.
struct LinkFace {
  float area = 0.0f;
  glm::vec3 centre{0.0f};
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
  glm::vec3 axisU{1.0f, 0.0f, 0.0f};
  float inertiaU = 0.0f; // m^4, about axisU
  float inertiaV = 0.0f; // m^4, about axisV
  float reachU = 0.0f;   // m along axisU
  float reachV = 0.0f;   // m along axisV
};
// The face of a flat polygon (its corners in order) with this normal.
LinkFace linkFace(const std::vector<glm::vec3> &polygon, glm::vec3 normal);
// The face moved by a rigid transform (a piece's frame to the world).
LinkFace movedFace(const LinkFace &face, const glm::mat4 &where);

struct StressPiece {
  glm::vec3 centre{0.0f}; // its mass's middle
  float mass = 0.0f;      // kg
};
// A link between pieces a and b, or from a to the ground (b = -1). A contact
// only rests (between objects, on the ground): it's four springs across its
// face that only push, and any that would pull lifts off, so a slab on two
// walls sits on them instead of being clamped, walls side by side stand on
// their own, and a leaning wall lifts its light edge and stands on the rest
// until its weight leaves its base and it tips.
struct StressLink {
  int a = 0;
  int b = -1;
  LinkFace face;
  bool contact = false;
};

// What a link carries: its spring's force along the normal (positive
// pulling), across the face, its bending and twist; and from those the
// stress on its stretched edge (pull), its squeezed edge (push) and across
// it (slide), each positive the way it acts. Zero for a link of a piece with
// no way to the ground.
struct LinkStress {
  float normalForce = 0.0f; // N
  float shearForce = 0.0f;  // N
  float bending = 0.0f;     // N m
  float twist = 0.0f;       // N m
  float pull = 0.0f;        // Pa
  float push = 0.0f;        // Pa
  float slide = 0.0f;       // Pa
  float area = 0.0f;        // m^2, the face's
  // Its bending about the face's u and v axes (N m) and the face's reach
  // along them, for tipping; and for a contact the share of its springs
  // lifted off (1: it's open and holds nothing).
  float bendU = 0.0f;
  float bendV = 0.0f;
  float reachU = 0.0f;
  float reachV = 0.0f;
  float liftedShare = 0.0f;
};

// Weight put on a piece by what rests on it (debris, a character): a force
// through its middle and a torque about it.
struct LiveLoad {
  int piece = 0;
  glm::vec3 force{0.0f};
  glm::vec3 torque{0.0f};
};

class StressSolver {
public:
  // The pieces, their links and gravity (each piece loaded by its weight).
  // Pieces with no path to the ground are left out.
  void build(const std::vector<StressPiece> &pieces, const std::vector<StressLink> &links,
             glm::vec3 gravity = glm::vec3(0.0f, -9.81f, 0.0f));
  // Start from these shifts and turns (an earlier answer) instead of zero.
  void setStart(const std::vector<glm::vec3> &shifts, const std::vector<glm::vec3> &turns);
  // More load on a piece: a force through its middle and a torque.
  void addLoad(int piece, glm::vec3 force, glm::vec3 torque = glm::vec3(0.0f));
  // What rests on the pieces now, instead of what rested before; the answer
  // so far is where the next solve starts. Pieces outside the solve are
  // skipped.
  void setLiveLoads(const std::vector<LiveLoad> &loads);
  // Up to maxIterations more steps; true once the answer has settled (what's
  // left unbalanced under tolerance times the load).
  bool solve(int maxIterations, float tolerance = 1e-4f);
  int iterations() const { return iterations_; }
  bool settled() const { return settled_; }
  std::vector<glm::vec3> shifts() const;
  std::vector<glm::vec3> turns() const;
  LinkStress stress(int link) const;

private:
  struct Spring {
    int a = 0, b = -1; // into the solved pieces (b = -1: the ground)
    glm::dvec3 point{0.0}, normal{0.0}, axisU{0.0}, axisV{0.0};
    double kNormal = 0, kShear = 0, kTwist = 0, kBendU = 0, kBendV = 0;
    double area = 0, inertiaU = 0, inertiaV = 0, reachU = 0, reachV = 0;
    bool contact = false;
    int firstPoint = 0; // a contact's four push-only springs in points_
  };
  struct Point {
    glm::dvec3 position{0.0};
    double kNormal = 0, kShear = 0;
    bool active = true; // pressed; lifted off when it would pull
  };
  glm::dvec3 relative(const Spring &sp, const glm::dvec3 &at, const std::vector<glm::dvec3> &shift,
                      const std::vector<glm::dvec3> &turn) const;
  void rebuildBlocks();
  bool liftContacts(); // true if any of the contacts' springs changed
  void findLoose();
  void multiply(const std::vector<glm::dvec3> &shift, const std::vector<glm::dvec3> &turn,
                std::vector<glm::dvec3> &outForce, std::vector<glm::dvec3> &outTorque) const;
  void precondition(const std::vector<glm::dvec3> &force, const std::vector<glm::dvec3> &torque,
                    std::vector<glm::dvec3> &outShift, std::vector<glm::dvec3> &outTurn) const;
  void restart();

  std::vector<int> solvedOf_;       // piece -> solved index, or -1
  std::vector<int> springOf_;       // link -> spring, or -1
  std::vector<glm::dvec3> centres_; // by solved index
  std::vector<Spring> springs_;
  std::vector<Point> points_;
  std::vector<char> loose_; // by solved index: held by nothing now (all its contacts lifted)
  int rounds_ = 0;          // contact rounds since the last build
  std::vector<glm::dmat3> shiftBlock_, turnBlock_; // inverted diagonal blocks
  std::vector<glm::dvec3> loadForce_, loadTorque_; // gravity and addLoad
  std::vector<glm::dvec3> liveForce_, liveTorque_; // setLiveLoads
  std::vector<glm::dvec3> shift_, turn_;
  // Conjugate gradient state, kept between solve() calls.
  std::vector<glm::dvec3> rForce_, rTorque_, zShift_, zTurn_, pShift_, pTurn_;
  double rz_ = 0.0, loadNorm_ = 0.0;
  bool started_ = false;
  bool settled_ = false;
  int iterations_ = 0;
};

// What a link holds out against (Pa), and whether it only rests: a contact
// never pulls (its springs lift off instead), slides once the force across it
// passes friction times the force pressing it, and tips once it would have
// to carry a turning force its pressure can't (its weight past its edge).
struct LinkLimits {
  float pull = 1.0f;
  float push = 1.0f;
  float slide = 1.0f;
  bool contact = false;
  float friction = 0.5f;
};
constexpr float kTipShare = 0.25f;
// How far over its limits a link is: the worst of its pull, push and slide
// against theirs; over 1 breaks it (or opens a contact).
float overload(const LinkStress &stress, const LinkLimits &limits);
// Only the tipping part of a contact's overload (0 for a glued link or a
// contact pressed wider than a line): over 1, it's tipping off its edge.
float tipping(const LinkStress &stress, const LinkLimits &limits);
// How long a link has been over its limits: before plus seconds while its
// overload is over 1, back to 0 once it isn't.
float overloadTime(float before, float overload, float seconds);

#pragma once

// Tiny test runner, so we don't need a test framework.
//   TEST(name) { ... }        defines and registers a test
//   CHECK(cond)               records a failure and keeps the test running
//   CHECK_NEAR(a, b, eps)     the same, for numbers within eps of each other
// tests/main.cpp runs them all and returns non-zero if any check failed,
// which is what CTest looks at.

#include <cmath>
#include <cstdio>
#include <vector>

namespace test {

struct Case {
  const char *name;
  void (*fn)();
};

inline std::vector<Case> &registry() {
  static std::vector<Case> cases;
  return cases;
}

// Failed checks in the running test.
inline int &currentFailures() {
  static int failures = 0;
  return failures;
}

struct Registrar {
  Registrar(const char *name, void (*fn)()) { registry().push_back({name, fn}); }
};

} // namespace test

#define TEST(name)                                                             \
  static void name();                                                          \
  static test::Registrar name##Registrar(#name, name);                         \
  static void name()

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("    %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);  \
      ++test::currentFailures();                                               \
    }                                                                          \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
  do {                                                                         \
    const double checkA = (a), checkB = (b);                                   \
    if (!(std::fabs(checkA - checkB) <= (eps))) {                              \
      std::printf("    %s:%d: %s = %.6f, expected %.6f (within %g)\n",         \
                  __FILE__, __LINE__, #a, checkA, checkB, (double)(eps));      \
      ++test::currentFailures();                                               \
    }                                                                          \
  } while (0)

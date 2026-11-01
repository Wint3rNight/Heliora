#include "TestRunner.h"

#include <cstring>

// Runs every registered test, or only those whose name contains argv[1].
int main(int argc, char **argv) {
  const char *filter = argc > 1 ? argv[1] : nullptr;
  int failedTests = 0;
  int ranTests = 0;
  for (const test::Case &c : test::registry()) {
    if (filter && !std::strstr(c.name, filter))
      continue;
    test::currentFailures() = 0;
    c.fn();
    ++ranTests;
    if (test::currentFailures() > 0) {
      ++failedTests;
      std::printf("[FAIL] %s\n", c.name);
    } else {
      std::printf("[ OK ] %s\n", c.name);
    }
  }
  std::printf("%d of %d tests passed\n", ranTests - failedTests, ranTests);
  return failedTests == 0 ? 0 : 1;
}

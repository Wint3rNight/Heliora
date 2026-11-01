cmake_minimum_required(VERSION 3.20)  # Jolt's own CMake needs 3.20

project(VulkanRenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find Dependencies
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

# meshoptimizer — mesh simplification for LOD generation
include(FetchContent)
FetchContent_Declare(
  meshoptimizer
  GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
  GIT_TAG        v0.21
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(meshoptimizer)

# Dear ImGui (docking branch) — overlay UI
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        docking
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)
target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC Vulkan::Vulkan glfw)

# Jolt Physics, the one engine we don't write ourselves. Its options are plain
# variables set before FetchContent_MakeAvailable (its option() calls keep
# them, policy CMP0077). Its samples and tests only build when Jolt is the
# top-level project, so they're off already.
set(ENABLE_INSTALL OFF)
# The GPU compute backends are for Jolt's hair simulation. We have no hair.
set(JPH_USE_VK OFF)
set(JPH_USE_DX12 OFF)
set(JPH_USE_MTL OFF)
set(JPH_USE_CPU_COMPUTE OFF)
# Our own timers measure the step, and we don't use Jolt's ObjectStream file
# format.
set(PROFILER_IN_DEBUG_AND_RELEASE OFF)
set(ENABLE_OBJECT_STREAM OFF)
# On in Debug and Release for the F2 collision-shape view. A shipping build
# should turn it off, it adds a few branches to Jolt's narrow phase.
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE ON)
# -Werror on code we don't own breaks the build with every new gcc warning.
set(ENABLE_ALL_WARNINGS OFF)
# Debug symbols only in Debug. In Release they just cost build RAM.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(GENERATE_DEBUG_SYMBOLS ON)
else()
  set(GENERATE_DEBUG_SYMBOLS OFF)
endif()
# No link-time optimization until it's measured. Its link step runs a lot of
# compiler processes at once, and this machine's RAM doesn't like that.
set(INTERPROCEDURAL_OPTIMIZATION OFF)
# Jolt adds -mavx2 -mfma (PUBLIC) to everything that links it, so the game
# needs a CPU with AVX2 and FMA.
FetchContent_Declare(
  JoltPhysics
  GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
  GIT_TAG        v5.6.0
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  Build
)
FetchContent_MakeAvailable(JoltPhysics)

# Source File Management - Simplified
# Note: Using GLOB is okay for small projects, but listing them is safer.
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/*.h"
        "include/*.h"
        "include/*.hpp"
)

# Vulkan-free game code, shared by the game and the tests so it compiles
# once. Jolt is PUBLIC so everything linking this gets the same Jolt defines
# and CPU flags.
set(HELIORA_CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PhysicsWorld.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PrimitiveMesh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Scene.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneNode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ScenePhysics.cpp
)
list(REMOVE_ITEM ALL_SOURCES ${HELIORA_CORE_SOURCES})
add_library(heliora_core STATIC ${HELIORA_CORE_SOURCES})
target_include_directories(heliora_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(heliora_core
        PUBLIC glm::glm Jolt
        PRIVATE spdlog::spdlog
)

# Create the Executable
add_executable(${PROJECT_NAME} ${ALL_SOURCES})

# Include Paths - Using PUBLIC/PRIVATE correctly
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Linking
target_link_libraries(${PROJECT_NAME} PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        spdlog::spdlog
        Threads::Threads
        heliora_core
)

# Vulkan-style depth [0,1] for all glm projection matrices
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_FORCE_DEPTH_ZERO_TO_ONE)

if(TARGET assimp::assimp)
        target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)
else()
        target_include_directories(${PROJECT_NAME} PRIVATE ${ASSIMP_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${ASSIMP_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE meshoptimizer imgui_lib)

# Tests: no GPU needed. Run with `ctest --test-dir build --output-on-failure`.
enable_testing()
add_executable(heliora_tests
        tests/main.cpp
        tests/FixedStepTests.cpp
        tests/PhysicsWorldTests.cpp
        tests/PrimitiveMeshTests.cpp
        tests/SceneNodeTests.cpp
        tests/ScenePhysicsTests.cpp
        tests/SceneTests.cpp
)
target_link_libraries(heliora_tests PRIVATE heliora_core)
add_test(NAME heliora_tests COMMAND heliora_tests)

message(STATUS "Vulkan SDK: ${Vulkan_INCLUDE_DIRS}")

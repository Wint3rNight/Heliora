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
# Link-time optimization for Jolt, off by default: its link step runs several
# compiler processes at once, which this machine's RAM doesn't like. With
# Makefiles and --parallel 2 gcc keeps it to two. Jolt's objects then need LTO
# on our side as well to link.
option(HELIORA_JOLT_LTO "Build Jolt (and the game) with link-time optimization" OFF)
set(INTERPROCEDURAL_OPTIMIZATION ${HELIORA_JOLT_LTO})
if(HELIORA_JOLT_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()
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
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AnimationClip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AssetImport.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CharacterAnimator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CharacterMovement.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CharacterRoster.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CharacterFile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CharacterSettings.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Controls.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/EditHistory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/GizmoMath.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PhysicsWorld.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Picking.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PlayerController.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PlayerStart.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PoseBlend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Prefab.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PrimitiveMesh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Scene.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneNode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ScenePhysics.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneViewport.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SettingsFile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Skeleton.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Skinning.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Spawning.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ShadowCascades.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdPersonCamera.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Thumbnails.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/BreakGraph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Fracture.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Frustum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Rubble.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Stress.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Ballistics.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Bullets.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Weapon.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/WeaponFile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/HitZones.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PieceDamage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/EffectGeometry.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Particles.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Inertialize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/FootIk.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioMixer.cpp
)
list(REMOVE_ITEM ALL_SOURCES ${HELIORA_CORE_SOURCES})
add_library(heliora_core STATIC ${HELIORA_CORE_SOURCES})
# PhysicsWorld.cpp subclasses Jolt's debug renderer, and Jolt is built with
# -fno-rtti, so it has no typeinfo for that base class to link against.
set_source_files_properties(src/PhysicsWorld.cpp PROPERTIES
        COMPILE_OPTIONS -fno-rtti)
target_include_directories(heliora_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(heliora_core
        PUBLIC glm::glm Jolt
        PRIVATE spdlog::spdlog meshoptimizer
)
# Vulkan-style depth [0,1] for all glm projection matrices. PUBLIC so the core,
# the game and the tests all build glm's projections the same way.
target_compile_definitions(heliora_core PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

# Create the Executable
add_executable(${PROJECT_NAME} ${ALL_SOURCES})

# Include Paths - Using PUBLIC/PRIVATE correctly
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# miniaudio (step 1.5c), one header plus stb_vorbis, built in AudioDevice.cpp
# only; it opens the system's sound libraries at run time, hence dl.
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio)
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
set_source_files_properties(src/AudioDevice.cpp PROPERTIES COMPILE_OPTIONS "-w")

# Linking
target_link_libraries(${PROJECT_NAME} PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        spdlog::spdlog
        Threads::Threads
        heliora_core
)


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
        tests/AnimationClipTests.cpp
        tests/AssetImportTests.cpp
        tests/CharacterAnimatorTests.cpp
        tests/CharacterMovementTests.cpp
        tests/CharacterRosterTests.cpp
        tests/CharacterFileTests.cpp
        tests/CharacterSettingsTests.cpp
        tests/ControlsTests.cpp
        tests/EditHistoryTests.cpp
        tests/FixedStepTests.cpp
        tests/BreakGraphTests.cpp
        tests/FractureTests.cpp
        tests/FrustumTests.cpp
        tests/GizmoTests.cpp
        tests/PhysicsCharacterTests.cpp
        tests/PhysicsWorldTests.cpp
        tests/PickingTests.cpp
        tests/PlayerStartTests.cpp
        tests/PlayerControllerTests.cpp
        tests/PoseBlendTests.cpp
        tests/PrefabTests.cpp
        tests/PrimitiveMeshTests.cpp
        tests/SceneNodeTests.cpp
        tests/ScenePhysicsTests.cpp
        tests/SceneViewportTests.cpp
        tests/SceneTests.cpp
        tests/SettingsFileTests.cpp
        tests/SkeletonTests.cpp
        tests/SkinningTests.cpp
        tests/SpawningTests.cpp
        tests/ShadowCascadesTests.cpp
        tests/ThirdPersonCameraTests.cpp
        tests/ThumbnailsTests.cpp
        tests/RubbleTests.cpp
        tests/StressTests.cpp
        tests/BallisticsTests.cpp
        tests/BulletsTests.cpp
        tests/WeaponFileTests.cpp
        tests/WeaponTests.cpp
        tests/HitZonesTests.cpp
        tests/PieceDamageTests.cpp
        tests/EffectGeometryTests.cpp
        tests/ParticlesTests.cpp
        tests/InertializeTests.cpp
        tests/FootIkTests.cpp
        tests/AudioMixerTests.cpp
)
target_link_libraries(heliora_tests PRIVATE heliora_core)
# Tests that check the shipped data files read them from the source tree.
target_compile_definitions(heliora_tests PRIVATE HELIORA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME heliora_tests COMMAND heliora_tests)

message(STATUS "Vulkan SDK: ${Vulkan_INCLUDE_DIRS}")

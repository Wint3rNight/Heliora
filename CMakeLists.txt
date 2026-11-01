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

# Jolt Physics: the physics engine (the agreed exception to building our own
# systems). Jolt's options are plain variables set before
# FetchContent_MakeAvailable; its option() calls keep them (policy CMP0077).
# Jolt's sample and test programs only build when it is the top-level project.
set(ENABLE_INSTALL OFF)
# The GPU compute backends exist for Jolt's hair simulation; Heliora has none.
set(JPH_USE_VK OFF)
set(JPH_USE_DX12 OFF)
set(JPH_USE_MTL OFF)
set(JPH_USE_CPU_COMPUTE OFF)
# Our own CPU timers measure the physics step, and Jolt's ObjectStream file
# format is unused, so neither costs time or code.
set(PROFILER_IN_DEBUG_AND_RELEASE OFF)
set(ENABLE_OBJECT_STREAM OFF)
# Kept in Debug and Release for the F2 collision-shape view; a shipping build
# turns it off (it adds a few branches to Jolt's narrow phase).
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE ON)
# -Werror on code we don't own would break the build on each new gcc warning.
set(ENABLE_ALL_WARNINGS OFF)
# Debug symbols only where we debug: in Release they only cost build RAM.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(GENERATE_DEBUG_SYMBOLS ON)
else()
  set(GENERATE_DEBUG_SYMBOLS OFF)
endif()
# Link-time optimization is off until measured: its link step runs many
# compiler processes at once, which this machine's RAM does not tolerate.
set(INTERPROCEDURAL_OPTIMIZATION OFF)
# Note: Jolt adds -mavx2 -mfma (PUBLIC) to every target that links it, so the
# whole game requires a CPU with AVX2 and FMA.
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
        Jolt
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

message(STATUS "Vulkan SDK: ${Vulkan_INCLUDE_DIRS}")

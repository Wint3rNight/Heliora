cmake_minimum_required(VERSION 3.16)

project(VulkanRenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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

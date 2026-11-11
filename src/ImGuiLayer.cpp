#include "ImGuiLayer.h"

#include "Scene.h"
#include "ScenePhysics.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <span>
#include <stdexcept>

namespace {
constexpr const char *kFontPath = "Resources/Fonts/Inter-Regular.ttf";
constexpr float kFontSize = 14.0f;

// 0xRRGGBB to an ImGui colour.
ImVec4 hex(uint32_t rgb, float alpha = 1.0f) {
  return ImVec4(static_cast<float>((rgb >> 16) & 0xff) / 255.0f,
                static_cast<float>((rgb >> 8) & 0xff) / 255.0f,
                static_cast<float>(rgb & 0xff) / 255.0f, alpha);
}

// The palette. Red marks what you can change, purple marks where you are.
constexpr uint32_t kBase0 = 0x131016; // panels
constexpr uint32_t kBase1 = 0x1a161e; // focused title bars, popups, tabs
constexpr uint32_t kBase2 = 0x221d27; // input boxes, slider tracks
constexpr uint32_t kBase3 = 0x2b2531; // buttons, hovered inputs
constexpr uint32_t kBase4 = 0x352e3c; // pressed inputs, scrollbar
constexpr uint32_t kBorder = 0x2a2430;
constexpr uint32_t kText = 0xe6dee8;
constexpr uint32_t kTextDim = 0x857a8e;
constexpr uint32_t kRed = 0xb0243a;
constexpr uint32_t kRedBright = 0xc93a50;
constexpr uint32_t kPurple = 0x7b4b94;
constexpr uint32_t kPurpleBright = 0x9a63b8;

// The shapes come from passivestar's Godot Minimal theme: calm panels, 5 px
// corners, Inter. The colours are gothic: near-black with a purple tint.
void applyTheme() {
  ImGuiStyle &style = ImGui::GetStyle();
  ImGui::StyleColorsDark(&style); // fills any slot not set below

  style.WindowPadding = ImVec2(12.0f, 10.0f);
  style.FramePadding = ImVec2(8.0f, 4.0f);
  style.ItemSpacing = ImVec2(8.0f, 5.0f);
  style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
  style.IndentSpacing = 14.0f;
  style.ScrollbarSize = 10.0f;
  style.GrabMinSize = 10.0f;

  style.WindowBorderSize = 1.0f;
  style.ChildBorderSize = 1.0f;
  style.PopupBorderSize = 1.0f;
  style.FrameBorderSize = 0.0f;
  style.TabBorderSize = 0.0f;
  style.TabBarBorderSize = 1.0f;
  style.TabBarOverlineSize = 1.0f;
  style.SeparatorTextBorderSize = 1.0f;
  style.SeparatorTextPadding = ImVec2(0.0f, 4.0f);

  style.WindowRounding = 5.0f;
  style.ChildRounding = 5.0f;
  style.FrameRounding = 5.0f;
  style.PopupRounding = 5.0f;
  style.ScrollbarRounding = 5.0f;
  style.TabRounding = 5.0f;
  style.GrabRounding = 3.0f; // sits inside a 5 px frame, so a bit less

  style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
  style.WindowMenuButtonPosition = ImGuiDir_None;

  ImVec4 *c = style.Colors;
  c[ImGuiCol_Text] = hex(kText);
  c[ImGuiCol_TextDisabled] = hex(kTextDim);
  c[ImGuiCol_WindowBg] = hex(kBase0, 0.94f);
  c[ImGuiCol_ChildBg] = hex(0, 0.0f);
  c[ImGuiCol_PopupBg] = hex(kBase1, 0.98f);
  c[ImGuiCol_Border] = hex(kBorder);
  c[ImGuiCol_BorderShadow] = hex(0, 0.0f);
  c[ImGuiCol_FrameBg] = hex(kBase2);
  c[ImGuiCol_FrameBgHovered] = hex(kBase3);
  c[ImGuiCol_FrameBgActive] = hex(kBase4);
  c[ImGuiCol_TitleBg] = hex(kBase0);
  c[ImGuiCol_TitleBgActive] = hex(kBase1);
  c[ImGuiCol_TitleBgCollapsed] = hex(kBase0);
  c[ImGuiCol_MenuBarBg] = hex(kBase1);
  c[ImGuiCol_ScrollbarBg] = hex(0, 0.0f);
  c[ImGuiCol_ScrollbarGrab] = hex(kBase4);
  c[ImGuiCol_ScrollbarGrabHovered] = hex(0x463d4f);
  c[ImGuiCol_ScrollbarGrabActive] = hex(kPurple);
  c[ImGuiCol_CheckMark] = hex(kRedBright);
  c[ImGuiCol_SliderGrab] = hex(kRed);
  c[ImGuiCol_SliderGrabActive] = hex(kRedBright);
  c[ImGuiCol_Button] = hex(kBase3);
  c[ImGuiCol_ButtonHovered] = hex(kBase4);
  c[ImGuiCol_ButtonActive] = hex(kRed);
  c[ImGuiCol_Header] = hex(kPurple, 0.30f);
  c[ImGuiCol_HeaderHovered] = hex(kPurple, 0.45f);
  c[ImGuiCol_HeaderActive] = hex(kPurple, 0.60f);
  c[ImGuiCol_Separator] = hex(kBorder);
  c[ImGuiCol_SeparatorHovered] = hex(kPurple);
  c[ImGuiCol_SeparatorActive] = hex(kPurpleBright);
  c[ImGuiCol_ResizeGrip] = hex(0, 0.0f);
  c[ImGuiCol_ResizeGripHovered] = hex(kPurple, 0.60f);
  c[ImGuiCol_ResizeGripActive] = hex(kPurpleBright);
  c[ImGuiCol_InputTextCursor] = hex(kText);
  c[ImGuiCol_Tab] = hex(kBase1);
  c[ImGuiCol_TabHovered] = hex(kPurple, 0.45f);
  c[ImGuiCol_TabSelected] = hex(0x2e2238);
  c[ImGuiCol_TabSelectedOverline] = hex(kPurpleBright);
  c[ImGuiCol_TabDimmed] = hex(kBase0);
  c[ImGuiCol_TabDimmedSelected] = hex(kBase2);
  c[ImGuiCol_TabDimmedSelectedOverline] = hex(kPurple, 0.50f);
  c[ImGuiCol_DockingPreview] = hex(kPurple, 0.50f);
  c[ImGuiCol_DockingEmptyBg] = hex(kBase0);
  c[ImGuiCol_PlotLines] = hex(kPurpleBright);
  c[ImGuiCol_PlotLinesHovered] = hex(kRedBright);
  c[ImGuiCol_PlotHistogram] = hex(kPurple);
  c[ImGuiCol_PlotHistogramHovered] = hex(kRedBright);
  c[ImGuiCol_TableHeaderBg] = hex(kBase1);
  c[ImGuiCol_TableBorderStrong] = hex(kBorder);
  c[ImGuiCol_TableBorderLight] = hex(kBase2);
  c[ImGuiCol_TableRowBg] = hex(0, 0.0f);
  c[ImGuiCol_TableRowBgAlt] = hex(0xffffff, 0.03f);
  c[ImGuiCol_TextLink] = hex(kPurpleBright);
  c[ImGuiCol_TextSelectedBg] = hex(kPurple, 0.35f);
  c[ImGuiCol_TreeLines] = hex(kBorder);
  c[ImGuiCol_DragDropTarget] = hex(kRedBright);
  c[ImGuiCol_DragDropTargetBg] = hex(kRed, 0.15f);
  c[ImGuiCol_UnsavedMarker] = hex(kRedBright);
  c[ImGuiCol_NavCursor] = hex(kPurpleBright);
  c[ImGuiCol_NavWindowingHighlight] = hex(kPurpleBright, 0.70f);
  c[ImGuiCol_NavWindowingDimBg] = hex(0, 0.50f);
  c[ImGuiCol_ModalWindowDimBg] = hex(0, 0.55f);

  // ImGui asserts on a missing font file, so check first. The game runs from
  // build/, hence the second path.
  for (const std::string path : {std::string(kFontPath),
                                 "../" + std::string(kFontPath)}) {
    if (std::filesystem::exists(path)) {
      ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str(), kFontSize);
      style.FontSizeBase = kFontSize;
      return;
    }
  }
  spdlog::warn("[ui] {} not found, using ImGui's built-in font", kFontPath);
}

const char *presentModeName(VkPresentModeKHR mode) {
  switch (mode) {
  case VK_PRESENT_MODE_IMMEDIATE_KHR:
    return "IMMEDIATE";
  case VK_PRESENT_MODE_MAILBOX_KHR:
    return "MAILBOX";
  case VK_PRESENT_MODE_FIFO_KHR:
    return "FIFO";
  case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
    return "FIFO_RELAXED";
  default:
    return "unknown";
  }
}
} // namespace

void ImGuiLayer::init(GLFWwindow *window, VulkanDevice &device,
                      VkRenderPass renderPass, uint32_t imageCount) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  applyTheme();

  ImGui_ImplGlfw_InitForVulkan(window, true);

  QueueFamilyIndices qi = device.getQueueFamilies();
  ImGui_ImplVulkan_InitInfo info = {};
  info.ApiVersion = VK_API_VERSION_1_3;
  info.Instance = device.getInstance();
  info.PhysicalDevice = device.getPhysicalDevice();
  info.Device = device.getLogicalDevice();
  info.QueueFamily = static_cast<uint32_t>(qi.graphicsFamily);
  info.Queue = device.getGraphicsQueue();
  info.DescriptorPoolSize = 16;
  info.MinImageCount = 2;
  info.ImageCount = imageCount;
  info.PipelineInfoMain.RenderPass = renderPass;
  info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&info);
  initialized = true;
}

void ImGuiLayer::cleanup(VkDevice device) {
  cleanupFramebuffers(device);
  if (!initialized)
    return;

  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  initialized = false;
}

void ImGuiLayer::createFramebuffers(
    VkDevice device, VkRenderPass renderPass, VkExtent2D extent,
    const std::vector<SwapChainImage> &images) {
  cleanupFramebuffers(device);
  framebuffers.resize(images.size(), VK_NULL_HANDLE);
  for (size_t i = 0; i < images.size(); ++i) {
    VkImageView view = images[i].imageView.get();
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &view;
    fbci.width = extent.width;
    fbci.height = extent.height;
    fbci.layers = 1;
    if (vkCreateFramebuffer(device, &fbci, nullptr, &framebuffers[i]) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create ImGui framebuffer");
  }
}

void ImGuiLayer::cleanupFramebuffers(VkDevice device) {
  for (VkFramebuffer fb : framebuffers) {
    if (fb != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device, fb, nullptr);
  }
  framebuffers.clear();
}

bool ImGuiLayer::wantsMouse() const {
  return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiLayer::wantsTextInput() const {
  return ImGui::GetCurrentContext() && ImGui::GetIO().WantTextInput;
}

void ImGuiLayer::buildUi(DebugUiContext &ui) {
  // Frame-time history accumulates even while hidden so the graph shows
  // truthful recent history after un-hiding, not a stale pre-hide splice.
  frameTimeGraphData[frameTimeGraphOffset] =
      static_cast<float>(ui.metrics.getLastFrameTimeMs());
  frameTimeGraphOffset = (frameTimeGraphOffset + 1) % 128;

  if (!visible)
    return; // F1 screenshot mode: no windows -> empty draw list this frame

  // One sidebar, full height, with the tabs as two rows of pills on top.
  constexpr float kSidebarWidth = 380.0f;
  constexpr float kLabelWidth = 150.0f; // room for the longest slider label
  const ImVec2 display = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
  ImGui::SetNextWindowSize(
      ImVec2(kSidebarWidth, std::max(200.0f, display.y - 20.0f)),
      ImGuiCond_Always);
  ImGui::Begin("##sidebar", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoScrollbar |
                   ImGuiWindowFlags_NoScrollWithMouse);
  drawTabPills();
  ImGui::Separator();
  // The tab's contents scroll on their own, so the pills stay put.
  ImGui::BeginChild("##tab");
  ImGui::PushItemWidth(-kLabelWidth);
  switch (activeTab) {
  case SidebarTab::Scene:
    drawSceneTab(ui);
    break;
  case SidebarTab::Render:
    drawRenderTab(ui);
    break;
  case SidebarTab::Lighting:
    drawLightingTab(ui);
    break;
  case SidebarTab::Debug:
    drawDebugTab(ui);
    break;
  case SidebarTab::Physics:
    drawPhysicsTab(ui);
    break;
  case SidebarTab::Performance:
    drawPerformanceTab(ui);
    break;
  case SidebarTab::Camera:
    drawCameraTab(ui);
    break;
  }
  ImGui::PopItemWidth();
  ImGui::EndChild();
  ImGui::End();
  drawToolbar(ui);
  drawInspector(ui);
}

namespace {

// A key that's either set on the entry or left to its material and the
// defaults. The box sets it; unticked, the field shows the value in use,
// greyed out.
bool optionalFloat(const char *label, std::optional<float> &value,
                   float inUse, float speed, float lo, float hi,
                   const char *format) {
  bool changed = false;
  ImGui::PushID(label);
  bool on = value.has_value();
  if (ImGui::Checkbox("##set", &on)) {
    value = on ? std::optional<float>(inUse) : std::nullopt;
    changed = true;
  }
  ImGui::SameLine();
  float shown = value.value_or(inUse);
  ImGui::BeginDisabled(!on);
  if (ImGui::DragFloat(label, &shown, speed, lo, hi, format,
                       ImGuiSliderFlags_AlwaysClamp) &&
      on) {
    value = shown;
    changed = true;
  }
  ImGui::EndDisabled();
  ImGui::PopID();
  return changed;
}

} // namespace

// The selected entry's settings, on the right, in edit mode. Edits go to a
// copy first and are only kept if the parser would accept them, so a saved
// scene always loads.
void ImGuiLayer::drawInspector(DebugUiContext &ui) {
  auto flushRebuild = [&] {
    if (inspectorRebuildPending && !ImGui::IsAnyItemActive()) {
      inspectorRebuildPending = false;
      if (ui.onEntryChanged)
        ui.onEntryChanged(inspectorPendingRef);
    }
  };
  if (!ui.scene || !ui.selection) {
    flushRebuild();
    return;
  }
  const SceneEntryRef ref = *ui.selection;
  SceneDescription &scene = *ui.scene;
  const bool isGrid = ref.kind == SceneEntryRef::Kind::Grid;
  const size_t count = isGrid ? scene.grids.size() : scene.models.size();
  if (ref.index < 0 || ref.index >= static_cast<int>(count)) {
    flushRebuild();
    return;
  }
  if (ref != inspectorShownRef) {
    inspectorShownRef = ref;
    inspectorProblem.clear();
  }
  SceneGridEntry *grid = isGrid ? &scene.grids[ref.index] : nullptr;
  SceneModelEntry &entry = isGrid ? grid->model : scene.models[ref.index];
  SceneModelEntry edited = entry;
  SceneGridEntry editedGrid = isGrid ? *grid : SceneGridEntry{};
  bool moved = false;   // placement only: applied at once
  bool changed = false; // needs the entry rebuilt
  bool split = false;
  bool deleting = false;

  const ImVec2 display = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(display.x - 10.0f, 10.0f), ImGuiCond_Always,
                          ImVec2(1.0f, 0.0f));
  // 0 height fits the contents.
  ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_Always);
  ImGui::Begin("##inspector", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoSavedSettings);
  constexpr ImGuiSliderFlags kClamp = ImGuiSliderFlags_AlwaysClamp;
  static constexpr const char *kShapes[] = {"box", "sphere", "capsule",
                                            "plane"};
  const std::string what =
      edited.shape ? kShapes[static_cast<int>(*edited.shape)]
                   : std::filesystem::path(edited.path).filename().string();
  if (isGrid) {
    const glm::ivec3 n = grid->count;
    ImGui::Text("Grid %d: %s, %d copies", ref.index, what.c_str(),
                n.x * n.y * n.z);
  } else {
    ImGui::Text("Model %d: %s", ref.index, what.c_str());
  }
  ImGui::PushItemWidth(-110.0f);

  ImGui::SeparatorText("Transform");
  moved |= ImGui::DragFloat3("Position", &edited.transform.position.x, 0.05f,
                             0.0f, 0.0f, "%.2f");
  moved |= ImGui::DragFloat3("Rotation", &edited.transform.rotationDeg.x,
                             0.5f, 0.0f, 0.0f, "%.1f\u00b0");
  // Scale shows at once, but the body's size only changes in the rebuild.
  if (ImGui::DragFloat("Scale", &edited.transform.scale, 0.01f, 0.01f, 100.0f,
                       "%.2f", kClamp)) {
    moved = true;
    changed = true;
  }

  if (edited.shape) {
    ImGui::SeparatorText("Shape");
    int shape = static_cast<int>(*edited.shape);
    if (ImGui::Combo("Shape", &shape, kShapes, 4)) {
      edited.shape = static_cast<PrimitiveShape>(shape);
      changed = true;
    }
    switch (*edited.shape) {
    case PrimitiveShape::Box:
      changed |= ImGui::DragFloat3("Size", &edited.size.x, 0.01f, 0.01f,
                                   1000.0f, "%.2f", kClamp);
      break;
    case PrimitiveShape::Plane: {
      glm::vec2 size(edited.size.x, edited.size.z);
      if (ImGui::DragFloat2("Size", &size.x, 0.05f, 0.01f, 10000.0f, "%.2f",
                            kClamp)) {
        edited.size.x = size.x;
        edited.size.z = size.y;
        changed = true;
      }
      break;
    }
    case PrimitiveShape::Capsule:
      changed |= ImGui::DragFloat("Height", &edited.height, 0.01f, 0.02f,
                                  1000.0f, "%.2f", kClamp);
      [[fallthrough]];
    case PrimitiveShape::Sphere:
      changed |= ImGui::DragFloat("Radius", &edited.radius, 0.01f, 0.01f,
                                  1000.0f, "%.2f", kClamp);
      break;
    }
    changed |= ImGui::ColorEdit3("Colour", &edited.color.x);
  } else {
    ImGui::SeparatorText("Model file");
    ImGui::TextWrapped("%s", edited.path.c_str());
  }

  ImGui::SeparatorText("Physics");
  static constexpr const char *kBodies[] = {"none", "static", "dynamic",
                                            "kinematic"};
  int body = edited.body ? 1 + static_cast<int>(*edited.body) : 0;
  if (ImGui::Combo("Body", &body, kBodies, 4)) {
    if (body == 0) {
      // These only mean something with a body.
      edited.body.reset();
      edited.mass.reset();
      edited.density.reset();
      edited.material.reset();
      edited.friction.reset();
      edited.restitution.reset();
      edited.collider.reset();
    } else {
      edited.body = static_cast<PhysicsMotion>(body - 1);
    }
    changed = true;
  }
  if (edited.body) {
    const std::span<const PhysicsMaterial> materials = physicsMaterials();
    const char *materialName =
        edited.material ? edited.material->c_str() : "none";
    if (ImGui::BeginCombo("Material", materialName)) {
      if (ImGui::Selectable("none", !edited.material)) {
        edited.material.reset();
        changed = true;
      }
      for (const PhysicsMaterial &m : materials)
        if (ImGui::Selectable(m.name, edited.material == m.name)) {
          edited.material = m.name;
          changed = true;
        }
      ImGui::EndCombo();
    }
    // What the body gets from its material and the defaults right now.
    const PhysicsBodyDesc inUse = bodyDescFor(
        edited, nullptr, ui.physics ? ui.physics->defaultFriction : 0.2f);
    // The weight Jolt gave it; a body that doesn't exist yet (just given
    // one, not rebuilt) starts from 1 kg, since 0 isn't allowed.
    const float massInUse = ui.selectionMass > 0.0f ? ui.selectionMass
                            : inUse.mass > 0.0f     ? inUse.mass
                                                    : 1.0f;
    changed |= optionalFloat("Mass", edited.mass, massInUse, 0.5f, 0.01f,
                             1.0e6f, "%.1f kg");
    changed |= optionalFloat("Density", edited.density, inUse.density, 5.0f,
                             1.0f, 30000.0f, "%.0f kg/m\u00b3");
    changed |= optionalFloat("Friction", edited.friction, inUse.friction,
                             0.01f, 0.0f, 10.0f, "%.2f");
    changed |= optionalFloat("Restitution", edited.restitution,
                             inUse.restitution, 0.01f, 0.0f, 1.0f, "%.2f");
    if (!edited.shape) {
      static constexpr const char *kColliders[] = {"auto", "hull", "mesh",
                                                   "box"};
      int collider = static_cast<int>(
          edited.collider.value_or(SceneCollider::Auto));
      if (ImGui::Combo("Collider", &collider, kColliders, 4)) {
        // auto is the default, so it's left out of the file.
        if (collider == 0)
          edited.collider.reset();
        else
          edited.collider = static_cast<SceneCollider>(collider);
        changed = true;
      }
    }
    if (ui.selectionMass > 0.0f)
      ImGui::TextDisabled(isGrid ? "Weighs %.1f kg each" : "Weighs %.1f kg",
                          ui.selectionMass);
  }

  if (isGrid) {
    ImGui::SeparatorText("Grid");
    changed |= ImGui::DragInt3("Count", &editedGrid.count.x, 0.1f, 1, 64,
                               "%d", kClamp);
    moved |= ImGui::DragFloat3("Spacing", &editedGrid.spacing.x, 0.01f, 0.01f,
                               100.0f, "%.2f", kClamp);
    moved |= ImGui::DragFloat("Layout rotation", &editedGrid.layoutRotation,
                              0.5f, -360.0f, 360.0f, "%.1f\u00b0", kClamp);
    split = ImGui::Button("Split into objects");
  }
  ImGui::PopItemWidth();

  if (moved || changed) {
    std::optional<std::string> problem;
    if (isGrid) {
      editedGrid.model = edited;
      problem = problemWith(editedGrid);
      const glm::ivec3 n = editedGrid.count;
      const uint32_t maxBodies =
          ui.physics ? ui.physics->config.maxBodies : 16384u;
      if (!problem && edited.body &&
          uint32_t(n.x) * uint32_t(n.y) * uint32_t(n.z) > maxBodies)
        problem = std::format("more copies than the physics world holds ({})",
                              maxBodies);
    } else {
      problem = problemWith(edited);
    }
    if (problem) {
      inspectorProblem = *problem;
    } else {
      inspectorProblem.clear();
      if (isGrid)
        *grid = editedGrid;
      else
        entry = edited;
      if (moved && ui.onEntryMoved)
        ui.onEntryMoved(ref);
      if (changed) {
        inspectorRebuildPending = true;
        inspectorPendingRef = ref;
      }
    }
  }
  if (!inspectorProblem.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, hex(kRedBright));
    ImGui::TextWrapped("%s", inspectorProblem.c_str());
    ImGui::PopStyleColor();
  }
  ImGui::Separator();
  deleting = ImGui::Button("Delete");
  ImGui::End();

  if (split || deleting) {
    // The entry is going, so a rebuild of it would be wasted.
    inspectorRebuildPending = false;
    if (split && ui.onGridSplit)
      ui.onGridSplit(ref);
    else if (deleting && ui.onEntryDeleted)
      ui.onEntryDeleted(ref);
    return;
  }
  flushRebuild();
}

// A small bar at the top centre: Play/Stop and which mode the game is in.
void ImGuiLayer::drawToolbar(DebugUiContext &ui) {
  const ImVec2 display = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(display.x * 0.5f, 10.0f), ImGuiCond_Always,
                          ImVec2(0.5f, 0.0f));
  ImGui::Begin("##toolbar", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoScrollbar |
                   ImGuiWindowFlags_AlwaysAutoResize |
                   ImGuiWindowFlags_NoSavedSettings);
  if (ImGui::Button(ui.playing ? "Stop" : "Play") && ui.onTogglePlay)
    ui.onTogglePlay();
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled(ui.playing ? "Playing (F5 stops)" : "Editing (F5 plays)");
  ImGui::End();
}

// Pills wrap onto a second row when the first is full. The selected one is
// filled purple, the rest are text until hovered.
void ImGuiLayer::drawTabPills() {
  static constexpr std::pair<SidebarTab, const char *> kTabs[] = {
      {SidebarTab::Scene, "Scene"},
      {SidebarTab::Render, "Render"},
      {SidebarTab::Lighting, "Lighting"},
      {SidebarTab::Debug, "Debug"},
      {SidebarTab::Physics, "Physics"},
      {SidebarTab::Performance, "Performance"},
      {SidebarTab::Camera, "Camera"},
  };
  const ImGuiStyle &style = ImGui::GetStyle();
  const float rowLimit = ImGui::GetContentRegionAvail().x;
  float rowWidth = 0.0f;
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,
                      ImGui::GetFrameHeight() * 0.5f);
  for (const auto &[tab, label] : kTabs) {
    const float width =
        ImGui::CalcTextSize(label).x + style.FramePadding.x * 2.0f;
    if (rowWidth > 0.0f &&
        rowWidth + style.ItemSpacing.x + width <= rowLimit) {
      ImGui::SameLine();
      rowWidth += style.ItemSpacing.x + width;
    } else {
      rowWidth = width;
    }
    const bool selected = tab == activeTab;
    ImGui::PushStyleColor(ImGuiCol_Text, hex(selected ? kText : kTextDim));
    ImGui::PushStyleColor(ImGuiCol_Button,
                          selected ? hex(kPurple) : hex(0, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          selected ? hex(kPurple) : hex(kBase3));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, hex(kPurpleBright));
    if (ImGui::Button(label))
      activeTab = tab;
    ImGui::PopStyleColor(4);
  }
  ImGui::PopStyleVar();
}

void ImGuiLayer::drawSceneTab(DebugUiContext &ui) {
  if (ui.sceneNames && !ui.sceneNames->empty()) {
    const std::vector<std::string> &names = *ui.sceneNames;
    int count = static_cast<int>(names.size());
    int selected = ui.activeSceneIndex;
    const char *preview = (selected >= 0 && selected < count)
                              ? names[selected].c_str()
                              : "<select scene>";
    if (ImGui::BeginCombo("Scene", preview)) {
      for (int i = 0; i < count; i++) {
        bool isSelected = (i == selected);
        if (ImGui::Selectable(names[i].c_str(), isSelected) && !isSelected &&
            ui.onSceneSelected)
          ui.onSceneSelected(i);
        if (isSelected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
  }
  if (ui.sceneError && !ui.sceneError->empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, hex(kRedBright));
    ImGui::TextWrapped("%s", ui.sceneError->c_str());
    ImGui::PopStyleColor();
  }
}

// Save writes one section of the settings file; Reset puts that section
// back to the code defaults without saving.
void ImGuiLayer::drawSettingsButtons(DebugUiContext &ui, const char *section) {
  if (ImGui::Button("Save settings") && ui.onSaveSettings)
    ui.onSaveSettings(section);
  ImGui::SameLine();
  if (ImGui::Button("Reset to defaults") && ui.onResetSettings)
    ui.onResetSettings(section);
  if (ui.settingsStatus && !ui.settingsStatus->empty())
    ImGui::TextDisabled("%s", ui.settingsStatus->c_str());
}

void ImGuiLayer::drawRenderTab(DebugUiContext &ui) {
  drawSettingsButtons(ui, "render");
  ImGui::BeginDisabled();
  ImGui::TextWrapped("Also saves FOV and draw distance (Camera), normal "
                     "strength (Debug), and LOD and point-shadow far "
                     "(Lighting).");
  ImGui::EndDisabled();
  ImGui::Separator();
  if (ImGui::SliderFloat("AA variance", &ui.specAAVariance, 0.0f, 2.0f,
                         "%.3f"))
    ui.sceneUbo.qualityToggles.z = ui.specAAVariance;
  if (ImGui::SliderFloat("AA threshold", &ui.specAAThreshold, 0.0f, 1.0f,
                         "%.3f"))
    ui.sceneUbo.qualityToggles.w = ui.specAAThreshold;
  if (ImGui::SliderFloat("IBL roughness floor", &ui.iblRoughnessFloor, 0.0f,
                         1.0f, "%.3f"))
    ui.sceneUbo.qualityToggles.x = ui.iblRoughnessFloor;
  ImGui::SliderFloat("Min surface roughness", &ui.minSurfaceRoughness, 0.0f,
                     1.0f, "%.2f");
  ImGui::SliderFloat("Sky occlusion floor", &ui.skyOcclusionFloor, 0.0f,
                     1.0f, "%.2f");
  ImGui::SliderFloat("CSM far", &ui.csmFar, 100.0f, 5000.0f, "%.0f");
  ImGui::Checkbox("Shadow front-face cull", &ui.shadowFrontFaceCull);
  ImGui::Checkbox("Cull shadow casters", &ui.cullShadowCasters);
  ImGui::SliderFloat("IBL / sky intensity", &ui.iblIntensity, 0.0f, 2.0f,
                     "%.2f");
  ImGui::Checkbox("Auto-exposure", &ui.autoExposureEnabled);
  ImGui::SameLine();
  ImGui::Text("(%.2fx)", ui.autoExposureAdaptedValue);
  ImGui::SliderFloat("Exposure (EV)", &ui.exposureEV, -3.0f, 3.0f, "%+.2f");
  ImGui::SliderFloat("SSGI intensity", &ui.ssgiIntensity, 0.0f, 2.0f,
                     "%.2f");
  ImGui::SliderInt("SSGI samples", &ui.ssgiSamples, 4, 12);
  ImGui::SeparatorText("Bloom");
  if (ImGui::Checkbox("Bloom", &ui.enableBloom) &&
      ui.onTaaHistoryInvalidated)
    ui.onTaaHistoryInvalidated();
  if (ui.enableBloom) {
    float bloomSensitivity = (4.0f - ui.bloomThreshold) / (4.0f - 0.2f);
    if (ImGui::SliderFloat("Bloom sensitivity", &bloomSensitivity, 0.0f, 1.0f,
                           "%.2f")) {
      ui.bloomThreshold = glm::mix(4.0f, 0.2f, bloomSensitivity);
    }
    ImGui::SliderFloat("Bloom intensity", &ui.bloomIntensity, 0.0f, 1.0f,
                       "%.3f");
    ImGui::SliderFloat("Bloom radius", &ui.bloomRadius, 0.5f, 4.0f, "%.2f");
  }
  ImGui::SeparatorText("TAA / SSR");
  ImGui::Checkbox("SSR reflections", &ui.ssrEnabled);
  ImGui::Checkbox("TAA", &ui.taaEnabled);
  ImGui::Checkbox("Responsive TAA", &ui.responsiveTaa);
  ImGui::SeparatorText("Edge / Alpha / Shadow");
  ImGui::SliderFloat("Sharpness", &ui.sharpness, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("Edge AA", &ui.edgeAA, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("Alpha dither", &ui.alphaDither, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("Shadow softness", &ui.shadowSoftness, 0.5f, 2.5f,
                     "%.2f");
  ImGui::SliderFloat("Contact grounding", &ui.contactGrounding, 0.0f, 1.0f,
                     "%.2f");
  ImGui::SeparatorText("Day / night");
  ImGui::Checkbox("Day/night cycle", &ui.dayNightEnable);
  ImGui::SliderFloat("Sim hour", &ui.dayNightHour, 0.0f, 24.0f, "%.2f h");
  ImGui::SliderFloat("Speed (sim-h / real-s)", &ui.dayNightSpeed, 0.1f,
                     600.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

  // Below here nothing is saved: these start at the code defaults every
  // launch (see VulkanRenderer::loadSettings).
  ImGui::SeparatorText("Lighting isolation (not saved)");
  ImGui::Checkbox("Sun direct", &ui.enableSunDirect);
  ImGui::Checkbox("Point lights", &ui.enablePointLights);
  ImGui::Checkbox("Spot lights", &ui.enableSpotLights);
  ImGui::Checkbox("IBL ambient", &ui.enableIblAmbient);
  ImGui::Checkbox("SSGI bounce", &ui.enableSsgiBounce);
  ImGui::SeparatorText("Diagnostics (not saved)");
  ImGui::Checkbox("GPU-driven G-buffer", &ui.gpuDrivenEnabled);
  ImGui::Checkbox("HZB occlusion cull", &ui.hzbCullingEnabled);
  ImGui::SliderInt("GPU min candidates", &ui.gpuDrivenMinCandidates, 0, 2048);
  ImGui::Text("GPU candidates: %u | submitted: %u",
              ui.gpuDrivenCandidateCount, ui.gpuDrivenMeshCount);
  ImGui::Text("GPU path: %s",
              ui.gpuDrivenLastFrameUsed ? "indirect" : "direct fallback");
  ImGui::Checkbox("Threaded CPU G-buffer", &ui.threadedGBufferEnabled);
  ImGui::SameLine();
  ImGui::Text("%u worker(s)", ui.threadedGBufferWorkers);
  if (ImGui::Checkbox("Uncapped present", &ui.preferMailboxPresent) &&
      ui.onPresentModeChanged) {
    ui.onPresentModeChanged(ui.preferMailboxPresent);
  }
  ImGui::SameLine();
  ImGui::Text("actual: %s", presentModeName(ui.activePresentMode));
}

void ImGuiLayer::drawLightingTab(DebugUiContext &ui) {
  if (ImGui::CollapsingHeader("Directional Light",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    // The cycle drives the sun while it runs, so edits here wouldn't show.
    ImGui::BeginDisabled(ui.dayNightEnable);
    glm::vec3 dir = ui.sun.direction;
    if (ImGui::DragFloat3("Dir##sun", &dir.x, 0.01f, -1.0f, 1.0f) &&
        glm::length(dir) > 0.001f)
      ui.sun.direction = glm::normalize(dir);
    ImGui::ColorEdit3("Color##sun", &ui.sun.color.x);
    ImGui::SliderFloat("Intensity##sun", &ui.sun.intensity, 0.0f, 5.0f);
    ImGui::EndDisabled();
  }
  if (ImGui::CollapsingHeader("Fog", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("Density##fog", &ui.fog.density, 0.0f, 0.02f, "%.4f");
    ImGui::SliderFloat("Falloff##fog", &ui.fog.falloff, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Max##fog", &ui.fog.max, 0.0f, 1.0f, "%.2f");
  }
  int pointCount = ui.sceneUbo.lightCounts.x;
  for (int i = 0; i < pointCount; i++) {
    char label[32];
    snprintf(label, sizeof(label), "Point Light %d", i);
    if (ImGui::CollapsingHeader(label)) {
      glm::vec3 pos = glm::vec3(ui.sceneUbo.pointLights[i].position);
      char id[24];
      snprintf(id, sizeof(id), "Pos##pt%d", i);
      if (ImGui::DragFloat3(id, &pos.x, 0.1f))
        ui.sceneUbo.pointLights[i].position = glm::vec4(pos, 1.0f);
      glm::vec3 col = glm::vec3(ui.sceneUbo.pointLights[i].colorIntensity);
      snprintf(id, sizeof(id), "Color##pt%d", i);
      if (ImGui::ColorEdit3(id, &col.x))
        ui.sceneUbo.pointLights[i].colorIntensity =
            glm::vec4(col, ui.sceneUbo.pointLights[i].colorIntensity.a);
      float intens = ui.sceneUbo.pointLights[i].colorIntensity.a;
      snprintf(id, sizeof(id), "Intensity##pt%d", i);
      if (ImGui::SliderFloat(id, &intens, 0.0f, 10.0f))
        ui.sceneUbo.pointLights[i].colorIntensity.a = intens;
    }
  }
  if (ImGui::CollapsingHeader("Post & Perf")) {
    ImGui::SliderFloat("LOD near", &ui.lodNear, 1.0f, 60.0f);
    ImGui::SliderFloat("LOD far", &ui.lodFar, ui.lodNear + 1.0f, 200.0f);
    if (ImGui::SliderFloat("Pt shadow far", &ui.pointShadowFar, 5.0f,
                           100.0f)) {
      ui.sceneUbo.shadowParams.x = ui.pointShadowFar;
      if (ui.onPointShadowChanged)
        ui.onPointShadowChanged();
    }
  }
}

void ImGuiLayer::drawDebugTab(DebugUiContext &ui) {
  const char *debugModes[] = {"None",          "Albedo",      "Normals",
                              "Metallic",      "Roughness",   "Depth",
                              "Shadow vis",    "SSAO factor", "Direct only",
                              "Indirect only", "Direct (no shadow)",
                              "SSGI bounce",   "SSGI raw",    "Bloom only"};
  if (ImGui::Combo("G-Buffer", &ui.debugMode, debugModes, 14))
    ui.sceneUbo.debugMode = ui.debugMode;
  ImGui::Checkbox("Use geometric normal only", &ui.useGeomNormalOnly);
  ImGui::BeginDisabled(ui.useGeomNormalOnly);
  ImGui::SliderFloat("Normal strength", &ui.normalStrength, 0.0f, 1.5f,
                     "%.2f");
  ImGui::EndDisabled();
  const MaterialProbeResult &probe = ui.materialProbe;
  if (probe.aoClothDepthValid.w > 0.5f) {
    const float normalLen =
        std::sqrt(probe.normalRoughness.x * probe.normalRoughness.x +
                  probe.normalRoughness.y * probe.normalRoughness.y +
                  probe.normalRoughness.z * probe.normalRoughness.z);
    ImGui::SeparatorText("Material Probe");
    ImGui::Text("Alb %.2f %.2f %.2f | M %.2f", probe.albedoMetallic.x,
                probe.albedoMetallic.y, probe.albedoMetallic.z,
                probe.albedoMetallic.w);
    ImGui::Text("Rough %.2f | AO %.2f | Cloth %.0f",
                probe.normalRoughness.w, probe.aoClothDepthValid.x,
                probe.aoClothDepthValid.y);
    ImGui::Text("N len %.2f | var %.3f | dev %.3f", normalLen,
                probe.geomVarianceFinal.y, probe.geomVarianceFinal.x);
  } else {
    ImGui::TextDisabled("Material Probe: no surface");
  }
}

void ImGuiLayer::drawPhysicsTab(DebugUiContext &ui) {
  if (!ui.physics) {
    ImGui::TextDisabled("No physics world");
    return;
  }
  PhysicsPanel &physics = *ui.physics;
  ImGui::Text("Bodies: %u | awake: %u", physics.bodies,
              physics.awakeBodies);
  ImGui::Text("Step time: %.3f ms",
              ui.metrics.getCpuPhaseTimeMs(
                  PerformanceMetrics::CpuPhase::Physics));
  if (physics.droppedContactSteps > 0)
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f),
                       "Dropped contacts in %u steps",
                       physics.droppedContactSteps);
  // Pause and Step are for play mode; in edit mode physics doesn't run.
  ImGui::BeginDisabled(!ui.playing);
  ImGui::Checkbox("Pause", &physics.paused);
  ImGui::SameLine();
  ImGui::BeginDisabled(!physics.paused);
  if (ImGui::Button("Step"))
    physics.stepRequested = true;
  ImGui::EndDisabled();
  ImGui::EndDisabled();
  if (!ui.playing)
    ImGui::TextDisabled("Editing: physics waits for Play (F5)");
  ImGui::Checkbox("Wireframes (F2)", &physics.showWireframes);
  ImGui::Text("F throws a ball (%u thrown)", physics.thrownBalls);

  ImGui::SeparatorText("Settings");
  drawSettingsButtons(ui, "physics");
  // AlwaysClamp holds typed-in values (Ctrl+click) to the range too.
  constexpr ImGuiSliderFlags kClamp = ImGuiSliderFlags_AlwaysClamp;
  PhysicsWorld::Config &config = physics.config;
  bool changed = false;
  changed |= ImGui::DragFloat3("Gravity", &config.gravity.x, 0.05f, -100.0f,
                               100.0f, "%.2f", kClamp);
  changed |= ImGui::SliderInt("Steps per second", &physics.stepsPerSecond, 10,
                              480, "%d", kClamp);
  changed |= ImGui::SliderInt("Max catch-up steps", &physics.maxCatchUpSteps,
                              1, 16, "%d", kClamp);
  changed |= ImGui::SliderInt("Velocity passes", &config.velocitySteps, 2, 64,
                              "%d", kClamp);
  changed |= ImGui::SliderInt("Position passes", &config.positionSteps, 1, 64,
                              "%d", kClamp);
  changed |= ImGui::SliderFloat("Sleep delay", &config.sleepDelay, 0.0f, 5.0f,
                                "%.2f s", kClamp);
  changed |= ImGui::SliderFloat("Sleep speed", &config.sleepSpeed, 0.0f, 0.5f,
                                "%.3f m/s", kClamp);
  changed |= ImGui::SliderFloat("Penetration slop", &config.penetrationSlop,
                                0.0f, 0.1f, "%.3f m", kClamp);
  if (changed)
    physics.settingsChanged = true;

  ImGui::SeparatorText("Applies on the next scene load");
  ImGui::SliderFloat("Default friction", &physics.defaultFriction, 0.0f, 2.0f,
                     "%.2f", kClamp);
  const uint32_t threadsMin = 0, threadsMax = 64;
  ImGui::SliderScalar("Worker threads", ImGuiDataType_U32,
                      &config.workerThreads, &threadsMin, &threadsMax, "%u",
                      kClamp);
  const uint32_t limitMin = 64, bodiesMax = 1u << 20, pairsMax = 1u << 22,
                 contactsMax = 1u << 17;
  ImGui::DragScalar("Max bodies", ImGuiDataType_U32, &config.maxBodies,
                    64.0f, &limitMin, &bodiesMax, "%u", kClamp);
  ImGui::DragScalar("Max body pairs", ImGuiDataType_U32, &config.maxBodyPairs,
                    256.0f, &limitMin, &pairsMax, "%u", kClamp);
  ImGui::DragScalar("Max contacts", ImGuiDataType_U32,
                    &config.maxContactConstraints, 128.0f, &limitMin,
                    &contactsMax, "%u", kClamp);
}

void ImGuiLayer::drawPerformanceTab(DebugUiContext &ui) {
  ImGui::Text("CPU submit FPS: %.1f | CPU: %.2f ms | Avg: %.2f ms",
              ui.metrics.getAverageFps(), ui.metrics.getLastFrameTimeMs(),
              ui.metrics.getAverageFrameTimeMs());
  ImGui::PlotLines("##ft", frameTimeGraphData, 128, frameTimeGraphOffset,
                   "Frame time (ms)", 0.0f, 50.0f, ImVec2(-FLT_MIN, 60));
  ImGui::Text("Shadow: %.2f | Pt: %.2f ms",
              ui.metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Shadow),
              ui.metrics.getPassGpuTimeMs(
                  PerformanceMetrics::GpuPass::PointShadow));
  ImGui::Text("GBuffer: %.2f | SSGI: %.2f ms",
              ui.metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::GBuffer),
              ui.metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::SSGI));
  ImGui::Text("Lit: %.2f | Bloom: %.2f ms",
              ui.metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Lit),
              ui.metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::Bloom));
  ImGui::Text("AutoExp: %.2f | Comp: %.2f ms",
              ui.metrics.getPassGpuTimeMs(
                  PerformanceMetrics::GpuPass::AutoExposure),
              ui.metrics.getPassGpuTimeMs(
                  PerformanceMetrics::GpuPass::Composite));
  ImGui::Text("ImGui: %.2f ms",
              ui.metrics.getPassGpuTimeMs(PerformanceMetrics::GpuPass::ImGui));
  ImGui::Text("GPU total: %.2f | avg %.2f ms",
              ui.metrics.getLastGpuTimeMs(), ui.metrics.getAverageGpuTimeMs());
  ImGui::Text("CPU wait/acq/pres: %.2f | %.2f | %.2f ms",
              ui.metrics.getCpuPhaseTimeMs(
                  PerformanceMetrics::CpuPhase::WaitFence),
              ui.metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Acquire),
              ui.metrics.getCpuPhaseTimeMs(
                  PerformanceMetrics::CpuPhase::Present));
  ImGui::Text("CPU phys/upd/rec/sub: %.2f | %.2f | %.2f | %.2f ms",
              ui.metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Physics),
              ui.metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Update),
              ui.metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Record),
              ui.metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Submit));
  ImGui::Text("CPU active: %.2f | sync wait: %.2f ms",
              ui.metrics.getCpuActiveTimeMs(), ui.metrics.getCpuSyncWaitTimeMs());
  ImGui::Text("CPU measured: %.2f | avg %.2f ms",
              ui.metrics.getCpuPhaseTotalMs(),
              ui.metrics.getAverageCpuPhaseTotalMs());
  ImGui::Text("Draws: %u  |  Tris: %uk", ui.metrics.getLastDrawCallCount(),
              ui.metrics.getLastTriangleCount() / 1000);
  ImGui::Text("Submits: %u | GBuf binds: %u | dyn: %u",
              ui.metrics.getLastQueueSubmitCount(),
              ui.metrics.getLastPipelineBindCount(),
              ui.metrics.getLastDynamicStateChangeCount());
  auto vram = PerformanceMetrics::queryVram(ui.allocator);
  ImGui::Text("VRAM: %llu MiB / %llu MiB",
              static_cast<unsigned long long>(vram.usedBytes >> 20),
              static_cast<unsigned long long>(vram.budgetBytes >> 20));
}

void ImGuiLayer::drawCameraTab(DebugUiContext &ui) {
  ImGui::Text("Pos: (%.1f, %.1f, %.1f)", ui.cameraPos.x, ui.cameraPos.y,
              ui.cameraPos.z);
  ImGui::SliderFloat("Speed", &ui.cameraSpeed, 0.5f, 50.0f);
  if (ImGui::SliderFloat("FOV", &ui.cameraFov, 30.0f, 120.0f) &&
      ui.onProjectionChanged)
    ui.onProjectionChanged();
  if (ImGui::SliderFloat("Draw Dist", &ui.drawDistance, 100.0f, 20000.0f) &&
      ui.onProjectionChanged)
    ui.onProjectionChanged();
}

void ImGuiLayer::record(VkCommandBuffer cmd, VkRenderPass renderPass,
                        VkExtent2D extent, uint32_t imageIndex,
                        DebugUiContext &ui,
                        const std::function<void(VkCommandBuffer)> &beforeUi) {
  if (!initialized)
    return;
  if (imageIndex >= framebuffers.size())
    throw std::runtime_error("Invalid ImGui framebuffer index");

  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  buildUi(ui);

  ImGui::Render();

  VkClearValue clear = {};
  VkRenderPassBeginInfo rpbi = {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = renderPass;
  rpbi.framebuffer = framebuffers[imageIndex];
  rpbi.renderArea = {{0, 0}, extent};
  rpbi.clearValueCount = 1;
  rpbi.pClearValues = &clear;
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  if (beforeUi)
    beforeUi(cmd);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
  vkCmdEndRenderPass(cmd);
}

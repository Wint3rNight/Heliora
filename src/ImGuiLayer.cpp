#include "ImGuiLayer.h"

#include "Scene.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
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
}

void ImGuiLayer::drawRenderTab(DebugUiContext &ui) {
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
  ImGui::SliderFloat("IBL / sky intensity", &ui.iblIntensity, 0.0f, 2.0f,
                     "%.2f");
  ImGui::Checkbox("Auto-exposure", &ui.autoExposureEnabled);
  ImGui::SameLine();
  ImGui::Text("(%.2fx)", ui.autoExposureAdaptedValue);
  ImGui::SliderFloat("Exposure (EV)", &ui.exposureEV, -3.0f, 3.0f, "%+.2f");
  ImGui::SliderFloat("SSGI intensity", &ui.ssgiIntensity, 0.0f, 2.0f,
                     "%.2f");
  ImGui::SliderInt("SSGI samples", &ui.ssgiSamples, 4, 12);
  ImGui::SeparatorText("Lighting isolation");
  ImGui::Checkbox("Sun direct", &ui.enableSunDirect);
  ImGui::Checkbox("Point lights", &ui.enablePointLights);
  ImGui::Checkbox("Spot lights", &ui.enableSpotLights);
  ImGui::Checkbox("IBL ambient", &ui.enableIblAmbient);
  ImGui::Checkbox("SSGI bounce", &ui.enableSsgiBounce);
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
  ImGui::Separator();
  ImGui::Checkbox("Day/night cycle", &ui.dayNightEnable);
  ImGui::SliderFloat("Sim hour", &ui.dayNightHour, 0.0f, 24.0f, "%.2f h");
  ImGui::SliderFloat("Speed (sim-h / real-s)", &ui.dayNightSpeed, 0.1f,
                     600.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
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
  ImGui::Checkbox("Pause", &physics.paused);
  ImGui::SameLine();
  ImGui::BeginDisabled(!physics.paused);
  if (ImGui::Button("Step"))
    physics.stepRequested = true;
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Reset scene"))
    physics.resetRequested = true;
  ImGui::Checkbox("Wireframes (F2)", &physics.showWireframes);
  ImGui::Text("F throws a ball (%u thrown)", physics.thrownBalls);
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

#include "ImGuiLayer.h"

#include "GizmoMath.h"
#include "Picking.h"
#include "Scene.h"
#include "ScenePhysics.h"
#include "Spawning.h"

#include <imgui.h>
#include <imgui_internal.h> // the dock builder, for the default layout
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

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
  // Docked panels ignore per-window size limits; the dock dividers use this
  // instead, so a panel stays wide enough for its slider labels (and the
  // scene area keeps at least this much too).
  style.WindowMinSize = ImVec2(240.0f, 120.0f);
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
  this->window = window;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // No imgui.ini in whatever folder the game runs from: setLayoutFile
  // points it at Resources before the first frame.
  ImGui::GetIO().IniFilename = nullptr;
  registerPanelSettings();
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

bool ImGuiLayer::editInProgress() const {
  return gizmoDrag.active ||
         (ImGui::GetCurrentContext() && ImGui::IsAnyItemActive());
}

// Once nothing is held, the edit is one finished step: a drag gives one undo
// step, not one per frame.
void ImGuiLayer::finishSceneEdit(DebugUiContext &ui) {
  if (sceneEditOpen && !ImGui::IsAnyItemActive() && !gizmoDrag.active) {
    sceneEditOpen = false;
    if (ui.onEditFinished)
      ui.onEditFinished();
  }
}

namespace {

// In the View menu's order, and the names their windows (and the layout
// file) know them by.
constexpr const char *kPanelNames[] = {"Scene",     "Render",  "Lighting",
                                       "Camera",    "Inspector", "Physics",
                                       "Performance", "Debug"};

// The dockspace's id, the same every run so the layout file finds it.
ImGuiID dockspaceId() { return ImHashStr("HelioraDockspace"); }

} // namespace

void ImGuiLayer::setLayoutFile(std::string path) {
  layoutPath = std::move(path);
  ImGui::GetIO().IniFilename = layoutPath.c_str();
}

// Which panels are open, saved in the layout file next to ImGui's own
// window and dock entries:
//   [HelioraPanels][Open]
//   Scene=1
void ImGuiLayer::registerPanelSettings() {
  ImGuiSettingsHandler handler;
  handler.TypeName = "HelioraPanels";
  handler.TypeHash = ImHashStr("HelioraPanels");
  handler.UserData = this;
  handler.ReadOpenFn = [](ImGuiContext *, ImGuiSettingsHandler *,
                          const char *) -> void * {
    return reinterpret_cast<void *>(1); // one entry, no state needed
  };
  handler.ReadLineFn = [](ImGuiContext *, ImGuiSettingsHandler *h, void *,
                          const char *line) {
    auto *layer = static_cast<ImGuiLayer *>(h->UserData);
    char name[32] = {};
    int open = 1;
    if (std::sscanf(line, "%31[^=]=%d", name, &open) != 2)
      return;
    for (int i = 0; i < kPanelCount; ++i)
      if (std::strcmp(name, kPanelNames[i]) == 0)
        layer->panelOpen[i] = open != 0;
  };
  handler.WriteAllFn = [](ImGuiContext *, ImGuiSettingsHandler *h,
                          ImGuiTextBuffer *out) {
    const auto *layer = static_cast<const ImGuiLayer *>(h->UserData);
    out->appendf("[%s][Open]\n", h->TypeName);
    for (int i = 0; i < kPanelCount; ++i)
      out->appendf("%s=%d\n", kPanelNames[i], layer->panelOpen[i] ? 1 : 0);
    out->append("\n");
  };
  ImGui::AddSettingsHandler(&handler);
}

// The layout a first run (or Reset layout) gets: the old sidebar's tabs on
// the left, the Inspector on the right above the physics and performance
// panels, and the scene in between.
void ImGuiLayer::buildDefaultLayout(unsigned int dockspace) {
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::DockBuilderRemoveNode(dockspace);
  ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspace, viewport->WorkSize);
  const float width = std::max(viewport->WorkSize.x, 1.0f);
  ImGuiID left = 0, rest = 0, right = 0, centre = 0, top = 0, bottom = 0;
  ImGui::DockBuilderSplitNode(dockspace, ImGuiDir_Left, 380.0f / width, &left,
                              &rest);
  ImGui::DockBuilderSplitNode(rest, ImGuiDir_Right,
                              360.0f / std::max(width - 380.0f, 1.0f), &right,
                              &centre);
  ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, 0.5f, &top, &bottom);
  for (const char *name : {"Scene", "Render", "Lighting", "Camera"})
    ImGui::DockBuilderDockWindow(name, left);
  ImGui::DockBuilderDockWindow("Inspector", top);
  for (const char *name : {"Physics", "Performance", "Debug"})
    ImGui::DockBuilderDockWindow(name, bottom);
  ImGui::DockBuilderFinish(dockspace);
}

void ImGuiLayer::buildUi(DebugUiContext &ui) {
  // Frame-time history accumulates even while hidden so the graph shows
  // truthful recent history after un-hiding, not a stale pre-hide splice.
  frameTimeGraphData[frameTimeGraphOffset] =
      static_cast<float>(ui.metrics.getLastFrameTimeMs());
  frameTimeGraphOffset = (frameTimeGraphOffset + 1) % 128;

  if (!visible) {
    // F1 screenshot mode: no windows -> empty draw list this frame, and the
    // scene gets the whole window.
    centralArea.reset();
    gizmoHovered = false;
    endGizmoDrag(ui);
    flushInspectorRebuild(ui);
    finishSceneEdit(ui);
    return;
  }

  // The menu bar takes the top of the window, and the dockspace the rest.
  drawMenuBar(ui);
  const ImGuiID dockspace = dockspaceId();
  // A missing or unreadable layout file leaves no dockspace node to load.
  if (resetLayoutRequested || !ImGui::DockBuilderGetNode(dockspace)) {
    buildDefaultLayout(dockspace);
    if (resetLayoutRequested)
      std::fill(std::begin(panelOpen), std::end(panelOpen), true);
    resetLayoutRequested = false;
    focusDefaultTabs = true;
    ImGui::MarkIniSettingsDirty();
  }
  ImGui::DockSpaceOverViewport(dockspace, ImGui::GetMainViewport(),
                               ImGuiDockNodeFlags_PassthruCentralNode |
                                   ImGuiDockNodeFlags_NoDockingOverCentralNode);
  if (const ImGuiDockNode *central =
          ImGui::DockBuilderGetCentralNode(dockspace))
    centralArea = SceneViewport{glm::vec2(central->Pos.x, central->Pos.y),
                                glm::vec2(central->Size.x, central->Size.y)};
  else
    centralArea.reset();

  constexpr float kLabelWidth = 150.0f; // room for the longest slider label
  auto panel = [&](Panel which, auto &&draw) {
    const int i = static_cast<int>(which);
    if (!panelOpen[i])
      return;
    // The first tabs shown in the default layout.
    if (focusDefaultTabs &&
        (which == Panel::Scene || which == Panel::Physics))
      ImGui::SetNextWindowFocus();
    bool open = true;
    if (ImGui::Begin(kPanelNames[i], &open)) {
      ImGui::PushItemWidth(-kLabelWidth);
      draw();
      ImGui::PopItemWidth();
    }
    ImGui::End();
    if (!open) {
      panelOpen[i] = false;
      ImGui::MarkIniSettingsDirty();
    }
  };
  panel(Panel::Scene, [&] { drawSceneTab(ui); });
  panel(Panel::Render, [&] { drawRenderTab(ui); });
  panel(Panel::Lighting, [&] { drawLightingTab(ui); });
  panel(Panel::Camera, [&] { drawCameraTab(ui); });
  panel(Panel::Inspector, [&] { drawInspector(ui); });
  panel(Panel::Physics, [&] { drawPhysicsTab(ui); });
  panel(Panel::Performance, [&] { drawPerformanceTab(ui); });
  panel(Panel::Debug, [&] { drawDebugTab(ui); });
  focusDefaultTabs = false;
  // Even with the Inspector closed or behind another tab.
  flushInspectorRebuild(ui);

  drawGizmo(ui);
  finishSceneEdit(ui);
  drawPrompt(ui);
}

// The renderer's question as a modal window in the middle of the screen,
// which blocks the rest of the UI until it's answered. Esc is Cancel.
void ImGuiLayer::drawPrompt(DebugUiContext &ui) {
  static constexpr const char *kPopup = "##prompt";
  const UiPrompt *prompt = ui.prompt;
  if (!prompt || prompt->kind == UiPrompt::Kind::None) {
    // Answered: close the modal if it's still up.
    if (promptShown != UiPrompt::Kind::None &&
        ImGui::BeginPopupModal(kPopup)) {
      ImGui::CloseCurrentPopup();
      ImGui::EndPopup();
    }
    promptShown = UiPrompt::Kind::None;
    return;
  }
  if (promptShown == UiPrompt::Kind::None) {
    ImGui::OpenPopup(kPopup);
    if (prompt->kind == UiPrompt::Kind::SaveAs)
      std::snprintf(saveAsName, sizeof(saveAsName), "%s",
                    ui.sceneName ? ui.sceneName->c_str() : "");
  }
  // The name field takes the keyboard when Save as opens, after Replace is
  // cancelled, and after a name is refused, so typing goes straight in.
  if (prompt->kind == UiPrompt::Kind::SaveAs &&
      promptShown != UiPrompt::Kind::SaveAs)
    focusSaveAsName = true;
  if (promptShown != prompt->kind)
    promptOpenedFrame = ImGui::GetFrameCount();
  promptShown = prompt->kind;
  const ImVec2 display = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(display.x * 0.5f, display.y * 0.4f),
                          ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopup, nullptr,
                              ImGuiWindowFlags_AlwaysAutoResize |
                                  ImGuiWindowFlags_NoTitleBar |
                                  ImGuiWindowFlags_NoMove))
    return;
  auto answer = [&](PromptAnswer a) {
    if (ui.onPromptAnswer)
      ui.onPromptAnswer(a, saveAsName);
  };
  auto button = [&](const char *label, PromptAnswer a) {
    if (ImGui::Button(label))
      answer(a);
    ImGui::SameLine();
  };
  const char *name = prompt->name.c_str();
  switch (prompt->kind) {
  case UiPrompt::Kind::UnsavedQuit:
    ImGui::TextUnformatted("Quit with unsaved changes?");
    if (prompt->unsavedScene)
      ImGui::BulletText("The scene '%s'", name);
    if (prompt->unsavedSettings)
      ImGui::BulletText("Render or physics settings");
    button("Save and quit", PromptAnswer::Save);
    button("Quit without saving", PromptAnswer::DontSave);
    button("Cancel", PromptAnswer::Cancel);
    break;
  case UiPrompt::Kind::UnsavedSwitch:
    ImGui::Text("Save the changes to '%s' before switching scenes?", name);
    button("Save", PromptAnswer::Save);
    button("Don't save", PromptAnswer::DontSave);
    button("Cancel", PromptAnswer::Cancel);
    break;
  case UiPrompt::Kind::SaveAs:
    ImGui::TextUnformatted("Save the scene as");
    if (std::exchange(focusSaveAsName, false))
      ImGui::SetKeyboardFocusHere();
    ImGui::SetNextItemWidth(320.0f);
    if (ImGui::InputText("##name", saveAsName, sizeof(saveAsName),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
      answer(PromptAnswer::Save);
      focusSaveAsName = true; // still here if the name was refused
    }
    ImGui::TextDisabled("Letters, digits, spaces, - and _. It goes in "
                        "Resources/Scenes.");
    if (ImGui::Button("Save")) {
      answer(PromptAnswer::Save);
      focusSaveAsName = true;
    }
    ImGui::SameLine();
    button("Cancel", PromptAnswer::Cancel);
    break;
  case UiPrompt::Kind::Replace:
    ImGui::Text("%s already exists. Replace it?", name);
    button("Replace", PromptAnswer::Replace);
    button("Cancel", PromptAnswer::Cancel);
    break;
  case UiPrompt::Kind::None:
    break;
  }
  ImGui::NewLine();
  if (!prompt->error.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, hex(kRedBright));
    ImGui::TextWrapped("%s", prompt->error.c_str());
    ImGui::PopStyleColor();
  }
  // Not on the frame it opened: Esc to quit opens it, and ImGui sees that
  // same press this frame.
  if (ImGui::GetFrameCount() > promptOpenedFrame &&
      ImGui::IsKeyPressed(ImGuiKey_Escape, false))
    answer(PromptAnswer::Cancel);
  ImGui::EndPopup();
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

// A change that rebuilds the entry, once no field is held, so a drag
// rebuilds once, when it ends.
void ImGuiLayer::flushInspectorRebuild(DebugUiContext &ui) {
  if (inspectorRebuildPending && !ImGui::IsAnyItemActive()) {
    inspectorRebuildPending = false;
    if (ui.onEntryChanged)
      ui.onEntryChanged(inspectorPendingRef);
  }
}

namespace {

constexpr ImGuiSliderFlags kClamp = ImGuiSliderFlags_AlwaysClamp;
constexpr const char *kShapeNames[] = {"box", "sphere", "capsule", "plane"};

// An Inspector edit: moved is placement only, applied at once; changed needs
// the entry rebuilt.
struct InspectorEdit {
  bool moved = false;
  bool changed = false;
};

void transformFields(NodeTransform &t, InspectorEdit &edit) {
  ImGui::SeparatorText("Transform");
  edit.moved |= ImGui::DragFloat3("Position", &t.position.x, 0.05f, 0.0f, 0.0f,
                                  "%.2f");
  edit.moved |= ImGui::DragFloat3("Rotation", &t.rotationDeg.x, 0.5f, 0.0f,
                                  0.0f, "%.1f\u00b0");
  // Scale shows at once, but the body's size only changes in the rebuild.
  if (ImGui::DragFloat("Scale", &t.scale, 0.01f, 0.01f, 100.0f, "%.2f",
                       kClamp)) {
    edit.moved = true;
    edit.changed = true;
  }
}

// A generated shape's kind, size and colour, or a model file's path.
void shapeFields(SceneModelEntry &edited, InspectorEdit &edit) {
  if (edited.shape) {
    ImGui::SeparatorText("Shape");
    int shape = static_cast<int>(*edited.shape);
    if (ImGui::Combo("Shape", &shape, kShapeNames, 4)) {
      edited.shape = static_cast<PrimitiveShape>(shape);
      edit.changed = true;
    }
    switch (*edited.shape) {
    case PrimitiveShape::Box:
      edit.changed |= ImGui::DragFloat3("Size", &edited.size.x, 0.01f, 0.01f,
                                        1000.0f, "%.2f", kClamp);
      break;
    case PrimitiveShape::Plane: {
      glm::vec2 size(edited.size.x, edited.size.z);
      if (ImGui::DragFloat2("Size", &size.x, 0.05f, 0.01f, 10000.0f, "%.2f",
                            kClamp)) {
        edited.size.x = size.x;
        edited.size.z = size.y;
        edit.changed = true;
      }
      break;
    }
    case PrimitiveShape::Capsule:
      edit.changed |= ImGui::DragFloat("Height", &edited.height, 0.01f, 0.02f,
                                       1000.0f, "%.2f", kClamp);
      [[fallthrough]];
    case PrimitiveShape::Sphere:
      edit.changed |= ImGui::DragFloat("Radius", &edited.radius, 0.01f, 0.01f,
                                       1000.0f, "%.2f", kClamp);
      break;
    }
    edit.changed |= ImGui::ColorEdit3("Colour", &edited.color.x);
  } else {
    ImGui::SeparatorText("Model file");
    ImGui::TextWrapped("%s", edited.path.c_str());
  }
}

// The body, its material and the numbers that override it, with what's in
// use right now shown in the empty fields.
void physicsFields(const DebugUiContext &ui, SceneModelEntry &edited,
                   bool isGrid, InspectorEdit &edit) {
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
    edit.changed = true;
  }
  if (edited.body) {
    const std::span<const PhysicsMaterial> materials = physicsMaterials();
    const char *materialName =
        edited.material ? edited.material->c_str() : "none";
    if (ImGui::BeginCombo("Material", materialName)) {
      if (ImGui::Selectable("none", !edited.material)) {
        edited.material.reset();
        edit.changed = true;
      }
      for (const PhysicsMaterial &m : materials)
        if (ImGui::Selectable(m.name, edited.material == m.name)) {
          edited.material = m.name;
          edit.changed = true;
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
    edit.changed |= optionalFloat("Mass", edited.mass, massInUse, 0.5f, 0.01f,
                                  1.0e6f, "%.1f kg");
    edit.changed |= optionalFloat("Density", edited.density, inUse.density,
                                  5.0f, 1.0f, 30000.0f, "%.0f kg/m\u00b3");
    edit.changed |= optionalFloat("Friction", edited.friction, inUse.friction,
                                  0.01f, 0.0f, 10.0f, "%.2f");
    edit.changed |= optionalFloat("Restitution", edited.restitution,
                                  inUse.restitution, 0.01f, 0.0f, 1.0f, "%.2f");
    if (!edited.shape) {
      static constexpr const char *kColliders[] = {"auto", "hull", "mesh",
                                                   "box"};
      int collider =
          static_cast<int>(edited.collider.value_or(SceneCollider::Auto));
      if (ImGui::Combo("Collider", &collider, kColliders, 4)) {
        // auto is the default, so it's left out of the file.
        if (collider == 0)
          edited.collider.reset();
        else
          edited.collider = static_cast<SceneCollider>(collider);
        edit.changed = true;
      }
    }
    if (ui.selectionMass > 0.0f)
      ImGui::TextDisabled(isGrid ? "Weighs %.1f kg each" : "Weighs %.1f kg",
                          ui.selectionMass);
  }
}

// A grid's count, spacing and layout rotation. True when Split was pressed.
bool gridFields(SceneGridEntry &grid, InspectorEdit &edit) {
  ImGui::SeparatorText("Grid");
  edit.changed |= ImGui::DragInt3("Count", &grid.count.x, 0.1f, 1, 64, "%d",
                                  kClamp);
  edit.moved |= ImGui::DragFloat3("Spacing", &grid.spacing.x, 0.01f, 0.01f,
                                  100.0f, "%.2f", kClamp);
  edit.moved |= ImGui::DragFloat("Layout rotation", &grid.layoutRotation, 0.5f,
                                 -360.0f, 360.0f, "%.1f\u00b0", kClamp);
  return ImGui::Button("Split into objects");
}

} // namespace

// The selected entry's settings, in the Inspector panel. Edits go to a copy
// first and are only kept if the parser would accept them, so a saved scene
// always loads.
void ImGuiLayer::drawInspector(DebugUiContext &ui) {
  if (!ui.scene || !ui.selection) {
    ImGui::TextDisabled(ui.playing ? "Stop (F5) to edit."
                                   : "Nothing selected. Click an object "
                                     "in the scene.");
    return;
  }
  const SceneEntryRef ref = *ui.selection;
  SceneDescription &scene = *ui.scene;
  const bool isGrid = ref.kind == SceneEntryRef::Kind::Grid;
  const size_t count = isGrid ? scene.grids.size() : scene.models.size();
  if (ref.index < 0 || ref.index >= static_cast<int>(count))
    return;
  if (ref != inspectorShownRef) {
    inspectorShownRef = ref;
    inspectorProblem.clear();
  }
  // Each entry's fields get their own IDs, so a field still being typed in
  // can't carry over to another entry when the selection changes.
  ImGui::PushID(static_cast<int>(ref.kind));
  ImGui::PushID(ref.index);
  const SceneGridEntry *grid = isGrid ? &scene.grids[ref.index] : nullptr;
  SceneModelEntry edited = isGrid ? grid->model : scene.models[ref.index];
  SceneGridEntry editedGrid = isGrid ? *grid : SceneGridEntry{};

  const std::string what =
      edited.shape ? kShapeNames[static_cast<int>(*edited.shape)]
                   : std::filesystem::path(edited.path).filename().string();
  if (isGrid) {
    const glm::ivec3 n = grid->count;
    ImGui::Text("Grid %d: %s, %lld copies", ref.index, what.c_str(),
                static_cast<long long>(n.x) * n.y * n.z);
  } else {
    ImGui::Text("Model %d: %s", ref.index, what.c_str());
  }
  ImGui::PushItemWidth(-110.0f);
  InspectorEdit edit;
  transformFields(edited.transform, edit);
  shapeFields(edited, edit);
  physicsFields(ui, edited, isGrid, edit);
  const bool split = isGrid && gridFields(editedGrid, edit);
  ImGui::PopItemWidth();

  if (edit.moved || edit.changed)
    commitInspectorEdit(ui, ref, edited, editedGrid, edit.moved, edit.changed);
  if (!inspectorProblem.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, hex(kRedBright));
    ImGui::TextWrapped("%s", inspectorProblem.c_str());
    ImGui::PopStyleColor();
  }
  ImGui::Separator();
  const bool deleting = ImGui::Button("Delete");
  ImGui::PopID();
  ImGui::PopID();

  if (split || deleting) {
    // The entry is going, so a rebuild of it would be wasted.
    inspectorRebuildPending = false;
    if (split && ui.onGridSplit)
      ui.onGridSplit(ref);
    else if (deleting && ui.onEntryDeleted)
      ui.onEntryDeleted(ref);
  }
}

// Keeps an Inspector edit if the parser would take it (and, for a grid, if
// the physics world has room for its bodies), otherwise shows why not.
void ImGuiLayer::commitInspectorEdit(DebugUiContext &ui, SceneEntryRef ref,
                                     SceneModelEntry edited,
                                     SceneGridEntry editedGrid, bool moved,
                                     bool changed) {
  const bool isGrid = ref.kind == SceneEntryRef::Kind::Grid;
  std::optional<std::string> problem;
  if (isGrid) {
    editedGrid.model = edited;
    problem = problemWith(editedGrid);
    // Every body in the world counts, against the live world's limit
    // (the Physics tab's only applies at the next load). What the world
    // holds for this grid now, not the scene's count: a drag of Count
    // changes the scene every frame but the world only at its end.
    const glm::ivec3 n = editedGrid.count;
    const uint32_t after =
        edited.body ? uint32_t(n.x) * uint32_t(n.y) * uint32_t(n.z) : 0u;
    if (!problem && !gridFitsBodyLimit(ui.bodiesInWorld, ui.selectionBodies,
                                       after, ui.bodyLimit))
      problem = std::format("more bodies than the physics world holds ({})",
                            ui.bodyLimit);
  } else {
    problem = problemWith(edited);
  }
  if (problem) {
    inspectorProblem = *problem;
  } else {
    inspectorProblem.clear();
    sceneEditOpen = true;
    if (isGrid)
      ui.scene->grids[ref.index] = editedGrid;
    else
      ui.scene->models[ref.index] = edited;
    if (moved && ui.onEntryMoved)
      ui.onEntryMoved(ref);
    if (changed) {
      inspectorRebuildPending = true;
      inspectorPendingRef = ref;
    }
  }
}

// The bar across the top: the File and View menus, then Play/Stop, the
// gizmo's mode and Snap, and the scene's name on the right.
void ImGuiLayer::drawMenuBar(DebugUiContext &ui) {
  if (!ImGui::BeginMainMenuBar())
    return;
  if (ImGui::BeginMenu("File")) {
    if (ImGui::MenuItem("Save", "Ctrl+S") && ui.onSave)
      ui.onSave();
    if (ImGui::MenuItem("Save as...", "Ctrl+Shift+S") && ui.onSaveAs)
      ui.onSaveAs();
    ImGui::Separator();
    if (ImGui::MenuItem("Quit", "Esc") && ui.onQuit)
      ui.onQuit();
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("View")) {
    for (int i = 0; i < kPanelCount; ++i)
      if (ImGui::MenuItem(kPanelNames[i], nullptr, &panelOpen[i]))
        ImGui::MarkIniSettingsDirty();
    ImGui::Separator();
    if (ImGui::MenuItem("Reset layout"))
      resetLayoutRequested = true;
    ImGui::EndMenu();
  }
  ImGui::Separator();
  if (ImGui::Button(ui.playing ? "Stop" : "Play") && ui.onTogglePlay)
    ui.onTogglePlay();
  ImGui::SetItemTooltip(ui.playing ? "F5 stops" : "F5 plays");
  ImGui::TextDisabled(ui.playing ? "Playing" : "Editing");
  if (!ui.playing) {
    // The gizmo's mode, for the touchpad as much as for keys 1, 2, 3.
    static constexpr std::pair<GizmoMode, const char *> kModes[] = {
        {GizmoMode::Move, "Move"},
        {GizmoMode::Rotate, "Rotate"},
        {GizmoMode::Scale, "Scale"}};
    int key = 1;
    for (const auto &[mode, label] : kModes) {
      const bool selected = mode == gizmoMode;
      ImGui::PushStyleColor(ImGuiCol_Button,
                            selected ? hex(kPurple) : hex(kBase3));
      if (ImGui::Button(label))
        gizmoMode = mode;
      ImGui::PopStyleColor();
      ImGui::SetItemTooltip("Key %d", key++);
    }
    ImGui::Checkbox("Snap", &gizmoSnap);
    ImGui::SetItemTooltip(
        "Or hold Ctrl: 0.25 m steps, 15\u00b0 steps and 0.1 scale steps");
  }
  if (ui.sceneName) {
    // On the right.
    const char *unsaved = ui.sceneUnsaved ? " (unsaved)" : "";
    const float width = ImGui::CalcTextSize(ui.sceneName->c_str()).x +
                        ImGui::CalcTextSize(unsaved).x +
                        ImGui::GetStyle().ItemSpacing.x * 2.0f;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX(),
                             ImGui::GetWindowWidth() - width));
    ImGui::TextUnformatted(ui.sceneName->c_str());
    if (ui.sceneUnsaved) {
      ImGui::SameLine(0.0f, 4.0f);
      ImGui::PushStyleColor(ImGuiCol_Text, hex(kPurpleBright));
      ImGui::TextUnformatted("(unsaved)");
      ImGui::PopStyleColor();
      ImGui::SetItemTooltip("Ctrl+S saves");
    }
  }
  ImGui::EndMainMenuBar();
}

namespace {

// Handles keep this size on screen however far away the object is.
constexpr float kGizmoPixels = 110.0f;     // arrow length, ring radius
constexpr float kScaleRingPixels = 70.0f;  // the scale handle's radius
constexpr float kGrabPixels = 8.0f;        // how close counts as on it
constexpr int kRingSegments = 48;
constexpr float kMoveSnap = 0.25f;  // metres
constexpr float kRotateSnap = 15.0f; // degrees
constexpr float kScaleSnap = 0.1f;
constexpr glm::vec3 kAxes[3] = {
    {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
constexpr uint32_t kAxisColours[3] = {0xe0505f, 0x7ac74f, 0x4f8fe6};
constexpr uint32_t kGizmoHot = 0xf5d36b; // under the cursor or dragged

ImU32 colourU32(uint32_t rgb) { return ImGui::GetColorU32(hex(rgb)); }

// The ring around axis, as points on screen. A point behind the camera is
// left out, and the segments either side of it aren't drawn.
std::array<std::optional<glm::vec2>, kRingSegments>
ringOnScreen(glm::vec3 centre, glm::vec3 axis, float radius,
             glm::vec2 viewport, const glm::mat4 &viewProj) {
  const glm::vec3 helper = std::abs(axis.y) < 0.9f ? glm::vec3(0, 1, 0)
                                                   : glm::vec3(1, 0, 0);
  const glm::vec3 u = glm::normalize(glm::cross(axis, helper));
  const glm::vec3 v = glm::cross(axis, u);
  std::array<std::optional<glm::vec2>, kRingSegments> points;
  for (int i = 0; i < kRingSegments; ++i) {
    const float a = 2.0f * std::numbers::pi_v<float> * i / kRingSegments;
    points[i] = screenFromWorld(
        centre + radius * (std::cos(a) * u + std::sin(a) * v), viewport,
        viewProj);
  }
  return points;
}

} // namespace

void ImGuiLayer::endGizmoDrag(DebugUiContext &ui) {
  if (!gizmoDrag.active)
    return;
  // The scale showed at once; the body's size changes in this rebuild.
  if (gizmoDrag.scaled && ui.onEntryChanged)
    ui.onEntryChanged(gizmoDrag.ref);
  gizmoDrag = {};
}

// The handles are drawn in 2D over the frame, under the panels, and a click
// is tested against them on screen. Dragging turns the cursor into a ray and
// the ray into a placement, with the maths in GizmoMath. Only the selected
// object is looked at, so it costs nothing per object in the scene.
void ImGuiLayer::drawGizmo(DebugUiContext &ui) {
  gizmoHovered = false;
  ImGuiIO &io = ImGui::GetIO();
  if (!ui.playing && !io.WantTextInput && !gizmoDrag.active) {
    if (ImGui::IsKeyPressed(ImGuiKey_1, false))
      gizmoMode = GizmoMode::Move;
    if (ImGui::IsKeyPressed(ImGuiKey_2, false))
      gizmoMode = GizmoMode::Rotate;
    if (ImGui::IsKeyPressed(ImGuiKey_3, false))
      gizmoMode = GizmoMode::Scale;
  }

  const std::optional<GizmoView> view = gizmoView(ui);
  if (!view) {
    endGizmoDrag(ui);
    return;
  }
  const int hot = gizmoDrag.active ? gizmoDrag.axis : hoveredHandle(*view);
  gizmoHovered = hot >= 0;

  if (!gizmoDrag.active && hot >= 0 && !view->justSelected &&
      ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    startGizmoDrag(ui, *view, hot);
  if (gizmoDrag.active && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    endGizmoDrag(ui);
  } else if (gizmoDrag.active &&
             ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    // Nothing moves until the cursor has gone ImGui's drag distance (6 px)
    // from where it grabbed, so a plain click, or a tap that wobbles a
    // pixel, changes nothing: no snapping, no unsaved marker, no undo step.
    dragGizmo(ui, *view);
  }
  if (gizmoHovered)
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  drawGizmoHandles(*view, hot);
}

// A point in the scene area's pixels, as a window position for drawing.
ImVec2 ImGuiLayer::GizmoView::at(glm::vec2 p) const {
  return ImVec2(p.x + area.origin.x, p.y + area.origin.y);
}

// A grid only turns about y, through its layout rotation.
bool ImGuiLayer::GizmoView::axisShown(int axis, GizmoMode mode) const {
  return !(mode == GizmoMode::Rotate && grid && axis != 1);
}

// An arrow pointing almost straight at the camera is too short to grab or
// drag sensibly, so it's hidden.
std::optional<glm::vec2> ImGuiLayer::GizmoView::arrowTip(int axis) const {
  const std::optional<glm::vec2> tip =
      screenFromWorld(centre + kAxes[axis] * length, area.size, viewProj);
  if (!tip || glm::distance(*tip, centrePixel) < 0.2f * kGizmoPixels)
    return std::nullopt;
  return tip;
}

// The selected entry and where its handles sit this frame, in the scene
// area's own pixels. Nothing when there are no handles to show: no
// selection, the cursor captured for flying, a prompt open, a drag that lost
// its object, or the object behind the camera.
std::optional<ImGuiLayer::GizmoView> ImGuiLayer::gizmoView(DebugUiContext &ui) {
  GizmoView view;
  // The selected entry. The renderer only passes a selection in edit mode.
  if (ui.scene && ui.selection) {
    const SceneEntryRef ref = *ui.selection;
    if (ref.kind == SceneEntryRef::Kind::Grid && ref.index >= 0 &&
        ref.index < static_cast<int>(ui.scene->grids.size())) {
      view.grid = &ui.scene->grids[ref.index];
      view.entry = &view.grid->model;
    } else if (ref.kind == SceneEntryRef::Kind::Model && ref.index >= 0 &&
               ref.index < static_cast<int>(ui.scene->models.size())) {
      view.entry = &ui.scene->models[ref.index];
    }
  }
  // main.cpp selects before the UI runs, so on the click that selected this
  // object its handles are new, and the click mustn't grab one.
  const std::optional<SceneEntryRef> selected =
      view.entry ? ui.selection : std::nullopt;
  view.justSelected = selected != gizmoLastSelection;
  gizmoLastSelection = selected;
  // While flying, the cursor is hidden and its position means nothing.
  const bool cursorFree =
      window && glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL;
  const bool asking = ui.prompt && ui.prompt->kind != UiPrompt::Kind::None;
  if (!view.entry || !cursorFree || asking || !ImGui::IsMousePosValid() ||
      (gizmoDrag.active && gizmoDrag.ref != *ui.selection))
    return std::nullopt;

  view.area = ui.sceneArea;
  view.viewProj = ui.viewProj;
  // A grid turns about its first copy's spot, so the handles sit there.
  view.centre = view.entry->transform.position;
  const std::optional<glm::vec2> centrePixel =
      screenFromWorld(view.centre, view.area.size, view.viewProj);
  if (!centrePixel || view.area.size.y <= 0.0f)
    return std::nullopt; // behind the camera
  view.centrePixel = *centrePixel;
  // The world length that shows as kGizmoPixels at the centre's depth (w):
  // the projection's y scale times half the viewport's height is how many
  // pixels one metre covers one metre away.
  const float depth = (view.viewProj * glm::vec4(view.centre, 1.0f)).w;
  const float pixelsPerMetre =
      std::abs(ui.sceneUbo.projection[1][1]) * view.area.size.y * 0.5f / depth;
  view.length = kGizmoPixels / pixelsPerMetre;
  const ImGuiIO &io = ImGui::GetIO();
  view.windowMouse = glm::vec2(io.MousePos.x, io.MousePos.y);
  view.mouse = view.area.toScene(view.windowMouse);
  view.ray = rayFromScreen(view.mouse, view.area.size,
                           glm::inverse(view.viewProj));
  return view;
}

// Which handle the cursor is on, the nearest within kGrabPixels: an axis, 0
// for the scale ring, or -1 for none. None while the cursor is off the scene,
// on a panel, or using a widget.
int ImGuiLayer::hoveredHandle(const GizmoView &view) const {
  if (!view.area.contains(view.windowMouse) ||
      ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
      ImGui::IsAnyItemActive())
    return -1;
  int hot = -1;
  float nearest = kGrabPixels;
  auto consider = [&](int handle, float distance) {
    if (distance < nearest) {
      nearest = distance;
      hot = handle;
    }
  };
  for (int axis = 0; axis < 3; ++axis) {
    if (!view.axisShown(axis, gizmoMode))
      continue;
    if (gizmoMode == GizmoMode::Move) {
      if (const std::optional<glm::vec2> tip = view.arrowTip(axis))
        consider(axis, distanceToSegment(view.mouse, view.centrePixel, *tip));
    } else if (gizmoMode == GizmoMode::Rotate) {
      const auto ring = ringOnScreen(view.centre, kAxes[axis], view.length,
                                     view.area.size, view.viewProj);
      for (int i = 0; i < kRingSegments; ++i) {
        const auto &a = ring[i];
        const auto &b = ring[(i + 1) % kRingSegments];
        if (a && b)
          consider(axis, distanceToSegment(view.mouse, *a, *b));
      }
    }
  }
  if (gizmoMode == GizmoMode::Scale)
    consider(0, std::abs(glm::distance(view.mouse, view.centrePixel) -
                         kScaleRingPixels));
  return hot;
}

// Grabs a handle: remembers where the object started and where on the axis
// or ring the cursor took hold, so the drag moves it by the difference.
void ImGuiLayer::startGizmoDrag(DebugUiContext &ui, const GizmoView &view,
                                int handle) {
  GizmoDrag drag;
  drag.active = true;
  drag.axis = handle;
  drag.ref = *ui.selection;
  drag.start = view.entry->transform;
  drag.startLayoutRotation = view.grid ? view.grid->layoutRotation : 0.0f;
  drag.grabbedPixel = view.mouse;
  drag.centrePixel = view.centrePixel;
  bool grabbed = true;
  if (gizmoMode == GizmoMode::Move) {
    const std::optional<glm::vec3> point =
        closestPointOnAxis(view.ray, view.centre, kAxes[handle]);
    grabbed = point.has_value();
    drag.grabbed = point.value_or(view.centre);
  } else if (gizmoMode == GizmoMode::Rotate) {
    const std::optional<float> angle =
        angleAroundAxis(view.ray, view.centre, kAxes[handle]);
    grabbed = angle.has_value();
    drag.lastAngle = angle.value_or(0.0f);
  }
  if (grabbed)
    gizmoDrag = drag;
}

// The object follows the cursor: along the axis, around the ring, or by the
// scale ring's distance, snapped while snapping is on or Ctrl is held.
void ImGuiLayer::dragGizmo(DebugUiContext &ui, const GizmoView &view) {
  const bool snapping = gizmoSnap || ImGui::GetIO().KeyCtrl;
  const int axis = gizmoDrag.axis;
  SceneGridEntry editedGrid = view.grid ? *view.grid : SceneGridEntry{};
  SceneModelEntry edited = *view.entry;
  NodeTransform &t = edited.transform;
  switch (gizmoMode) {
  case GizmoMode::Move:
    // The line through the object along the axis doesn't change as it
    // slides along it, so the current position works as well as the
    // start.
    if (const std::optional<glm::vec3> point =
            closestPointOnAxis(view.ray, view.centre, kAxes[axis])) {
      t.position = gizmoDrag.start.position + (*point - gizmoDrag.grabbed);
      // Snapped onto the 0.25 m grid, not by 0.25 m steps, so things
      // line up with each other.
      if (snapping)
        t.position[axis] = snap(t.position[axis], kMoveSnap);
    }
    break;
  case GizmoMode::Rotate: {
    if (const std::optional<float> angle =
            angleAroundAxis(view.ray, view.centre, kAxes[axis])) {
      // The change since last frame, taken the short way round, so
      // passing the -180/180 seam doesn't jump a full turn.
      gizmoDrag.turned += std::remainder(*angle - gizmoDrag.lastAngle,
                                         2.0f * std::numbers::pi_v<float>);
      gizmoDrag.lastAngle = *angle;
    }
    float degrees = glm::degrees(gizmoDrag.turned);
    if (snapping)
      degrees = snap(degrees, kRotateSnap); // by 15 degree steps
    if (view.grid) {
      editedGrid.layoutRotation =
          std::remainder(gizmoDrag.startLayoutRotation + degrees, 360.0f);
    } else if (axis == 1) {
      // About world y it's just more yaw, since rotationFromDegrees
      // applies y first, so the other two numbers stay as typed.
      t.rotationDeg = gizmoDrag.start.rotationDeg;
      t.rotationDeg.y = std::remainder(t.rotationDeg.y + degrees, 360.0f);
    } else {
      t.rotationDeg = degreesFromRotation(
          glm::angleAxis(glm::radians(degrees), kAxes[axis]) *
          rotationFromDegrees(gizmoDrag.start.rotationDeg));
    }
    break;
  }
  case GizmoMode::Scale: {
    float scale = gizmoDrag.start.scale *
                  scaleFromDrag(gizmoDrag.centrePixel,
                                gizmoDrag.grabbedPixel, view.mouse);
    if (snapping)
      scale = snap(scale, kScaleSnap); // onto 0.1 steps
    t.scale = glm::clamp(scale, 0.01f, 100.0f);
    break;
  }
  }
  commitGizmoEdit(ui, view, edited, editedGrid);
}

// The dragged values go into the scene only if the parser would take them,
// as in the Inspector; then the renderer moves the object to match.
void ImGuiLayer::commitGizmoEdit(DebugUiContext &ui, const GizmoView &view,
                                 const SceneModelEntry &edited,
                                 SceneGridEntry editedGrid) {
  bool changed = false;
  if (view.grid) {
    editedGrid.model = edited;
    if (editedGrid != *view.grid && !problemWith(editedGrid)) {
      *view.grid = editedGrid;
      changed = true;
    }
  } else if (edited != *view.entry && !problemWith(edited)) {
    *view.entry = edited;
    changed = true;
  }
  if (changed) {
    sceneEditOpen = true;
    gizmoDrag.scaled |= gizmoMode == GizmoMode::Scale;
    if (ui.onEntryMoved)
      ui.onEntryMoved(gizmoDrag.ref);
  }
}

// The handles at the object's spot after this frame's edit, the hovered or
// dragged one highlighted.
void ImGuiLayer::drawGizmoHandles(const GizmoView &view, int hot) const {
  // Drawn after the edit, so the handles sit on the object's new spot.
  const glm::vec3 now = view.entry->transform.position;
  const std::optional<glm::vec2> nowPixel =
      screenFromWorld(now, view.area.size, view.viewProj);
  if (!nowPixel)
    return;
  ImDrawList *draw = ImGui::GetBackgroundDrawList();
  draw->PushClipRect(view.at(glm::vec2(0.0f)), view.at(view.area.size),
                     true);
  auto colourOf = [&](int handle, uint32_t rgb) {
    return colourU32(handle == hot ? kGizmoHot : rgb);
  };
  for (int axis = 0; axis < 3 && gizmoMode != GizmoMode::Scale; ++axis) {
    if (!view.axisShown(axis, gizmoMode))
      continue;
    const ImU32 colour = colourOf(axis, kAxisColours[axis]);
    const float thickness = axis == hot ? 4.0f : 2.5f;
    if (gizmoMode == GizmoMode::Move) {
      const std::optional<glm::vec2> tip =
          screenFromWorld(now + kAxes[axis] * view.length, view.area.size,
                          view.viewProj);
      if (!tip || glm::distance(*tip, *nowPixel) < 0.2f * kGizmoPixels)
        continue;
      const glm::vec2 along = glm::normalize(*tip - *nowPixel);
      const glm::vec2 across(-along.y, along.x);
      const glm::vec2 point = *tip + along * 14.0f;
      draw->AddLine(view.at(*nowPixel), view.at(*tip), colour, thickness);
      draw->AddTriangleFilled(view.at(point), view.at(*tip + across * 6.0f),
                              view.at(*tip - across * 6.0f), colour);
    } else {
      const auto ring =
          ringOnScreen(now, kAxes[axis], view.length, view.area.size,
                       view.viewProj);
      for (int i = 0; i < kRingSegments; ++i) {
        const auto &a = ring[i];
        const auto &b = ring[(i + 1) % kRingSegments];
        if (a && b)
          draw->AddLine(view.at(*a), view.at(*b), colour, thickness);
      }
    }
  }
  if (gizmoMode == GizmoMode::Scale)
    draw->AddCircle(view.at(*nowPixel), kScaleRingPixels, colourOf(0, kText),
                    kRingSegments, hot == 0 ? 4.0f : 2.5f);
  draw->AddCircleFilled(view.at(*nowPixel), 3.0f, colourU32(kText));
  draw->PopClipRect();
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
  // Save writes the layout, never where physics moved things.
  if (ImGui::Button("Save") && ui.onSave)
    ui.onSave();
  ImGui::SetItemTooltip("Ctrl+S");
  ImGui::SameLine();
  if (ImGui::Button("Save as...") && ui.onSaveAs)
    ui.onSaveAs();
  ImGui::SetItemTooltip("Ctrl+Shift+S");
  if (ui.saveStatus && !ui.saveStatus->empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", ui.saveStatus->c_str());
  }

  // The spawn list, edit mode only: a new object lands where the camera
  // looks and gets selected.
  ImGui::SeparatorText("Add an object");
  if (ui.playing)
    ImGui::TextDisabled("Stop (F5) to add objects.");
  ImGui::BeginDisabled(ui.playing);
  static constexpr std::pair<PrimitiveShape, const char *> kShapes[] = {
      {PrimitiveShape::Box, "Box"},
      {PrimitiveShape::Sphere, "Sphere"},
      {PrimitiveShape::Capsule, "Capsule"},
      {PrimitiveShape::Plane, "Plane"}};
  for (const auto &[shape, label] : kShapes) {
    if (shape != PrimitiveShape::Box)
      ImGui::SameLine();
    if (ImGui::Button(label) && ui.onSpawn)
      ui.onSpawn(newShapeEntry(shape));
  }
  if (ui.modelFiles) {
    for (const std::string &file : *ui.modelFiles) {
      const std::string name =
          std::filesystem::path(file).filename().string() + "##" + file;
      // A path the scene file couldn't read back (a '#') can't be added.
      const std::optional<std::string> problem =
          problemWith(newModelFileEntry(file));
      ImGui::BeginDisabled(problem.has_value());
      if (ImGui::Selectable(name.c_str()) && ui.onSpawn)
        ui.onSpawn(newModelFileEntry(file));
      ImGui::EndDisabled();
      ImGui::SetItemTooltip("%s", problem ? problem->c_str() : file.c_str());
    }
    if (ui.modelFiles->empty())
      ImGui::TextDisabled("No .gltf or .glb files in Resources/Models.");
  }
  if (ImGui::Button("Rescan models") && ui.onRescanModels)
    ui.onRescanModels();
  ImGui::EndDisabled();
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
    // The sun and fog are saved with the scene, so they're undo steps too.
    // Each edit goes into a copy that's kept only if the parser would read
    // it back; Ctrl+click typing isn't clamped otherwise, and the colour
    // fields take negatives.
    constexpr ImGuiSliderFlags kClamp = ImGuiSliderFlags_AlwaysClamp;
    SceneSun sun = ui.sun;
    bool sunChanged = false;
    glm::vec3 dir = sun.direction;
    if (ImGui::DragFloat3("Dir##sun", &dir.x, 0.01f, -1.0f, 1.0f, "%.3f",
                          kClamp) &&
        glm::length(dir) > 0.001f) {
      sun.direction = glm::normalize(dir);
      sunChanged = true;
    }
    sunChanged |= ImGui::ColorEdit3("Color##sun", &sun.color.x);
    sunChanged |= ImGui::SliderFloat("Intensity##sun", &sun.intensity, 0.0f,
                                     5.0f, "%.3f", kClamp);
    if (sunChanged && !problemWith(sun)) {
      ui.sun = sun;
      sceneEditOpen = true;
    }
    ImGui::EndDisabled();
  }
  if (ImGui::CollapsingHeader("Fog", ImGuiTreeNodeFlags_DefaultOpen)) {
    constexpr ImGuiSliderFlags kClamp = ImGuiSliderFlags_AlwaysClamp;
    SceneFog fog = ui.fog;
    bool fogChanged = false;
    fogChanged |= ImGui::SliderFloat("Density##fog", &fog.density, 0.0f,
                                     0.02f, "%.4f", kClamp);
    fogChanged |= ImGui::SliderFloat("Falloff##fog", &fog.falloff, 0.0f, 1.0f,
                                     "%.2f", kClamp);
    fogChanged |=
        ImGui::SliderFloat("Max##fog", &fog.max, 0.0f, 1.0f, "%.2f", kClamp);
    if (fogChanged && !problemWith(fog)) {
      ui.fog = fog;
      sceneEditOpen = true;
    }
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
  ImGui::Text("CPU ui/phys/upd/rec/sub: %.2f | %.2f | %.2f | %.2f | %.2f ms",
              ui.metrics.getCpuPhaseTimeMs(PerformanceMetrics::CpuPhase::Ui),
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

void ImGuiLayer::buildFrame(DebugUiContext &ui) {
  if (!initialized)
    return;
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  buildUi(ui);
  ImGui::Render();
}

void ImGuiLayer::record(VkCommandBuffer cmd, VkRenderPass renderPass,
                        VkExtent2D extent, uint32_t imageIndex,
                        const std::function<void(VkCommandBuffer)> &beforeUi) {
  if (!initialized)
    return;
  if (imageIndex >= framebuffers.size())
    throw std::runtime_error("Invalid ImGui framebuffer index");

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

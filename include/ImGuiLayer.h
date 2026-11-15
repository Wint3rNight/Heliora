#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <functional>
#include <string>
#include <vector>

#include "PerformanceMetrics.h"
#include "PhysicsWorld.h"
#include "Scene.h"
#include "Utilities.h"
#include "VulkanDevice.h"

// The Physics panel. The panel sets the requests, the renderer acts on them
// and fills in the counts every frame.
struct PhysicsPanel {
  bool paused = false;
  bool stepRequested = false;  // one step while paused
  bool showWireframes = false; // also F2
  uint32_t bodies = 0;
  uint32_t awakeBodies = 0;
  uint32_t droppedContactSteps = 0;
  uint32_t thrownBalls = 0;

  // Physics settings, saved in the settings file. The renderer hands
  // config's live values to the world when settingsChanged is set; the
  // limits, threads and default friction wait for the next scene load.
  PhysicsWorld::Config config;
  int stepsPerSecond = 60;
  int maxCatchUpSteps = 4; // steps a slow frame may run before time slips
  float defaultFriction = 0.2f;
  bool settingsChanged = false;
};

// A question that has to be answered before something can happen: unsaved
// changes before quitting or switching scenes, the name for Save as, or
// whether Save as may replace a file. The renderer sets it and acts on the
// answer; the UI shows it as a modal window.
struct UiPrompt {
  enum class Kind { None, UnsavedQuit, UnsavedSwitch, SaveAs, Replace };
  Kind kind = Kind::None;
  std::string name;             // the scene, or the file Replace is about
  bool unsavedScene = false;    // quitting: what isn't saved
  bool unsavedSettings = false;
  std::string error;            // why the last answer didn't work
};
enum class PromptAnswer { Save, DontSave, Replace, Cancel };

struct DebugUiContext {
  PerformanceMetrics &metrics;
  VmaAllocator allocator = VK_NULL_HANDLE;
  SceneUniformBuffer &sceneUbo;

  glm::vec3 &cameraPos;
  float &cameraSpeed;
  float &cameraFov;
  float &drawDistance;
  float &lodNear;
  float &lodFar;
  float &pointShadowFar;
  SceneSun &sun; // the current scene's, so edits are saved with it
  SceneFog &fog;
  int &debugMode;
  float &specAAVariance;
  float &specAAThreshold;
  float &iblRoughnessFloor;
  bool &shadowFrontFaceCull;
  bool &cullShadowCasters;
  float &csmFar;
  float &iblIntensity;
  float &exposureEV;
  float &minSurfaceRoughness;
  float &skyOcclusionFloor;
  float &ssgiIntensity;
  int &ssgiSamples;
  bool &enableSunDirect;
  bool &enablePointLights;
  bool &enableSpotLights;
  bool &enableIblAmbient;
  bool &enableSsgiBounce;
  bool &enableBloom;
  float &bloomThreshold;
  float &bloomIntensity;
  float &bloomRadius;
  bool &ssrEnabled;
  bool &taaEnabled;
  bool &responsiveTaa;
  bool &preferMailboxPresent;
  float &sharpness;
  float &edgeAA;
  float &alphaDither;
  float &shadowSoftness;
  float &contactGrounding;
  bool &dayNightEnable;
  float &dayNightSpeed;
  float &dayNightHour;
  float &normalStrength;
  bool &useGeomNormalOnly;
  bool &gpuDrivenEnabled;
  bool &hzbCullingEnabled;
  int &gpuDrivenMinCandidates;
  bool &threadedGBufferEnabled;
  bool &autoExposureEnabled;

  // Scene picker: display names of discovered .scene files and the active
  // selection. Selection changes go through onSceneSelected, which defers
  // the actual switch to the top of the next frame.
  const std::vector<std::string> *sceneNames = nullptr;
  int activeSceneIndex = -1;

  float autoExposureAdaptedValue = 1.0f;
  MaterialProbeResult materialProbe = {};
  uint32_t gpuDrivenCandidateCount = 0;
  uint32_t gpuDrivenMeshCount = 0;
  bool gpuDrivenLastFrameUsed = false;
  uint32_t threadedGBufferWorkers = 0;
  VkPresentModeKHR activePresentMode = VK_PRESENT_MODE_FIFO_KHR;

  std::function<void()> onProjectionChanged;
  std::function<void()> onPointShadowChanged;
  std::function<void()> onTaaHistoryInvalidated;
  std::function<void(bool)> onPresentModeChanged;
  std::function<void(int)> onSceneSelected;

  PhysicsPanel *physics = nullptr; // no panel when null
  // The live physics world's bodies and limit, for the grid count check.
  uint32_t bodiesInWorld = 0;
  uint32_t bodyLimit = 0;

  // The Save settings and Reset to defaults buttons, for one section of the
  // settings file: "render" or "physics".
  std::function<void(const char *)> onSaveSettings;
  std::function<void(const char *)> onResetSettings;
  const std::string *settingsStatus = nullptr; // the last button's result

  // The toolbar's Play/Stop button, and why the last scene pick failed.
  bool playing = true;
  std::function<void()> onTogglePlay;
  const std::string *sceneError = nullptr;

  // The Inspector: the scene it edits and the selection, which the renderer
  // only passes in edit mode. Moving takes effect at once; other changes,
  // Delete and Split ask the renderer to rebuild that entry.
  SceneDescription *scene = nullptr;
  std::optional<SceneEntryRef> selection;
  float selectionMass = 0.0f; // kg, one copy for a grid; 0 without a body
  std::function<void(SceneEntryRef)> onEntryMoved;
  std::function<void(SceneEntryRef)> onEntryChanged;
  std::function<void(SceneEntryRef)> onEntryDeleted;
  std::function<void(SceneEntryRef)> onGridSplit;

  // The gizmo's camera: the unjittered projection times view, the one
  // picking uses, so the handles sit where the object is on screen.
  glm::mat4 viewProj{1.0f};

  // An edit to the scene (a drag, a typed value, the sun or fog) is done:
  // nothing is held any more, so it's one undo step.
  std::function<void()> onEditFinished;

  // The Scene tab's spawn list: the model files under Resources/Models, and
  // what was picked, which the renderer adds at the top of the next frame.
  const std::vector<std::string> *modelFiles = nullptr;
  std::function<void(SceneModelEntry)> onSpawn;
  std::function<void()> onRescanModels;

  // Saving: the scene's name and the toolbar's unsaved marker, the Save and
  // Save as buttons, the last save's result, and the question on screen
  // (text is the name typed for Save as).
  const std::string *sceneName = nullptr;
  bool sceneUnsaved = false;
  std::function<void()> onSave;
  std::function<void()> onSaveAs;
  const std::string *saveStatus = nullptr;
  const UiPrompt *prompt = nullptr;
  std::function<void(PromptAnswer, const std::string &text)> onPromptAnswer;
};

class ImGuiLayer {
public:
  void init(GLFWwindow *window, VulkanDevice &device, VkRenderPass renderPass,
            uint32_t imageCount);
  void cleanup(VkDevice device);

  void createFramebuffers(VkDevice device, VkRenderPass renderPass,
                          VkExtent2D extent,
                          const std::vector<SwapChainImage> &images);
  void cleanupFramebuffers(VkDevice device);

  // beforeUi draws inside the same render pass, under the panels (debug
  // lines use it).
  void record(VkCommandBuffer cmd, VkRenderPass renderPass, VkExtent2D extent,
              uint32_t imageIndex, DebugUiContext &ui,
              const std::function<void(VkCommandBuffer)> &beforeUi = {});

  bool wantsMouse() const;
  // A text box has the keyboard, so typed letters aren't game keys.
  bool wantsTextInput() const;
  // The cursor is on one of the gizmo's handles, or dragging one, so a left
  // click there isn't a click to select.
  bool gizmoHasMouse() const { return gizmoHovered || gizmoDrag.active; }
  // A field or a handle is being dragged, so undo would fight it.
  bool editInProgress() const;

  // Screenshot mode: hides every panel (empty ImGui draw list) so captures
  // are HUD-free. Rendering itself is unaffected.
  void toggleVisibility() { visible = !visible; }
  bool isVisible() const { return visible; }

private:
  enum class SidebarTab {
    Scene,
    Render,
    Lighting,
    Debug,
    Physics,
    Performance,
    Camera
  };

  void buildUi(DebugUiContext &ui);
  void drawTabPills();
  void drawToolbar(DebugUiContext &ui);
  void drawInspector(DebugUiContext &ui);
  void drawGizmo(DebugUiContext &ui);
  void endGizmoDrag(DebugUiContext &ui);
  void finishSceneEdit(DebugUiContext &ui);
  void drawPrompt(DebugUiContext &ui);
  void drawSceneTab(DebugUiContext &ui);
  void drawRenderTab(DebugUiContext &ui);
  void drawLightingTab(DebugUiContext &ui);
  void drawDebugTab(DebugUiContext &ui);
  void drawPhysicsTab(DebugUiContext &ui);
  void drawSettingsButtons(DebugUiContext &ui, const char *section);
  void drawPerformanceTab(DebugUiContext &ui);
  void drawCameraTab(DebugUiContext &ui);

  SidebarTab activeTab = SidebarTab::Performance;

  // A change that rebuilds the entry waits until no field is held, so a
  // drag rebuilds once, when it ends.
  bool inspectorRebuildPending = false;
  SceneEntryRef inspectorPendingRef;
  SceneEntryRef inspectorShownRef;
  std::string inspectorProblem; // why the last edit was refused
  // The scene was edited and the edit isn't finished yet (something is
  // still held), so the undo step waits.
  bool sceneEditOpen = false;
  // The prompt on screen, so a new one opens the modal once, and the name
  // typed for Save as, kept while Replace asks.
  UiPrompt::Kind promptShown = UiPrompt::Kind::None;
  // The frame the question last opened or changed on. The Esc press that
  // opened it (main.cpp's quit key) reaches ImGui on that same frame.
  int promptOpenedFrame = -1;
  char saveAsName[64] = {};
  bool focusSaveAsName = false;

  // The gizmo: handles on the selected object, in edit mode. Keys 1, 2, 3
  // or the toolbar pick the mode; Ctrl or the toolbar's Snap box snaps.
  enum class GizmoMode { Move, Rotate, Scale };
  GizmoMode gizmoMode = GizmoMode::Move;
  bool gizmoSnap = false;
  bool gizmoHovered = false; // last frame, which is what a click goes by
  // Last frame's selection: a click that selects an object doesn't also
  // grab the handles that only appear with it.
  std::optional<SceneEntryRef> gizmoLastSelection;
  // What a drag started from. Each frame works out the new placement from
  // these and the cursor, rather than adding up small steps, so rounding
  // doesn't drift and snapping stays on its steps.
  struct GizmoDrag {
    bool active = false;
    int axis = 0; // x, y, z; the scale handle has none
    SceneEntryRef ref;
    NodeTransform start;
    float startLayoutRotation = 0.0f; // a grid's
    glm::vec3 grabbed{0.0f};      // move: the point on the axis grabbed
    float lastAngle = 0.0f;       // rotate: radians, last frame
    float turned = 0.0f;          // rotate: radians, can pass a full turn
    glm::vec2 grabbedPixel{0.0f}; // scale
    glm::vec2 centrePixel{0.0f};  // scale
    bool scaled = false;          // the body needs rebuilding at the end
  };
  GizmoDrag gizmoDrag;
  GLFWwindow *window = nullptr; // to tell if the cursor is free

  bool visible = true;

  std::vector<VkFramebuffer> framebuffers;
  float frameTimeGraphData[128] = {};
  int frameTimeGraphOffset = 0;
  bool initialized = false;
};

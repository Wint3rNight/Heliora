#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <filesystem>
#include <map>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "AssetImport.h"
#include "CharacterAnimator.h"
#include "CharacterMovement.h"
#include "Controls.h"
#include "Outliner.h"
#include "PerformanceMetrics.h"
#include "Picking.h"
#include "PhysicsWorld.h"
#include "Scene.h"
#include "SceneViewport.h"
#include "ThirdPersonCamera.h"
#include "Utilities.h"
#include "VulkanDevice.h"

// The Physics panel. The panel sets the requests, the renderer acts on them
// and fills in the counts every frame.
// The Sound section of the Render tab (step 1.5c): the player's volume and
// mute, saved in settings.cfg, and what the mixer reports while playing.
struct SoundPanel {
  float volume = 1.0f;
  bool mute = false;
  float windowTop = 0.0f; // dB: the loudest sound heard lately
  int voices = 0;
  float ringing = 0.0f;   // seconds the ears still ring
  bool open = false;      // a sound device or file is running
};

struct PhysicsPanel {
  bool paused = false;
  bool stepRequested = false;  // one step while paused
  bool showWireframes = false; // also F2
  bool showLoads = false;      // weight on breakables' links, in play
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

// The Character tab: the character's and its camera's tuning, saved in the
// settings file under [character]. The renderer hands them to the character
// when changed is set; the body (radius, heights, mass) waits for the next
// Play, since the capsule in the physics world was made at its size.
struct CharacterPanel {
  CharacterTuning character;
  ThirdPersonCameraTuning camera;
  AnimatorTuning animation;
  bool changed = false;
  // What the animated character is doing, for the Animation section: the
  // renderer fills it each frame while playing; no state when it isn't.
  struct PlayingClip {
    const char *name;
    float weight; // its share of the pose, 0 to 1
    float time;   // seconds into it
  };
  const char *animState = nullptr;
  std::vector<PlayingClip> playing;
  // The .character file: the tuning above (movement, animation timing,
  // body) and the Clips section's choices, saved together by Save
  // character. The files to pick clips from, and what the tab asked for.
  CharacterDescription description;
  std::vector<std::string> clipFiles;
  std::optional<ClipSlot> clipChanged; // set by the tab, applied by the renderer
  bool clipsEdited = false;   // since the file was read or saved
  bool unsaved = false;       // the tuning or the clips differ from the file
  bool saveCharacter = false; // the Save character button
  bool resetTuning = false;   // the Reset to defaults button
  std::string status;
  std::string variantNote; // the open scene's variant and what it overrides
  // Draws the animated character's bones as lines in play mode, instead of
  // its body. Not saved: a debug view (HELIORA_SHOW_SKELETON=1 starts with it).
  bool showSkeleton = false;
  bool characterLoaded = false; // whether there's a skeleton to show
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
  CharacterPanel *character = nullptr; // likewise
  ControlBindings *controls = nullptr;  // likewise
  SoundPanel *sound = nullptr;          // likewise
  // The live physics world's bodies and limit, for the grid count check.
  uint32_t bodiesInWorld = 0;
  uint32_t bodyLimit = 0;
  // Bodies the world holds for an entry now (a grid's copies), for edits
  // that change how many it needs.
  std::function<uint32_t(SceneEntryRef)> entryBodies;

  // The Save settings and Reset to defaults buttons, for one section of the
  // settings file: "render", "physics" or "character".
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
  std::optional<SceneEntryRef> selection; // the primary, the one shown
  // Everything selected (the primary last); null while playing. An edit to
  // the primary is made on every selected entry of its kind.
  const std::vector<SceneEntryRef> *selected = nullptr;
  // The Outliner: a row clicked (mode 0 alone, 1 with Ctrl, 2 with Shift
  // over the rows in the order shown), a row double-clicked, and Delete.
  std::function<void(SceneEntryRef, int, const std::vector<SceneEntryRef> &)> onSelectEntry;
  std::function<void(SceneEntryRef)> onFrameEntry;
  std::function<void()> onDeleteSelection;
  float selectionMass = 0.0f; // kg, one copy for a grid; 0 without a body
  uint32_t selectionBodies = 0; // bodies the world holds for it right now
  std::function<void(SceneEntryRef)> onEntryMoved;
  // Several moved at once: placed together, the scene updated once.
  std::function<void(const std::vector<SceneEntryRef> &)> onEntriesMoved;
  std::function<void(SceneEntryRef)> onEntryChanged;
  std::function<void(SceneEntryRef)> onEntryDeleted;
  std::function<void(SceneEntryRef)> onGridSplit;
  std::function<void()> onDuplicate; // the selection, beside itself

  // The gizmo's camera: the unjittered projection times view, the one
  // picking uses, so the handles sit where the object is on screen.
  glm::mat4 viewProj{1.0f};
  // Where the scene is in the window: the handles are worked out in its
  // pixels and drawn there, clipped to it.
  SceneViewport sceneArea;
  // Playing and aiming: the crosshair in the scene area's middle.
  bool crosshair = false;
  // A gun in his hands: the rounds in it (the chamber's and the
  // magazine's), the rounds in his pouch, its mode, a reload under way.
  bool ammo = false;
  int ammoInGun = 0;
  int ammoPouch = 0;
  const char *fireMode = ""; // "AUTO", "SEMI", or "" for a gun with one
  int grenades = -1;         // hand grenades left, or below 0 to leave out
  int charges = -1;          // demolition blocks left, the same
  bool reloading = false;
  // A prompt at the bottom middle (in reach of a weapons crate), or empty.
  const char *playPrompt = "";

  // An edit to the scene (a drag, a typed value, the sun or fog) is done:
  // nothing is held any more, so it's one undo step.
  std::function<void()> onEditFinished;

  // The Scene tab's spawn list: the model files under Resources/Models, and
  // what was picked, which the renderer adds at the top of the next frame.
  // Player start places the scene's one start, or moves it.
  const std::vector<std::string> *modelFiles = nullptr;
  std::function<void(SceneModelEntry)> onSpawn;
  // Prefabs: an object file as a placeable copy, and the Inspector's buttons.
  enum class PrefabAction { Make, Apply, Revert, Unpack };
  std::function<std::optional<SceneModelEntry>(const std::string &)> objectEntry;
  std::function<void(SceneEntryRef, PrefabAction)> onPrefabAction;
  // Dragging from the Assets panel: over the scene each frame (window
  // pixel, turned to the surface or upright), let go there, or left it.
  std::function<void(const SceneModelEntry &, glm::vec2, bool)> onAssetHover;
  std::function<void(const SceneModelEntry &, glm::vec2, bool)> onAssetDrop;
  std::function<void()> onAssetDragEnd;
  // Asset pictures: the atlas texture, and where an asset's picture sits in
  // it (u0, v0, u1, v1), or nothing yet (asking draws it soon).
  // Importing model files (dropped on the window or picked with Import),
  // and what the last import did.
  std::function<void(std::vector<std::string>)> onImportFiles;
  const std::string *assetStatus = nullptr;
  ImTextureID thumbnailAtlas = 0;
  std::function<std::optional<glm::vec4>(const SceneModelEntry &)> thumbnail;
  // Clips: a clip file's still in the atlas, the clip under the mouse (said
  // each frame it's hovered), and the live preview image with the clip it
  // shows.
  std::function<std::optional<glm::vec4>(const std::string &)> clipThumbnail;
  std::function<void(const std::string &)> onClipHover;
  ImTextureID livePreview = 0;
  const std::string *livePreviewClip = nullptr;
  // Characters: the .character files to pick from, and the variants of one
  // already read; from the Assets panel, one dragged over the scene (each
  // frame) or let go there, or double-clicked (no pixel: where the camera
  // looks); a file's picture.
  const std::vector<std::string> *characterFiles = nullptr;
  std::function<std::vector<std::string>(const std::string &)> characterVariants;
  std::function<void(const std::string &, glm::vec2)> onCharacterHover;
  std::function<void(const std::string &, std::optional<glm::vec2>)> onCharacterDrop;
  std::function<std::optional<glm::vec4>(const std::string &)> characterThumbnail;
  std::function<void()> onSpawnPlayerStart;
  std::function<void()> onRescanModels;

  // Saving: the scene's name and the toolbar's unsaved marker, the Save and
  // Save as buttons, the last save's result, and the question on screen
  // (text is the name typed for Save as).
  const std::string *sceneName = nullptr;
  bool sceneUnsaved = false;
  std::function<void()> onSave;
  std::function<void()> onSaveAs;
  std::function<void()> onQuit; // File > Quit, which asks like Esc does
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

  // Builds this frame's UI (panels, gizmo, prompts) into ImGui's draw data.
  // Runs at the start of the frame, before anything is recorded.
  void buildFrame(DebugUiContext &ui);
  // Records the draw data buildFrame made. beforeUi draws inside the same
  // render pass, under the panels (debug lines use it).
  void record(VkCommandBuffer cmd, VkRenderPass renderPass, VkExtent2D extent,
              uint32_t imageIndex,
              const std::function<void(VkCommandBuffer)> &beforeUi = {});

  bool wantsMouse() const;
  // A text box has the keyboard, so typed letters aren't game keys.
  bool wantsTextInput() const;
  // Shows a folder in the Assets panel, read afresh (after an import).
  void showAssetFolder(const std::string &folder) {
    assetFolder = folder;
    assetListings.clear();
    assetSearch[0] = '\0';
  }
  // An image ImGui can draw, sampled in this layout.
  ImTextureID addTexture(VkImageView view);
  // The cursor is on one of the gizmo's handles, or dragging one, so a left
  // click there isn't a click to select.
  bool gizmoHasMouse() const { return gizmoHovered || gizmoDrag.active; }
  // A field or a handle is being dragged, so undo would fight it.
  bool editInProgress() const;

  // Screenshot mode: hides every panel (empty ImGui draw list) so captures
  // are HUD-free. Rendering itself is unaffected.
  void toggleVisibility() { visible = !visible; }
  bool isVisible() const { return visible; }

  // Where the panel layout is saved, in ImGui's own format. Set before the
  // first frame, which is when ImGui reads it.
  void setLayoutFile(std::string path);
  // The space the docked panels left for the scene last frame, in window
  // pixels, or nothing with the UI hidden (then the scene has the window).
  std::optional<SceneViewport> layoutSceneArea() const { return centralArea; }

private:
  // The docked layout: each panel is an ImGui window, and the scene is the
  // dockspace's central node, which nothing can be docked over.
  enum class Panel {
    Scene,
    Render,
    Lighting,
    Camera,
    Inspector,
    Physics,
    Character,
    Controls,
    Performance,
    Debug,
    Assets,
    Outliner,
    Count
  };
  static constexpr int kPanelCount = static_cast<int>(Panel::Count);

  void buildUi(DebugUiContext &ui);
  void drawMenuBar(DebugUiContext &ui);
  void buildDefaultLayout(unsigned int dockspace);
  void registerPanelSettings();
  void drawInspector(DebugUiContext &ui);
  void drawOutliner(DebugUiContext &ui);
  // The other selected entries of ref's kind, for an edit to ref.
  std::vector<SceneEntryRef> othersLike(const DebugUiContext &ui, SceneEntryRef ref) const;
  void drawPlayerStartInspector(DebugUiContext &ui);
  void drawCharacterInspector(DebugUiContext &ui);
  void flushInspectorRebuild(DebugUiContext &ui);
  void commitInspectorEdit(DebugUiContext &ui, SceneEntryRef ref,
                           SceneModelEntry edited, SceneGridEntry editedGrid,
                           bool moved, bool changed);
  void drawGizmo(DebugUiContext &ui);
  void drawCrosshair(const DebugUiContext &ui) const;
  void drawAmmo(const DebugUiContext &ui) const;
  void drawPlayPrompt(const DebugUiContext &ui) const;
  void endGizmoDrag(DebugUiContext &ui);
  // The others moved, for placing them with the primary.
  std::vector<SceneEntryRef> moveOthersWithGizmo(DebugUiContext &ui, const NodeTransform &was,
                                                 const NodeTransform &now, float layoutTurn);
  // Places entries that moved: together if there are several.
  void entriesMoved(DebugUiContext &ui, const std::vector<SceneEntryRef> &refs);
  // How many more bodies the physics world can take this frame (an edit
  // that adds more is refused), and what an entry's bodies are.
  int64_t bodyRoom(const DebugUiContext &ui) const;
  void finishSceneEdit(DebugUiContext &ui);
  void drawPrompt(DebugUiContext &ui);
  void drawSceneTab(DebugUiContext &ui);
  void drawRenderTab(DebugUiContext &ui);
  void drawLightingTab(DebugUiContext &ui);
  void drawDebugTab(DebugUiContext &ui);
  void drawPhysicsTab(DebugUiContext &ui);
  void drawCharacterTab(DebugUiContext &ui);
  void drawControlsTab(DebugUiContext &ui);
  void drawAssetsPanel(DebugUiContext &ui);
  void updateAssetDrag(DebugUiContext &ui);
  void updateCharacterDrag(DebugUiContext &ui);
  void drawAssetFolderTree(const std::string &folder);
  const FolderListing &listingOf(const std::string &folder);
  void drawImportPicker(DebugUiContext &ui);
  void drawClipsSection(DebugUiContext &ui, CharacterPanel &panel);
  // A tile in the Assets panel or the clip picker: a shape or model file
  // (entry), an animation clip (clip, the file), or a character file.
  struct AssetTile {
    std::string label;
    std::string tooltip;
    SceneModelEntry entry;
    std::string clip;
    bool usable = true;
    std::string character;
  };
  // Draws tiles in rows, pictures where there are some, the hovered clip
  // playing; placing lets models be dragged in or double-clicked. The tile
  // clicked this frame, if any.
  std::optional<int> drawAssetTiles(DebugUiContext &ui, const std::vector<AssetTile> &tiles,
                                    bool placing, const std::string &chosenClip = {});
  std::optional<ClipSlot> clipPickerSlot; // the move the clip picker chooses for
  void drawSettingsButtons(DebugUiContext &ui, const char *section);
  void drawPerformanceTab(DebugUiContext &ui);
  void drawCameraTab(DebugUiContext &ui);

  // Every panel starts open, new ones included: a layout file only closes
  // the ones it says were closed.
  static constexpr std::array<bool, kPanelCount> allOpen() {
    std::array<bool, kPanelCount> open{};
    open.fill(true);
    return open;
  }
  std::array<bool, kPanelCount> panelOpen = allOpen();
  bool resetLayoutRequested = false;
  bool focusDefaultTabs = false; // once, after the default layout is built
  std::string layoutPath;        // ImGui keeps a pointer to it
  std::optional<SceneViewport> centralArea;

  // A change that rebuilds the entry waits until no field is held, so a
  // drag rebuilds once, when it ends.
  bool inspectorRebuildPending = false;
  std::vector<SceneEntryRef> inspectorPendingRefs;
  SceneEntryRef inspectorShownRef;
  std::string inspectorProblem; // why the last edit was refused
  // The Outliner: its search text and filters, and the primary it last
  // showed, so a pick in the scene scrolls the list to it.
  char outlinerSearch[128] = {};
  OutlinerFilter outlinerFilter;
  std::optional<SceneEntryRef> outlinerShownPrimary;
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
  // The selected entry and where its handles sit this frame, in the scene
  // area's own pixels: worked out once by gizmoView, shared by the steps
  // after it.
  struct GizmoView {
    SceneModelEntry *entry = nullptr;
    SceneGridEntry *grid = nullptr; // set when the entry is a grid's model
    // Or the player start or a character, which only move and turn about y.
    ScenePlayerStart *player = nullptr;
    SceneCharacterEntry *character = nullptr;
    bool justSelected = false; // selected by this frame's click
    SceneViewport area;
    glm::mat4 viewProj{1.0f};
    glm::vec3 centre{0.0f};
    glm::vec2 centrePixel{0.0f};
    float length = 0.0f; // world length of an arrow
    glm::vec2 windowMouse{0.0f};
    glm::vec2 mouse{0.0f}; // in the scene area's pixels
    PickRay ray;
    ImVec2 at(glm::vec2 p) const;
    bool axisShown(int axis, GizmoMode mode) const;
    bool scales() const { return player == nullptr && character == nullptr; }
    std::optional<glm::vec2> arrowTip(int axis) const;
    // What's selected, as a placement: the player start's yaw is a turn
    // about y and its scale is 1.
    NodeTransform transform() const;
  };
  std::optional<GizmoView> gizmoView(DebugUiContext &ui);
  int hoveredHandle(const GizmoView &view) const;
  void startGizmoDrag(DebugUiContext &ui, const GizmoView &view, int handle);
  void dragGizmo(DebugUiContext &ui, const GizmoView &view);
  void commitGizmoEdit(DebugUiContext &ui, const GizmoView &view,
                       const NodeTransform &placement, float layoutRotation);
  void drawGizmoHandles(const GizmoView &view, int hot) const;
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
  GLFWwindow *window = nullptr; // to tell if the cursor is free, and to read keys
  // The Controls tab waiting for a key for this action. Armed once nothing
  // is held, so the click that started it doesn't count.
  std::optional<Action> rebinding;
  bool rebindArmed = false;
  std::string rebindNote; // why the last key wasn't taken
  char clipFilter[64] = ""; // the Clips section's file filter
  // The Assets panel: what's being dragged (until it's dropped or let go),
  // whether drops lie along the surface, and whether an old layout file
  // has been given a place for the panel yet.
  std::optional<SceneModelEntry> assetDrag;
  std::optional<std::string> characterDrag; // a .character file instead
  bool turnToSurface = true;
  bool assetsPlaced = false;
  // The folder the Assets panel shows ("" for the built-in shapes), each
  // folder's listing until the next rescan, the search text, and the
  // Import picker's folder and choice.
  std::string assetFolder = "Resources/Models";
  std::map<std::string, FolderListing> assetListings;
  char assetSearch[64] = "";
  bool importPickerOpen = false;
  std::filesystem::path importFolder;
  std::filesystem::path importChoice;

  bool visible = true;

  std::vector<VkFramebuffer> framebuffers;
  float frameTimeGraphData[128] = {};
  int frameTimeGraphOffset = 0;
  bool initialized = false;
};

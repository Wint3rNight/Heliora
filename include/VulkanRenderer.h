#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "AutoExposurePass.h"
#include "BloomPass.h"
#include "CompositePass.h"
#include "CharacterImport.h"
#include "DebugLinePass.h"
#include "SkinningPass.h"
#include "DescriptorManager.h"
#include "EditHistory.h"
#include "FixedStepClock.h"
#include "GpuDrivenGBufferPass.h"
#include "ImGuiLayer.h"
#include "MaterialProbePass.h"
#include "Model.h"
#include "ModelManager.h"
#include "PerformanceMetrics.h"
#include "PhysicsWorld.h"
#include "Player.h"
#include "RenderResources.h"
#include "RenderPassManager.h"
#include "Scene.h"
#include "SceneNode.h"
#include "SceneViewport.h"
#include "SettingsFile.h"
#include "ShadowPass.h"
#include "SsaoPass.h"
#include "SsgiPass.h"
#include "TextureManager.h"
#include "Utilities.h"
#include "VulkanDevice.h"
#include "VulkanPipeline.h"
#include "VulkanPipelineCache.h"
#include "VulkanSwapchain.h"

struct InstancedDrawable {
  int modelId;
  std::vector<InstanceData> instances;
  AllocatedBuffer instanceBuffer;
  // World-space bounding sphere covering all instances. Used for whole-batch
  // frustum culling and LOD selection.
  glm::vec3 groupCenter = glm::vec3(0.0f);
  float groupRadius = 0.0f;
};

class CommandThreadPool;

class VulkanRenderer {
public:
  VulkanRenderer();

  using CameraPresetCallback =
      std::function<void(glm::vec3 position, float yaw, float pitch,
                         float speed)>;

  int init(GLFWwindow *newWindow);
  void draw();
  int createMeshModel(const std::string &modelFile);
  // A model for a scene entry's generated shape (box, sphere, capsule or
  // plane). Entries with the same shape, size and colour share one model.
  int createPrimitiveModel(const SceneModelEntry &entry);

  // --- Scene management ---
  // Scans `directory` for *.scene files (also checked under ../ because the
  // binary runs from build/). Call once after init(), before loadSceneAt().
  void discoverScenes(const std::string &directory = "Resources/Scenes");
  // The model files the spawn list offers.
  void discoverModels(const std::string &folder = "Resources/Models");
  bool hasScenes() const { return !availableScenes.empty(); }
  // Immediate load: waits for the device to go idle, clears the scene graph
  // and instanced drawables, then builds the described scene. Models are
  // cached by path, so returning to a scene reuses its GPU geometry.
  // moveCamera puts the camera at the scene's pose; Stop leaves it be.
  void loadScene(const SceneDescription &desc, bool moveCamera = true);
  // Reads the scene's file again first, so hand edits show up. Returns false
  // (and leaves the current scene untouched) when the index is out of range
  // or the file doesn't parse; the Scene tab shows why.
  bool loadSceneAt(int index);
  // Deferred load, safe to call from UI callbacks that run mid-record; the
  // switch happens at the top of the next draw().
  // Asks first when the current scene has unsaved changes.
  void requestSceneLoad(int index);
  // Ctrl+S saves the scene to its file, Ctrl+Shift+S asks for a name. Both
  // write the layout, so saving while playing is safe.
  void requestSave();
  void requestSaveAs();
  // Esc and the window's close button: quits at once with nothing unsaved,
  // otherwise asks. The main loop runs until quitApproved().
  void requestQuit();
  bool quitApproved() const { return quitting; }
  // A question is on screen, so game keys wait.
  bool promptOpen() const { return prompt.kind != UiPrompt::Kind::None; }
  // Runs the fixed physics steps this frame owes, then moves the nodes of
  // bodies that moved to their interpolated poses. Call once per frame
  // before updateScene() with the frame time in seconds.
  void stepPhysics(float frameSeconds);
  // F2: wireframes of the physics shapes over the frame.
  void togglePhysicsDebug() {
    physicsPanel.showWireframes = !physicsPanel.showWireframes;
  }
  // F: throws a ball from the camera along direction. Play mode only.
  void throwBall(glm::vec3 from, glm::vec3 direction);
  // Advances node animations (in play mode) and recomputes global
  // transforms. Call once per frame before draw() with the frame time in
  // seconds.
  void updateScene(float frameSeconds);

  // Edit mode: physics and animations stand still and everything sits where
  // the layout puts it. Play mode: they run from the layout. The game starts
  // in play mode.
  bool isPlaying() const { return playing; }
  // F5 and the toolbar button. Stop rebuilds the scene from the layout,
  // without moving the camera, at the top of the next draw().
  void togglePlayMode() { playToggleRequested = true; }
  // Play mode's keys and mouse for this frame; stepPhysics hands them to the
  // character.
  void setPlayerControls(const PlayerControls &controls) {
    playerControls = controls;
  }
  // The fly camera, every frame. With no [player] in the scene the
  // character starts on the ground below it, facing its way.
  void setFlyCamera(glm::vec3 position, float yawDegrees) {
    flyCameraPosition = position;
    flyCameraYaw = yawDegrees;
  }
  // While playing there's a character, and the view comes from its camera.
  bool hasPlayer() const { return player != nullptr; }
  // The player's keys, rebound in the Controls tab.
  const ControlBindings &controls() const { return controlBindings; }
  const ThirdPersonCamera &playerCamera() const { return player->camera(); }
  // Ctrl+Z and Ctrl+Shift+Z: one finished edit back or forward, rebuilt the
  // way Stop rebuilds, at the top of the next draw(). Edit mode only, and
  // not in the middle of a drag.
  void undoEdit() { requestHistoryStep(HistoryStep::Undo); }
  void redoEdit() { requestHistoryStep(HistoryStep::Redo); }
  // Ctrl+D and the Inspector's Duplicate: a copy of the selected object or
  // grid, beside it on the camera's right, then selected. Edit mode only,
  // not in the middle of a drag; made at the top of the next draw(), like a
  // spawn.
  void duplicateSelection() {
    if (!playing && selection && !imguiLayer.editInProgress())
      duplicateRequested = true;
  }
  // Left click in edit mode: selects what's under the pixel (window
  // coordinates), or nothing. A click on any copy in a grid selects the grid.
  // A click outside the scene's area does nothing.
  void selectAt(glm::vec2 windowPixel);
  // Whether a window pixel is on the scene, not on a panel around it.
  bool sceneContains(glm::vec2 windowPixel) const {
    return sceneArea.contains(windowPixel);
  }

  SceneNode &getRootNode();
  void updateCameraView(const glm::mat4 &viewMatrix,
                        const glm::vec3 &cameraPosition);
  void addInstancedModel(int modelId, const std::vector<glm::mat4> &transforms);
  void cleanup();
  void notifyResize();

  void setImGuiCameraInfo(glm::vec3 pos, float speed);
  void setFov(float fovDegrees);
  void setDrawDistance(float dist);
  void setCameraPresetCallback(CameraPresetCallback callback);
  float getCameraSpeed() const { return imguiCameraSpeed; }
  bool imguiWantsMouse() const { return imguiLayer.wantsMouse(); }
  bool imguiWantsTextInput() const { return imguiLayer.wantsTextInput(); }
  bool gizmoHasMouse() const { return imguiLayer.gizmoHasMouse(); }
  void toggleUiVisibility() { imguiLayer.toggleVisibility(); }
  // HELIORA_CAPTURE: copies the scene's finished picture in the next draw(),
  // and finishCapture() saves it as a PNG once that frame is done. True
  // when it has (or failed and logged why).
  void requestCapture(const std::string &path);
  bool finishCapture();
  bool isUiVisible() const { return imguiLayer.isVisible(); }

  ~VulkanRenderer();

private:
  GLFWwindow *window = nullptr;
  int currentFrame = 0;

  SceneNode rootNode;
  SceneUniformBuffer sceneUbo = {};

  // Physics for the loaded scene. loadScene() makes a new world, so a scene
  // switch starts clean. A body's userData indexes physicsLinks, the node it
  // moves.
  std::unique_ptr<PhysicsWorld> physics;
  FixedStepClock physicsClock;
  // The character, while playing. After physics, so it goes first: it takes
  // its character out of the world. loadScene() removes it before it
  // replaces the world, and puts a new one in if still playing.
  std::unique_ptr<Player> player;
  PlayerControls playerControls;
  SceneNode *playerNode = nullptr; // its capsule, owned by rootNode
  int playerModelId = -1;          // standing, and crouched below
  int crouchedPlayerModelId = -1;
  glm::vec3 flyCameraPosition{0.0f};
  float flyCameraYaw = -90.0f; // degrees, Camera's convention
  static SceneModelEntry playerCapsuleEntry(float radius, float height);
  // The player's look and moves, from Resources/Characters/ganfaul.character,
  // read once at startup. Without its files there's none, and the capsule
  // stays.
  std::optional<CharacterAsset> characterAsset;
  std::string characterPath; // the .character file it came from, as found
  // The tuning as the character file last had it, to tell unsaved edits.
  std::pair<CharacterTuning, AnimatorTuning> savedTuning;
  std::optional<std::string> notedVariant; // the variant variantNote describes
  // The file each move's clip was read from, so the Clips section only
  // re-reads a clip when its file changes.
  std::array<std::string, static_cast<size_t>(ClipSlot::Count)> loadedClipFiles;
  int characterModelId = -1; // made at the first scene load, kept after
  double characterClipTime = 0.0; // seconds into the clip playing
  // This frame's skinning matrices, worked out while posing and uploaded
  // in draw() once the frame slot's fence says it's free.
  std::vector<glm::mat4> characterSkinning;
  std::vector<PhysicsDebugVertex> skeletonLines; // Show skeleton, play mode
  // Picks and blends the character's clips from what the controller does.
  // Made with the character at Play, gone with it.
  std::unique_ptr<CharacterAnimator> animator;
  SkinningPass skinningPass;
  void loadCharacter();
  std::array<RootMotion, 5> characterDodgePaths() const;
  void applyClipChange(ClipSlot slot);
  bool saveCharacterFile();
  std::pair<CharacterTuning, AnimatorTuning> playedTuning() const;
  void updateCharacterUnsaved();
  void updateVariantNote();
  void createCharacterModel();
  // The scale from the model file's units to the spawned body's height.
  float characterScale() const;
  // Each bone in the skeleton's space, for this frame.
  std::vector<glm::mat4> characterPose() const;
  // The character's skeleton as lines where it's drawn this frame.
  void updateSkeletonLines(const glm::mat4 &placement,
                           const std::vector<glm::mat4> &model);
  // The capsules for the Character tab's body sizes, made if they aren't
  // yet. A new mesh waits for the GPU to go idle first.
  void ensurePlayerModels();
  void spawnPlayer();
  void posePlayer(float alpha, float frameSeconds);
  bool playing = true;
  bool playToggleRequested = false;
  // Animation time. Runs only while playing, and goes back to 0 with the
  // layout. double for the same reason as main.cpp's frame clock.
  double playSeconds = 0.0;
  // What's selected in edit mode. Its outline and the player start's are
  // the editor's lines for DebugLinePass, worked out when the selection or
  // the layout changes, not every frame.
  std::optional<SceneEntryRef> selection;
  std::vector<PhysicsDebugVertex> editorLines;
  std::vector<PhysicsDebugVertex> debugLineScratch; // wireframes + outline
  float selectionMass = 0.0f; // kg, of one copy for a grid; 0 if no body
  // Bodies the world holds for the selection now (a grid's copies), which
  // a drag of its Count runs ahead of until the rebuild.
  uint32_t selectionBodies = 0;
  void updateEditorLines();
  bool entryHasBody(SceneEntryRef ref) const;

  // Building and changing single entries of currentScene, so an edit never
  // reloads the whole scene.
  int modelFor(const SceneModelEntry &entry);
  void addEntryNode(const SceneModelEntry &entry, int modelId,
                    SceneEntryRef ref);
  void addEntry(SceneEntryRef ref);        // its nodes and bodies
  void removeEntryNodes(SceneEntryRef ref); // and its bodies
  void placeEntry(SceneEntryRef ref); // moves them to the entry's placement
  // Moves node entry indices of kind from `from` on by `by`, after entries
  // before them were added or removed.
  void shiftEntryRefs(SceneEntryRef::Kind kind, int from, int by);
  // Edits that rebuild nodes or bodies wait for the top of draw(), where
  // nothing in flight can be using them and a new mesh can be uploaded.
  enum class EntryOp { Rebuild, Delete, Split };
  std::vector<std::pair<EntryOp, SceneEntryRef>> pendingEntryOps;
  void applyEntryOp(EntryOp op, SceneEntryRef ref);
  // A copy of currentScene after every finished edit, for undo. Loading
  // another scene starts a new history; Stop and undo keep it.
  EditHistory history;
  enum class HistoryStep { None, Undo, Redo };
  HistoryStep historyRequest = HistoryStep::None;
  void requestHistoryStep(HistoryStep step);
  void applyHistoryStep();
  void recordEdit() {
    history.push(currentScene);
    updateUnsaved();
  }
  // The spawn list: what it offers, and what was picked, added at the top of
  // the next draw() since a new model may need a GPU upload.
  std::vector<std::string> modelFiles;
  std::optional<SceneModelEntry> pendingSpawn;
  void spawnEntry(SceneModelEntry entry);
  void addNewEntry(SceneModelEntry entry);
  // Dragging from the Assets panel: this frame's hover or drop, and the
  // preview node showing where it would land.
  struct AssetPlacement {
    SceneModelEntry entry;
    glm::vec2 windowPixel{0.0f};
    bool turnToSurface = true;
  };
  std::optional<AssetPlacement> assetHover;
  std::optional<AssetPlacement> assetDrop;
  bool assetDragEnded = false;
  SceneNode *assetPreview = nullptr;
  int assetPreviewModel = -1;
  NodeTransform dropPlacement(const SceneModelEntry &entry, int modelId, glm::vec2 windowPixel,
                              bool turnToSurface);
  void showAssetPreview(const SceneModelEntry &entry, glm::vec2 windowPixel, bool turnToSurface);
  void hideAssetPreview();
  void dropAsset(SceneModelEntry entry, glm::vec2 windowPixel, bool turnToSurface);
  bool pendingPlayerStart = false; // the list's Player start, same timing
  void placePlayerStart();
  bool duplicateRequested = false; // Ctrl+D, same timing
  void applyDuplicate();
  // A removed body leaves an empty slot, so the others keep their index.
  struct PhysicsLink {
    SceneNode *node = nullptr;
    PhysicsBodyId body;
  };
  std::vector<PhysicsLink> physicsLinks;
  DebugLinePass debugLinePass;
  PhysicsPanel physicsPanel;
  // The character's and its camera's tuning (the Character tab). The player
  // start's outline and clicks use its body sizes too.
  CharacterPanel characterPanel;
  ControlBindings controlBindings;
  // Sums for the [physics] log line, printed every 120 frames.
  struct PhysicsLogWindow {
    int frames = 0;
    int steps = 0;
    double stepMs = 0.0; // inside Jolt's update
    double poseMs = 0.0; // copying moved poses into nodes
  } physicsLog;
  int ballModelId = -1; // made at scene load, so throwing never builds a mesh
  static SceneModelEntry thrownBallEntry();
  // The camera's near plane, shared by the projection and the debug lines.
  // 10 cm, so a third-person camera close to a wall or the character doesn't
  // cut into them. Reversed-Z keeps far depth precise with it this close.
  static constexpr float kCameraNear = 0.1f;

  // --- Settings (Resources/settings.cfg, shared by every scene) ---
  std::string settingsPath;
  SettingsFile settingsFile;     // the file as last read or saved
  SettingsFile settingsDefaults; // the code defaults, for Reset to defaults
  std::vector<SettingBinding> settingBindings;
  std::string settingsStatus; // shown under the settings buttons
  void loadSettings();
  bool saveSettings(std::string_view section); // false if it failed
  void resetSettings(std::string_view section);

  // --- Scene library ---
  std::vector<SceneDescription> availableScenes;
  // The loaded scene as it stands now. The Lighting tab's sun and fog sliders
  // edit this copy, not the one read from the file.
  SceneDescription currentScene;
  std::vector<std::string> availableSceneNames; // display names for ImGui
  int activeSceneIndex = -1;
  int pendingSceneIndex = -1;   // set by the UI, applied at next draw()
  std::string sceneError; // why the last scene pick failed, for the UI

  // --- Saving the scene ---
  // The scene as last loaded or saved: unsaved changes are any difference.
  // Checked when the scene changes, not every frame.
  SceneDescription savedScene;
  bool sceneUnsaved = false;
  void updateUnsaved() { sceneUnsaved = currentScene != savedScene; }
  std::string scenesDirectory; // where Save as puts new scenes
  std::string saveStatus;      // the last save's result, for the Scene tab
  UiPrompt prompt;             // the question on screen, if any
  int promptSceneIndex = -1;   // the switch waiting on the prompt
  std::string replaceName, replacePath; // the Save as waiting on Replace
  bool quitting = false;
  bool saveScene(); // to currentScene.filePath; false if it failed
  bool saveSceneAs(const std::string &path, const std::string &name);
  void answerPrompt(PromptAnswer answer, const std::string &text);
  std::vector<std::string> unsavedSettingsSections() const;
  std::string activeEnvironmentHdr; // HDRI backing the current sky/IBL
  VkFormat shadowDepthFormat = VK_FORMAT_UNDEFINED;

  // --- G-buffer (per swapchain image) ---
  VkFormat gBufferDepthFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage> gBuffer0Images, gBuffer1Images, gBuffer2Images,
      gBufferDepthImages;
  std::vector<ImageViewHandle> gBuffer0Views, gBuffer1Views, gBuffer2Views,
      gBufferDepthViews;
  std::vector<VkFramebuffer> gBufferFramebuffers;

  // --- Lit buffer (post-PBR HDR, sampled by SSR composite pass) ---
  VkFormat litFormat = VK_FORMAT_UNDEFINED;
  std::vector<AllocatedImage> litImages;
  std::vector<ImageViewHandle> litViews;
  std::vector<VkFramebuffer> litFramebuffers;
  VkSampler litSampler = VK_NULL_HANDLE;

  // --- TAA history (HDR, persistent across frames) ---
  // Same ring sizing as SSGI: MAX_FRAMES_DRAWS + 1 avoids overwriting a
  // history image that a still-in-flight frame is sampling.
  std::vector<AllocatedImage> taaHistoryImages;   // size MAX_FRAMES_DRAWS + 1
  std::vector<ImageViewHandle> taaHistoryViews;   // size MAX_FRAMES_DRAWS + 1
  // Sampler used by the TAA shader to read history-prev and depth.
  // CLAMP_TO_EDGE so reprojected UVs that drift just past the edge fall
  // back gracefully (treated as off-screen by the shader's bounds check).
  VkSampler taaSampler = VK_NULL_HANDLE;
  bool autoExpEnabled = true;   // ImGui toggle; off -> manual EV only.

  // --- IBL resources ---
  int iblSkyboxImageIndex = -1; // index into TextureManager::textureImages
  int irradianceImageIndex = -1;
  int prefilteredEnvImageIndex = -1;
  int brdfLutImageIndex = -1;
  int ssaoNoiseImageIndex = -1;
  VkSampler iblSampler = VK_NULL_HANDLE;
  VkSampler ssaoNoiseSampler = VK_NULL_HANDLE;

  // --- Instanced drawables ---
  std::vector<InstancedDrawable> instancedDrawables;
  // This frame's meshes with their world bounds (collectMeshInstances):
  // opaque, and blended for the transparent pass.
  std::vector<MeshInstance> frameMeshes;
  std::vector<MeshInstance> frameBlendedMeshes;

  // --- GPU-driven rendering controls (Phase 7.6) ---
  bool imguiGpuDrivenEnabled = true;
  bool imguiHzbCullingEnabled = true;
  int imguiGpuDrivenMinCandidates = 256;
  bool imguiThreadedGBufferEnabled = false;

  // --- Multi-threaded command recording (Phase 7.4) ---
  struct ThreadCommandFrameResources {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer gBufferSecondary = VK_NULL_HANDLE;
  };
  std::vector<std::vector<ThreadCommandFrameResources>> threadedCommandFrames;
  std::unique_ptr<CommandThreadPool> commandThreadPool;
  uint32_t threadedCommandWorkerCount = 0;

  // --- Subsystems ---
  VulkanDevice device;
  VulkanSwapchain swapchain;

  // The scene has its own size and place in the window (step 1.1c): its
  // images are sceneExtent, and each frame its finished picture is copied
  // into sceneArea, stretched while the area is being resized.
  VkExtent2D sceneExtent = {};
  SceneViewport sceneArea;
  std::optional<SceneResizeTracker> sceneResize;
  VkFormat sceneColorFormat = VK_FORMAT_UNDEFINED; // HDR, between SSR and TAA
  std::vector<AllocatedImage> sceneColorImages;    // one per swap image
  std::vector<ImageViewHandle> sceneColorViews;
  std::vector<AllocatedImage> sceneOutputImages; // the finished picture
  // A requested capture: Requested until the frame records the copy of the
  // window into captureBuffer, after the UI pass, Recorded until
  // finishCapture writes it out.
  enum class CaptureState { None, Requested, Recorded };
  CaptureState captureState = CaptureState::None;
  std::string capturePath;
  AllocatedBuffer captureBuffer;
  VkExtent2D captureExtent = {};
  void recordCapture(VkCommandBuffer cmd, uint32_t imageIndex);
  std::vector<ImageViewHandle> sceneOutputViews;
  RenderPassManager renderPassManager;
  DescriptorManager descriptorManager;
  VulkanPipeline pipeline;
  VulkanPipelineCache pipelineCache;
  TextureManager textureManager;
  ModelManager modelManager;
  PerformanceMetrics metrics;
  RenderResources renderResources;
  GpuDrivenGBufferPass gpuDrivenGBufferPass;
  MaterialProbePass materialProbePass;
  BloomPass bloomPass;
  AutoExposurePass autoExposurePass;
  ShadowPass shadowPass;
  // The point-shadow cube is created UNDEFINED and only its render pass moves
  // it to the layout lit.frag's binding expects, so it must render at least
  // once even if no shadowed point light is active (see recordShadowCommands).
  bool pointShadowMapInitialized = false;
  SsaoPass ssaoPass;
  SsgiPass ssgiPass;
  CompositePass compositePass;

  // --- Synchronization ---
  // imageAvailable & drawFences are sized by MAX_FRAMES_DRAWS (frames in
  // flight). renderFinished is sized by swapchain image count and indexed
  // by the acquired imageIndex — the presentation engine binds its wait
  // to the swap image, not to our frame-in-flight slot. Indexing this by
  // currentFrame caused a validation hit when the swap engine had not
  // finished presenting the image whose semaphore we tried to reuse.
  // imagesInFlight maps swap image -> frame fence. frameImageInFlight is the
  // reverse owner map, used to clear stale image entries when a frame fence is
  // reused for a different acquired image.
  // See https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkSemaphore> asyncComputeTimeline;
  std::vector<uint64_t> asyncComputeTimelineValue;
  std::vector<VkFence> drawFences;
  std::vector<VkFence> imagesInFlight;
  std::vector<uint32_t> frameImageInFlight;

  // Split graphics/compute command buffers for Phase 7.5 async SSAO.
  // swapchain.getCommandBuffer(image) records the G-buffer submit; these
  // additional primaries record shadow work and post-lighting work.
  std::vector<VkCommandBuffer> shadowCommandBuffers;
  std::vector<VkCommandBuffer> postCommandBuffers;
  std::vector<VkCommandBuffer> ssaoCommandBuffers;
  bool framebufferResized = false;

  // --- ImGui ---
  ImGuiLayer imguiLayer;
  CameraPresetCallback cameraPresetCallback;
  glm::vec3 imguiCameraPos = {};
  float imguiCameraSpeed = 15.0f;
  float imguiCameraFov = 45.0f;
  // Metric world since node-transform baking: Sponza is ~30 m long, so a
  // 200 m far plane covers everything with depth precision to spare.
  float imguiDrawDistance = 200.0f;
  float imguiLodNear = 15.0f;
  float imguiLodFar = 45.0f;
  float imguiPointShadowFar = 40.0f;
  int imguiDebugMode = 0;
  float imguiSpecAAVariance = 1.60f;       // tunable
  // Karis-style spec-AA threshold cap. 0.18 (the plan's KAPPA) leaves
  // visible per-pixel sparkle on Sponza's dense cloth weave / foliage at
  // glancing angles. Bumping to 0.5 lets the kernel saturate more, folding
  // sub-pixel normal variance into much higher effective roughness on
  // high-frequency surfaces while leaving low-variance surfaces unchanged.
  float imguiSpecAAThreshold = 1.0f;       // tunable
  // IBL specular mip floor. This only affects global cubemap specular, not
  // direct-light highlights. Keeping it moderately high prevents the
  // environment map's bright texels from reading as rectangular local lights
  // on glossy indoor floors.
  float imguiIblRoughnessFloor = 0.65f;    // tunable
  bool imguiShadowFrontFaceCull = false;   // off → CULL_NONE in shadow pass.
                                           // Front-face cull silently drops
                                           // any single-sided wall whose only
                                           // triangle faces the sun, so the
                                           // floor's shadow lookup misses the
                                           // wall as an occluder and reads as
                                           // lit. NONE captures both winding
                                           // orientations; slope-scaled bias
                                           // + receiver-plane gradient
                                           // already handle the acne that
                                           // front-cull was protecting.
  bool imguiCullShadowCasters = true;       // skip casters outside a
                                           // cascade's (or cube face's) clip
                                           // volume; each cascade reaches
                                           // csmFar towards the sun, so only
                                           // what the rasterizer would clip
                                           // goes (tests/FrustumTests.cpp).
  float imguiCsmFar = 80.0f;               // cascade far plane (metric world)
  float imguiIblIntensity = 1.0f;          // manual IBL/sky multiplier
  // Exposure stops applied BEFORE the ACES tonemap in second.frag. ACES has
  // a steep toe — without a pre-scale, midtones get crushed to black on
  // shadowed Sponza interiors. +1 to +2 stops typically matches the warm
  // reference look once the sky-occlusion proxy is also lifted.
  float imguiExposureEV = 0.0f;            // exposure stops, -3..+3
  // Min-roughness floor applied at the G-buffer write for non-metallic
  // surfaces only. Sponza glTF's roughness maps bottom out too low — that
  // produces the wet-floor banner reflection and stone-pillar glitter.
  // 0 = disabled (use authored roughness); ~0.35–0.5 is the Sponza sweet
  // spot. Metals (metallic > 0.5) are untouched so trim still reads glossy.
  float imguiMinSurfaceRoughness = 0.60f;
  // Sky-occlusion proxy floor — bottom of the IBL multiplier in shadowed
  // pixels. 0.30 = old conservative ("dead-black interior"), 0.55 = new
  // default ("warm interior"), 1.0 = unbounded (full sky everywhere).
  float imguiSkyOcclusionFloor = 0.55f;
  // SSGI (screen-space one-bounce diffuse) intensity. 0 = disabled.
  // Dropped 1.0 → 0.6 alongside the per-sample firefly clamp tightening
  // and the close-range fade: 1.0 over-emphasized the residual SSGI grain
  // on sun-lit floor pixels once auto-exposure started amplifying. 0.6
  // keeps the warm-bounce feel without the visible noise floor.
  float imguiSsgiIntensity = 0.6f;
  // SSGI sample quality. 8 is the default performance/quality balance; 12 is
  // the old high-quality path, useful for A/B checks when chasing artifacts.
  int imguiSsgiSamples = 8;
  bool imguiEnableSunDirect = true;
  // Off until lights come to the editor. The point-shadow pass only runs
  // while this is on.
  bool imguiEnablePointLights = false;
  bool imguiEnableSpotLights = true;
  bool imguiEnableIblAmbient = true;
  bool imguiEnableSsgiBounce = true;
  bool imguiEnableBloom = true;
  float imguiBloomThreshold = 0.8f;
  float imguiBloomIntensity = 0.12f;
  float imguiBloomRadius = 1.25f;
  bool imguiSsrEnabled = false;            // off by default: avoids rough Sponza SSR leaks
  bool imguiTaaEnabled = true;             // SSAO/SSGI/PCF noise relies on TAA to average out
  bool imguiResponsiveTaa = true;          // drop history while camera moves
  bool imguiPreferMailboxPresent = true;   // on = MAILBOX / uncapped present
  // CAS-style sharpening strength. Dropped 0.4 → 0.10 — the previous 0.4
  // was amplifying the residual SSGI/spec-AA noise into a visible chunky
  // pattern on the floor. 0.10 still restores TAA-softening on real edges
  // without exaggerating noise. Push back toward 0.3–0.4 only if the
  // image looks too soft after the noise-reduction pass.
  float imguiSharpness = 0.10f;
  float imguiEdgeAA = 0.35f;              // post/TAA edge cleanup
  float imguiAlphaDither = 0.45f;         // alpha-test cutout stabilization
  float imguiShadowSoftness = 1.25f;      // PCF radius multiplier
  float imguiContactGrounding = 0.45f;    // SSAO contact-strength multiplier
  bool imguiDayNightEnable = false;        // day/night animation
  float imguiDayNightSpeed = 60.0f;        // sim-hours per real-second
  float imguiDayNightHour = 12.0f;         // current sim-hour [0..24)
  float imguiNormalStrength = 0.55f;       // Sponza normals are over-sharp at 1.0
  bool imguiUseGeomNormalOnly = false;     // P2 diag: bypass normal-map sampling
  MaterialProbeResult materialProbeResult = {};
  glm::uvec2 materialProbePixel = glm::uvec2(0);

  void updateMaterialProbePixel();

  // --- TAA state ---
  // Frame counter drives the Halton index and temporal history ring slots.
  // Jitter wraps every 8 frames; history slots wrap by their ring size.
  uint32_t taaFrameCounter = 0;
  // Un-jittered baseline projection. Updated only by rebuildProjection().
  // Each draw() starts by copying this into sceneUbo.projection BEFORE
  // applying the per-frame Halton jitter, so the jitter doesn't accumulate
  // across frames into a random-walk drift of the projection matrix.
  glm::mat4 taaBaseProjection = glm::mat4(1.0f);
  // Jittered VP of the previous frame. Used by the TAA shader to
  // reproject each pixel's world position to where it landed in last
  // frame's image (which was rendered with last frame's jitter).
  glm::mat4 taaPrevViewProj = glm::mat4(1.0f);
  bool taaHistoryValid = false;            // dropped on swapchain recreate / first frame
  glm::vec3 taaLastCameraPos = glm::vec3(0.0f);
  glm::mat4 taaLastView = glm::mat4(1.0f);
  bool taaHasLastCamera = false;
  bool cameraMovedThisFrame = false;

  // --- Init helpers ---
  void createSynchronization();
  void createAsyncFrameCommandBuffers();
  void cleanupAsyncFrameCommandBuffers();
  void createThreadedCommandResources();
  void cleanupThreadedCommandResources();
  void registerSwapchainResources();
  // Every image the scene's size decides, and what points at them.
  void createSceneTargets();
  void cleanupSceneTargets();
  void recreateSceneTargets(VkExtent2D extent);
  void createSceneColorImages();
  void updateSceneArea();
  void copySceneToWindow(VkCommandBuffer cmd, uint32_t imageIndex);
  void noteGBufferFinalLayouts(uint32_t imageIndex);
  void noteLitFinalLayout(uint32_t imageIndex);
  void noteCompositeFinalLayouts(uint32_t imageIndex, uint32_t historyIndex);
  void createGBuffer();
  void cleanupGBuffer();
  void createLitResources();
  void createLitFramebuffers();
  void cleanupLitFramebuffers();
  void cleanupLitResources();
  void createTaaResources();
  void cleanupTaaResources();
  void initIBL();
  void cleanupIBL();
  void rebuildProjection();
  void applySunAndFog();

  // --- Per-frame ---
  // One frame, in the order draw() runs them.
  void buildUi();
  void applyPendingChanges();
  void waitForFrameSlot();
  bool acquireSwapImage(uint32_t &imageIndex);
  void waitForSwapImage(uint32_t imageIndex);
  void updateFrameUniforms();
  void updateDayNightSun();
  void packRenderSettings();
  void applyTaaJitter();
  void recordFrame(uint32_t imageIndex);
  void submitFrame(uint32_t imageIndex);
  bool presentFrame(uint32_t imageIndex);
  void recordGBufferCommands(uint32_t currentImage);
  void collectMeshInstances();
  bool prepareGpuDrivenFrame(uint32_t currentImage,
                             const glm::vec4 frustumPlanes[6],
                             uint32_t minCandidates);
  void collectGBufferDrawItems(std::vector<GBufferDrawItem> &drawItems,
                               const glm::vec4 frustumPlanes[6]);
  void recordGBufferDraws(VkCommandBuffer drawCmd,
                          const std::vector<GBufferDrawItem> &items,
                          size_t begin, size_t end, const VkViewport &viewport,
                          const VkRect2D &scissor, VkDescriptorSet vpSet,
                          VkDescriptorSet bindlessSet);
  void recordGBufferThreaded(VkCommandBuffer cmd, uint32_t currentImage,
                             const VkRenderPassBeginInfo &rpbi,
                             const std::vector<GBufferDrawItem> &items,
                             const VkViewport &viewport,
                             const VkRect2D &scissor, VkDescriptorSet vpSet,
                             VkDescriptorSet bindlessSet);
  void recordShadowCommands(VkCommandBuffer cmd);
  void recordSsaoComputeCommands(VkCommandBuffer cmd, uint32_t currentImage);
  void recordPostCommands(VkCommandBuffer cmd, uint32_t currentImage);
  void recordGBufferPass(VkCommandBuffer cmd, uint32_t currentImage,
                         const VkViewport &viewport,
                         const VkRect2D &scissor);
  void recordTransparentPass(VkCommandBuffer cmd, uint32_t currentImage,
                             const VkViewport &viewport,
                             const VkRect2D &scissor);
  void recordImGuiCommands(VkCommandBuffer cmd, uint32_t imageIndex);
  void recreateSwapChain();
};

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
#include <map>
#include <unordered_set>
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
#include "BreakGraph.h"
#include "Rubble.h"
#include "Stress.h"
#include "CharacterRoster.h"
#include "PlayerController.h"
#include "Powers.h"
#include "Prefab.h"
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
#include "ThumbnailRenderer.h"
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
  // Model files from outside the project, copied into Resources/Models
  // (dropped on the window, or the Assets panel's Import), at the next
  // frame's edit step.
  void importFiles(std::vector<std::string> paths) {
    pendingImports.insert(pendingImports.end(), paths.begin(), paths.end());
  }
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
  // A 5 kg ball, or with heavy a 60 kg one, for breaking things.
  void throwBall(glm::vec3 from, glm::vec3 direction, bool heavy = false);
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
  // While playing, the player possesses a character and the view comes from
  // its camera.
  bool hasPlayer() const { return playerController && playerController->body(); }
  // The player's keys, rebound in the Controls tab.
  const ControlBindings &controls() const { return controlBindings; }
  const ThirdPersonCamera &playerCamera() const { return playerController->camera(); }
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
  // While playing: the player's brain, and the bodies of every character in
  // play (CharacterRoster.h). The controller comes first so it outlives what
  // it possesses, and both come after physics so the bodies leave the world
  // before it goes. loadScene() clears them before it replaces the world.
  std::unique_ptr<PlayerController> playerController;
  std::unique_ptr<CharacterRoster> roster;
  PlayerControls playerControls;
  int playerModelId = -1;          // the capsule standing, and crouched below
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
  std::vector<PhysicsDebugVertex> skeletonLines; // Show skeleton, play mode
  // Bends every character's mesh each frame: one mesh per character file,
  // one copy per character drawn, and each file's copy for its pictures.
  SkinningPass skinningPass;

  // A character file in use: read once, shared by every character made
  // from it. The default one (Ganfaul's) is the Character tab's: its
  // description and asset are the tab's own, edited live; the others keep
  // theirs here.
  struct CharacterFileData {
    std::string file; // as scenes name it
    const CharacterDescription *description = nullptr;
    const CharacterAsset *asset = nullptr; // none: drawn as a capsule
    CharacterDescription ownDescription;
    std::optional<CharacterAsset> ownAsset;
    int skinMesh = -1;
    // Copies of its mesh to draw at once: a model with its own vertex
    // buffer, and the skinning copy that bends into it. Made as more are
    // needed and kept, so rebuilding the scene reuses them.
    struct Copy {
      int model = -1;
      int skin = -1;
    };
    std::vector<Copy> copies;
    // The copy its pictures pose (clip stills, the live clip, its own
    // tile), skinned into a buffer of its own that only the thumbnail
    // renderer draws.
    int previewSkin = -1;
    AllocatedBuffer previewVertices;
  };
  std::vector<std::unique_ptr<CharacterFileData>> characterFiles; // [0] the default
  std::vector<std::string> characterFileNames; // Resources/Characters/*.character
  // The file a scene names, read the first time (the GPU must be idle), or
  // nothing if it can't be read.
  CharacterFileData *characterFile(const std::string &file);
  CharacterFileData *defaultCharacter() {
    return characterFiles.empty() ? nullptr : characterFiles.front().get();
  }
  // Its mesh in the skinning pass and the model its copies are made like.
  void prepareCharacterMesh(CharacterFileData &data);
  int characterCopy(CharacterFileData &data, size_t index);

  // A character drawn in the scene: a [character], or with none ticked the
  // player at the player start. Edit mode: standing in its idle pose. Play
  // mode: a body in the roster, animated.
  struct LiveCharacter {
    SceneEntryRef entry; // Kind::None for the player start's
    CharacterFileData *file = nullptr;
    std::string variant;
    CharacterTuning tuning; // what it plays with: its file, its variant
    AnimatorTuning animation;
    float scale = 1.0f;  // the model file's units to its body's height
    SceneNode *node = nullptr; // owned by rootNode
    int copy = -1; // into file->copies; -1 is the capsule
    int body = -1; // in the roster, while playing
    std::unique_ptr<CharacterAnimator> animator; // while playing, with a mesh
    // This frame's skinning matrices, worked out while posing and uploaded
    // in draw() once the frame slot's fence says it's free.
    std::vector<glm::mat4> skinning;
  };
  std::vector<LiveCharacter> characters;
  int playedCharacter = -1; // the one the player possesses, into characters
  // Every character from the scene, in edit or play mode: the old ones'
  // nodes and bodies go first. The GPU must be idle (a copy may be new).
  void buildCharacters();
  void clearCharacters();
  // Edit mode: a character's node at its entry's place, after an edit.
  void placeCharacter(int index);
  void poseCharacters(float alpha, float frameSeconds);
  std::pair<CharacterTuning, AnimatorTuning> tuningFor(const CharacterFileData &data,
                                                       const std::string &variant) const;
  // The variant the player's character plays: the ticked one's, or the
  // player start's.
  std::string playedVariant() const;
  std::array<RootMotion, 5> dodgePathsFor(const LiveCharacter &character) const;
  void loadCharacter();
  void applyClipChange(ClipSlot slot);
  bool saveCharacterFile();
  void updateCharacterUnsaved();
  void updateVariantNote();
  // The character's skeleton as lines where it's drawn this frame.
  void updateSkeletonLines(const LiveCharacter &character, const glm::mat4 &placement,
                           const std::vector<glm::mat4> &model);
  // The capsules for the Character tab's body sizes, made if they aren't
  // yet. A new mesh waits for the GPU to go idle first.
  void ensurePlayerModels();
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
  std::vector<std::string> pendingImports;
  std::string assetStatus;
  void runPendingImports();
  std::optional<SceneModelEntry> pendingSpawn;
  // Prefabs: the object files read so far, and an Inspector button to act
  // on at the next edit step.
  std::map<std::string, ObjectAsset> objectAssets;
  const ObjectAsset *objectAsset(const std::string &file);
  std::string objectsFolderOnDisk() const;
  void resolvePrefabs(SceneDescription &scene);
  std::optional<std::pair<SceneEntryRef, DebugUiContext::PrefabAction>> pendingPrefabAction;
  void applyPrefabAction(SceneEntryRef ref, DebugUiContext::PrefabAction action);
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
  // The Assets panel's pictures: drawn one a frame as the panel asks,
  // cached on disk.
  ThumbnailRenderer thumbnails;
  std::string thumbnailFolder;
  ImTextureID thumbnailTexture = 0;
  std::vector<std::pair<std::string, SceneModelEntry>> thumbnailRequests;
  std::vector<std::pair<std::string, std::string>> clipRequests; // key, clip file
  std::vector<std::pair<std::string, std::string>> characterRequests; // key, .character
  // The clip under the mouse in a tile, playing live; clips read for
  // previews, kept for a while.
  ImTextureID liveTexture = 0;
  std::string hoveredClip;
  std::string liveClip;
  double liveClipTime = 0.0;
  std::vector<glm::mat4> liveSkinning;
  std::map<std::string, CharacterClipData> previewClips;
  std::string clipThumbnailKey(const std::string &file) const;
  const CharacterClipData *previewClip(const std::string &file);
  std::string characterThumbnailKey(const std::string &file) const;
  std::vector<ThumbnailDraw> characterPreviewDraws(const CharacterFileData &data);
  // A posed character's picture, framed around its joints.
  bool renderCharacterThumbnail(CharacterFileData &data, const std::string &key,
                                const std::vector<glm::mat4> &model);
  void prepareLivePreview();
  std::chrono::steady_clock::time_point liveTick;
  std::unordered_set<std::string> thumbnailsFailed; // not asked again
  std::string thumbnailKeyFor(const SceneModelEntry &entry) const;
  void makeNextThumbnail();
  int assetPreviewModel = -1;
  NodeTransform dropPlacement(const SceneModelEntry &entry, int modelId, glm::vec2 windowPixel,
                              bool turnToSurface);
  void showAssetPreview(const SceneModelEntry &entry, glm::vec2 windowPixel, bool turnToSurface);
  void hideAssetPreview();
  void dropAsset(SceneModelEntry entry, glm::vec2 windowPixel, bool turnToSurface);
  // Characters from the Assets panel: standing on the surface under the
  // cursor, facing away from the camera, shown meanwhile as a capsule's
  // outline. Double-click puts one where the camera looks.
  struct CharacterPlacement {
    std::string file;
    std::optional<glm::vec2> windowPixel; // none: where the camera looks
  };
  std::optional<CharacterPlacement> characterHover;
  std::optional<CharacterPlacement> characterDrop;
  std::optional<SceneCharacterEntry> characterDragShown; // its outline
  SceneCharacterEntry characterPlacement(const CharacterPlacement &placement);
  void dropCharacter(const CharacterPlacement &placement);
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

  // Destruction (step 1.3): breakable boxes are cut into pieces
  // (Fracture.h), each piece a node, held by bodies that each carry a group
  // of pieces: one per object at first. In play the pieces form one graph
  // (BreakGraph.h): hits break links, and what's left splits into groups
  // that stay put if anchored or fall as one moving body. A breakable
  // body's userData has kBreakBodyFlag set and indexes breakBodies.
  static constexpr uint64_t kBreakBodyFlag = 1ull << 62;
  struct Breakable {
    SceneEntryRef entry;
    std::string key; // the cut's: size, pieces, seed
    std::shared_ptr<const Fracture> fracture;
    int firstPiece = 0;
    float strength = kDefaultBreakStrength;
    bool moving = false; // a dynamic body, never anchored
    float volume = 0.0f;
    float density = 1000.0f; // kg/m^3, for its chunks' masses
    FracturePattern pattern = FracturePattern::Clustered;
    glm::vec3 grain{1.0f, 0.0f, 0.0f}; // its longest side, in its own space
    glm::vec3 colour{0.8f};
    // Weight: what its links hold out against (its material's strengths,
    // scaled by its break_strength), its friction, and whether it's glued to
    // the breakables it touches.
    LinkLimits limits;
    bool glued = false;
  };
  struct BreakPiece {
    int object = 0; // into breakables
    // Its shape in its object's space: one of the first cut's pieces, or a
    // piece cut from one while playing.
    std::shared_ptr<const FracturePiece> shape;
    SceneNode *node = nullptr;
    int body = -1; // into breakBodies
    // Its mesh's frame (the object's space) inside its body's frame.
    glm::vec3 localPosition{0.0f};
    glm::quat localRotation{1.0f, 0.0f, 0.0f, 0.0f};
    bool anchor = false; // rests on something that never breaks
    bool alive = true;   // false once cut again or turned to rubble
  };
  struct BreakBody {
    PhysicsBodyId id; // invalid once replaced; indices stay
    std::vector<int> pieces;
    bool fixed = false; // static
    double madeAt = 0.0; // playSeconds, for turning small pieces to rubble
  };
  // Rubble (step 1.3c): a layer of rubble over the ground (Rubble.h),
  // drawn and walked on in 4 m tiles, each a node with a mesh rewritten in
  // place and a static height field. Its bodies' userData has
  // kRubbleBodyFlag set.
  static constexpr uint64_t kRubbleBodyFlag = 1ull << 61;
  struct RubbleTileDraw {
    SceneNode *node = nullptr;
    int modelId = -1;
    PhysicsBodyId body;
    std::vector<float> heights; // what its body was given, to see how much it rose
  };
  // Rubble on its way into the layer, fed in over kFeedSeconds so a pile
  // grows rather than jumps (a height field rising past what rests on it
  // would push it out of its back).
  struct RubbleFeed {
    glm::vec2 at{0.0f};
    float volume = 0.0f; // m^3 still to come
    float rate = 0.0f;   // m^3 a second
    float radius = 0.0f;
    float castTop = 0.0f; // where rays down for the ground start
    glm::vec3 colour{0.5f};
  };
  // A small piece that settled: no body any more, it lies on the pile as a
  // stone and follows the surface down if the pile slumps away under it.
  struct RubbleStone {
    SceneNode *node = nullptr;
    glm::vec2 at{0.0f}; // its middle, across the ground
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    float below = 0.0f; // from its node down to its lowest point
  };
  // A piece queued to be cut where it was hit, in its own space.
  struct PendingSplit {
    int piece = 0;
    glm::vec3 hit{0.0f};
    glm::vec3 push{0.0f};
    float impulse = 0.0f;
  };
  std::vector<Breakable> breakables;
  std::vector<BreakPiece> breakPieces;
  std::vector<BreakBody> breakBodies;
  std::vector<BreakLink> breakLinks;
  std::vector<glm::vec3> breakLinkPoints; // in piece a's mesh frame
  // Weight (step 1.3d, Stress.h): every link's face in piece a's mesh frame,
  // and whether it only rests (between objects that aren't glued); a link
  // from a piece to the fixed thing it rests on (the ground) for each face
  // that does. A piece is an anchor while one of its ground links holds.
  std::vector<LinkFace> breakLinkFaces;
  std::vector<char> breakLinkContact;
  struct GroundLink {
    int piece = 0;
    LinkFace face; // in the piece's mesh frame
    bool broken = false;
  };
  std::vector<GroundLink> groundLinks;
  std::vector<std::vector<int>> pieceGroundLinks; // by piece
  // The solve over the pieces that stay put, redone when the structure
  // changes and spread over frames; each piece's last shift and turn start
  // the next one. Each link's overload is kept for the load view.
  StressSolver stressSolver;
  bool stressDirty = false;
  bool stressFirst = false; // the first settle after Play only reports
  bool stressJudged = false; // settled and its links judged: nothing to do until a change
  std::vector<glm::vec3> pieceShift, pieceTurn;
  std::vector<int> stressPieceOf; // by piece, into the solve or -1
  std::vector<int> stressOrder;   // by solve index, the piece
  struct SolvedLink {
    int index = 0;       // into breakLinks or groundLinks
    bool ground = false;
  };
  std::vector<SolvedLink> solvedLinks;
  std::vector<float> solvedLoad; // overload, by solved link
  std::vector<PhysicsDebugVertex> loadLines;
  double stressStarted = 0.0; // ms spent on the current solve so far
  // Live loads (C2): what moving bodies rest on, by the moving body (kept
  // while it sleeps), and each piece's load from them and from characters,
  // smoothed, and as last handed to the solve. A link breaks once it's been
  // over its limits for kOverloadSeconds.
  std::unordered_map<uint32_t, std::vector<PhysicsRest>> restsOn;
  std::vector<glm::vec3> liveForce, liveTorque;               // by piece, smoothed
  std::vector<glm::vec3> liveAppliedForce, liveAppliedTorque; // by piece
  std::vector<float> overloadSeconds;                         // by solved link
  float sinceJudged = 0.0f;                                   // s
  float stressFrame = 0.0f;                                   // s, this frame's
  int loggedIterations = -1; // the solve's count when last logged
  void gatherLiveLoads();
  void applyLiveLoads();
  std::vector<std::vector<int>> pieceLinks; // by piece, its links
  std::map<std::string, std::shared_ptr<const Fracture>> fractures; // cut once each
  std::unique_ptr<RubbleLayer> rubbleLayer;
  std::map<RubbleKey, RubbleTileDraw> rubbleTiles;
  std::vector<RubbleKey> rubbleWaiting; // changed, waiting for a re-mesh
  std::vector<RubbleFeed> rubbleFeed;
  std::vector<RubbleStone> rubbleStones;
  float rubbleCastTop = 50.0f; // m: the ground rays of the deposit being made start here
  float rubbleRemeshWait = 0.0f; // s until tiles may be re-meshed again
  std::vector<PendingSplit> pendingSplits;
  // Cuts' uploads the GPU may still be doing; dropped once their fence says
  // it's done (UploadBatch).
  std::vector<std::unique_ptr<UploadBatch>> uploadsInFlight;
  uint32_t playSeed = 0;     // drawn at every Play: no two breaks alike
  uint32_t splitsMade = 0;   // for each cut's own seed
  int runtimePieces = 0;     // pieces cut while playing, against the budget
  bool breakablesDirty = false; // rebuilt at the top of the next frame
  bool breakGraphReady = false; // the links across objects and anchors, on Play
  // Every breakable from the scene, intact. The GPU must be idle.
  void rebuildBreakables();
  void clearBreakables();
  void prepareBreakGraph();
  // After the physics steps: hits on breakable bodies break links and cut
  // pieces, groups that changed get new bodies, small settled pieces turn
  // to rubble.
  void handleImpacts();
  // A hit that broke links this frame: freed pieces near it share it, and
  // what hit carries on with the rest.
  struct BreakHit {
    glm::vec3 point{0.0f};
    glm::vec3 push{0.0f}; // into the object, unit
    float impulse = 0.0f;
    PhysicsBodyId other;
    glm::vec3 otherVelocity{0.0f};
  };
  void regroupBreakables(const std::vector<BreakHit> &hits);
  void splitPiece(const PendingSplit &split);
  void settleRubble();
  // Each frame in play: the weight on what stays put, and the links it
  // breaks (step 1.3d).
  void updateStress(float seconds);
  // The faces of a placed piece that rest against something fixed, by index.
  std::vector<int> restingFaces(const PlacedPiece &piece) const;
  void addGroundLinks(int piece, const PlacedPiece &placed);
  LinkLimits linkLimits(int link) const;
  LinkLimits groundLimits(int piece) const;
  // Rubble into the layer around where (its ground found below it), fed in
  // over a moment.
  void addRubble(glm::vec3 where, float volume, float radius, glm::vec3 colour);
  // Each frame in play: rubble fed in, slumped, and the changed tiles
  // re-meshed and their height fields set, a few a frame.
  void updateRubble(float seconds);
  int pieceModel(const Breakable &object, const FracturePiece &shape, const std::string &key);
  PhysicsPose pieceWorld(const BreakPiece &piece) const;
  int addBreakBody(const std::vector<int> &pieces, const std::vector<PhysicsPose> &worlds,
                   bool fixed, glm::vec3 velocity, glm::vec3 spin);
  void poseBreakBody(const BreakBody &body, const PhysicsPose &pose);
  void placeBreakable(SceneEntryRef ref);
  void appendPieceLines(SceneEntryRef ref);
  int breakableOf(uint64_t userData) const; // into breakables, or -1
  int pieceAt(const BreakBody &body, glm::vec3 point) const; // the piece a point is on, or -1

  // The powers (step 1.4a, Powers.h; src/PowerPlay.cpp): with the aim held
  // the player grabs what the crosshair is on, holds it out along the aim,
  // throws it, tears pieces out of what they're part of, and bursts.
  // Something the power can take: a whole moving body (a scene object's or
  // a breakable chunk's), or one piece of a breakable, torn out first.
  struct PowerTarget {
    int id = -1; // the controller's: index * 4, plus 0 a piece, 1 a chunk's body, 2 a scene body
    GrabKind kind = GrabKind::Lift;
    PhysicsBodyId body;
    int piece = -1; // the piece; for a chunk's body, one it carries
    glm::vec3 centre{0.0f};
    float radius = 0.0f;
  };
  PowerController powerController;
  std::vector<PowerTarget> powerCandidates; // near the player, this frame
  std::optional<PowerTarget> powerTarget;    // under the crosshair and in sight
  PowerTarget powerGrabbed;                  // what was grabbed, while it's held
  // The moving body held out (lifting, holding, a throw not yet gone), a
  // piece it carries so it's found again when a regroup replaces it, where
  // its lift began and how long ago, its tumble, the wheel's distance, and
  // how long it's been stuck away from the hold point.
  PhysicsBodyId heldBody;
  int heldPiece = -1;
  std::vector<int> heldPieces; // the held chunk's, for another to follow if heldPiece is cut
  glm::vec3 heldFrom{0.0f};
  float heldSeconds = 0.0f;
  glm::vec3 heldAxis{0.0f, 1.0f, 0.0f};
  float heldDistance = 0.0f;
  float heldAway = 0.0f;
  glm::vec3 holdTarget{0.0f}; // this frame's hold point
  // A tear under way: the piece's links and ground links, when each gives
  // way, and how far along it is.
  struct PowerTear {
    std::vector<int> links, grounds;
    std::vector<float> linkTimes, groundTimes;
    float seconds = 0.0f;
    float total = 0.0f;
    bool done = false; // every link it had is broken, and it's still not free to lift
  } powerTear;

  // The last throw's path, for its trail.
  struct PowerTrail {
    PhysicsBodyId body;
    std::vector<glm::vec3> points;
    float seconds = 0.0f;
  } powerTrail;
  float revealSince = -1.0f; // s since Q, or -1
  glm::vec3 revealFrom{0.0f};
  float burstSince = -1.0f; // s since the blast, or -1
  glm::vec3 burstFrom{0.0f}, burstAim{0.0f, 0.0f, -1.0f};
  std::vector<PhysicsImpact> powerImpacts; // the burst's hits, for handleImpacts
  // Nodes shaken last frame (a tear, a strain): put back where their
  // bodies are before this frame's shake.
  std::vector<int> shakenPieces;
  int shakenLink = -1; // into physicsLinks
  glm::vec3 handPosition{0.0f}; // the player's right hand this frame
  int handBone = -2;            // into the skeleton; -1 none, -2 not looked for
  std::vector<PhysicsDebugVertex> powerLines;
  uint32_t powerDraws = 0; // for the tumble axes, varied each grab and throw
  void clearPowers();
  // Each frame in play, before the steps: what's near and under the
  // crosshair, the controller, and what it says to do.
  void updatePowers(float seconds);
  // Each physics step: the held body pulled toward the hold point.
  void holdPowers(float stepSeconds);
  // After the poses: the piece being torn out, or what's strained against,
  // shaken where it's drawn.
  void shakePowers();
  // After the characters are posed: the powers as lines, for now.
  void drawPowerLines();
  void gatherPowerCandidates(glm::vec3 around, float radius);
  bool followHeld(); // false once the held body is gone for good
  void startTear(int piece);
  bool advanceTear(float seconds); // true once the piece is loose and light enough
  void throwHeld();
  void burstPower();
  glm::vec3 bodyCentre(PhysicsBodyId body) const; // its bounding box's middle
  glm::vec3 heldCentre() const;
  // Where the power's light goes: what's held, or the piece torn at or
  // strained against; nothing without a light out.
  std::optional<glm::vec3> powerReach() const;
  float pieceMass(int piece) const;
  glm::vec3 pieceCentre(int piece) const;
  glm::vec3 randomAxis();
  // The played character's (its variant's), or the Character tab's.
  const PowerTuning &powerTuning() const;
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

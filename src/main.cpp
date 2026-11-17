#include "Camera.h"
#include "InputManager.h"
#include "VulkanRenderer.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

GLFWwindow *window;
VulkanRenderer vulkanRenderer;
InputManager inputManager;

// The scene file sets the real start pose when it loads.
Camera camera(glm::vec3(0.0f, 2.0f, 0.0f));

// Timing
float deltaTime = 0.0f;
// double, not float: glfwGetTime() is seconds since init, and float's ~2 ms
// resolution past 4.5 hours of uptime would quantize deltaTime into visible
// camera/animation stutter. The delta itself is small and safe as float.
double lastFrame = 0.0;

// typing: a text box in the UI has the keyboard, so letters and Esc are
// its, not the game's (Esc leaves the box instead of quitting).
void processInput(bool typing) {
  // Esc quits, asking first if something isn't saved. While a question is
  // on screen, Esc is its Cancel.
  const bool escPressed = inputManager.wasKeyJustPressed(GLFW_KEY_ESCAPE);
  if (typing || vulkanRenderer.promptOpen())
    return;
  if (escPressed)
    vulkanRenderer.requestQuit();
  // Ctrl+S saves, so with Ctrl held the letters aren't movement.
  if (inputManager.isKeyPressed(GLFW_KEY_LEFT_CONTROL) ||
      inputManager.isKeyPressed(GLFW_KEY_RIGHT_CONTROL))
    return;
  // While playing, WASD walks the character (readPlayerControls).
  if (vulkanRenderer.hasPlayer())
    return;

  if (inputManager.isKeyPressed(GLFW_KEY_W))
    camera.ProcessKeyboard(FORWARD, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_S))
    camera.ProcessKeyboard(BACKWARD, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_A))
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (inputManager.isKeyPressed(GLFW_KEY_D))
    camera.ProcessKeyboard(RIGHT, deltaTime);
}

void initWindow(const std::string &wName = "Vulkan Renderer",
                const int width = 800, const int height = 600) {

  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
  if (!glfwInit()) {
    throw std::runtime_error("Failed to init GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // Enable window resizing

  window = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }

  // InputManager handles cursor callback and resize callback
  inputManager.init(window);
}

// Who has the mouse: Tab switches between flying the camera (cursor hidden)
// and the UI (cursor free, panels live). In UI mode, holding the right mouse
// button on the scene flies too; Tab stays for the touchpad, which can't hold
// a right click.
struct FlyState {
  bool cameraMode = true;
  bool rightMouseFlying = false;
  bool flying() const { return cameraMode || rightMouseFlying; }
};

// The startup scene: argv[1] if there is one (an index into the scenes found,
// or a path to a .scene file), otherwise the first scene found. False when
// there's no scene at all. Scenes are data-driven: Resources/Scenes/*.scene
// files describe the models, transforms, animations, rings and camera pose,
// and the Scene panel switches between them.
bool loadStartupScene(int argc, char **argv) {
  vulkanRenderer.discoverScenes("Resources/Scenes");
  vulkanRenderer.discoverModels("Resources/Models");
  bool sceneLoaded = false;
  if (argc > 1) {
    const std::string arg = argv[1];
    try {
      if (!arg.empty() &&
          arg.find_first_not_of("0123456789") == std::string::npos) {
        // An out-of-range index must NOT count as loaded, or the fallback
        // below is skipped and the app runs with an empty scene.
        sceneLoaded = vulkanRenderer.loadSceneAt(std::stoi(arg));
      } else {
        vulkanRenderer.loadScene(SceneIO::loadFromFile(arg));
        sceneLoaded = true;
      }
    } catch (const std::exception &e) {
      std::cerr << "Failed to load requested scene '" << arg
                << "': " << e.what() << std::endl;
    }
  }
  if (!sceneLoaded && vulkanRenderer.hasScenes())
    sceneLoaded = vulkanRenderer.loadSceneAt(0);
  return sceneLoaded;
}

// Tab, right-mouse flying, and a prompt taking the mouse back: a question
// needs the cursor and the panels.
void updateFlying(FlyState &fly, bool typing, bool asking) {
  if (asking && fly.flying()) {
    fly.cameraMode = false;
    fly.rightMouseFlying = false;
    inputManager.setCursorCaptured(false);
  }
  if (asking && !vulkanRenderer.isUiVisible())
    vulkanRenderer.toggleUiVisibility();

  // Tab toggles between camera fly-through (cursor hidden) and UI mode
  // (cursor visible, ImGui panels interactive).
  // WantCaptureMouse can't drive this because GLFW_CURSOR_DISABLED delivers
  // only raw deltas — ImGui never sees absolute positions and can't detect
  // hover.
  if (inputManager.wasKeyJustPressed(GLFW_KEY_TAB) && !typing && !asking) {
    fly.cameraMode = !fly.cameraMode;
    fly.rightMouseFlying = false;
    inputManager.setCursorCaptured(fly.cameraMode);
    // Entering UI mode with the HUD hidden would strand the user in a
    // dead-looking state (free cursor, nothing to click, no mouse-look) —
    // un-hide the panels so UI mode always shows a UI.
    if (!fly.cameraMode && !vulkanRenderer.isUiVisible())
      vulkanRenderer.toggleUiVisibility();
  }

  // Only a press that starts on the scene flies; one on a panel stays
  // with the panel. Read every frame, so a press made in camera mode
  // doesn't count as new after Tab.
  const bool rightPressed =
      inputManager.wasMouseButtonJustPressed(GLFW_MOUSE_BUTTON_RIGHT);
  // While playing, the right mouse aims instead.
  if (!fly.cameraMode && !vulkanRenderer.hasPlayer()) {
    if (rightPressed && !asking && !vulkanRenderer.imguiWantsMouse() &&
        vulkanRenderer.sceneContains(inputManager.getMousePosition())) {
      fly.rightMouseFlying = true;
      inputManager.setCursorCaptured(true);
    } else if (fly.rightMouseFlying &&
               !inputManager.isMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT)) {
      fly.rightMouseFlying = false;
      inputManager.setCursorCaptured(false);
    }
  }
}

// The editor's mouse and keys: selecting with a click, undo and redo, save,
// and the function keys.
void handleEditorKeys(const FlyState &fly, bool typing, bool asking) {
  // A left click in edit mode, with the cursor free and on the scene (off
  // the panels and the gizmo's handles), selects what's under it. While a
  // text field has the keyboard, the click only ends the typing, so the
  // value goes to the object it was typed for.
  const bool leftPressed =
      inputManager.wasMouseButtonJustPressed(GLFW_MOUSE_BUTTON_LEFT);
  if (leftPressed && !typing && !fly.cameraMode && !fly.rightMouseFlying &&
      !asking && !vulkanRenderer.isPlaying() &&
      !vulkanRenderer.imguiWantsMouse() && !vulkanRenderer.gizmoHasMouse()) {
    vulkanRenderer.selectAt(inputManager.getMousePosition());
  }

  // Ctrl+Z undoes and Ctrl+Shift+Z redoes, in edit mode (the renderer
  // checks), unless a text box has the keyboard: it has its own undo.
  const bool zPressed = inputManager.wasKeyJustPressed(GLFW_KEY_Z);
  const bool sPressed = inputManager.wasKeyJustPressed(GLFW_KEY_S);
  const bool dPressed = inputManager.wasKeyJustPressed(GLFW_KEY_D);
  const bool ctrl = inputManager.isKeyPressed(GLFW_KEY_LEFT_CONTROL) ||
                    inputManager.isKeyPressed(GLFW_KEY_RIGHT_CONTROL);
  const bool shift = inputManager.isKeyPressed(GLFW_KEY_LEFT_SHIFT) ||
                     inputManager.isKeyPressed(GLFW_KEY_RIGHT_SHIFT);
  // Not while the character has the mouse: Ctrl is crouch there, and
  // crouch-walking backwards mustn't save.
  const bool characterHasMouse = vulkanRenderer.hasPlayer() && fly.cameraMode;
  const bool shortcuts = ctrl && !typing && !asking && !characterHasMouse;
  if (zPressed && shortcuts) {
    if (shift)
      vulkanRenderer.redoEdit();
    else
      vulkanRenderer.undoEdit();
  }
  // Ctrl+D duplicates the selection, in edit mode (the renderer checks).
  if (dPressed && shortcuts)
    vulkanRenderer.duplicateSelection();
  // Ctrl+S saves the scene and Ctrl+Shift+S saves it under a new name,
  // in either mode: what's saved is the layout.
  if (sPressed && shortcuts) {
    if (shift)
      vulkanRenderer.requestSaveAs();
    else
      vulkanRenderer.requestSave();
  }

  // F5 switches between play mode and edit mode.
  if (inputManager.wasKeyJustPressed(GLFW_KEY_F5) && !asking)
    vulkanRenderer.togglePlayMode();
  // F1 toggles HUD-free screenshot mode (hides all ImGui panels).
  if (inputManager.wasKeyJustPressed(GLFW_KEY_F1) && !asking)
    vulkanRenderer.toggleUiVisibility();
  // F2 shows the physics shapes as wireframes.
  if (inputManager.wasKeyJustPressed(GLFW_KEY_F2) && !asking)
    vulkanRenderer.togglePhysicsDebug();
  // F throws a ball where the camera looks, in camera mode only so a
  // click-and-type in a panel never throws. Play mode only (throwBall).
  // With a character it starts from its shoulders, not from the camera
  // behind it, or the first thing it would hit is the character.
  if (fly.cameraMode && !typing && !asking &&
      inputManager.wasKeyJustPressed(GLFW_KEY_F)) {
    if (vulkanRenderer.hasPlayer()) {
      const ThirdPersonCamera &view = vulkanRenderer.playerCamera();
      vulkanRenderer.throwBall(view.pivot(), view.forward());
    } else {
      vulkanRenderer.throwBall(camera.Position, camera.Front);
    }
  }
}

// The character's controls, while it has the mouse (camera mode). With the
// cursor free for the panels it stands still.
PlayerControls readPlayerControls(const FlyState &fly, bool typing,
                                  bool asking) {
  // Read every frame, so a press made with the cursor free doesn't count
  // as new after Tab.
  const bool jump = inputManager.wasKeyJustPressed(GLFW_KEY_SPACE);
  const bool dodge = inputManager.wasKeyJustPressed(GLFW_KEY_LEFT_ALT);
  PlayerControls controls;
  if (!fly.cameraMode || typing || asking)
    return controls;
  controls.forward = inputManager.isKeyPressed(GLFW_KEY_W);
  controls.back = inputManager.isKeyPressed(GLFW_KEY_S);
  controls.left = inputManager.isKeyPressed(GLFW_KEY_A);
  controls.right = inputManager.isKeyPressed(GLFW_KEY_D);
  controls.sprint = inputManager.isKeyPressed(GLFW_KEY_LEFT_SHIFT);
  controls.crouch = inputManager.isKeyPressed(GLFW_KEY_LEFT_CONTROL);
  controls.aim = inputManager.isMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT);
  controls.jump = jump;
  controls.dodge = dodge;
  // InputManager's y counts up; the controls' counts down, like the screen.
  const glm::vec2 mouse = inputManager.getMouseDelta();
  controls.mouse = glm::vec2(mouse.x, -mouse.y);
  controls.wheel = inputManager.getScrollDelta();
  return controls;
}

void turnCamera(const FlyState &fly) {
  // The mouse turns the camera only while flying, and while playing it
  // turns the character's camera instead.
  glm::vec2 mouseDelta = inputManager.getMouseDelta();
  if (fly.flying() && !vulkanRenderer.hasPlayer() &&
      (mouseDelta.x != 0.0f || mouseDelta.y != 0.0f)) {
    camera.ProcessMouseMovement(mouseDelta.x, mouseDelta.y);
  }
}

int main(int argc, char **argv) {
  initWindow("Vulkan Renderer", 1920, 1080);

  if (vulkanRenderer.init(window) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  // HELIORA_PHYSICS_DEBUG=1 starts with the physics wireframes on (same as
  // F2), for runs nobody is pressing keys in.
  if (std::getenv("HELIORA_PHYSICS_DEBUG"))
    vulkanRenderer.togglePhysicsDebug();

  camera.MovementSpeed = 15.0f;
  vulkanRenderer.setCameraPresetCallback(
      [](glm::vec3 position, float yaw, float pitch, float speed) {
        camera.SetPose(position, yaw, pitch);
        camera.MovementSpeed = speed;
      });
  FlyState fly;
  inputManager.setCursorCaptured(true);

  // HELIORA_CAPTURE=out.png renders the startup scene in edit mode (nothing
  // moves) with the panels hidden and no input, saves the picture after
  // HELIORA_CAPTURE_FRAMES frames (300 by default, so TAA and exposure
  // settle) and quits. Two builds captured the same way can be compared.
  // HELIORA_CAPTURE_PLAY=1 stays in play mode instead, character and all,
  // and HELIORA_CAPTURE_UI=1 keeps the panels, to check the editor itself.
  // HELIORA_CAPTURE_KEYS presses the character's keys on a script, so its
  // moves can be checked too (scriptedControls in Player.h).
  const char *capturePath = std::getenv("HELIORA_CAPTURE");
  const char *captureKeys = std::getenv("HELIORA_CAPTURE_KEYS");
  const char *captureFramesText = std::getenv("HELIORA_CAPTURE_FRAMES");
  const int captureFrames =
      captureFramesText ? std::max(1, std::atoi(captureFramesText)) : 300;
  if (capturePath) {
    if (!std::getenv("HELIORA_CAPTURE_PLAY"))
      vulkanRenderer.togglePlayMode();
    if (!std::getenv("HELIORA_CAPTURE_UI"))
      vulkanRenderer.toggleUiVisibility();
    inputManager.setCursorCaptured(false);
  }
  int frameNumber = 0;

  if (!loadStartupScene(argc, argv)) {
    // Nothing to show without a scene, so say why instead of opening an
    // empty world.
    std::cerr << "No .scene files found in Resources/Scenes" << std::endl;
    vulkanRenderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  while (!vulkanRenderer.quitApproved()) {
    double currentFrame = glfwGetTime();
    deltaTime = static_cast<float>(currentFrame - lastFrame);
    lastFrame = currentFrame;

    inputManager.pollEvents();
    // The window's close button asks first, the same as Esc.
    if (inputManager.shouldClose()) {
      glfwSetWindowShouldClose(window, GLFW_FALSE);
      vulkanRenderer.requestQuit();
    }
    // A text box has the keyboard, or a question waits for an answer: the
    // game's keys wait.
    const bool typing = vulkanRenderer.imguiWantsTextInput();
    const bool asking = vulkanRenderer.promptOpen();
    if (!capturePath) {
      updateFlying(fly, typing, asking);
      processInput(typing);
      handleEditorKeys(fly, typing, asking);
      turnCamera(fly);
      vulkanRenderer.setPlayerControls(readPlayerControls(fly, typing, asking));
    } else if (captureKeys) {
      vulkanRenderer.setPlayerControls(scriptedControls(captureKeys, frameNumber));
    }

    if (inputManager.wasResized()) {
      vulkanRenderer.notifyResize();
      inputManager.resetResizedFlag();
    }

    // Physics first, so nodes it moved get their matrices in updateScene.
    vulkanRenderer.setFlyCamera(camera.Position, camera.Yaw);
    vulkanRenderer.stepPhysics(deltaTime);
    // Advances node animations (play mode only) and global transforms.
    vulkanRenderer.updateScene(deltaTime);

    // The character's camera while playing, the fly camera otherwise. Feed
    // ImGui the camera state; the speed is read back in case the slider
    // changed it.
    if (vulkanRenderer.hasPlayer()) {
      const ThirdPersonCamera &view = vulkanRenderer.playerCamera();
      vulkanRenderer.updateCameraView(view.view(), view.position());
      vulkanRenderer.setImGuiCameraInfo(view.position(), camera.MovementSpeed);
    } else {
      vulkanRenderer.updateCameraView(camera.GetViewMatrix(), camera.Position);
      vulkanRenderer.setImGuiCameraInfo(camera.Position, camera.MovementSpeed);
    }
    if (capturePath && ++frameNumber == captureFrames)
      vulkanRenderer.requestCapture(capturePath);
    vulkanRenderer.draw();
    camera.MovementSpeed = vulkanRenderer.getCameraSpeed();
    if (capturePath && vulkanRenderer.finishCapture())
      break;
  }

  vulkanRenderer.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

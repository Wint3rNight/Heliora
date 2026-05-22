#include "VulkanDevice.h"
#include "VulkanDebug.h"
#include <cstring>
#include <iostream>
#include <spdlog/spdlog.h>

// --- Phase 7.1: debug-utils function pointer definitions ---
PFN_vkCmdBeginDebugUtilsLabelEXT  g_vkCmdBeginDebugUtilsLabel  = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT    g_vkCmdEndDebugUtilsLabel    = nullptr;
PFN_vkCmdInsertDebugUtilsLabelEXT g_vkCmdInsertDebugUtilsLabel = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT  g_vkSetDebugUtilsObjectName  = nullptr;

namespace {
// wanted vulkan validation layer
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// the callback for validation layer messages
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {

  spdlog::error("Validation Layer: {}", pCallbackData->pMessage);
  return VK_FALSE; // Always return false per Vulkan spec
}

// helper to check validation layer support
inline bool checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : validationLayers) {
    bool layerFound = false;
    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }
    if (!layerFound)
      return false;
  }
  return true;
}

// helper to populate the debug messenger create info struct
inline void populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback; // Point to our callback above
}
} // namespace

#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

// --- Debug messenger helpers ---

static VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void
DestroyDebugUtilsMessengerEXT(VkInstance instance,
                              VkDebugUtilsMessengerEXT debugMessenger,
                              const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
    func(instance, debugMessenger, pAllocator);
}

// --- Public interface ---

void VulkanDevice::init(GLFWwindow *newWindow) {
  window = newWindow;
  createInstance();
  createSurface();
  selectPhysicalDevice();
  createLogicalDevice();
  loadDebugFunctions();
  createVmaAllocator();
  createCommandPool();
}

void VulkanDevice::cleanup() {
  vmaDestroyAllocator(allocator);
  vkDestroyCommandPool(mainDevice.logicalDevice, graphicsCommandPool, nullptr);
  vkDestroyDevice(mainDevice.logicalDevice, nullptr);
  vkDestroySurfaceKHR(instance, surface, nullptr);

  if (enableValidationLayers) {
    DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  }

  vkDestroyInstance(instance, nullptr);
}

QueueFamilyIndices VulkanDevice::getQueueFamilies() const {
  return getQueueFamilies(mainDevice.physicalDevice);
}

// --- Instance creation ---

void VulkanDevice::createInstance() {
  // Check if the system/drivers actually have the validation layers
  if (enableValidationLayers && !checkValidationLayerSupport()) {
    throw std::runtime_error("Validation layers requested, but not available!");
  }

  // info about the application
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan App";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  // create information for vkinstance
  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  // Get required extensions from GLFW
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> instanceExtensions(
      glfwExtensions, glfwExtensions + glfwExtensionCount);

  // Always request debug_utils: validation messenger uses it in debug builds,
  // and RenderDoc / GPU profilers need it for pass labels in all builds.
  instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // Verify all requested extensions are supported by the GPU
  if (!checkInstanceExtensionSupport(&instanceExtensions)) {
    throw std::runtime_error("VkInstance does not support required extensions");
  }

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  createInfo.ppEnabledExtensionNames = instanceExtensions.data();

  // Hook the layers and the early debug messenger into creation
  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    // pNext part allows us to debug the actual creation and destruction
    // of the instance
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.pNext = nullptr;
  }

  // create the vulkan instance
  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }

  // Setup the "permanent" debug messenger for the rest of the application
  if (enableValidationLayers) {
    if (CreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, nullptr,
                                     &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("Failed to set up debug messenger!");
    }
  }
}

// --- Surface creation ---

void VulkanDevice::createSurface() {
  VkResult result =
      glfwCreateWindowSurface(instance, window, nullptr, &surface);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }
}

// --- Logical device creation ---

void VulkanDevice::createLogicalDevice() {
  // get the queue family indices for the physical device
  QueueFamilyIndices indices = getQueueFamilies(mainDevice.physicalDevice);

  // vector for queue creation information
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  // set for family indices to avoid duplicate queue create info
  std::set<int> queueFamilyIndicies = {indices.graphicsFamily,
                                       indices.presentationFamily};

  // queues the logical device needs to create and info to do so
  for (int queueFamilyIndex : queueFamilyIndicies) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex =
        queueFamilyIndex; // index of the queue family to create a queue from
    queueCreateInfo.queueCount = 1; // number of queues to create
    float priority = 1.0f;
    queueCreateInfo.pQueuePriorities =
        &priority; // priority of the queues to create

    queueCreateInfos.push_back(queueCreateInfo);
  }

  // info about the logical device to create
  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(
      queueCreateInfos.size()); // number of queue create infos
  deviceCreateInfo.pQueueCreateInfos =
      queueCreateInfos.data(); // list of queue create infos
  deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(
      deviceExtensions.size()); // number of device extensions to enable
  deviceCreateInfo.ppEnabledExtensionNames =
      deviceExtensions.data(); // list of enabled device extensions

  // Physical-device features. Switched from the legacy `pEnabledFeatures`
  // single-struct form to `VkPhysicalDeviceFeatures2` + a pNext chain so we
  // can enable VK_EXT_descriptor_indexing's bindless capabilities (Phase
  // 7.2). The base struct still carries samplerAnisotropy; the chained
  // descriptor-indexing struct adds:
  //   - shaderSampledImageArrayNonUniformIndexing: lets the shader index
  //     the bindless array with a value that diverges across the warp
  //     (per-mesh material index from a push constant).
  //   - descriptorBindingPartiallyBound: most of the 4096 slots stay empty
  //     until a material registers; partial-bound lets that work without
  //     binding to every slot.
  //   - descriptorBindingSampledImageUpdateAfterBind: lets us write new
  //     image descriptors into the set after it has been bound for use in
  //     a command buffer (texture streaming, lazy loads).
  //   - descriptorBindingVariableDescriptorCount: the last binding in the
  //     set is the variable-size texture array; this allows it.
  //   - runtimeDescriptorArray: GLSL `texture2D textures[]` (no fixed size).
  VkPhysicalDeviceDescriptorIndexingFeatures diFeatures = {};
  diFeatures.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
  diFeatures.shaderSampledImageArrayNonUniformIndexing       = VK_TRUE;
  diFeatures.descriptorBindingPartiallyBound                 = VK_TRUE;
  diFeatures.descriptorBindingSampledImageUpdateAfterBind    = VK_TRUE;
  diFeatures.descriptorBindingVariableDescriptorCount        = VK_TRUE;
  diFeatures.runtimeDescriptorArray                          = VK_TRUE;

  VkPhysicalDeviceFeatures2 features2 = {};
  features2.sType        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext        = &diFeatures;
  features2.features.samplerAnisotropy = VK_TRUE;

  deviceCreateInfo.pEnabledFeatures = nullptr; // mutually exclusive w/ pNext
  deviceCreateInfo.pNext            = &features2;

  // create the logical device
  VkResult result = vkCreateDevice(mainDevice.physicalDevice, &deviceCreateInfo,
                                   nullptr, &mainDevice.logicalDevice);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device");
  }

  // queues are created at the same time as the logical device
  vkGetDeviceQueue(mainDevice.logicalDevice, indices.graphicsFamily, 0,
                   &graphicsQueue);
  vkGetDeviceQueue(mainDevice.logicalDevice, indices.presentationFamily, 0,
                   &presentationQueue);
}

// --- Phase 7.1: debug-utils function pointer loader ---

void VulkanDevice::loadDebugFunctions() {
  VkDevice dev = mainDevice.logicalDevice;
#define LOAD(name, type) \
  g_##name = reinterpret_cast<type>( \
      vkGetDeviceProcAddr(dev, #name "EXT"));
  LOAD(vkCmdBeginDebugUtilsLabel,  PFN_vkCmdBeginDebugUtilsLabelEXT)
  LOAD(vkCmdEndDebugUtilsLabel,    PFN_vkCmdEndDebugUtilsLabelEXT)
  LOAD(vkCmdInsertDebugUtilsLabel, PFN_vkCmdInsertDebugUtilsLabelEXT)
  LOAD(vkSetDebugUtilsObjectName,  PFN_vkSetDebugUtilsObjectNameEXT)
#undef LOAD
  if (g_vkCmdBeginDebugUtilsLabel)
    spdlog::info("VK_EXT_debug_utils loaded — RenderDoc labels active");
  else
    spdlog::warn("VK_EXT_debug_utils not available — labels disabled");
}

// --- Command pool creation ---

void VulkanDevice::createCommandPool() {
  // get indices of queue families to create command pool for
  QueueFamilyIndices queueFamilyIndices =
      getQueueFamilies(mainDevice.physicalDevice);

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex =
      queueFamilyIndices.graphicsFamily; // command buffers from this pool will
                                         // be submitted to this queue family
  // create graphics queue family command pool
  VkResult result = vkCreateCommandPool(mainDevice.logicalDevice, &poolInfo,
                                        nullptr, &graphicsCommandPool);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool");
  }
}

void VulkanDevice::createVmaAllocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice = mainDevice.physicalDevice;
  allocatorInfo.device = mainDevice.logicalDevice;
  allocatorInfo.instance = instance;
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

  if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
    spdlog::critical("Failed to create VMA Allocator");
    throw std::runtime_error("Failed to create VMA Allocator");
  }
}

// --- Physical device selection ---

void VulkanDevice::selectPhysicalDevice() {
  // enumerate physical devices the vkinstance can access
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("Failed to find a GPU with Vulkan support");
  }

  // get list of physical devices
  std::vector<VkPhysicalDevice> deviceList(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, deviceList.data());

  VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
  int highestScore = -1;
  for (const auto &device : deviceList) {
    if (checkDeviceSuitable(device)) {
      int score = rateDeviceSuitability(device);
      if (score > highestScore) {
        bestDevice = device;
        highestScore = score;
      }
    }
  }

  if (bestDevice != VK_NULL_HANDLE) {
    mainDevice.physicalDevice = bestDevice;
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(mainDevice.physicalDevice, &deviceProperties);

    spdlog::info("Selected GPU: {}", deviceProperties.deviceName);
    spdlog::info("Device Type: {}",
                 static_cast<int>(deviceProperties.deviceType));
    spdlog::info("Total Devices Found: {}", deviceCount);
    spdlog::info("API Version: {}.{}.{}",
                 VK_VERSION_MAJOR(deviceProperties.apiVersion),
                 VK_VERSION_MINOR(deviceProperties.apiVersion),
                 VK_VERSION_PATCH(deviceProperties.apiVersion));
  } else {
    spdlog::critical("Failed to find a suitable GPU");
    throw std::runtime_error("Failed to find a suitable GPU");
  }
}

int VulkanDevice::rateDeviceSuitability(VkPhysicalDevice device) {
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  int score = 0;
  // Discrete GPUs have a significant performance advantage
  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1000;
  }
  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
    score += 100;
  }
  return score;
}

// --- Support / checker functions ---

bool VulkanDevice::checkInstanceExtensionSupport(
    std::vector<const char *> *checkExtensions) {

  // get all available extensions
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

  // create a list of all vkextension properties
  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                         extensions.data());

  // check if all extensions in checkExtensions are in the list of available
  for (const auto &checkExtension : *checkExtensions) {
    bool hasExtension = false;

    for (const auto &extension : extensions) {
      if (strcmp(checkExtension, extension.extensionName) == 0) {
        hasExtension = true;
        break;
      }
    }
    if (!hasExtension) {
      return false;
    }
  }
  return true;
}

bool VulkanDevice::checkDeviceExtensionSupport(VkPhysicalDevice device) {
  // get device extension count
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);
  if (extensionCount == 0) {
    return false;
  }
  // populate list of extensions
  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       extensions.data());
  // check for extensions
  for (const auto &deviceExtension : deviceExtensions) {
    bool hasExtension = false;
    for (const auto &extension : extensions) {
      if (strcmp(deviceExtension, extension.extensionName) == 0) {
        hasExtension = true;
        break;
      }
    }
    if (!hasExtension) {
      return false;
    }
  }
  return true;
}

bool VulkanDevice::checkDeviceSuitable(VkPhysicalDevice device) {
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  QueueFamilyIndices indices = getQueueFamilies(device);

  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainValid = false;

  if (extensionsSupported) {
    SwapChainDetails swapChainDetails = getSwapChainDetails(device);
    swapChainValid = !swapChainDetails.formats.empty() &&
                     !swapChainDetails.presentModes.empty();
  }

  return indices.isValid() && extensionsSupported && swapChainValid &&
         deviceFeatures.samplerAnisotropy;
}

QueueFamilyIndices
VulkanDevice::getQueueFamilies(VkPhysicalDevice device) const {
  QueueFamilyIndices indices;

  // get all the queue families of the device
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilyList(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilyList.data());

  // go through each queue family and check if it has the required queues
  int i = 0;

  for (const auto &queueFamily : queueFamilyList) {
    // check if queue family has graphics capabilities
    if (queueFamily.queueCount > 0 &&
        queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i; // set graphics family index
    }

    // check if queue family supports presentation
    VkBool32 presentationSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                         &presentationSupport);
    if (queueFamily.queueCount > 0 && presentationSupport) {
      indices.presentationFamily = i; // set presentation family index
    }

    // check if queue family is valid, if so break loop
    if (indices.isValid()) {
      break;
    }

    i++;
  }
  return indices;
}

SwapChainDetails
VulkanDevice::getSwapChainDetails(VkPhysicalDevice device) const {
  SwapChainDetails swapChainDetails;

  // capabilities
  // getting the surface capabilities of the device and surface
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      device, surface, &swapChainDetails.surfaceCapabilities);

  // formats
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

  // if format count is not zero, get list of surface formats
  if (formatCount != 0) {
    swapChainDetails.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         swapChainDetails.formats.data());
  }

  // presentation modes
  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                            nullptr);

  // if presentation mode count is not zero, get list of presentation modes
  if (presentModeCount != 0) {
    swapChainDetails.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &presentModeCount,
        swapChainDetails.presentModes.data());
  }

  return swapChainDetails;
}

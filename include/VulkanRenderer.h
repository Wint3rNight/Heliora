#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan_core.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "stb_image.h"

#include "Mesh.h"
#include "MeshModel.h"
#include "Utilities.h"
#include "VulkanValidation.h"

class VulkanRenderer {
public:
  VulkanRenderer();

  int init(GLFWwindow *newWindow);
  void draw();
  int createMeshModel(const std::string &modelFile);
  void updateModel(int modelId, const glm::mat4 &newModel);
  void updateCameraView(const glm::mat4 &viewMatrix);
  void cleanup();

  ~VulkanRenderer();

private:
  GLFWwindow *window = nullptr;

  int currentFrame = 0;

  // scene objects
  std::vector<MeshModel> modelList;

  // scene settings
  struct UboViewProjection {
    glm::mat4 projection;
    glm::mat4 view;
  } uboViewProjection;

  // vk components
  // main
  VkInstance instance = VK_NULL_HANDLE;

  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
  struct {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice = VK_NULL_HANDLE;
  } mainDevice;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue presentationQueue = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;

  std::vector<SwapChainImage> swapChainImages;
  std::vector<VkFramebuffer> swapChainFramebuffers;
  std::vector<VkCommandBuffer> commandBuffers;

  std::vector<VkImage> colorBufferImage;
  std::vector<VkDeviceMemory> colorBufferImageMemory;
  std::vector<VkImageView> colorBufferImageView;

  std::vector<VkImage> depthBufferImage;
  std::vector<VkDeviceMemory> depthBufferImageMemory;
  std::vector<VkImageView> depthBufferImageView;

  std::vector<VkImageView> textureImageViews;

  // descriptors
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout samplerSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout inputSetLayout = VK_NULL_HANDLE;
  VkPushConstantRange pushConstantRange = {};

  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool samplerDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool inputDescriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> descriptorSets;
  std::vector<VkDescriptorSet> samplerDescriptorSets;
  std::vector<VkDescriptorSet> inputDescriptorSets;

  std::vector<VkBuffer> vpUniformBuffers;
  std::vector<VkDeviceMemory> vpUniformBufferMemory;



  // Assets

  VkSampler textureSampler = VK_NULL_HANDLE;
  std::vector<VkImage> textureImages;
  std::vector<VkDeviceMemory> textureImageMemory;

  // pipeline
  VkPipeline graphicsPipeline = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkPipeline secondPipeline = VK_NULL_HANDLE;
  VkPipelineLayout secondPipelineLayout = VK_NULL_HANDLE;

  // pools
  VkCommandPool graphicsCommandPool = VK_NULL_HANDLE;

  // utils
  VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D swapChainExtent = {};

  // synchronization
  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkFence> drawFences;
  std::vector<VkFence> imagesInFlight;

  // vulkan functions
  // create functions
  void createInstance();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();
  void recreateSwapChain();
  void cleanupSwapChain();
  void createRenderPass();
  void createDescriptorSetLayout();
  void createPushConstantRange();
  void createGraphicsPipeline();
  void createColorBufferImage();
  void createDepthBufferImage();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSynchronization();
  void createTextureSampler();

  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  void createInputDescriptorSets();

  void updateUniformBuffers(uint32_t imageIndex);

  // record functions
  void recordCommands(uint32_t currentImage);

  // get functions
  void getPhysicalDevice();
  int rateDeviceSuitability(VkPhysicalDevice device);

  // support functions
  // checker functions
  bool checkInstanceExtensionSupport(std::vector<const char *> *checkExtensions);
  bool checkDeviceExtensionSupport(VkPhysicalDevice device);
  bool checkDeviceSuitable(VkPhysicalDevice device);

  // getter functions
  QueueFamilyIndices getQueueFamilies(VkPhysicalDevice device);
  SwapChainDetails getSwapChainDetails(VkPhysicalDevice device);

  // choose functions
  VkSurfaceFormatKHR
  chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats);
  VkPresentModeKHR chooseBestPresentationMode(
      const std::vector<VkPresentModeKHR> &presentationModes);
  VkExtent2D
  chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities);
  VkFormat chooseSupportedFormat(const std::vector<VkFormat> &formats,
                                 VkImageTiling tiling,
                                 VkFormatFeatureFlags featuresFlags);

  // support create functions
  VkImage createImage(uint32_t width, uint32_t height, VkFormat formats,
                      VkImageTiling tiling, VkImageUsageFlags useFlags,
                      VkMemoryPropertyFlags propFlags,
                      VkDeviceMemory *imageMemory);

  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags);

  VkShaderModule createShaderModule(const std::vector<char> &code);

  int createTextureImage(const std::string &filename);
  int createTexture(const std::string &filename);
  int createTextureDescriptor(VkImageView textureImage);

  // Loader functions
  stbi_uc *loadTextureFile(const std::string &filename, int *width, int *height,
                           VkDeviceSize *imageSize);
};

#pragma once

#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>

#include "Utilities.h"

struct Model {
  glm::mat4 model;
};

class Mesh {
public:
  Mesh();
  Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice,
       VkQueue transferQueue, VkCommandPool transferCommandPool,
       std::vector<Vertex> *vertices, std::vector<uint32_t> *indices, int newTexId);

  void setModel(const glm::mat4 &newModel);
  const Model &getModel() const;

  int getTexId() const;

  int getVertexCount() const;
  VkBuffer getVertexBuffer() const;
  int getIndexCount() const;
  VkBuffer getIndexBuffer() const;

  void destroyBuffers();

  ~Mesh();

private:
  Model model;

  int texId;

  int vertexCount;
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;

  int indexCount;
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;

  VkPhysicalDevice physicalDevice;
  VkDevice device;

  void createVertexBuffer(VkQueue transferQueue,
                          VkCommandPool transferCommandPool,
                          std::vector<Vertex> *vertices);
  void createIndexBuffer(VkQueue transferQueue,
                         VkCommandPool transferCommandPool,
                         std::vector<uint32_t> *indices);
};

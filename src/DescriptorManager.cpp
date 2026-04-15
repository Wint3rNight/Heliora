#include "DescriptorManager.h"
#include "Utilities.h"
#include "VulkanSwapchain.h"

#include <cstring>
#include <stdexcept>
#include <vector>

// --- UBO struct (matches shader layout) ---
struct UboViewProjection {
  glm::mat4 projection;
  glm::mat4 view;
};

// --- Public interface ---

void DescriptorManager::init(VkDevice device, VkPhysicalDevice physicalDevice,
                             size_t swapchainImageCount) {
  createDescriptorSetLayout(device);
  createPushConstantRange();
  createUniformBuffers(device, physicalDevice, swapchainImageCount);
  createDescriptorPool(device, swapchainImageCount);
  createDescriptorSets(device, swapchainImageCount);
}

void DescriptorManager::cleanup(VkDevice device, size_t swapchainImageCount) {
  // Destroy input descriptor pool and layout
  vkDestroyDescriptorPool(device, inputDescriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, inputSetLayout, nullptr);

  // Destroy sampler descriptor pool and layout
  vkDestroyDescriptorPool(device, samplerDescriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, samplerSetLayout, nullptr);

  // Destroy uniform descriptor pool and layout
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

  // Destroy uniform buffers
  for (size_t i = 0; i < swapchainImageCount; i++) {
    vkDestroyBuffer(device, vpUniformBuffers[i], nullptr);
    vkFreeMemory(device, vpUniformBufferMemory[i], nullptr);
  }
}

void DescriptorManager::updateUniformBuffer(VkDevice device, size_t imageIndex,
                                            const void *data, size_t size) {
  void *mapped;
  vkMapMemory(device, vpUniformBufferMemory[imageIndex], 0, size, 0, &mapped);
  memcpy(mapped, data, size);
  vkUnmapMemory(device, vpUniformBufferMemory[imageIndex]);
}

int DescriptorManager::createTextureDescriptor(VkDevice device,
                                               VkImageView textureImageView,
                                               VkSampler sampler) {
  VkDescriptorSet descriptorSet;
  // allocate descriptor set for texture
  VkDescriptorSetAllocateInfo setAllocInfo = {};
  setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  setAllocInfo.descriptorPool =
      samplerDescriptorPool;           // descriptor pool to allocate from
  setAllocInfo.descriptorSetCount = 1; // number of descriptor sets to allocate
  setAllocInfo.pSetLayouts =
      &samplerSetLayout; // layout to use for allocated descriptor set

  // allocate descriptor set
  VkResult result =
      vkAllocateDescriptorSets(device, &setAllocInfo, &descriptorSet);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor set for texture");
  }

  // texture image info
  VkDescriptorImageInfo imageInfo = {};
  imageInfo.imageLayout =
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // layout of image during shader
                                                 // access
  imageInfo.imageView = textureImageView; // image view to write to descriptor
  imageInfo.sampler = sampler;           // sampler to write to descriptor

  // descriptor write info
  VkWriteDescriptorSet descriptorWrite = {};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = descriptorSet; // descriptor set to update
  descriptorWrite.dstBinding = 0;         // binding of the descriptor to update
  descriptorWrite.dstArrayElement = 0;    // first index in array to update
  descriptorWrite.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // type of descriptor
  descriptorWrite.descriptorCount = 1;     // number of descriptors to update
  descriptorWrite.pImageInfo = &imageInfo; // image info to write to descriptor

  // update descriptor set with texture image info
  vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

  // add descriptor set to vector for reference
  samplerDescriptorSets.push_back(descriptorSet);
  // return descriptor set index for reference
  return static_cast<int>(samplerDescriptorSets.size() - 1);
}

void DescriptorManager::recreateInputSets(VkDevice device,
                                          const VulkanSwapchain &swapchain) {
  vkResetDescriptorPool(device, inputDescriptorPool, 0);
  createInputDescriptorSets(device, swapchain);
}

// --- Descriptor set layout creation ---

void DescriptorManager::createDescriptorSetLayout(VkDevice device) {
  // UNIFORM VALUES DESCRIPTOR SET LAYOUT

  // mvp binding info
  VkDescriptorSetLayoutBinding vpLayoutBinding = {};
  vpLayoutBinding.binding = 0; // binding number referenced in the shader
  vpLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // type of resource (uniform buffer)
  vpLayoutBinding.descriptorCount =
      1; // number of resources for binding, can be more than 1 for arrays
  vpLayoutBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT; // shader stage to bind the resources to
  vpLayoutBinding.pImmutableSamplers =
      nullptr; // used for image sampling, not used for uniform buffers

  std::vector<VkDescriptorSetLayoutBinding> layoutBindings = {vpLayoutBinding};

  // create info for descriptor set layout creation
  VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
  layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCreateInfo.bindingCount = static_cast<uint32_t>(
      layoutBindings.size()); // number of bindings in the descriptor set
  layoutCreateInfo.pBindings =
      layoutBindings.data(); // list of bindings in the descriptor set

  // create the descriptor set layout
  VkResult result = vkCreateDescriptorSetLayout(device, &layoutCreateInfo,
                                                nullptr, &descriptorSetLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor set layout");
  }

  // CREATE TEXTURE SAMPLER DESCRIPTOR SET LAYOUT
  // texture binding info
  VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
  samplerLayoutBinding.binding = 0; // binding number referenced in the shader
  samplerLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // type of resource
  samplerLayoutBinding.descriptorCount =
      1; // number of resources for binding, can be more than 1 for arrays
  samplerLayoutBinding.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT; // shader stage to bind the resources to
  samplerLayoutBinding.pImmutableSamplers =
      nullptr; // used for image sampling, not used for uniform buffers

  // create a descriptor set layout with given binding for texture sampler
  VkDescriptorSetLayoutCreateInfo textureLayoutCreateInfo = {};
  textureLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  textureLayoutCreateInfo.bindingCount = 1;
  textureLayoutCreateInfo.pBindings =
      &samplerLayoutBinding; // list of bindings in the descriptor set

  // create the descriptor set layout
  result = vkCreateDescriptorSetLayout(device, &textureLayoutCreateInfo,
                                       nullptr, &samplerSetLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to create texture sampler descriptor set layout");
  }

  // CREATE INPUT ATTACHMENT IMAGE DESCRIPTOR SET LAYOUT

  // image input attachment binding info
  VkDescriptorSetLayoutBinding colorInputLayoutBinding = {};
  colorInputLayoutBinding.binding = 0;
  colorInputLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; // type of resource
  colorInputLayoutBinding.descriptorCount =
      1; // number of resources for binding, can be more than 1 for arrays
  colorInputLayoutBinding.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT; // shader stage to bind the resources to
  colorInputLayoutBinding.pImmutableSamplers =
      nullptr; // used for image sampling, not used for uniform buffers

  // depth input attachment binding info
  VkDescriptorSetLayoutBinding depthInputLayoutBinding = {};
  depthInputLayoutBinding.binding = 1;
  depthInputLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; // type of resource
  depthInputLayoutBinding.descriptorCount =
      1; // number of resources for binding, can be more than 1 for arrays
  depthInputLayoutBinding.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT; // shader stage to bind the resources to
  depthInputLayoutBinding.pImmutableSamplers =
      nullptr; // used for image sampling, not used for uniform buffers

  // array on input attachment bindings
  std::vector<VkDescriptorSetLayoutBinding> inputBindings = {
      colorInputLayoutBinding, depthInputLayoutBinding};

  // create a descriptor set layout with given bindings for input attachments
  VkDescriptorSetLayoutCreateInfo inputLayoutCreateInfo = {};
  inputLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  inputLayoutCreateInfo.bindingCount = static_cast<uint32_t>(
      inputBindings.size()); // number of bindings in the descriptor set
  inputLayoutCreateInfo.pBindings =
      inputBindings.data(); // list of bindings in the descriptor set

  // create the descriptor set layout
  result = vkCreateDescriptorSetLayout(device, &inputLayoutCreateInfo, nullptr,
                                       &inputSetLayout);
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to create input attachment descriptor set layout");
  }
}

// --- Push constant range ---

void DescriptorManager::createPushConstantRange() {
  // define the push constant range(no creation needed because it is not a
  // vulkan object)
  pushConstantRange.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT; // shader stage to bind the push constant to
  pushConstantRange.offset = 0;   // offset of the push constant range
  pushConstantRange.size = sizeof(glm::mat4); // size of the push constant range
}

// --- Uniform buffer creation ---

void DescriptorManager::createUniformBuffers(VkDevice device,
                                             VkPhysicalDevice physicalDevice,
                                             size_t swapchainImageCount) {
  VkDeviceSize vpBufferSize = sizeof(UboViewProjection);

  vpUniformBuffers.resize(swapchainImageCount);
  vpUniformBufferMemory.resize(swapchainImageCount);

  for (size_t i = 0; i < swapchainImageCount; i++) {
    createBuffer(physicalDevice, device, vpBufferSize,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &vpUniformBuffers[i], &vpUniformBufferMemory[i]);
  }
}

// --- Descriptor pool creation ---

void DescriptorManager::createDescriptorPool(VkDevice device,
                                             size_t swapchainImageCount) {
  // CREATE UNIFORM DESCRIPTOR POOL
  VkDescriptorPoolSize vpPoolSize = {};
  vpPoolSize.type =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // type of resource in pool
  vpPoolSize.descriptorCount = static_cast<uint32_t>(
      vpUniformBuffers.size()); // number of descriptors in pool

  std::vector<VkDescriptorPoolSize> descriptorPoolSizes = {vpPoolSize};

  // create info for descriptor pool creation
  VkDescriptorPoolCreateInfo poolCreateInfo = {};
  poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCreateInfo.maxSets = static_cast<uint32_t>(
      swapchainImageCount); // max number of descriptor
                            // sets that can be allocated
  poolCreateInfo.poolSizeCount = static_cast<uint32_t>(
      descriptorPoolSizes.size()); // number of descriptor types in pool
  poolCreateInfo.pPoolSizes =
      descriptorPoolSizes.data(); // list of descriptor types and counts in pool

  // create descriptor pool
  VkResult result =
      vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }

  // CREATE COMBINED IMAGE SAMPLER DESCRIPTOR POOL
  // texture sampler pool size info
  VkDescriptorPoolSize samplerPoolSize = {};
  samplerPoolSize.type =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // type of resource in pool
  samplerPoolSize.descriptorCount =
      MAX_OBJECTS; // number of descriptors in pool

  // create info for combined image sampler descriptor pool creation
  VkDescriptorPoolCreateInfo samplerPoolCreateInfo = {};
  samplerPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  samplerPoolCreateInfo.maxSets = MAX_OBJECTS; // max number of descriptor sets
  samplerPoolCreateInfo.poolSizeCount =
      1; // number of descriptor types in pool
  samplerPoolCreateInfo.pPoolSizes =
      &samplerPoolSize; // list of descriptor types and counts in pool

  // create combined image sampler descriptor pool
  result = vkCreateDescriptorPool(device, &samplerPoolCreateInfo, nullptr,
                                  &samplerDescriptorPool);
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to create combined image sampler descriptor pool");
  }

  // CREATE INPUT ATTACHMENT DESCRIPTOR POOL
  //  color attachment pool size info
  VkDescriptorPoolSize colorInputPoolSize = {};
  colorInputPoolSize.type =
      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; // type of resource in pool
  colorInputPoolSize.descriptorCount =
      static_cast<uint32_t>(swapchainImageCount);

  // depth attachment pool size info
  VkDescriptorPoolSize depthInputPoolSize = {};
  depthInputPoolSize.type =
      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; // type of resource in pool
  depthInputPoolSize.descriptorCount =
      static_cast<uint32_t>(swapchainImageCount);

  // list of pool sizes for input attachment descriptor pool
  std::vector<VkDescriptorPoolSize> inputPoolSizes = {colorInputPoolSize,
                                                      depthInputPoolSize};

  // create info for input attachment descriptor pool creation
  VkDescriptorPoolCreateInfo inputPoolCreateInfo = {};
  inputPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  inputPoolCreateInfo.maxSets = static_cast<uint32_t>(
      swapchainImageCount); // max number of descriptor sets
  inputPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(
      inputPoolSizes.size()); // number of descriptor types in pool
  inputPoolCreateInfo.pPoolSizes =
      inputPoolSizes.data(); // list of descriptor types and counts in pool

  // create input attachment descriptor pool
  result = vkCreateDescriptorPool(device, &inputPoolCreateInfo, nullptr,
                                  &inputDescriptorPool);
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to create input attachment descriptor pool");
  }
}

// --- Descriptor set creation ---

void DescriptorManager::createDescriptorSets(VkDevice device,
                                             size_t swapchainImageCount) {
  descriptorSets.resize(swapchainImageCount);

  std::vector<VkDescriptorSetLayout> setLayouts(swapchainImageCount,
                                                descriptorSetLayout);

  VkDescriptorSetAllocateInfo setAllocInfo = {};
  setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  setAllocInfo.descriptorPool =
      descriptorPool; // descriptor pool to allocate from
  setAllocInfo.descriptorSetCount =
      static_cast<uint32_t>(swapchainImageCount);
  setAllocInfo.pSetLayouts =
      setLayouts.data(); // layout to use for each allocated descriptor set

  // allocate descriptor sets, multiple
  VkResult result = vkAllocateDescriptorSets(device, &setAllocInfo,
                                             descriptorSets.data());
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor sets");
  }

  // update descriptor sets with buffer info for each swap chain image
  for (size_t i = 0; i < swapchainImageCount; i++) {
    // view projection descriptor
    //  buffer info and data offset info
    VkDescriptorBufferInfo vpBufferInfo = {};
    vpBufferInfo.buffer = vpUniformBuffers[i]; // buffer to bind to descriptor
    vpBufferInfo.offset = 0;                   // offset of data in buffer
    vpBufferInfo.range = sizeof(UboViewProjection); // size of data in buffer

    // data about connection bwtn descriptor set and buffer info to update
    // descriptor set with
    VkWriteDescriptorSet vpSetWrite = {};
    vpSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vpSetWrite.dstSet = descriptorSets[i]; // descriptor set to update
    vpSetWrite.dstBinding = 0;
    vpSetWrite.dstArrayElement = 0; // first index in array to update
    vpSetWrite.descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // type of descriptor
    vpSetWrite.descriptorCount =
        1; // number of descriptors to update (size of array)
    vpSetWrite.pBufferInfo =
        &vpBufferInfo; // buffer info to write to descriptor

    // list of descriptor set writes to update
    std::vector<VkWriteDescriptorSet> setWrites = {vpSetWrite};

    // update the descriptor set with new buffer info
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(setWrites.size()),
                           setWrites.data(), 0, nullptr);
  }
}

// --- Input descriptor set creation ---

void DescriptorManager::createInputDescriptorSets(
    VkDevice device, const VulkanSwapchain &swapchain) {
  size_t imageCount = swapchain.getImageCount();
  inputDescriptorSets.resize(imageCount);

  std::vector<VkDescriptorSetLayout> setLayouts(imageCount, inputSetLayout);

  // create info for allocating descriptor sets
  VkDescriptorSetAllocateInfo setAllocInfo = {};
  setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  setAllocInfo.descriptorPool =
      inputDescriptorPool; // descriptor pool to allocate from
  setAllocInfo.descriptorSetCount =
      static_cast<uint32_t>(imageCount);
  setAllocInfo.pSetLayouts =
      setLayouts.data(); // list of descriptor set layouts to use for each
                         // allocated descriptor set

  // allocate descriptor sets
  VkResult result = vkAllocateDescriptorSets(device, &setAllocInfo,
                                             inputDescriptorSets.data());
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to allocate input attachment descriptor sets");
  }

  // update descriptor sets with image info for each swap chain image
  for (size_t i = 0; i < imageCount; i++) {
    // color attachment descriptor
    VkDescriptorImageInfo colorAttachmentDescriptor = {};
    colorAttachmentDescriptor.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorAttachmentDescriptor.imageView = swapchain.getColorBufferView(i);
    colorAttachmentDescriptor.sampler = VK_NULL_HANDLE;

    // color attachment descriptor write
    VkWriteDescriptorSet colorWrite = {};
    colorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    colorWrite.dstSet = inputDescriptorSets[i]; // descriptor set to update
    colorWrite.dstBinding = 0;
    colorWrite.dstArrayElement = 0;
    colorWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    colorWrite.descriptorCount = 1;
    colorWrite.pImageInfo = &colorAttachmentDescriptor;

    // depth attachment descriptor write
    VkDescriptorImageInfo depthAttachmentDescriptor = {};
    depthAttachmentDescriptor.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depthAttachmentDescriptor.imageView = swapchain.getDepthBufferView(i);
    depthAttachmentDescriptor.sampler = VK_NULL_HANDLE;

    // depth attachment descriptor
    VkWriteDescriptorSet depthWrite = {};
    depthWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    depthWrite.dstSet = inputDescriptorSets[i];
    depthWrite.dstBinding = 1;
    depthWrite.dstArrayElement = 0;
    depthWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    depthWrite.descriptorCount = 1;
    depthWrite.pImageInfo = &depthAttachmentDescriptor;

    // list of descriptor set writes to update
    std::vector<VkWriteDescriptorSet> setWrites = {colorWrite, depthWrite};

    // update the descriptor set with new image info
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(setWrites.size()),
                           setWrites.data(), 0, nullptr);
  }
}

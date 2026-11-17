#include "SkinningPass.h"

#include "VulkanDebug.h"

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

struct SkinningPush {
  uint32_t vertexCount;
  uint32_t arenaFirstVertex;
  uint32_t writeArena;
};

VkShaderModule loadComputeSpv(VkDevice device, const std::string &relPath) {
  std::string found;
  for (const std::string &candidate : {relPath, "../" + relPath})
    if (std::filesystem::exists(candidate)) {
      found = candidate;
      break;
    }
  if (found.empty())
    throw std::runtime_error("Compute shader not found: " + relPath);
  const std::vector<char> code = readFile(found);
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shader module: " + found);
  return module;
}

// Device-local storage holding data, through a staging copy.
void uploadStorage(VulkanDevice &device, const void *data, VkDeviceSize size,
                   AllocatedBuffer &out) {
  AllocatedBuffer staging;
  createBuffer(device.getAllocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VMA_MEMORY_USAGE_AUTO, RENDER_UPLOAD_ALLOCATION_FLAGS, &staging);
  uploadToAllocation(device.getAllocator(), staging.getAllocation(), data, size);
  createBuffer(device.getAllocator(), size,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, RENDER_DEVICE_ALLOCATION_FLAGS,
               &out);
  copyBuffer(device.getLogicalDevice(), device.getGraphicsQueue(),
             device.getGraphicsCommandPool(), staging.get(), out.get(), size);
}

void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage,
             VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
             VkAccessFlags2 dstAccess) {
  VkMemoryBarrier2 b{};
  b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
  b.srcStageMask = srcStage;
  b.srcAccessMask = srcAccess;
  b.dstStageMask = dstStage;
  b.dstAccessMask = dstAccess;
  VkDependencyInfo dependency{};
  dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

} // namespace

void SkinningPass::create(VulkanDevice &newDevice, VkPipelineCache pipelineCache) {
  cleanup();
  device = &newDevice;
  const VkDevice dev = device->getLogicalDevice();

  std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layoutCI{};
  layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutCI.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(dev, &layoutCI, nullptr, &setLayout) != VK_SUCCESS)
    throw std::runtime_error("Failed to create the skinning set layout");

  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.size = sizeof(SkinningPush);
  VkPipelineLayoutCreateInfo pipelineLayoutCI{};
  pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount = 1;
  pipelineLayoutCI.pSetLayouts = &setLayout;
  pipelineLayoutCI.pushConstantRangeCount = 1;
  pipelineLayoutCI.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(dev, &pipelineLayoutCI, nullptr, &pipelineLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create the skinning pipeline layout");

  const VkShaderModule shader = loadComputeSpv(dev, "Shaders/skinning.comp.spv");
  VkComputePipelineCreateInfo pipelineCI{};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineCI.stage.module = shader;
  pipelineCI.stage.pName = "main";
  pipelineCI.layout = pipelineLayout;
  const VkResult result =
      vkCreateComputePipelines(dev, pipelineCache, 1, &pipelineCI, nullptr, &pipeline);
  vkDestroyShaderModule(dev, shader, nullptr);
  if (result != VK_SUCCESS)
    throw std::runtime_error("Failed to create the skinning pipeline");
}

void SkinningPass::cleanup() {
  if (!device)
    return;
  const VkDevice dev = device->getLogicalDevice();
  for (Copy &copy : copies)
    if (copy.pool)
      vkDestroyDescriptorPool(dev, copy.pool, nullptr);
  copies.clear();
  meshes.clear();
  if (pipeline)
    vkDestroyPipeline(dev, pipeline, nullptr);
  if (pipelineLayout)
    vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
  if (setLayout)
    vkDestroyDescriptorSetLayout(dev, setLayout, nullptr);
  pipeline = VK_NULL_HANDLE;
  pipelineLayout = VK_NULL_HANDLE;
  setLayout = VK_NULL_HANDLE;
  device = nullptr;
}

int SkinningPass::addMesh(const std::vector<Vertex> &bindVertices,
                          const std::vector<VertexSkin> &skin, size_t boneCount) {
  if (!device || bindVertices.empty() || skin.size() != bindVertices.size())
    return -1;
  Mesh mesh;
  uploadStorage(*device, bindVertices.data(), sizeof(Vertex) * bindVertices.size(),
                mesh.bind);
  uploadStorage(*device, skin.data(), sizeof(VertexSkin) * skin.size(), mesh.skin);
  mesh.vertexCount = static_cast<uint32_t>(bindVertices.size());
  mesh.matrixCount = std::max<size_t>(boneCount, 1);
  meshes.push_back(std::move(mesh));
  return static_cast<int>(meshes.size()) - 1;
}

int SkinningPass::addCopy(int mesh) {
  if (!device || mesh < 0 || mesh >= static_cast<int>(meshes.size()))
    return -1;
  const VkDevice dev = device->getLogicalDevice();
  Copy copy;
  copy.mesh = mesh;
  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                static_cast<uint32_t>(5 * MAX_FRAMES_DRAWS)};
  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.maxSets = MAX_FRAMES_DRAWS;
  poolCI.poolSizeCount = 1;
  poolCI.pPoolSizes = &poolSize;
  if (vkCreateDescriptorPool(dev, &poolCI, nullptr, &copy.pool) != VK_SUCCESS)
    throw std::runtime_error("Failed to create a skinning descriptor pool");
  std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_DRAWS, setLayout);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = copy.pool;
  allocInfo.descriptorSetCount = MAX_FRAMES_DRAWS;
  allocInfo.pSetLayouts = layouts.data();
  copy.sets.resize(MAX_FRAMES_DRAWS);
  if (vkAllocateDescriptorSets(dev, &allocInfo, copy.sets.data()) != VK_SUCCESS) {
    vkDestroyDescriptorPool(dev, copy.pool, nullptr);
    throw std::runtime_error("Failed to allocate the skinning descriptor sets");
  }
  // Identity until the first pose: the bind pose.
  const size_t matrixCount = meshes[mesh].matrixCount;
  const std::vector<glm::mat4> identity(matrixCount, glm::mat4(1.0f));
  copy.matrixBuffers.resize(MAX_FRAMES_DRAWS);
  for (AllocatedBuffer &buffer : copy.matrixBuffers) {
    createBuffer(device->getAllocator(), sizeof(glm::mat4) * matrixCount,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                 RENDER_UPLOAD_ALLOCATION_FLAGS, &buffer);
    uploadToAllocation(device->getAllocator(), buffer.getAllocation(),
                       identity.data(), sizeof(glm::mat4) * matrixCount);
  }
  copies.push_back(std::move(copy));
  return static_cast<int>(copies.size()) - 1;
}

size_t SkinningPass::vertexCount(int copy) const {
  return hasCopy(copy) ? meshes[copies[copy].mesh].vertexCount : 0;
}

void SkinningPass::updateMatrices(int copy, int frame,
                                  const std::vector<glm::mat4> &skinning) {
  if (!hasCopy(copy))
    return;
  const size_t matrixCount = meshes[copies[copy].mesh].matrixCount;
  if (skinning.size() < matrixCount)
    return;
  uploadToAllocation(device->getAllocator(),
                     copies[copy].matrixBuffers[frame].getAllocation(),
                     skinning.data(), sizeof(glm::mat4) * matrixCount);
}

void SkinningPass::record(VkCommandBuffer cmd, int frame,
                          const std::vector<Target> &targets) {
  std::vector<const Target *> valid;
  for (const Target &target : targets)
    if (hasCopy(target.copy) && target.meshVertices != VK_NULL_HANDLE)
      valid.push_back(&target);
  if (valid.empty())
    return;

  vkdbgBeginLabel(cmd, "Skinning", 0.9f, 0.5f, 0.2f);
  // The frames before drew from these vertices (vertex input) on this queue;
  // they're done reading before the shader writes.
  barrier(cmd,
          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
              VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
          0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  for (const Target *target : valid) {
    const Copy &copy = copies[target->copy];
    const Mesh &mesh = meshes[copy.mesh];
    const bool writeArena = target->arena != VK_NULL_HANDLE && target->arenaFirstVertex >= 0;
    // This slot's set was last used by the frame two before, which its
    // fence has finished; the shared buffer may have been replaced since.
    const std::array<VkDescriptorBufferInfo, 5> infos = {{
        {mesh.bind.get(), 0, VK_WHOLE_SIZE},
        {mesh.skin.get(), 0, VK_WHOLE_SIZE},
        {copy.matrixBuffers[frame].get(), 0, VK_WHOLE_SIZE},
        {target->meshVertices, 0, VK_WHOLE_SIZE},
        {writeArena ? target->arena : target->meshVertices, 0, VK_WHOLE_SIZE},
    }};
    std::array<VkWriteDescriptorSet, 5> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = copy.sets[frame];
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device->getLogicalDevice(),
                           static_cast<uint32_t>(writes.size()), writes.data(), 0,
                           nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                            1, &copy.sets[frame], 0, nullptr);
    const SkinningPush push{mesh.vertexCount,
                            static_cast<uint32_t>(std::max(target->arenaFirstVertex, 0)),
                            writeArena ? 1u : 0u};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);
    // Copies write different vertices, so they need nothing between them.
    vkCmdDispatch(cmd, (mesh.vertexCount + 63) / 64, 1, 1);
  }
  // This frame's G-buffer and, later on this queue, its shadows read them.
  barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
              VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
          VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
  vkdbgEndLabel(cmd);
}

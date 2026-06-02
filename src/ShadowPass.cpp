#include "ShadowPass.h"

#include "VulkanDebug.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace {
void extractFrustumPlanes(const glm::mat4 &vp, glm::vec4 planes[6]) {
  glm::mat4 t = glm::transpose(vp);
  planes[0] = t[3] + t[0];
  planes[1] = t[3] - t[0];
  planes[2] = t[3] + t[1];
  planes[3] = t[3] - t[1];
  planes[4] = t[3] + t[2];
  planes[5] = t[3] - t[2];
  for (int i = 0; i < 6; i++)
    planes[i] /= glm::length(glm::vec3(planes[i]));
}

bool sphereInFrustum(const glm::vec4 planes[6], glm::vec3 center,
                     float radius) {
  for (int i = 0; i < 6; i++) {
    if (glm::dot(glm::vec3(planes[i]), center) + planes[i].w < -radius)
      return false;
  }
  return true;
}

uint32_t shadowMaterialFlags(const Material &material) {
  return (material.isCloth ? 1u : 0u) | (material.alphaMasked ? 2u : 0u);
}

uint32_t alphaCutoff255(const Material &material) {
  return static_cast<uint32_t>(
      glm::clamp(material.alphaCutoff, 0.0f, 1.0f) * 255.0f + 0.5f);
}
} // namespace

void ShadowPass::create(VulkanDevice &newDevice, VkRenderPass newRenderPass,
                        VkFormat newDepthFormat) {
  cleanup();
  device = &newDevice;
  renderPass = newRenderPass;
  depthFormat = newDepthFormat;

  VkDevice dev = device->getLogicalDevice();

  VkImageCreateInfo imageCI{};
  imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCI.imageType = VK_IMAGE_TYPE_2D;
  imageCI.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
  imageCI.mipLevels = 1;
  imageCI.arrayLayers = NUM_CSM_CASCADES;
  imageCI.format = depthFormat;
  imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCI.usage =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  allocCI.flags = RENDER_DEVICE_ALLOCATION_FLAGS;

  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  if (vmaCreateImage(device->getAllocator(), &imageCI, &allocCI, &image,
                     &allocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create CSM depth array image");
  }
  csmDepthImage = AllocatedImage(device->getAllocator(), image, allocation);

  VkImageViewCreateInfo arrayViewCI{};
  arrayViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  arrayViewCI.image = csmDepthImage.get();
  arrayViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  arrayViewCI.format = depthFormat;
  arrayViewCI.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0,
                                  NUM_CSM_CASCADES};
  VkImageView view = VK_NULL_HANDLE;
  if (vkCreateImageView(dev, &arrayViewCI, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create CSM array view");
  csmArrayView = ImageViewHandle(dev, view);

  csmLayerViews.clear();
  csmFramebuffers.assign(NUM_CSM_CASCADES, VK_NULL_HANDLE);
  VkFramebufferCreateInfo framebufferCI{};
  framebufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferCI.renderPass = renderPass;
  framebufferCI.attachmentCount = 1;
  framebufferCI.width = SHADOW_MAP_SIZE;
  framebufferCI.height = SHADOW_MAP_SIZE;
  framebufferCI.layers = 1;
  for (int i = 0; i < NUM_CSM_CASCADES; i++) {
    VkImageViewCreateInfo layerViewCI = arrayViewCI;
    layerViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    layerViewCI.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1,
                                    static_cast<uint32_t>(i), 1};
    VkImageView layerView = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &layerViewCI, nullptr, &layerView) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create CSM layer view");
    }
    csmLayerViews.emplace_back(dev, layerView);
    framebufferCI.pAttachments = &layerView;
    if (vkCreateFramebuffer(dev, &framebufferCI, nullptr,
                            &csmFramebuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create CSM framebuffer");
    }
  }

  VkSamplerCreateInfo samplerCI{};
  samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCI.magFilter = VK_FILTER_LINEAR;
  samplerCI.minFilter = VK_FILTER_LINEAR;
  samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  samplerCI.maxLod = 1.0f;
  samplerCI.maxAnisotropy = 1.0f;
  if (vkCreateSampler(dev, &samplerCI, nullptr, &shadowSampler) != VK_SUCCESS)
    throw std::runtime_error("Failed to create shadow sampler");

  VkSamplerCreateInfo csmSamplerCI = samplerCI;
  csmSamplerCI.compareEnable = VK_TRUE;
  csmSamplerCI.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  if (vkCreateSampler(dev, &csmSamplerCI, nullptr, &csmShadowSampler) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create CSM compare sampler");
  }

  VkImageCreateInfo cubeCI = imageCI;
  cubeCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  cubeCI.extent = {POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE, 1};
  cubeCI.arrayLayers = 6;

  image = VK_NULL_HANDLE;
  allocation = VK_NULL_HANDLE;
  if (vmaCreateImage(device->getAllocator(), &cubeCI, &allocCI, &image,
                     &allocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create point shadow depth cubemap");
  }
  pointShadowDepthImage =
      AllocatedImage(device->getAllocator(), image, allocation);

  VkImageViewCreateInfo cubeViewCI{};
  cubeViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  cubeViewCI.image = pointShadowDepthImage.get();
  cubeViewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  cubeViewCI.format = depthFormat;
  cubeViewCI.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6};
  view = VK_NULL_HANDLE;
  if (vkCreateImageView(dev, &cubeViewCI, nullptr, &view) != VK_SUCCESS)
    throw std::runtime_error("Failed to create point shadow cubemap view");
  pointShadowCubeView = ImageViewHandle(dev, view);

  pointShadowFaceViews.clear();
  pointShadowFramebuffers.assign(6, VK_NULL_HANDLE);
  for (uint32_t face = 0; face < 6; ++face) {
    VkImageViewCreateInfo faceViewCI{};
    faceViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    faceViewCI.image = pointShadowDepthImage.get();
    faceViewCI.format = depthFormat;
    faceViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    faceViewCI.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, face, 1};
    VkImageView faceView = VK_NULL_HANDLE;
    if (vkCreateImageView(dev, &faceViewCI, nullptr, &faceView) != VK_SUCCESS)
      throw std::runtime_error("Failed to create point shadow face view");
    pointShadowFaceViews.emplace_back(dev, faceView);

    VkFramebufferCreateInfo pointFramebufferCI = framebufferCI;
    pointFramebufferCI.width = POINT_SHADOW_MAP_SIZE;
    pointFramebufferCI.height = POINT_SHADOW_MAP_SIZE;
    VkImageView attachment = pointShadowFaceViews.back().get();
    pointFramebufferCI.pAttachments = &attachment;
    if (vkCreateFramebuffer(dev, &pointFramebufferCI, nullptr,
                            &pointShadowFramebuffers[face]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create point shadow framebuffer");
    }
  }
}

void ShadowPass::cleanup() {
  if (!device)
    return;

  VkDevice dev = device->getLogicalDevice();
  for (VkFramebuffer framebuffer : pointShadowFramebuffers) {
    if (framebuffer != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, framebuffer, nullptr);
  }
  pointShadowFramebuffers.clear();
  pointShadowFaceViews.clear();
  pointShadowCubeView.reset();
  pointShadowDepthImage.reset();

  for (VkFramebuffer framebuffer : csmFramebuffers) {
    if (framebuffer != VK_NULL_HANDLE)
      vkDestroyFramebuffer(dev, framebuffer, nullptr);
  }
  csmFramebuffers.clear();
  csmLayerViews.clear();
  csmArrayView.reset();
  csmDepthImage.reset();

  if (shadowSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, shadowSampler, nullptr);
    shadowSampler = VK_NULL_HANDLE;
  }
  if (csmShadowSampler != VK_NULL_HANDLE) {
    vkDestroySampler(dev, csmShadowSampler, nullptr);
    csmShadowSampler = VK_NULL_HANDLE;
  }

  pointShadowMatrices.clear();
  renderPass = VK_NULL_HANDLE;
  depthFormat = VK_FORMAT_UNDEFINED;
  device = nullptr;
}

void ShadowPass::updateLightSpaceMatrices(SceneUniformBuffer &sceneUbo,
                                          float csmFarValue,
                                          float drawDistance) {
  const float nearPlane = 0.1f;
  const float csmFar = std::min(csmFarValue, drawDistance);
  const float farPlane = csmFar;
  const float lambda = 0.75f;

  float splits[NUM_CSM_CASCADES];
  for (int i = 0; i < NUM_CSM_CASCADES; i++) {
    float p = (i + 1) / float(NUM_CSM_CASCADES);
    float logSplit = nearPlane * std::pow(farPlane / nearPlane, p);
    float uniformSplit = nearPlane + (farPlane - nearPlane) * p;
    splits[i] = lambda * logSplit + (1.0f - lambda) * uniformSplit;
  }
  sceneUbo.cascadeSplits =
      glm::vec4(splits[0], splits[1], splits[2], splits[3]);

  glm::vec3 lightDir =
      glm::normalize(glm::vec3(sceneUbo.directionalLight.direction));
  glm::vec3 lightUp = (std::abs(lightDir.y) > 0.99f)
                          ? glm::vec3(0.0f, 0.0f, 1.0f)
                          : glm::vec3(0.0f, 1.0f, 0.0f);
  glm::mat4 lightView =
      glm::lookAt(-lightDir * 100.0f, glm::vec3(0.0f), lightUp);
  glm::mat4 invCam = glm::inverse(sceneUbo.projection * sceneUbo.view);

  auto viewDepthToNdcZ = [&](float depth) -> float {
    glm::vec4 clip =
        sceneUbo.projection * glm::vec4(0.0f, 0.0f, -depth, 1.0f);
    return clip.z / clip.w;
  };

  float prevSplit = nearPlane;
  for (int i = 0; i < NUM_CSM_CASCADES; i++) {
    float nearNdc = viewDepthToNdcZ(prevSplit);
    float farNdc = viewDepthToNdcZ(splits[i]);

    glm::vec3 corners[8];
    int k = 0;
    for (float nx : {-1.0f, 1.0f}) {
      for (float ny : {-1.0f, 1.0f}) {
        for (float nz : {nearNdc, farNdc}) {
          glm::vec4 world = invCam * glm::vec4(nx, ny, nz, 1.0f);
          corners[k++] = glm::vec3(world) / world.w;
        }
      }
    }

    glm::vec3 frustumCenter = glm::vec3(glm::inverse(sceneUbo.view)[3]);

    float sphereRadius = 0.0f;
    for (const glm::vec3 &corner : corners)
      sphereRadius =
          glm::max(sphereRadius, glm::length(corner - frustumCenter));

    glm::vec3 lsCenter =
        glm::vec3(lightView * glm::vec4(frustumCenter, 1.0f));
    float texelSize = 2.0f * sphereRadius / float(SHADOW_MAP_SIZE);
    if (texelSize > 0.0f) {
      lsCenter.x = std::floor(lsCenter.x / texelSize) * texelSize;
      lsCenter.y = std::floor(lsCenter.y / texelSize) * texelSize;
    }

    glm::vec3 minBounds(lsCenter.x - sphereRadius, lsCenter.y - sphereRadius,
                        lsCenter.z - sphereRadius - csmFar);
    glm::vec3 maxBounds(lsCenter.x + sphereRadius, lsCenter.y + sphereRadius,
                        lsCenter.z + sphereRadius + csmFar);

    glm::mat4 lightProj =
        glm::ortho(minBounds.x, maxBounds.x, minBounds.y, maxBounds.y,
                   minBounds.z, maxBounds.z);
    lightProj[1][1] *= -1.0f;
    sceneUbo.lightSpaceMatrices[i] = lightProj * lightView;

    prevSplit = splits[i];
  }
}

void ShadowPass::updatePointShadowMatrices(SceneUniformBuffer &sceneUbo) {
  pointShadowMatrices.resize(6);
  glm::vec3 lightPos = glm::vec3(sceneUbo.pointLights[0].position);
  float farPlane = sceneUbo.shadowParams.x;
  glm::mat4 proj =
      glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, farPlane);
  proj[1][1] *= -1.0f;

  pointShadowMatrices[0] =
      proj * glm::lookAt(lightPos, lightPos + glm::vec3(1, 0, 0),
                         glm::vec3(0, -1, 0));
  pointShadowMatrices[1] =
      proj * glm::lookAt(lightPos, lightPos + glm::vec3(-1, 0, 0),
                         glm::vec3(0, -1, 0));
  pointShadowMatrices[2] =
      proj * glm::lookAt(lightPos, lightPos + glm::vec3(0, 1, 0),
                         glm::vec3(0, 0, 1));
  pointShadowMatrices[3] =
      proj * glm::lookAt(lightPos, lightPos + glm::vec3(0, -1, 0),
                         glm::vec3(0, 0, -1));
  pointShadowMatrices[4] =
      proj * glm::lookAt(lightPos, lightPos + glm::vec3(0, 0, 1),
                         glm::vec3(0, -1, 0));
  pointShadowMatrices[5] =
      proj * glm::lookAt(lightPos, lightPos + glm::vec3(0, 0, -1),
                         glm::vec3(0, -1, 0));

  for (size_t i = 0; i < 6; ++i)
    sceneUbo.pointShadowMatrices[i] = pointShadowMatrices[i];
}

void ShadowPass::recordCsm(VkCommandBuffer cmd, SceneNode &rootNode,
                           ModelManager &modelManager,
                           const VulkanPipeline &pipeline,
                           const DescriptorManager &descriptorManager,
                           const SceneUniformBuffer &sceneUbo,
                           bool frontFaceCull, bool cullShadowCasters) {
  VkViewport viewport = {0,
                         0,
                         static_cast<float>(SHADOW_MAP_SIZE),
                         static_cast<float>(SHADOW_MAP_SIZE),
                         0,
                         1};
  VkRect2D scissor = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};

  // Sponza has thin shadow casters; reduced LODs can remove the exact
  // triangles that should block the sun, so keep CSM rendering at LOD0.
  auto cascadeLod = [](int cascade) {
    (void)cascade;
    return 0;
  };

  // Off means no culling, not back-face culling: authored wall winding is not
  // reliable enough for single-sided shadow casters.
  VkCullModeFlags defaultCull =
      frontFaceCull ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_NONE;
  VkCullModeFlags lastCull = defaultCull;

  auto renderNodeShadow = [&](auto &self, SceneNode *node, const glm::mat4 &lsm,
                              const glm::vec4 casterPlanes[6],
                              int lodIndex) -> void {
    if (node->getModelId() >= 0) {
      MeshModel *model = modelManager.getModel(node->getModelId());
      if (model) {
        const glm::mat4 transform = node->getGlobalTransform();
        float maxScale =
            glm::max(glm::length(glm::vec3(transform[0])),
                     glm::max(glm::length(glm::vec3(transform[1])),
                              glm::length(glm::vec3(transform[2]))));
        for (size_t meshIndex = 0; meshIndex < model->getMeshCount();
             ++meshIndex) {
          const Mesh *mesh = model->getMesh(meshIndex);
          glm::vec3 meshCenter = glm::vec3(
              transform * glm::vec4(mesh->boundingCenter, 1.0f));
          if (cullShadowCasters &&
              !sphereInFrustum(casterPlanes, meshCenter,
                               mesh->boundingRadius * maxScale)) {
            continue;
          }

          const Material &material = mesh->getMaterial();
          VkCullModeFlags wantCull =
              material.doubleSided ? VK_CULL_MODE_NONE : defaultCull;
          if (wantCull != lastCull) {
            vkCmdSetCullMode(cmd, wantCull);
            lastCull = wantCull;
          }

          ShadowPushConstants push{};
          push.model = transform;
          push.lightSpaceMatrix = lsm;
          push.albedoIdx = static_cast<uint32_t>(material.albedoTextureId);
          push.materialFlags = shadowMaterialFlags(material);
          push.alphaCutoff255 = alphaCutoff255(material);
          vkCmdPushConstants(cmd, pipeline.getShadowLayout(),
                             VK_SHADER_STAGE_VERTEX_BIT |
                                 VK_SHADER_STAGE_FRAGMENT_BIT,
                             0, sizeof(ShadowPushConstants), &push);
          int useLod = std::min(lodIndex, mesh->getLodCount() - 1);
          VkBuffer vertexBuffer[] = {mesh->getVertexBuffer()};
          VkDeviceSize offset[] = {0};
          vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffer, offset);
          vkCmdBindIndexBuffer(cmd, mesh->getIndexBuffer(useLod), 0,
                               VK_INDEX_TYPE_UINT32);
          vkCmdDrawIndexed(cmd, mesh->getIndexCount(useLod), 1, 0, 0, 0);
        }
      }
    }
    for (auto &child : node->getChildren())
      self(self, child.get(), lsm, casterPlanes, lodIndex);
  };

  for (int cascade = 0; cascade < NUM_CSM_CASCADES; cascade++) {
    VkClearValue clearValue{};
    clearValue.depthStencil.depth = 1.0f;

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPass;
    rpbi.framebuffer = csmFramebuffers[cascade];
    rpbi.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clearValue;

    char cascadeLabel[32];
    std::snprintf(cascadeLabel, sizeof(cascadeLabel), "Cascade %d", cascade);
    vkdbgBeginLabel(cmd, cascadeLabel, 1.0f, 0.55f + cascade * 0.1f, 0.1f);
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdSetCullMode(cmd, defaultCull);
    lastCull = defaultCull;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getShadowPipeline());
    VkDescriptorSet bindlessSet = descriptorManager.getBindlessSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getShadowLayout(), 0, 1, &bindlessSet, 0,
                            nullptr);
    glm::vec4 casterPlanes[6];
    extractFrustumPlanes(sceneUbo.lightSpaceMatrices[cascade], casterPlanes);
    renderNodeShadow(renderNodeShadow, &rootNode,
                     sceneUbo.lightSpaceMatrices[cascade], casterPlanes,
                     cascadeLod(cascade));
    vkCmdEndRenderPass(cmd);
    vkdbgEndLabel(cmd);
  }
}

void ShadowPass::recordPoint(VkCommandBuffer cmd, SceneNode &rootNode,
                             ModelManager &modelManager,
                             const VulkanPipeline &pipeline,
                             const DescriptorManager &descriptorManager,
                             bool frontFaceCull) {
  if (pointShadowMatrices.size() < 6)
    return;

  VkClearValue clearValue{};
  clearValue.depthStencil.depth = 1.0f;

  for (uint32_t face = 0; face < 6; ++face) {
    char faceLabel[32];
    std::snprintf(faceLabel, sizeof(faceLabel), "Point Shadow Face %u", face);
    vkdbgBeginLabel(cmd, faceLabel, 1.0f, 0.65f, 0.2f);
    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPass;
    rpbi.framebuffer = pointShadowFramebuffers[face];
    rpbi.renderArea.extent = {POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clearValue;

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {0,
                           0,
                           static_cast<float>(POINT_SHADOW_MAP_SIZE),
                           static_cast<float>(POINT_SHADOW_MAP_SIZE),
                           0,
                           1};
    VkRect2D scissor = {{0, 0},
                        {POINT_SHADOW_MAP_SIZE, POINT_SHADOW_MAP_SIZE}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    VkCullModeFlags defaultCull =
        frontFaceCull ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_NONE;
    vkCmdSetCullMode(cmd, defaultCull);
    VkCullModeFlags lastCull = defaultCull;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.getShadowPipeline());
    VkDescriptorSet bindlessSet = descriptorManager.getBindlessSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.getShadowLayout(), 0, 1, &bindlessSet, 0,
                            nullptr);

    auto renderNodeShadow = [&](auto &self, SceneNode *node) -> void {
      if (node->getModelId() >= 0) {
        MeshModel *model = modelManager.getModel(node->getModelId());
        if (model) {
          for (size_t meshIndex = 0; meshIndex < model->getMeshCount();
               ++meshIndex) {
            const Mesh *mesh = model->getMesh(meshIndex);
            const Material &material = mesh->getMaterial();
            VkCullModeFlags wantCull =
                material.doubleSided ? VK_CULL_MODE_NONE : defaultCull;
            if (wantCull != lastCull) {
              vkCmdSetCullMode(cmd, wantCull);
              lastCull = wantCull;
            }

            ShadowPushConstants push{};
            push.model = node->getGlobalTransform();
            push.lightSpaceMatrix = pointShadowMatrices[face];
            push.albedoIdx = static_cast<uint32_t>(material.albedoTextureId);
            push.materialFlags = shadowMaterialFlags(material);
            push.alphaCutoff255 = alphaCutoff255(material);
            vkCmdPushConstants(cmd, pipeline.getShadowLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT |
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ShadowPushConstants), &push);
            int useLod = 0;
            VkBuffer vertexBuffer[] = {mesh->getVertexBuffer()};
            VkDeviceSize offset[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffer, offset);
            vkCmdBindIndexBuffer(cmd, mesh->getIndexBuffer(useLod), 0,
                                 VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh->getIndexCount(useLod), 1, 0, 0, 0);
          }
        }
      }
      for (auto &child : node->getChildren())
        self(self, child.get());
    };

    renderNodeShadow(renderNodeShadow, &rootNode);
    vkCmdEndRenderPass(cmd);
    vkdbgEndLabel(cmd);
  }
}

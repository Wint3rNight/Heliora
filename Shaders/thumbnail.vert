#version 450

// The Assets panel's pictures: an asset drawn on its own into a small
// image, lit by one fixed light (ThumbnailRenderer).

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inCol;
layout(location = 2) in vec2 inTex;
layout(location = 3) in vec3 inNormal;

layout(push_constant) uniform Push {
    mat4 viewProj;  // the asset in its own space, framed (thumbnailViewProj)
    vec4 light;     // towards the light, xyz
    uvec4 texture;  // albedo index, alpha masked (0 or 1)
} push;

layout(location = 0) out vec3 fragCol;
layout(location = 1) out vec2 fragTex;
layout(location = 2) out vec3 fragNormal;

void main() {
    gl_Position = push.viewProj * vec4(inPos, 1.0);
    fragCol = inCol;
    fragTex = inTex;
    fragNormal = inNormal;
}

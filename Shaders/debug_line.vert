#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor; // RGBA8, unpacked by the vertex format

layout(push_constant) uniform Push {
    mat4 viewProj;   // unjittered: these lines are drawn after TAA
    vec2 invExtent;
    vec2 nearFar;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
    outColor = inColor;
}

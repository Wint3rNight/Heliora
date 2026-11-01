#version 450

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneDepth;

layout(push_constant) uniform Push {
    mat4 viewProj;
    vec2 invExtent;
    vec2 nearFar;
} pc;

// Distance from the camera for a depth-buffer value (0 to 1, standard Z).
float viewDistance(float depth) {
    float n = pc.nearFar.x, f = pc.nearFar.y;
    return n * f / (f - depth * (f - n));
}

void main() {
    float scene = viewDistance(texture(sceneDepth, gl_FragCoord.xy * pc.invExtent).r);
    float line  = viewDistance(gl_FragCoord.z);
    // A wireframe edge lies exactly on its mesh, and the scene depth was drawn
    // with TAA jitter, so allow 2 cm plus half a percent of the distance.
    bool hidden = line > scene + 0.02 + 0.005 * scene;
    outColor = vec4(inColor.rgb, inColor.a * (hidden ? 0.2 : 1.0));
}

#version 450

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneDepth;

layout(push_constant) uniform Push {
    mat4 viewProj;
    vec2 invAreaSize; // the scene area's size, inverted
    vec2 nearFar;
    vec2 areaOrigin;  // its corner in the window
} pc;

// Distance from the camera for a depth-buffer value (reversed-Z: 1 at the
// near plane, 0 at the far one).
float viewDistance(float depth) {
    float n = pc.nearFar.x, f = pc.nearFar.y;
    return n * f / (n + depth * (f - n));
}

void main() {
    vec2 uv = (gl_FragCoord.xy - pc.areaOrigin) * pc.invAreaSize;
    float scene = viewDistance(texture(sceneDepth, uv).r);
    float line  = viewDistance(gl_FragCoord.z);
    // A wireframe edge lies exactly on its mesh, and the scene depth was drawn
    // with TAA jitter, so allow 2 cm plus half a percent of the distance.
    bool hidden = line > scene + 0.02 + 0.005 * scene;
    outColor = vec4(inColor.rgb, inColor.a * (hidden ? 0.2 : 1.0));
}

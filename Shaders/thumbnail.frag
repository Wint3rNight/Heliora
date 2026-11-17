#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    mat4 viewProj;
    vec4 light;
    uvec4 texture;
} push;

layout(location = 0) in vec3 fragCol;
layout(location = 1) in vec2 fragTex;
layout(location = 2) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 albedo = texture(textures[nonuniformEXT(push.texture.x)], fragTex) *
                  vec4(fragCol, 1.0);
    if (push.texture.y != 0u && albedo.a < 0.5)
        discard;
    vec3 n = normalize(fragNormal);
    if (!gl_FrontFacing)
        n = -n;
    // A soft key light and some fill, so every side reads.
    float key = max(dot(n, normalize(push.light.xyz)), 0.0);
    vec3 lit = albedo.rgb * (0.35 + 0.8 * key);
    // The atlas is plain 8-bit and ImGui shows it as it is, so it's stored
    // already encoded for the screen.
    outColor = vec4(pow(clamp(lit, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}

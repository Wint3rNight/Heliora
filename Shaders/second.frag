#version 450

// ---- Scene UBO (set 0, binding 0) — same std140 layout as lit.frag ----
struct DirectionalLight {
    vec4 direction;
    vec4 colorIntensity;
};
struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 cutoffAngles;
};

layout(set = 0, binding = 0) uniform SceneUniformBuffer {
    mat4 projection;
    mat4 view;
    mat4 lightSpaceMatrices[4];
    mat4 pointShadowMatrices[6];
    vec4 cameraPosition;
    vec4 cascadeSplits;
    DirectionalLight directionalLight;
    PointLight pointLights[4];
    SpotLight spotLights[2];
    ivec4 lightCounts;
    vec4 shadowParams;
    mat4 invProj;
    mat4 invView;
    vec4 fogParams;
    int debugMode;
    vec4 qualityToggles;
    vec4 qualityToggles2;
    mat4 prevViewProj;     // prev frame's un-jittered VP
    vec4 taaParams;        // (jitter.x, jitter.y, taaEnable, historyValid) in NDC units
    vec4 viewportSize;     // (w, h, 1/w, 1/h)
} scene;

// ---- Set 1: input attachment (colorBuffer from subpass 0) ----
layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput inputColor;

// ---- Set 2: TAA inputs ----
layout(set = 2, binding = 0) uniform sampler2D historyPrev;
layout(set = 2, binding = 1) uniform sampler2D depthTex;

layout(location = 0) out vec4 outSwap;
layout(location = 1) out vec4 outHistory;

vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 tonemapAndEncode(vec3 hdr) {
    return pow(acesFilm(hdr), vec3(1.0 / 2.2));
}

// Reconstruct un-jittered world position from sub-pixel jitter + depth.
// scene.taaParams.xy is the jitter that was baked into projection.[2][0..1] —
// it shifts clip.xy by jitter*w, so post-divide NDC ends up at ndc+jitter.
// To undo, subtract jitter from the rasterized NDC before unprojecting.
vec3 reconstructWorldPos(vec2 uv, float depth, vec2 jitterNDC) {
    vec2 ndc = uv * 2.0 - 1.0 - jitterNDC;
    vec4 clip = vec4(ndc, depth, 1.0);
    vec4 viewPos = scene.invProj * clip;
    viewPos /= viewPos.w;
    vec4 worldPos = scene.invView * viewPos;
    return worldPos.xyz;
}

void main() {
    vec3 current = subpassLoad(inputColor).rgb;

    bool taaEnable     = scene.taaParams.z > 0.5;
    bool historyValid  = scene.taaParams.w > 0.5;

    // Passthrough when TAA is disabled or there's no usable history yet.
    if (!taaEnable || !historyValid) {
        outHistory = vec4(current, 1.0);
        outSwap    = vec4(tonemapAndEncode(current), 1.0);
        return;
    }

    vec2 uv     = gl_FragCoord.xy * scene.viewportSize.zw;
    float depth = texture(depthTex, uv).r;

    vec2 prevUV;
    if (depth >= 0.9999) {
        // Sky / far plane: world position is at infinity, reprojection is
        // unstable. Use current UV — sky is mostly low-frequency so this
        // still benefits from temporal accumulation.
        prevUV = uv;
    } else {
        vec3 worldPos = reconstructWorldPos(uv, depth, scene.taaParams.xy);
        vec4 prevClip = scene.prevViewProj * vec4(worldPos, 1.0);
        if (abs(prevClip.w) < 1e-6) {
            outHistory = vec4(current, 1.0);
            outSwap    = vec4(tonemapAndEncode(current), 1.0);
            return;
        }
        vec2 prevNDC = prevClip.xy / prevClip.w;
        prevUV = prevNDC * 0.5 + 0.5;
    }

    // Disocclusion / off-screen reject.
    if (any(lessThan(prevUV, vec2(0.0))) || any(greaterThan(prevUV, vec2(1.0)))) {
        outHistory = vec4(current, 1.0);
        outSwap    = vec4(tonemapAndEncode(current), 1.0);
        return;
    }

    vec3 history = texture(historyPrev, prevUV).rgb;
    // Guard against NaN/Inf seeping in from uninitialized history texels —
    // they'd contaminate every future frame via the mix.
    if (any(isnan(history)) || any(isinf(history))) {
        history = current;
    }

    // Karis-style temporal blend. 0.9 history weight gives ~10 frames of
    // effective accumulation. No neighborhood clamp yet — see plan notes;
    // requires switching colorBuffer off the input-attachment path.
    vec3 blended = mix(current, history, 0.9);

    outHistory = vec4(blended, 1.0);
    outSwap    = vec4(tonemapAndEncode(blended), 1.0);
}

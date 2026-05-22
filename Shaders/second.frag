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
    mat4 prevViewProj;     // JITTERED prev VP — multiply world pos to get prev jittered NDC
    vec4 taaParams;        // (jitter.x, jitter.y, taaEnable, historyValid) in NDC units
    vec4 viewportSize;     // (w, h, 1/w, 1/h)
} scene;

// Set 1: input attachment (colorBuffer from subpass 0) — zero-overhead center read.
layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput inputColor;

// Set 2: TAA samplers.
layout(set = 2, binding = 0) uniform sampler2D historyPrev;
layout(set = 2, binding = 1) uniform sampler2D depthTex;
// Same image as inputColor — but as a regular sampler so we can read neighbors
// with offsets for the 3x3 YCoCg AABB clamp.
layout(set = 2, binding = 2) uniform sampler2D currentTex;

layout(location = 0) out vec4 outSwap;
layout(location = 1) out vec4 outHistory;

vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 tonemapAndEncode(vec3 hdr) {
    // Pre-tonemap exposure multiplier. Lifts (or crushes) the HDR signal
    // before ACES — the steep ACES toe was eating shadow detail without a
    // pre-scale, leaving interior surfaces near-black. EV-stops authored
    // on the CPU side: qualityToggles2.x = exp2(EV).
    hdr *= scene.qualityToggles2.x;
    return pow(acesFilm(hdr), vec3(1.0 / 2.2));
}

// YCoCg conversion (Karis 2014). Better for AABB clamping than RGB because
// chroma is decorrelated from luma, so the box hugs the local color much
// tighter and the clamp rejects history closer to the surface boundary.
vec3 rgbToYCoCg(vec3 c) {
    float Y  = 0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
    float Co = 0.5  * c.r           - 0.5  * c.b;
    float Cg =-0.25 * c.r + 0.5 * c.g - 0.25 * c.b;
    return vec3(Y, Co, Cg);
}
vec3 yCoCgToRgb(vec3 c) {
    float Y = c.x, Co = c.y, Cg = c.z;
    return vec3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

// Reconstruct un-jittered view-space position from rasterized depth +
// JITTERED NDC. scene.invProj is the inverse of the JITTERED projection,
// so feeding the rasterized NDC (which is jittered) yields the original
// un-jittered surface point. No subtraction of taaParams.xy needed —
// that was the second-half of the blur bug.
vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = scene.invProj * clip;
    viewPos /= viewPos.w;
    vec4 worldPos = scene.invView * viewPos;
    return worldPos.xyz;
}

void main() {
    vec3 current = subpassLoad(inputColor).rgb;

    bool taaEnable    = scene.taaParams.z > 0.5;
    bool historyValid = scene.taaParams.w > 0.5;

    if (!taaEnable || !historyValid) {
        outHistory = vec4(current, 1.0);
        outSwap    = vec4(tonemapAndEncode(current), 1.0);
        return;
    }

    vec2 uv     = gl_FragCoord.xy * scene.viewportSize.zw;
    float depth = texture(depthTex, uv).r;

    vec2 prevUV;
    if (depth >= 0.9999) {
        // Sky / far plane: reprojection through depth is unstable. Use
        // same UV (sky moves only with camera rotation; for translation
        // the sky is at infinity, so static UV is the right reprojection).
        prevUV = uv;
    } else {
        vec3 worldPos = reconstructWorldPos(uv, depth);
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
    if (any(isnan(history)) || any(isinf(history))) {
        history = current;
    }

    // 3×3 YCoCg neighborhood statistics: raw min/max (the canonical
    // Karis 2014 AABB) and luma std-dev (used only to adapt the
    // history-blend weight, NOT to tighten the AABB). Tightening the
    // AABB via variance biases edges where the 3×3 neighborhood is
    // unbalanced — the dark side of an edge can never accumulate
    // because the variance box shrinks toward the brighter mean. Raw
    // min/max preserves both sides of an edge correctly.
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec3 ycMin = vec3( 1e30);
    vec3 ycMax = vec3(-1e30);
    float lumaSum  = 0.0;
    float lumaSum2 = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        vec3 n  = texelFetch(currentTex, px + ivec2(dx, dy), 0).rgb;
        vec3 yc = rgbToYCoCg(n);
        ycMin = min(ycMin, yc);
        ycMax = max(ycMax, yc);
        lumaSum  += yc.x;
        lumaSum2 += yc.x * yc.x;
    }
    float lumaMean   = lumaSum / 9.0;
    float lumaVar    = max(lumaSum2 / 9.0 - lumaMean * lumaMean, 0.0);
    float lumaStddev = sqrt(lumaVar);

    vec3 ycHist = rgbToYCoCg(history);

    // Lottes 2014 / Karis 2014 AABB clip-to-center. Project from the box
    // center through ycHist; if ycHist lies OUTSIDE the box, snap it to
    // the box face along that ray. If inside, leave it untouched — the
    // critical property that lets temporal accumulation converge on a
    // static camera. Box is the raw 3×3 neighborhood min/max in YCoCg.
    vec3 boxCenter = 0.5 * (ycMax + ycMin);
    vec3 boxExtent = 0.5 * (ycMax - ycMin) + vec3(1e-7);
    vec3 v = ycHist - boxCenter;
    vec3 vUnit = abs(v) / boxExtent;
    float maxAxis = max(max(vUnit.x, vUnit.y), vUnit.z);
    vec3 clippedYC = (maxAxis > 1.0) ? (boxCenter + v / maxAxis) : ycHist;
    history = yCoCgToRgb(clippedYC);

    // Variance-driven adaptive history weight. Tight neighborhood
    // (stddev small → sharp edge / clean surface) → lower history
    // weight so detail survives. Wide neighborhood (noisy textured
    // surface) → higher history weight so the noise actually averages
    // out.
    float historyWeight = mix(0.83, 0.95, smoothstep(0.0, 0.5, lumaStddev));

    vec3 blended = mix(current, history, historyWeight);

    outHistory = vec4(blended, 1.0);
    outSwap    = vec4(tonemapAndEncode(blended), 1.0);
}

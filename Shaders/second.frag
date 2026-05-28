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
layout(set = 2, binding = 3) uniform sampler2D bloomTex;

layout(location = 0) out vec4 outSwap;
layout(location = 1) out vec4 outHistory;

// AgX tonemap (Troy Sobotka, popularized by Blender 4.0+ / MrLixm's port).
// Replaces the ACES filmic curve. ACES has an aggressive toe that crushed
// midtone detail in Sponza interiors to black — AgX preserves the toe and
// gives a much more "natural daylight" look on indirect-lit surfaces.
// Output is already perceptually-encoded for an sRGB UNORM display, so
// no final pow(1/2.2) needed — that would double-encode and look washed
// out.
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return  15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         -  6.868   * x2 * x
         +  0.4298  * x2
         +  0.1191  * x
         -  0.00232;
}

vec3 agx(vec3 col) {
    // Input transform (inset) — narrows the gamut before the log/sigmoid so
    // out-of-gamut bright pixels (sky disc, light filaments) don't blow into
    // pure white / pure primary. Matrix from Sobotka / Eary canonical AgX.
    const mat3 agxInsetMatrix = mat3(
        0.842479062253094,  0.0784335999999992, 0.0792237451477643,
        0.0423282422610123, 0.878468636469772,  0.0791661274605434,
        0.0423756549057051, 0.0784336,          0.879142973793104
    );
    const float minEV = -12.47393;
    const float maxEV =   4.026069;
    col = agxInsetMatrix * col;
    col = clamp(log2(max(col, vec3(1e-6))), minEV, maxEV);
    col = (col - minEV) / (maxEV - minEV);
    return agxDefaultContrastApprox(col);
}

vec3 tonemapAndEncode(vec3 hdr) {
    // Pre-tonemap exposure multiplier. qualityToggles2.x = exp2(EV stops).
    hdr *= scene.qualityToggles2.x;
    return agx(hdr);
}

// AMD CAS-style contrast-adaptive sharpening, simplified.
// Sample center + 4-cross neighbors in DISPLAY space (post-AgX). Compute
// the laplacian (center vs average of neighbors) and add a fraction back
// to the center to undo perceptual softening from AgX + bloom + TAA
// blend. CAS-proper has a contrast-adaptive ramp; we ship the cheap
// fixed-strength version because it's a single shader and is good
// enough at our resolution.
//
// Strength = scene.shadowParams.z (the SSGI radius override slot wasn't
// being used; repurposing for sharpening since both ride on the same
// vec4 and we're out of qualityToggles slots).
//
// 0 = identity, 0.5 = light sharpen, 1.0 = strong sharpen / starts ringing.
vec3 applySharpen(vec3 center, ivec2 px) {
    float strength = scene.shadowParams.z;
    if (strength < 0.001) return center;

    vec3 n = texelFetch(currentTex, px + ivec2( 0, -1), 0).rgb;
    vec3 s = texelFetch(currentTex, px + ivec2( 0,  1), 0).rgb;
    vec3 e = texelFetch(currentTex, px + ivec2( 1,  0), 0).rgb;
    vec3 w = texelFetch(currentTex, px + ivec2(-1,  0), 0).rgb;

    // Tonemap the neighbors so we sharpen in the same perceptual space as
    // the center. Apply the same exposure pre-mult to keep things aligned.
    float ex = scene.qualityToggles2.x;
    n = agx(n * ex);
    s = agx(s * ex);
    e = agx(e * ex);
    w = agx(w * ex);

    vec3 avg = 0.25 * (n + s + e + w);
    vec3 lap = center - avg;            // positive on bright detail
    // Clamp the sharpening boost so single-pixel fireflies don't get
    // amplified into a bigger sparkle.
    lap = clamp(lap, vec3(-0.2), vec3(0.2));
    return clamp(center + lap * strength, vec3(0.0), vec3(1.0));
}

// Combined "send-to-display" path: tonemap then optionally sharpen. Only
// the swap-chain output is sharpened; the history image stays HDR linear
// so TAA reprojection math doesn't fight a post-tonemap perceptual signal.
vec3 finalEncode(vec3 hdr) {
    vec3 tm = tonemapAndEncode(hdr);
    return applySharpen(tm, ivec2(gl_FragCoord.xy));
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

// Linearized view-space Z from a depth-buffer sample. Used by the
// depth-disocclusion reject. Just the z component of the un-projected
// clip point — no need for the full world reconstruction.
float reconstructViewZ(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = scene.invProj * clip;
    return viewPos.z / viewPos.w;
}

// 5-tap optimized Catmull-Rom history sampler (Salvi 2016 / Jimenez UE4
// fork). Reconstructs the bilinear history texture with a sharper, less
// blurry filter — full bicubic Catmull-Rom is 16 taps; the trick is to
// reorganize the cubic weights so that adjacent pairs can use a single
// shifted bilinear tap. Total 5 bilinear taps vs 16 nearest.
//
// Result is the un-blurred history color; combined with the firefly-clamp
// in lit.frag and the YCoCg AABB below, ringing artifacts are bounded.
vec3 sampleHistoryCatmullRom(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 samplePos = uv * texSize;
    vec2 texPos1   = floor(samplePos - 0.5) + 0.5;
    vec2 f         = samplePos - texPos1;

    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    vec2 w12     = w1 + w2;
    vec2 offset12 = w2 / max(w12, vec2(1e-5));

    vec2 texPos0  = (texPos1 - 1.0)        / texSize;
    vec2 texPos3  = (texPos1 + 2.0)        / texSize;
    vec2 texPos12 = (texPos1 + offset12)   / texSize;

    vec3 result = vec3(0.0);
    result += textureLod(tex, vec2(texPos12.x, texPos0.y),  0.0).rgb * w12.x * w0.y;
    result += textureLod(tex, vec2(texPos0.x,  texPos12.y), 0.0).rgb * w0.x  * w12.y;
    result += textureLod(tex, vec2(texPos12.x, texPos12.y), 0.0).rgb * w12.x * w12.y;
    result += textureLod(tex, vec2(texPos3.x,  texPos12.y), 0.0).rgb * w3.x  * w12.y;
    result += textureLod(tex, vec2(texPos12.x, texPos3.y),  0.0).rgb * w12.x * w3.y;
    return max(result, vec3(0.0));
}

void main() {
    vec2 uv = gl_FragCoord.xy * scene.viewportSize.zw;
    vec3 current = subpassLoad(inputColor).rgb;
    vec3 bloom = texture(bloomTex, uv).rgb;
    if (scene.debugMode == 13) {
        outHistory = vec4(current, 1.0);
        outSwap = vec4(tonemapAndEncode(bloom * 8.0), 1.0);
        return;
    }

    bool enableBloom = (scene.lightCounts.z & (1 << 5)) != 0;
    if (enableBloom) {
        current += bloom;
    }

    bool taaEnable    = scene.taaParams.z > 0.5;
    bool historyValid = scene.taaParams.w > 0.5;

    if (!taaEnable || !historyValid) {
        outHistory = vec4(current, 1.0);
        outSwap    = vec4(finalEncode(current), 1.0);
        return;
    }

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
            outSwap    = vec4(finalEncode(current), 1.0);
            return;
        }
        vec2 prevNDC = prevClip.xy / prevClip.w;
        prevUV = prevNDC * 0.5 + 0.5;
    }

    // Off-screen reject.
    if (any(lessThan(prevUV, vec2(0.0))) || any(greaterThan(prevUV, vec2(1.0)))) {
        outHistory = vec4(current, 1.0);
        outSwap    = vec4(finalEncode(current), 1.0);
        return;
    }

    // Depth-based disocclusion reject. Without a prev-frame depth image we
    // can't compare prev-depth to current depth directly — but for a static
    // scene with camera motion (our case), we can use the CURRENT frame's
    // depth at prevUV as a proxy. If the surface that's now at prevUV is
    // at a very different view-Z than the reprojected surface, then the
    // pixel I'm reprojecting *from* isn't the same surface I'm shading
    // *now* → disocclusion → reject. View-Z compare scales tolerance with
    // distance so far surfaces aren't over-rejected by small fractional
    // depth wiggles.
    float depthAtPrevUV = texture(depthTex, prevUV).r;
    float viewZCur      = abs(reconstructViewZ(uv,     depth));
    float viewZPrev     = abs(reconstructViewZ(prevUV, depthAtPrevUV));
    // 10% relative tolerance: tight enough to catch silhouette
    // disocclusions, loose enough that sub-pixel jitter doesn't flap it.
    if (depthAtPrevUV < 0.9999 &&
        abs(viewZCur - viewZPrev) > max(0.1 * viewZCur, 0.05)) {
        outHistory = vec4(current, 1.0);
        outSwap    = vec4(finalEncode(current), 1.0);
        return;
    }

    vec3 history = sampleHistoryCatmullRom(historyPrev, prevUV,
                                           scene.viewportSize.xy);
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
        ivec2 nPx = px + ivec2(dx, dy);
        vec2 nUv = (vec2(nPx) + vec2(0.5)) * scene.viewportSize.zw;
        vec3 n  = texelFetch(currentTex, nPx, 0).rgb;
        if (enableBloom) {
            n += texture(bloomTex, nUv).rgb;
        }
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
    outSwap    = vec4(finalEncode(blended), 1.0);
}

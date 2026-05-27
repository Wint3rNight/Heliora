#version 450

// Composition subpass 0: samples the lit HDR buffer and augments smooth
// surfaces with true-color SSR (reflections of LIT scene, not albedo).

struct DirectionalLight { vec4 direction;  vec4 colorIntensity; };
struct PointLight       { vec4 position;   vec4 colorIntensity; };
struct SpotLight        { vec4 position;   vec4 direction; vec4 colorIntensity; vec4 cutoffAngles; };

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
} scene;

// Set 1 — extended G-buffer set:
// 0: gb0 (albedo + metallic)
// 1: gb1 (worldN + roughness)
// 2: gb2 (AO)
// 3: depth
// 4: lit (this is the post-PBR HDR buffer we're augmenting)
layout(set = 1, binding = 0) uniform sampler2D gBuffer0Sampler;
layout(set = 1, binding = 1) uniform sampler2D gBuffer1Sampler;
layout(set = 1, binding = 2) uniform sampler2D gBuffer2Sampler;
layout(set = 1, binding = 3) uniform sampler2D gBufferDepthSampler;
layout(set = 1, binding = 4) uniform sampler2D litSampler;

layout(location = 0) out vec4 outColor;

vec3 reconstructViewPos(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 v   = scene.invProj * ndc;
    return v.xyz / v.w;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Reflects the LIT color buffer along view-space reflection vector.
// Returns a Fresnel-weighted reflection contribution; 0 on miss.
vec3 computeSSR(vec2 uv, vec3 viewPos, vec3 viewN, float roughness,
                float metallic, vec3 F0) {
    // Only fire on near-mirror surfaces. Sponza fabric is rough — was getting
    // hit by SSR before and producing speckle.
    if (roughness > 0.18) return vec3(0.0);
    if (metallic < 0.25 && roughness > 0.08) return vec3(0.0);
    vec3 reflDir = normalize(reflect(normalize(viewPos), viewN));
    if (reflDir.z > 0.0) return vec3(0.0); // ray points back at camera

    const int   MAX_STEPS = 48;
    const float MAX_DIST  = 12.0;
    float stepSize = MAX_DIST / float(MAX_STEPS);

    // Scale-aware self-hit offset (B3): closer surfaces use smaller offset
    vec3 rayPos = viewPos + viewN * max(0.05, abs(viewPos.z) * 0.001);

    for (int i = 0; i < MAX_STEPS; i++) {
        rayPos += reflDir * stepSize;
        vec4 clip = scene.projection * vec4(rayPos, 1.0);
        if (clip.w <= 0.0) break;

        // No y-flip needed: projection already does [1][1] *= -1 so NDC y
        // matches Vulkan's screen-down convention; sUV.y * 0.5 + 0.5 gives
        // 0 at top, 1 at bottom — same as gl_FragCoord / texture UV.
        vec2 sUV = clip.xy / clip.w * 0.5 + 0.5;
        if (sUV.x < 0.01 || sUV.x > 0.99 || sUV.y < 0.01 || sUV.y > 0.99) break;

        float sceneDepth = texture(gBufferDepthSampler, sUV).r;
        float rayDepth   = clip.z / clip.w;

        // Depth-relative tolerance (B4)
        float tol = max(0.005, 0.05 * sceneDepth);
        if (rayDepth > sceneDepth && (rayDepth - sceneDepth) < tol) {
            // Refine via binary search
            vec3 lo = rayPos - reflDir * stepSize;
            vec3 hi = rayPos;
            for (int j = 0; j < 8; j++) {
                vec3 mid = (lo + hi) * 0.5;
                vec4 mc  = scene.projection * vec4(mid, 1.0);
                vec2 mUV = mc.xy / mc.w * 0.5 + 0.5;
                float mRay   = mc.z / mc.w;
                float mScene = texture(gBufferDepthSampler, mUV).r;
                if (mRay > mScene) hi = mid; else lo = mid;
            }
            vec4 fc  = scene.projection * vec4(lo, 1.0);
            vec2 fUV = fc.xy / fc.w * 0.5 + 0.5;

            float hitDepth = texture(gBufferDepthSampler, fUV).r;
            if (hitDepth >= 0.9999) return vec3(0.0);
            vec3 hitView = reconstructViewPos(fUV, hitDepth);
            float rayZ = -lo.z;
            float hitZ = -hitView.z;
            if (abs(hitZ - rayZ) > max(0.08 * hitZ, 0.05))
                return vec3(0.0);

            vec3 hitWorldN = texture(gBuffer1Sampler, fUV).xyz;
            if (dot(hitWorldN, hitWorldN) < 0.1) return vec3(0.0);
            vec3 hitViewN = normalize(mat3(scene.view) * normalize(hitWorldN));
            if (dot(hitViewN, -reflDir) < 0.05) return vec3(0.0);

            vec3 reflColor = min(texture(litSampler, fUV).rgb, vec3(3.0));

            vec2 edgeDist   = smoothstep(0.95, 1.0, abs(fUV * 2.0 - 1.0));
            float edgeFade  = 1.0 - clamp(max(edgeDist.x, edgeDist.y), 0.0, 1.0);
            float cosTheta  = max(dot(-normalize(viewPos), viewN), 0.0);
            float fresnel   = fresnelSchlick(cosTheta, F0).r;
            // pow(1-r, 4) sharpens the falloff so semi-rough surfaces get
            // very little SSR contribution (visible speckle goes away).
            float roughFade = pow(1.0 - roughness, 4.0);
            float ssrWeight = min(fresnel * roughFade * edgeFade, 0.35);

            return reflColor * ssrWeight;
        }
    }
    return vec3(0.0);
}

void main() {
    vec2 texSize = vec2(textureSize(litSampler, 0));
    vec2 uv      = gl_FragCoord.xy / texSize;

    vec3 lit = texture(litSampler, uv).rgb;
    if (scene.shadowParams.w < 0.5) {
        outColor = vec4(lit, 1.0);
        return;
    }

    // Skip SSR for sky pixels (depth at 1.0): no surface to reflect from.
    float depth = texture(gBufferDepthSampler, uv).r;
    if (depth >= 0.9999) {
        outColor = vec4(lit, 1.0);
        return;
    }

    // Reconstruct surface attributes for SSR
    vec4 g0       = texture(gBuffer0Sampler, uv);
    vec4 g1       = texture(gBuffer1Sampler, uv);
    vec3 albedo   = g0.rgb;
    float metallic  = g0.a;
    vec3 worldN   = normalize(g1.xyz);
    float roughness = g1.a;
    vec3 F0       = mix(vec3(0.04), albedo, metallic);

    vec3 viewPos = reconstructViewPos(uv, depth);
    vec3 viewN   = normalize(mat3(scene.view) * worldN);

    vec3 ssr = computeSSR(uv, viewPos, viewN, roughness, metallic, F0);
    // Additive blend — SSR contribution is already Fresnel- and edge-fade-weighted.
    outColor = vec4(lit + ssr, 1.0);
}

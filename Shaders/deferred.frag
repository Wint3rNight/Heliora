#version 450

// ---- Set 0: scene-global resources ----
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
} scene;

layout(set = 0, binding = 1) uniform sampler2DArray shadowMap;
layout(set = 0, binding = 2) uniform samplerCube pointShadowMap;
layout(set = 0, binding = 3) uniform samplerCube irradianceMap;
layout(set = 0, binding = 4) uniform samplerCube prefilteredEnvMap;
layout(set = 0, binding = 5) uniform sampler2D brdfLUT;
layout(set = 0, binding = 6) uniform sampler2D ssaoNoise;
layout(set = 0, binding = 7) uniform samplerCube skyboxCubemap;

// ---- Set 1: G-buffer samplers ----
layout(set = 1, binding = 0) uniform sampler2D gBuffer0Sampler;  // albedo + metallic
layout(set = 1, binding = 1) uniform sampler2D gBuffer1Sampler;  // world normal + roughness
layout(set = 1, binding = 2) uniform sampler2D gBuffer2Sampler;  // AO
layout(set = 1, binding = 3) uniform sampler2D gBufferDepthSampler;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const int IBL_PREFILTER_MIPS = 5;

// Hemisphere kernel in view space (16 samples)
const vec3 ssaoKernel[16] = vec3[16](
    vec3( 0.5381,  0.1856, -0.1495), vec3( 0.1379,  0.2486,  0.4430),
    vec3( 0.3371,  0.5679, -0.0057), vec3(-0.6999, -0.0451, -0.0019),
    vec3( 0.0689, -0.1598, -0.8547), vec3( 0.0560,  0.0069, -0.1843),
    vec3(-0.0146,  0.1402,  0.0762), vec3( 0.0100, -0.1924, -0.0344),
    vec3(-0.3577, -0.5301, -0.4358), vec3(-0.3169,  0.1063,  0.0158),
    vec3( 0.0103, -0.5869,  0.0046), vec3(-0.0897, -0.4940,  0.3287),
    vec3( 0.7119, -0.0154, -0.0918), vec3(-0.0533,  0.0596, -0.5411),
    vec3( 0.0352, -0.0631,  0.5460), vec3(-0.4776,  0.2847, -0.0271)
);

// ---- Position reconstruction ----
vec3 reconstructViewPos(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPosCl = scene.invProj * ndc;
    return viewPosCl.xyz / viewPosCl.w;
}

vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec3 viewPos = reconstructViewPos(uv, depth);
    vec4 worldPosCl = scene.invView * vec4(viewPos, 1.0);
    return worldPosCl.xyz / worldPosCl.w;
}

// ---- Cascaded Shadow Maps ----
float shadowFactor(vec3 worldPos, vec3 viewPos, vec3 N, vec3 L) {
    float viewDepth = -viewPos.z;

    int cascade = 3;
    if      (viewDepth < scene.cascadeSplits.x) cascade = 0;
    else if (viewDepth < scene.cascadeSplits.y) cascade = 1;
    else if (viewDepth < scene.cascadeSplits.z) cascade = 2;

    vec4 lsPos = scene.lightSpaceMatrices[cascade] * vec4(worldPos, 1.0);
    vec3 proj  = lsPos.xyz / lsPos.w;
    proj.xy    = proj.xy * 0.5 + 0.5;

    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 ||
        proj.y < 0.0 || proj.y > 1.0) return 0.0;

    // Tighter bias for near cascades (larger texel coverage = less acne)
    float biasScale = 1.0 / (scene.cascadeSplits[cascade] * 0.5 + 1.0);
    float bias = max(0.006 * biasScale * (1.0 - dot(N, L)), 0.0005);

    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y) {
            float d = texture(shadowMap, vec3(proj.xy + vec2(x, y) * texelSize, float(cascade))).r;
            shadow += proj.z - bias > d ? 1.0 : 0.0;
        }
    return shadow / 9.0;
}

int pointShadowFace(vec3 dir) {
    vec3 a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) return dir.x > 0.0 ? 0 : 1;
    if (a.y >= a.x && a.y >= a.z) return dir.y > 0.0 ? 2 : 3;
    return dir.z > 0.0 ? 4 : 5;
}

float pointShadowFactor(vec3 worldPos, vec3 lightPos) {
    vec3 offsets[20] = vec3[](
        vec3(1,1,1), vec3(1,-1,1), vec3(-1,-1,1), vec3(-1,1,1),
        vec3(1,1,-1), vec3(1,-1,-1), vec3(-1,-1,-1), vec3(-1,1,-1),
        vec3(1,1,0), vec3(1,-1,0), vec3(-1,-1,0), vec3(-1,1,0),
        vec3(1,0,1), vec3(-1,0,1), vec3(1,0,-1), vec3(-1,0,-1),
        vec3(0,1,1), vec3(0,-1,1), vec3(0,1,-1), vec3(0,-1,-1)
    );
    vec3 frag2Light = worldPos - lightPos;
    int face = pointShadowFace(frag2Light);
    vec4 lsPos = scene.pointShadowMatrices[face] * vec4(worldPos, 1.0);
    float depth = lsPos.z / lsPos.w;
    if (depth > 1.0) return 0.0;
    float shadow = 0.0;
    for (int i = 0; i < 20; ++i) {
        float d = texture(pointShadowMap, frag2Light + offsets[i] * 0.04).r;
        shadow += depth - 0.004 > d ? 1.0 : 0.0;
    }
    return shadow / 20.0;
}

// ---- Cook-Torrance BRDF ----
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a2 = roughness * roughness * roughness * roughness;
    float NdotH2 = max(dot(N, H), 0.0);
    NdotH2 *= NdotH2;
    float d = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * d * d);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) *
           pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 cookTorrance(vec3 albedo, vec3 N, vec3 V, vec3 L, vec3 F0,
                  float metallic, float roughness,
                  vec3 lightColor, float intensity) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    return (kD * albedo / PI + specular) * lightColor * intensity * NdotL;
}

// ---- SSAO ----
float computeSSAO(vec2 uv, vec3 viewPos, vec3 viewNormal) {
    vec2 noiseScale = vec2(textureSize(gBuffer0Sampler, 0)) / 4.0;
    vec3 randomVec  = normalize(texture(ssaoNoise, uv * noiseScale).rgb * 2.0 - 1.0);
    vec3 tangent    = normalize(randomVec - viewNormal * dot(randomVec, viewNormal));
    vec3 bitangent  = cross(viewNormal, tangent);
    mat3 TBN        = mat3(tangent, bitangent, viewNormal);

    float occlusion = 0.0;
    const float radius = 0.5;
    const float bias   = 0.025;

    for (int i = 0; i < 16; ++i) {
        vec3 samplePos = viewPos + TBN * ssaoKernel[i] * radius;

        vec4 offset = scene.projection * vec4(samplePos, 1.0);
        offset.xy  /= offset.w;
        offset.xy   = offset.xy * 0.5 + 0.5;
        offset.y    = 1.0 - offset.y;  // Vulkan NDC flip

        if (offset.x < 0.0 || offset.x > 1.0 ||
            offset.y < 0.0 || offset.y > 1.0) continue;

        float sampleDepth = texture(gBufferDepthSampler, offset.xy).r;
        vec4  sNdc        = vec4(offset.xy * 2.0 - 1.0, sampleDepth, 1.0);
        vec4  sViewCl     = scene.invProj * sNdc;
        float sLinearZ    = sViewCl.z / sViewCl.w;

        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(viewPos.z - sLinearZ));
        occlusion += (sLinearZ >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }
    return 1.0 - (occlusion / 16.0);
}

// ---- Bloom (brightness contribution from bright G-buffer albedo neighbors) ----
vec3 computeBloom(vec2 uv) {
    vec2 texelSize = 1.0 / vec2(textureSize(gBuffer0Sampler, 0));
    vec3  bloom      = vec3(0.0);
    float totalW     = 0.0;
    const float threshold = 1.5;
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec3  s   = texture(gBuffer0Sampler, uv + vec2(x, y) * texelSize * 2.0).rgb;
            float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
            if (lum > threshold) { bloom += s; totalW += 1.0; }
        }
    }
    return totalW > 0.0 ? bloom / totalW * 0.12 : vec3(0.0);
}

// ---- FXAA (edge anti-aliasing using G-buffer depth + albedo) ----
vec3 applyFXAA(vec2 uv, vec3 litColor) {
    vec2 texelSize = 1.0 / vec2(textureSize(gBuffer0Sampler, 0));
    const vec3 lumCoeff = vec3(0.2126, 0.7152, 0.0722);

    float dC = texture(gBufferDepthSampler, uv).r;
    float dN = texture(gBufferDepthSampler, uv + vec2(0.0,  texelSize.y)).r;
    float dS = texture(gBufferDepthSampler, uv - vec2(0.0,  texelSize.y)).r;
    float dE = texture(gBufferDepthSampler, uv + vec2(texelSize.x, 0.0)).r;
    float dW = texture(gBufferDepthSampler, uv - vec2(texelSize.x, 0.0)).r;

    float depthRange = max(max(max(dN, dS), max(dE, dW)), dC) -
                       min(min(min(dN, dS), min(dE, dW)), dC);
    if (depthRange < 0.0002) return litColor;

    // Check that it is also a shading edge (not just a smooth surface curve)
    vec3 nC = texture(gBuffer1Sampler, uv).rgb;
    vec3 nN = texture(gBuffer1Sampler, uv + vec2(0.0, texelSize.y)).rgb;
    if (dot(normalize(nC), normalize(nN)) > 0.98) return litColor;

    float edgeH = abs(dN + dS - 2.0 * dC);
    float edgeV = abs(dE + dW - 2.0 * dC);

    // Blend lit color with a brightness-ratio estimate from the neighboring albedo
    vec3 albedoC = texture(gBuffer0Sampler, uv).rgb;
    vec2 blendDir = edgeH > edgeV ? vec2(0.0, texelSize.y) : vec2(texelSize.x, 0.0);
    vec3 albedoA = texture(gBuffer0Sampler, uv - blendDir).rgb;
    vec3 albedoB = texture(gBuffer0Sampler, uv + blendDir).rgb;

    float lumC = max(dot(albedoC, lumCoeff), 0.001);
    float lumA = dot(albedoA, lumCoeff);
    float lumB = dot(albedoB, lumCoeff);
    vec3 blended = litColor * ((lumA + lumB) * 0.5) / lumC;

    return mix(litColor, clamp(blended, vec3(0.0), vec3(50.0)), 0.25);
}

// ---- Screen Space Reflections ----
// Ray-marches in view space, binary-searches the hit, returns Fresnel-weighted
// reflected albedo. Returns vec3(0) when no screen-space hit is found.
vec3 computeSSR(vec2 uv, vec3 viewPos, vec3 viewN, float roughness, vec3 F0) {
    if (roughness > 0.45) return vec3(0.0);

    vec3 reflDir = normalize(reflect(normalize(viewPos), viewN));
    if (reflDir.z > 0.0) return vec3(0.0); // pointing toward camera — skip

    const int   MAX_STEPS = 48;
    const float MAX_DIST  = 12.0;
    float stepSize = MAX_DIST / float(MAX_STEPS);

    vec3 rayPos = viewPos + viewN * 0.15; // offset to avoid self-hit

    for (int i = 0; i < MAX_STEPS; i++) {
        rayPos += reflDir * stepSize;

        vec4 clipPos = scene.projection * vec4(rayPos, 1.0);
        if (clipPos.w <= 0.0) break;

        vec2 sUV = clipPos.xy / clipPos.w * 0.5 + 0.5;
        sUV.y = 1.0 - sUV.y;
        if (sUV.x < 0.01 || sUV.x > 0.99 || sUV.y < 0.01 || sUV.y > 0.99) break;

        float sceneDepth = texture(gBufferDepthSampler, sUV).r;
        float rayDepth   = clipPos.z / clipPos.w;

        if (rayDepth > sceneDepth && (rayDepth - sceneDepth) < 0.15) {
            // Binary search to refine hit position
            vec3 lo = rayPos - reflDir * stepSize;
            vec3 hi = rayPos;
            for (int j = 0; j < 8; j++) {
                vec3 mid = (lo + hi) * 0.5;
                vec4 mc  = scene.projection * vec4(mid, 1.0);
                vec2 mUV = mc.xy / mc.w * 0.5 + 0.5;
                mUV.y    = 1.0 - mUV.y;
                float mRay   = mc.z / mc.w;
                float mScene = texture(gBufferDepthSampler, mUV).r;
                if (mRay > mScene) hi = mid; else lo = mid;
            }
            vec4 fc  = scene.projection * vec4(lo, 1.0);
            vec2 fUV = fc.xy / fc.w * 0.5 + 0.5;
            fUV.y    = 1.0 - fUV.y;

            if (texture(gBufferDepthSampler, fUV).r > 0.9999) return vec3(0.0); // hit sky

            vec3 reflColor = texture(gBuffer0Sampler, fUV).rgb;

            // Fade at screen edges and for rough surfaces
            vec2 edgeDist   = smoothstep(0.95, 1.0, abs(fUV * 2.0 - 1.0));
            float edgeFade  = 1.0 - clamp(max(edgeDist.x, edgeDist.y), 0.0, 1.0);
            float cosTheta  = max(dot(-normalize(viewPos), viewN), 0.0);
            float fresnel   = FresnelSchlick(cosTheta, F0).r;
            float ssrWeight = fresnel * (1.0 - roughness) * edgeFade;

            return reflColor * ssrWeight;
        }
    }
    return vec3(0.0);
}

// ---- Exponential height fog ----
vec3 applyHeightFog(vec3 color, vec3 worldPos) {
    vec3  camPos     = scene.cameraPosition.xyz;
    float dist       = length(worldPos - camPos);
    float heightDiff = worldPos.y - camPos.y;
    const float density  = 0.004;
    const float falloff  = 0.25;
    float fogAmt = density * exp(-falloff * camPos.y) *
                   (1.0 - exp(-falloff * heightDiff * dist)) /
                   (falloff * (abs(heightDiff) + 0.0001));
    fogAmt = clamp(fogAmt, 0.0, 0.6);
    vec3 fogColor = vec3(0.55, 0.62, 0.72);
    return mix(color, fogColor, fogAmt);
}

// ---- Main ----
void main() {
    vec2 texSize = vec2(textureSize(gBuffer0Sampler, 0));
    vec2 uv      = gl_FragCoord.xy / texSize;

    float depth = texture(gBufferDepthSampler, uv).r;

    // Background: sample skybox using reconstructed view direction
    if (depth >= 0.9999) {
        vec3 vPos   = reconstructViewPos(uv, depth);
        vec4 wPosCl = scene.invView * vec4(vPos, 1.0);
        vec3 viewDir = normalize(wPosCl.xyz / wPosCl.w - scene.cameraPosition.xyz);
        outColor = vec4(texture(skyboxCubemap, viewDir).rgb, 1.0);
        return;
    }

    vec4 g0 = texture(gBuffer0Sampler, uv);
    vec4 g1 = texture(gBuffer1Sampler, uv);
    float ao = texture(gBuffer2Sampler, uv).r;

    vec3  albedo   = g0.rgb;
    float metallic = g0.a;
    vec3  worldN   = normalize(g1.xyz);
    float roughness = g1.a;

    vec3 worldPos = reconstructWorldPos(uv, depth);
    vec3 viewPos  = reconstructViewPos(uv, depth);
    vec3 viewN    = normalize(mat3(scene.view) * worldN);
    vec3 V        = normalize(scene.cameraPosition.xyz - worldPos);
    vec3 F0       = mix(vec3(0.04), albedo, metallic);

    // SSAO
    float ssaoFactor = computeSSAO(uv, viewPos, viewN);

    // IBL ambient (diffuse irradiance + specular prefiltered)
    vec3 kS_ibl = FresnelSchlickRoughness(max(dot(worldN, V), 0.0), F0, roughness);
    vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);
    vec3 irradiance   = texture(irradianceMap, worldN).rgb;
    vec3 diffuseIBL   = kD_ibl * irradiance * albedo;

    vec3  R              = reflect(-V, worldN);
    float prefilteredLod = roughness * float(IBL_PREFILTER_MIPS - 1);
    vec3  prefilteredEnv = min(textureLod(prefilteredEnvMap, R, prefilteredLod).rgb, vec3(10.0));
    vec2  brdf           = texture(brdfLUT, vec2(max(dot(worldN, V), 0.0), roughness)).rg;
    vec3  specularIBL    = prefilteredEnv * (kS_ibl * brdf.x + brdf.y);

    // Screen-space reflections augment IBL specular for smooth surfaces
    vec3 ssrColor = computeSSR(uv, viewPos, viewN, roughness, F0);
    // Blend SSR over IBL: if SSR found a hit it's more accurate than the prefiltered map
    specularIBL = mix(specularIBL, ssrColor, step(0.001, length(ssrColor)));

    vec3 ambient = (diffuseIBL + specularIBL) * ao * ssaoFactor;
    vec3 lighting = ambient;

    // Directional light + PCF shadow
    vec3  sunDir    = normalize(-scene.directionalLight.direction.xyz);
    float sunShadow = shadowFactor(worldPos, viewPos, worldN, sunDir);
    lighting += (1.0 - sunShadow) * cookTorrance(albedo, worldN, V, sunDir, F0, metallic, roughness,
                    scene.directionalLight.colorIntensity.rgb,
                    scene.directionalLight.colorIntensity.a);

    // Point lights + omnidirectional shadow
    int pointCount = clamp(scene.lightCounts.x, 0, 4);
    for (int i = 0; i < pointCount; ++i) {
        vec3  toLight = scene.pointLights[i].position.xyz - worldPos;
        float dist    = length(toLight);
        vec3  L       = toLight / max(dist, 0.0001);
        float att     = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        float shadow  = (i == 0) ? pointShadowFactor(worldPos, scene.pointLights[i].position.xyz) : 0.0;
        lighting += (1.0 - shadow) * cookTorrance(albedo, worldN, V, L, F0, metallic, roughness,
                        scene.pointLights[i].colorIntensity.rgb,
                        scene.pointLights[i].colorIntensity.a * att);
    }

    // Spot lights
    int spotCount = clamp(scene.lightCounts.y, 0, 2);
    for (int i = 0; i < spotCount; ++i) {
        vec3  toLight = scene.spotLights[i].position.xyz - worldPos;
        float dist    = length(toLight);
        vec3  L       = toLight / max(dist, 0.0001);
        float theta   = dot(L, normalize(-scene.spotLights[i].direction.xyz));
        float eps     = scene.spotLights[i].cutoffAngles.x - scene.spotLights[i].cutoffAngles.y;
        float cone    = clamp((theta - scene.spotLights[i].cutoffAngles.y) / max(eps, 0.0001), 0.0, 1.0);
        float att     = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        lighting += cookTorrance(albedo, worldN, V, L, F0, metallic, roughness,
                       scene.spotLights[i].colorIntensity.rgb,
                       scene.spotLights[i].colorIntensity.a * att * cone);
    }

    // Bloom
    lighting += computeBloom(uv);

    // FXAA (depth-edge + albedo-luminance blend)
    lighting = applyFXAA(uv, lighting);

    // Exponential height fog
    lighting = applyHeightFog(lighting, worldPos);

    outColor = vec4(lighting, 1.0);
}

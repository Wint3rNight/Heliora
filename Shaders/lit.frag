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
    vec4 fogParams; // x=density, y=falloff, z=clampMax
    int debugMode;
    // x = sRGB albedo decode
    // y = specular AA enable, z = SPEC_AA_VARIANCE, w = SPEC_AA_THRESHOLD
    vec4 qualityToggles;
    // x = mipmap sampling enable (P7)
    // y = correlated Smith G enable (P3)
    // z = Vogel-disk PCF enable (P5b)
    vec4 qualityToggles2;
    // TAA state (consumed by second.frag — present here only to keep std140
    // offsets identical across all shaders that bind set=0,binding=0).
    mat4 prevViewProj;
    vec4 taaParams;
    vec4 viewportSize;
} scene;

layout(set = 0, binding = 1) uniform sampler2DArrayShadow shadowMap;
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
// Vogel-disk PCF kernel (Filament §5.4.2). Samples distributed via the
// golden angle to give near-uniform spatial coverage with arbitrary count;
// per-pixel rotation by interleaved gradient noise (Jimenez 2014) breaks
// the regular shadow-texel aliasing into high-freq noise.
vec2 vogelDisk(int i, int n, float phi) {
    const float GOLDEN = 2.39996323; // π * (3 - sqrt(5))
    float r     = sqrt((float(i) + 0.5) / float(n));
    float theta = float(i) * GOLDEN + phi;
    return vec2(r * cos(theta), r * sin(theta));
}

float interleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 *
                 fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

// PCF on a single cascade. Returns 1.0 = fully shadowed, 0.0 = lit,
// -1.0 = fragment is outside this cascade's shadow map (caller must fall back).
float sampleCascade(int cascade, vec3 worldPos, vec3 N, vec3 L) {
    vec4 lsPos = scene.lightSpaceMatrices[cascade] * vec4(worldPos, 1.0);
    vec3 proj  = lsPos.xyz / lsPos.w;
    proj.xy    = proj.xy * 0.5 + 0.5;

    // Receiver-plane depth gradient (Isidoro 2006, MJP "Shadow Sample Update").
    // Compute the slope of shadow-space depth w.r.t. shadow-space UV so each
    // off-center PCF tap can compare against the depth at *its* surface
    // position, not the center pixel's. Removes the Swiss-cheese pattern that
    // a single-depth Vogel kernel produces on grazing floors.
    //
    // Derivatives MUST be sampled before any flow-divergent return so all
    // four quad lanes contribute. At cascade boundaries this may give one
    // bogus pixel-wide band, hidden by the 15% cascade blend.
    vec2  duvdx   = dFdx(proj.xy);
    vec2  duvdy   = dFdy(proj.xy);
    float ddepdx  = dFdx(proj.z);
    float ddepdy  = dFdy(proj.z);

    if (proj.z > 1.0 || proj.z < 0.0 || proj.x < 0.0 || proj.x > 1.0 ||
        proj.y < 0.0 || proj.y > 1.0) return -1.0;

    // Solve [duvdx; duvdy]^T * grad = [ddepdx; ddepdy] for grad = d(depth)/d(uv).
    // Inverse of the screen→shadow-UV Jacobian. Clamp on near-singular det
    // (silhouettes and near-degenerate triangles) to avoid grad blowing up.
    //
    // Clamp at ±2: real grazing-angle gradients sit well under 1 even at
    // sunrise/sunset, but silhouette quads can spike grad arbitrarily large.
    // The earlier ±0.05 cap neutralized the fix on exactly the surfaces it
    // was meant to help.
    float det = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    vec2  grad;
    if (abs(det) > 1e-8) {
        grad = vec2( duvdy.y * ddepdx - duvdx.y * ddepdy,
                    -duvdy.x * ddepdx + duvdx.x * ddepdy) / det;
        grad = clamp(grad, vec2(-2.0), vec2(2.0));
    } else {
        grad = vec2(0.0);
    }

    float bias = max(0.0008 * float(cascade + 1) * (1.0 - dot(N, L)), 0.0005);
    vec2  texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);

    // 12-tap Vogel-disk PCF over a hardware compare sampler. Each tap's
    // reference depth is shifted along the receiver plane to match the
    // local surface slope.
    // 32-tap Vogel-disk PCF. The bumped sample count reduces the per-pixel
    // shadow quantization step from 1/12 (8.3%) to 1/32 (3.1%); below the
    // JND once multiplied by HDR sun intensity. Same radius and rotation as
    // before, so penumbra width is unchanged.
    const int SAMPLES = 32;
    float radius = (1.5 + float(cascade) * 0.5);
    float phi    = interleavedGradientNoise(gl_FragCoord.xy) * 6.2831853;
    float lit    = 0.0;
    for (int i = 0; i < SAMPLES; ++i) {
        vec2  off       = vogelDisk(i, SAMPLES, phi) * radius * texelSize;
        float tapDepth  = proj.z - bias + dot(grad, off);
        lit += texture(shadowMap,
                       vec4(proj.xy + off, float(cascade), tapDepth));
    }
    return 1.0 - lit / float(SAMPLES);
}

float shadowFactor(vec3 worldPos, vec3 viewPos, vec3 N, vec3 L) {
    float viewDepth = -viewPos.z;

    int primaryCascade = 3;
    if      (viewDepth < scene.cascadeSplits.x) primaryCascade = 0;
    else if (viewDepth < scene.cascadeSplits.y) primaryCascade = 1;
    else if (viewDepth < scene.cascadeSplits.z) primaryCascade = 2;

    // Fall back to progressively larger cascades when the fragment projects
    // outside the primary cascade's shadow map. This fixes fragments that are
    // below or beside the camera: view-Z depth is near zero for them (they
    // map to cascade 0) but the small near-cascade sphere AABB sits high in
    // the air and misses the geometry.
    int cascade = primaryCascade;
    float shadow = -1.0;
    for (int c = primaryCascade; c <= 3; ++c) {
        shadow = sampleCascade(c, worldPos, N, L);
        if (shadow >= 0.0) { cascade = c; break; }
    }
    if (shadow < 0.0) return 0.0;  // outside all cascades — assume lit

    // Blend with the next cascade in the last 15% of this cascade's range
    // to hide hard popping at boundaries.
    if (cascade < 3) {
        float thisSplit = scene.cascadeSplits[cascade];
        float prevSplit = (cascade == 0) ? 0.0 : scene.cascadeSplits[cascade - 1];
        float bandStart = mix(prevSplit, thisSplit, 0.85);
        if (viewDepth > bandStart) {
            float t = clamp((viewDepth - bandStart) / (thisSplit - bandStart),
                            0.0, 1.0);
            float shadowNext = sampleCascade(cascade + 1, worldPos, N, L);
            if (shadowNext >= 0.0)
                shadow = mix(shadow, shadowNext, t);
        }
    }
    return shadow;
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

// Heitz 2014 / Filament §4.4.2 — height-correlated Smith GGX visibility,
// already divided by 4*NoV*NoL, so the BRDF denominator collapses to D*V*F.
// Compared to the separable Schlick-GGX form this removes the energy excess
// at grazing angles that makes non-metals look 'wet' under sunlight.
float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
    float a2     = roughness * roughness * roughness * roughness;
    float ggxV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float ggxL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / max(ggxV + ggxL, 1e-5);
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
    // Clamp the dot products to a small positive epsilon. NoL <= 0 means the
    // light is below the horizon; bail. The 1e-4 floor on NoV/NoH stops the
    // denominator from blowing up at perfectly grazing views.
    float NoL = dot(N, L);
    if (NoL <= 0.0) return vec3(0.0);
    NoL = clamp(NoL, 1e-4, 1.0);
    vec3 H = normalize(V + L);
    float NoV = clamp(dot(N, V), 1e-4, 1.0);
    float NoH = clamp(dot(N, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);

    float NDF = DistributionGGX(N, H, roughness);
    vec3  F   = FresnelSchlick(VoH, F0);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    // Correlated Smith visibility absorbs the 4*NoV*NoL denominator.
    float Vis = V_SmithGGXCorrelated(NoV, NoL, roughness);
    vec3 specular = NDF * Vis * F;
    return (kD * albedo / PI + specular) * lightColor * intensity * NoL;
}

// ---- SSAO ----
float computeSSAO(vec2 uv, vec3 viewPos, vec3 viewNormal) {
    vec2 noiseScale = vec2(textureSize(gBuffer0Sampler, 0)) / 4.0;
    vec3 randomVec  = normalize(texture(ssaoNoise, uv * noiseScale).rgb * 2.0 - 1.0);
    vec3 tangent    = normalize(randomVec - viewNormal * dot(randomVec, viewNormal));
    vec3 bitangent  = cross(viewNormal, tangent);
    mat3 TBN        = mat3(tangent, bitangent, viewNormal);

    float occlusion = 0.0;
    const float radius = 1.0;
    const float bias   = 0.03;

    for (int i = 0; i < 16; ++i) {
        vec3 samplePos = viewPos + TBN * ssaoKernel[i] * radius;

        vec4 offset = scene.projection * vec4(samplePos, 1.0);
        offset.xy  /= offset.w;
        offset.xy   = offset.xy * 0.5 + 0.5;
        // No y-flip: projection's [1][1] *= -1 already aligns NDC with
        // Vulkan's screen-down convention, so this UV matches gl_FragCoord/H.

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

// ---- Bloom (bright-pass on estimated lit neighbors, not raw albedo) ----
// Samples albedo * NdotL * sunColor to approximate which neighbors are
// actually sunlit rather than just brightly-coloured in the texture.
vec3 computeBloom(vec2 uv) {
    vec2 texelSize = 1.0 / vec2(textureSize(gBuffer0Sampler, 0));
    vec3 bloom  = vec3(0.0);
    float totalW = 0.0;
    const float threshold = 2.0;   // threshold on estimated lit luminance
    vec3 sunDir   = normalize(-scene.directionalLight.direction.xyz);
    vec3 sunColor = scene.directionalLight.colorIntensity.rgb *
                    scene.directionalLight.colorIntensity.a;
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 sUV    = uv + vec2(x, y) * texelSize * 3.0;
            vec3 albedo = texture(gBuffer0Sampler, sUV).rgb;
            vec3 N      = texture(gBuffer1Sampler, sUV).xyz;
            if (dot(N, N) < 0.1) continue;   // skip sky/empty pixels
            float NdotL = max(dot(normalize(N), sunDir), 0.0);
            vec3  litEst = albedo * sunColor * NdotL;
            float lum    = dot(litEst, vec3(0.2126, 0.7152, 0.0722));
            if (lum > threshold) { bloom += litEst; totalW += 1.0; }
        }
    }
    return totalW > 0.0 ? bloom / totalW * 0.05 : vec3(0.0);
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
    if (depthRange < 0.005) return litColor;

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

// ---- Exponential height fog ----
// Exponential distance fog with height-based attenuation. The previous
// integral form degenerated to 0 when the ray was horizontal (heightDiff≈0),
// which is the common indoor case. This form is well-behaved everywhere.
vec3 applyHeightFog(vec3 color, vec3 worldPos) {
    vec3  camPos    = scene.cameraPosition.xyz;
    float dist      = length(worldPos - camPos);
    float density   = scene.fogParams.x;
    float falloff   = scene.fogParams.y;
    float clampMax  = scene.fogParams.z;
    float heightFactor = exp(-falloff * max(camPos.y, 0.0));
    float fogAmt = (1.0 - exp(-density * dist)) * heightFactor;
    fogAmt = clamp(fogAmt, 0.0, clampMax);
    vec3 fogColor = vec3(0.55, 0.62, 0.72);
    return mix(color, fogColor, fogAmt);
}

// ---- Main ----
void main() {
    vec2 texSize = vec2(textureSize(gBuffer0Sampler, 0));
    vec2 uv      = gl_FragCoord.xy / texSize;

    float depth = texture(gBufferDepthSampler, uv).r;

    // Background: sample skybox using reconstructed view direction.
    // Multiplied by the ambient intensity so the night sky goes dark instead
    // of staying frozen at baked daylight.
    if (depth >= 0.9999) {
        vec3 vPos   = reconstructViewPos(uv, depth);
        vec4 wPosCl = scene.invView * vec4(vPos, 1.0);
        vec3 viewDir = normalize(wPosCl.xyz / wPosCl.w - scene.cameraPosition.xyz);
        outColor = vec4(texture(skyboxCubemap, viewDir).rgb *
                        scene.qualityToggles2.w,
                        1.0);
        return;
    }

    vec4 g0 = texture(gBuffer0Sampler, uv);
    vec4 g1 = texture(gBuffer1Sampler, uv);
    float ao = texture(gBuffer2Sampler, uv).r;

    vec3  albedo   = g0.rgb;
    float metallic = g0.a;
    vec3  worldN   = normalize(g1.xyz);
    float roughness = g1.a;

    // G-buffer debug views — short-circuit full PBR when active
    if (scene.debugMode == 1) { outColor = vec4(albedo, 1.0); return; }
    if (scene.debugMode == 2) { outColor = vec4(worldN * 0.5 + 0.5, 1.0); return; }
    if (scene.debugMode == 3) { outColor = vec4(vec3(metallic), 1.0); return; }
    if (scene.debugMode == 4) { outColor = vec4(vec3(roughness), 1.0); return; }
    if (scene.debugMode == 5) { outColor = vec4(vec3(texture(gBufferDepthSampler, uv).r), 1.0); return; }

    vec3 worldPos = reconstructWorldPos(uv, depth);
    vec3 viewPos  = reconstructViewPos(uv, depth);
    vec3 viewN    = normalize(mat3(scene.view) * worldN);
    vec3 V        = normalize(scene.cameraPosition.xyz - worldPos);
    vec3 F0       = mix(vec3(0.04), albedo, metallic);

    // Geometric specular AA (Kaplanyan & Hill 2016 / Filament §4.7.1).
    // Pixel-scale normal variation is folded into the NDF roughness so
    // sub-pixel highlights stop sparkling. Variance/threshold are tunable
    // via qualityToggles.z/.w. Result is used for both direct *and* IBL.
    float perceptualRoughnessAA;
    {
        vec3  dndu     = dFdx(worldN);
        vec3  dndv     = dFdy(worldN);
        float variance = scene.qualityToggles.z *
                         (dot(dndu, dndu) + dot(dndv, dndv));
        float kernel   = min(2.0 * variance, scene.qualityToggles.w);
        float alpha    = roughness * roughness;
        perceptualRoughnessAA = sqrt(sqrt(alpha * alpha + kernel));
    }
    // Shader-wide roughness handle from here on. Both direct cookTorrance
    // and IBL pick from this value.
    roughness = perceptualRoughnessAA;

    // IBL-only roughness floor (qualityToggles.x). Screen-space dFdx(N)
    // under-reports variance when fine normal-map detail compresses into a
    // few screen pixels at oblique angles — curtains keep sparkling off-sun
    // even though AA "fired". A flat minimum on IBL mip selection escapes
    // that case without touching direct-light specular on smooth surfaces.
    float roughnessForIBL = max(perceptualRoughnessAA, scene.qualityToggles.x);

    // SSAO
    float ssaoFactor = computeSSAO(uv, viewPos, viewN);

    // IBL ambient (diffuse irradiance + specular prefiltered)
    vec3 kS_ibl = FresnelSchlickRoughness(max(dot(worldN, V), 0.0), F0, roughnessForIBL);
    vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);
    vec3 irradiance   = texture(irradianceMap, worldN).rgb;
    vec3 diffuseIBL   = kD_ibl * irradiance * albedo;

    vec3  R              = reflect(-V, worldN);
    // Karis split-sum LOD based on perceptual roughness.
    float roughLod       = roughnessForIBL * float(IBL_PREFILTER_MIPS - 1);
    // Filament-style geometric LOD based on screen-space derivative of R.
    // At silhouette pixels (where R swings wildly between adjacent fragments)
    // this drives the fetch to a higher mip, killing per-pixel sparkle on
    // normal-mapped surfaces that pure roughness-LOD can't reach.
    vec3  dRdx           = dFdx(R);
    vec3  dRdy           = dFdy(R);
    float envSize        = float(textureSize(prefilteredEnvMap, 0).x);
    float derivMag2      = max(dot(dRdx, dRdx), dot(dRdy, dRdy));
    float geomLod        = 0.5 * log2(max(derivMag2 * envSize * envSize, 1e-6));
    float prefilteredLod = clamp(max(roughLod, geomLod), 0.0,
                                 float(IBL_PREFILTER_MIPS - 1));
    vec3  prefilteredEnv = min(textureLod(prefilteredEnvMap, R, prefilteredLod).rgb, vec3(10.0));
    vec2  brdf           = texture(brdfLUT, vec2(max(dot(worldN, V), 0.0), roughnessForIBL)).rg;
    vec3  specularIBL    = prefilteredEnv * (kS_ibl * brdf.x + brdf.y);

    // Note: SSR is applied in a separate composite pass that samples this lit
    // image; here we keep IBL specular only.

    // Ambient (IBL diffuse + specular) is scaled by qualityToggles2.w. The
    // prefilter + irradiance cubemaps are baked once from the daytime sky,
    // so without this scale every glossy surface keeps reflecting baked
    // daylight even when the sun is below the horizon — the "everything
    // looks silver at night" bug.
    vec3 ambient = (diffuseIBL + specularIBL) * ao * ssaoFactor *
                   scene.qualityToggles2.w;

    // Directional light + PCF shadow
    vec3  sunDir    = normalize(-scene.directionalLight.direction.xyz);
    float sunShadow = shadowFactor(worldPos, viewPos, worldN, sunDir);
    vec3 sunDirectUnshadowed = cookTorrance(albedo, worldN, V, sunDir, F0, metallic, roughness,
                    scene.directionalLight.colorIntensity.rgb,
                    scene.directionalLight.colorIntensity.a);
    vec3 directLight = (1.0 - sunShadow) * sunDirectUnshadowed;

    // Point lights + omnidirectional shadow
    int pointCount = clamp(scene.lightCounts.x, 0, 4);
    for (int i = 0; i < pointCount; ++i) {
        vec3  toLight = scene.pointLights[i].position.xyz - worldPos;
        float dist    = length(toLight);
        vec3  L       = toLight / max(dist, 0.0001);
        float att     = 1.0 / (1.0 + 0.1 * dist * dist);
        float shadow  = (i == 0) ? pointShadowFactor(worldPos, scene.pointLights[i].position.xyz) : 0.0;
        directLight += (1.0 - shadow) * cookTorrance(albedo, worldN, V, L, F0, metallic, roughness,
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
        float att     = 1.0 / (1.0 + 0.1 * dist * dist);
        directLight += cookTorrance(albedo, worldN, V, L, F0, metallic, roughness,
                       scene.spotLights[i].colorIntensity.rgb,
                       scene.spotLights[i].colorIntensity.a * att * cone);
    }

    // Phase-2 diagnostic debug views: short-circuit before bloom/FXAA/fog so
    // post-effects don't corrupt the signal.
    //   6 = sun shadow visibility (1.0 = fully lit, 0.0 = fully shadowed)
    //   7 = SSAO factor (1.0 = no occlusion)
    //   8 = direct lighting only (no IBL, no ambient)
    //   9 = indirect lighting only (IBL ambient, with SSAO)
    if (scene.debugMode == 6) { outColor = vec4(vec3(1.0 - sunShadow), 1.0); return; }
    if (scene.debugMode == 7) { outColor = vec4(vec3(ssaoFactor), 1.0); return; }
    if (scene.debugMode == 8) { outColor = vec4(directLight, 1.0); return; }
    if (scene.debugMode == 9) { outColor = vec4(ambient, 1.0); return; }
    // mode 10: direct lighting with sunShadow forced to 0 — isolates whether
    // the floor dither lives in the shadow term or in cookTorrance itself.
    if (scene.debugMode == 10) { outColor = vec4(sunDirectUnshadowed, 1.0); return; }

    vec3 lighting = ambient + directLight;

    // Bloom
    lighting += computeBloom(uv);

    // FXAA (depth-edge + albedo-luminance blend)
    lighting = applyFXAA(uv, lighting);

    // Exponential height fog
    lighting = applyHeightFog(lighting, worldPos);

    outColor = vec4(lighting, 1.0);
}

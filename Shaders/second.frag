#version 450

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inputColor;

layout(location = 0) out vec4 outColor;

vec3 acesFilm(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr    = subpassLoad(inputColor).rgb;
    vec3 mapped = acesFilm(hdr);
    mapped      = pow(mapped, vec3(1.0 / 2.2));
    outColor    = vec4(mapped, 1.0);
}

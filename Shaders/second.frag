#version 450

layout(input_attachment_index = 0,binding = 0) uniform subpassInput inputColor; // color output from first subpass
layout(input_attachment_index = 1,binding = 1) uniform subpassInput inputDepth; // depth output from first subpass

layout(location = 0) out vec4 color; // output color of the fragment



void main() {
    color = subpassLoad(inputColor).rgba; // load color from input attachment
}

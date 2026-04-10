#version 450

layout(input_attachment_index = 0,binding = 0) uniform subpassInput inputColor; // color output from first subpass
layout(input_attachment_index = 1,binding = 1) uniform subpassInput inputDepth; // depth output from first subpass

layout(location = 0) out vec4 color; // output color of the fragment



void main() {
    int xHalf = 1366/2;
    if(gl_FragCoord.x>xHalf){
        float lowerBound = 0.98;
        float upperBound = 1;

        float depth = subpassLoad(inputDepth).r;
        float depthColorScaled = 1.0f -((depth-lowerBound) / (upperBound - lowerBound));
        color = vec4(subpassLoad(inputColor).rgb * depthColorScaled, 1.0f);
    }
    else{
        color = subpassLoad(inputColor).rgba;
    }
}

#version 420 core

// Attributes
layout (location = 0) in vec3 position;    // we can also use layout to specify the location of the attribute
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 color;


out vec3 fragColor;
out vec2 TexCoord;


void main(){
    //pass position uv coords on to the tesselation shaders
    vec4 pos = vec4(position, 1.0);
    gl_Position =  pos;
    TexCoord = uv;
    fragColor = color;
}
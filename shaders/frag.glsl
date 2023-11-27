#version 420 core

in vec3 fragColor;
in vec2 fragUV;
in float Height;

out vec4 outColor;

layout (binding=0) uniform sampler2D tex;   // attach tex to texture unit 0

void main(){
    vec3 color = vec3(1.0);
    //determine what we should color based on height we got from texture
    if(Height < 0.1)
    {
        color = vec3(0.255,0.42,0.875);
    }
    else  if(Height < 0.2)
    {
        color = vec3(0.243,0.643,0.941);
    }
    else  if(Height < 0.3)
    {
        color = vec3(0.992,0.925,0.729);
    }
    else  if(Height < 0.35)
    {
        color = vec3(0.561,0.784,0.051);
    }
    else  if(Height < 0.4)
    {
        color = vec3(0.4,0.541,0.075);
    }
    else  if(Height < 0.5)
    {
        color = vec3(0.569,0.463,0.294);
    }
    else  if(Height < 0.7)
    {
        color = vec3(0.278,0.263,0.239);
    }
     outColor = vec4 (color,1.0);
}
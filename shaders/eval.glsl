#version 420 core

layout(quads, equal_spacing, ccw) in;

uniform sampler2D heightMap;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform mat4 model_matrix;

in vec2 TextureCoord[];

out float Height;

void main()
{
    //patch coords
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    //point coords
    vec2 t1 = TextureCoord[0];
    vec2 t2 = TextureCoord[1];
    vec2 t3 = TextureCoord[2];
    vec2 t4 = TextureCoord[3];

    //perform interpolation to find the uv coords of the entire plane
    vec2 t5 = (t2 - t1) * u + t1;
    vec2 t6 = (t4 - t3) * u + t3;
    vec2 texCoord = (t6 - t5) * v + t5;
    
    //get our height value from the texture map we pass this to frag shader as well
    Height = texture(heightMap, texCoord).y;

    //point positions
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;
    vec4 p3 = gl_in[3].gl_Position;


    //patch surface normal
    vec4 uVec = p1 - p0;
    vec4 vVec = p2 - p0;
    vec4 normal = normalize( vec4( cross(vVec.xyz, uVec.xyz), 0) );

    //interpolation of positions for final position of patch
    vec4 p4 = (p1 - p0) * u + p0; 
    vec4 p5 = (p3 - p2) * u + p2;
    vec4 p = (p5 - p4) * v + p4 + normal * Height;

    //final position of patch
    gl_Position = projection_matrix * view_matrix * model_matrix * p;

 
}
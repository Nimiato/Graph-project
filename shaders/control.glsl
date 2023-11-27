#version 420 core

layout (vertices = 4) out;

in vec2 TexCoord[];

out vec2 TextureCoord[];

void main()
{
    //pass through both postition and uv coords for each vertex in the patch
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    TextureCoord[gl_InvocationID] = TexCoord[gl_InvocationID];

    //set the tesselation levels for the patch
    if (gl_InvocationID == 0)
    {
        gl_TessLevelOuter[0] = 2.0;
        gl_TessLevelOuter[1] = 4.0;
        gl_TessLevelOuter[2] = 6.0;
        gl_TessLevelOuter[3] = 8.0;

        gl_TessLevelInner[0] = 8.0;
        gl_TessLevelInner[1] = 8.0;
    }
}
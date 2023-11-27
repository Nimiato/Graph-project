import pygame as pg
from PIL import Image, ImageDraw
from OpenGL.GL import *
import numpy as np
import pyrr
from guiV2 import SimpleGUI
import shaderLoaderV3
import noise


pg.init()

# Set up OpenGL context version
pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)

# Create a window for graphics using OpenGL
width = 900
height = 500
pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF)

MAP_SIZE = (512,512)


#generating noise_map/heightmap and the max / min value for normalizing the values between 0 and 1.
def gen_noisemap(octaves,persistence,lacunarity,seed):
    noise_scale = MAP_SIZE[0]/2
    noisemap = np.zeros(MAP_SIZE)
    max_noise = 0
    min_noise = 0
    for i in range(MAP_SIZE[0]) :
        for j in range(MAP_SIZE[1]):
            noisemap[i][j] = (noise.pnoise2((i/noise_scale) + .0000000001, (j/noise_scale) + .0000000001, octaves, persistence, lacunarity, MAP_SIZE[0], MAP_SIZE[1], seed))
            if max_noise < noisemap[i][j]:
                max_noise = noisemap[i][j]
            if min_noise > noisemap[i][j]:
                min_noise = noisemap[i][j]
    return noisemap,max_noise,min_noise
noisemap,max_noise,min_noise = gen_noisemap(6,0.5,2,100)

#normalize noise between 0 and 1
def normalize_noise(noisemap,min_noise,max_noise):
    normal_noise = np.zeros_like(noisemap)
    for x in range(MAP_SIZE[0]):
        for y in range(MAP_SIZE[1]):
            normal_noise[x][y] = ((noisemap[x][y] - min_noise)/(max_noise - min_noise))
    return normal_noise

norm_noise = normalize_noise(noisemap,max_noise,min_noise)

#saving noise as img for use as texture we lose some detail here since PIL only accepts ints for image but that is okay :(
def gen_image(noisemap):
    image = Image.fromarray(np.uint8(noisemap*255))
    image.save("noisemap.png")

gen_image(norm_noise)

#vertex generation for our patches 
vertexes = []
xstep = 2/MAP_SIZE[0]
ystep = 2/MAP_SIZE[1]
for j in range(MAP_SIZE[0]-1):
    for i in range(MAP_SIZE[1]-1):
        xcomponent = -1 + (xstep*i)
        ycomponent = 0.0
        zcomponent = -1 + (ystep*j)
        x2 = -1 + (xstep*(i+1))
        z2 = -1 + (ystep*j)
        x3 = -1 + (xstep*i)
        z3 = -1 + (ystep*(j+1))
        x4 = -1 + (xstep*(i+1))
        z4 = -1 + (ystep*(j+1))
        #vertex 1
        vertexes.append(xcomponent)
        vertexes.append(ycomponent)
        vertexes.append(zcomponent)
        vertexes.append(0 + ((xstep/2)*i))
        vertexes.append(0 + ((ystep/2)*j))
        vertexes.append(0.0)
        vertexes.append(0.0)
        vertexes.append(0.0)
        #vertex 2
        vertexes.append(x2)
        vertexes.append(ycomponent)
        vertexes.append(z2)
        vertexes.append(0 + ((xstep/2)*i))
        vertexes.append(0 + ((ystep/2)*j))
        vertexes.append(0.0)
        vertexes.append(0.0)
        vertexes.append(0.0)
        #vertex 3
        vertexes.append(x3)
        vertexes.append(ycomponent)
        vertexes.append(z3)
        vertexes.append(0 + ((xstep/2)*i))
        vertexes.append(0 + ((ystep/2)*j))
        vertexes.append(0.0)
        vertexes.append(0.0)
        vertexes.append(0.0)
        #vertex 4
        vertexes.append(x4)
        vertexes.append(ycomponent)
        vertexes.append(z4)
        vertexes.append(0 + ((xstep/2)*i))
        vertexes.append(0 + ((ystep/2)*j))
        vertexes.append(0.0)
        vertexes.append(0.0)
        vertexes.append(0.0)
numpatches =(MAP_SIZE[0] - 1) * (MAP_SIZE[1] - 1) * 4

glClearColor(0.3, 0.4, 0.5, 1.0)
glEnable(GL_DEPTH_TEST)

#assign shaders
shaderProgram = shaderLoaderV3.ShaderProgram("project graph/shaders/vert.glsl", "project graph/shaders/frag.glsl", "project graph/shaders/control.glsl", "project graph/shaders/eval.glsl")
glUseProgram(shaderProgram.shader)    # Use the shader program

# Define the vertices
vertices = vertexes
vertices = np.array(vertices, dtype=np.float32)
size_position = 3       # x, y, z
size_texture = 2        # s, t
size_color = 3          # r, g, b

stride = (size_position + size_texture + size_color) * 4   # size of a single vertex in bytes
offset_position = 0                                 # offset of the position data
offset_texture = size_position * 4                  # offset of the texture data. Texture data starts after 3 floats (12 bytes) of position data
offset_color = (size_position + size_texture) * 4   # offset of the color data. Color data starts after 5 floats (20 bytes) of position and texture data
n_vertices = len(vertices) // (size_position + size_texture + size_color)  # number of vertices

vao = glGenVertexArrays(1)
glBindVertexArray(vao)
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)   # Upload the data to the GPU.

# Position attribute
position_loc = 0
glBindAttribLocation(shaderProgram.shader, position_loc, "position")
glVertexAttribPointer(position_loc, size_position, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc)

# texture attribute
texture_loc = 1
glBindAttribLocation(shaderProgram.shader, texture_loc, "uv")
glVertexAttribPointer(texture_loc, size_texture, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_texture))
glEnableVertexAttribArray(texture_loc)

# color attribute
color_loc = 2
glBindAttribLocation(shaderProgram.shader, color_loc, "color")
glVertexAttribPointer(color_loc, size_color, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_color))
glEnableVertexAttribArray(color_loc)

#set to 4 vertices in patch
glPatchParameteri(GL_PATCH_VERTICES, 4)

#read image data for storage as texture
filename = "noisemap.png"
img = pg.image.load(filename)
img_data = pg.image.tobytes(img, "RGB", True)
w, h = img.get_size()

# Create a texture object
texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_id)        # Bind the texture object. That is, make it the active one.
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)    # Set the texture wrapping parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)    # Set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# Upload the image data to the GPU
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

shaderProgram["heightMap"] = 0 #send texture to tesselation shaders



# Camera parameters
eye = (0,0,2.5)
target = (0,0,0)
up = (0,1,0)

fov = 60
aspect = width/height
near = 0.1
far = 10

view_mat  = pyrr.matrix44.create_look_at(eye, target, up)
projection_mat = pyrr.matrix44.create_perspective_projection_matrix(fov, aspect, near, far)
model_mat = pyrr.matrix44.create_identity()

gui = SimpleGUI("Assignment 7")

# Create a slider for the rotation angle around the Z axis
camera_rotY_slider = gui.add_slider("camera Y angle", -180, 180, 0, resolution=1)
camera_rotX_slider = gui.add_slider("camera X angle", -90, 90, 0, resolution=1)
fov_slider = gui.add_slider("fov", 25, 90, fov, resolution=1)
#sliders for real time changing of terrain
octave_slider = gui.add_slider("Octaves", 1, 10, 6, resolution=1)
persistance_slider = gui.add_slider("persistance", 0, 1, 0.5)
lacunarity_slider = gui.add_slider("Lacunarity", 1.5, 3.5, 2.0, resolution = 0.25)
seed_slider = gui.add_slider("Seed", 0, 500, 100, resolution=1)
#checkable values so we arent constantly generating noise maps and images in draw loop
current_octave = 6
current_persistance = 0.5
current_lacunarity = 2.0
current_seed = 100



draw = True
while draw:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            draw = False

    # Clear color buffer and depth buffer before drawing each frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    rotateY_mat = pyrr.matrix44.create_from_y_rotation(np.deg2rad(camera_rotY_slider.get_value()))
    rotateX_mat = pyrr.matrix44.create_from_x_rotation(np.deg2rad(camera_rotX_slider.get_value()))
    rotation_mat = pyrr.matrix44.multiply(rotateX_mat, rotateY_mat)

    rotated_eye = pyrr.matrix44.apply_to_vector(rotation_mat, eye)

    view_mat = pyrr.matrix44.create_look_at(rotated_eye, target, up)
    projection_mat = pyrr.matrix44.create_perspective_projection_matrix(fov_slider.get_value(),
                                                                        aspect, near,  far)

    #get value for realtime map changes
    octaves = octave_slider.get_value()
    persistance = persistance_slider.get_value()
    lacunarity = lacunarity_slider.get_value()
    seed = seed_slider.get_value()

    #check against current value so we arent always generating the same image/map
    if current_octave != octaves or current_persistance != persistance or lacunarity != current_lacunarity or seed != current_seed:
        #regenerate map and image based on new parameters
        new_noisemap,new_max_noise,new_min_noise = gen_noisemap(octaves,persistance,lacunarity,seed)
        new_norm_noise = normalize_noise(new_noisemap,new_max_noise,new_min_noise)
        gen_image(new_norm_noise)

        filename = "noisemap.png"
        img = pg.image.load(filename)
        img_data = pg.image.tobytes(img, "RGB", True)
        w, h = img.get_size()

        #send new texture to shaders
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

        #update check values
        current_octave = octaves
        current_lacunarity = lacunarity
        current_persistance = persistance
        current_seed = seed
    
    # Set uniforms
    shaderProgram["view_matrix"] = view_mat
    shaderProgram["projection_matrix"] = projection_mat
    shaderProgram["model_matrix"] = model_mat
    
    
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)


     # ***** Draw *****
    glBindVertexArray(vao)
    glDrawArrays(GL_PATCHES,0,numpatches)      # Draw the patches
    


    # Refresh the display to show what's been drawn
    pg.display.flip()


# Cleanup
glDeleteVertexArrays(1, [vao])
glDeleteBuffers(1, [vbo])
glDeleteProgram(shaderProgram.shader)

pg.quit()   # Close the graphics window
quit()      # Exit the program
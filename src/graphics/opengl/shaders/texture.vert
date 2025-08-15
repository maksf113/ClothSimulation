#version 450 core
layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_texUV;

out vec2 f_texUV;

void main()
{
	f_texUV = v_texUV;
	gl_Position = vec4(v_position.x, v_position.y, 0.0, 1.0);
}
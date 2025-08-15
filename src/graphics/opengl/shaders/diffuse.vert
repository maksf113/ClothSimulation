#version 450 core

layout (location = 0) in vec3 v_in_position;
layout (location = 1) in vec3 v_in_normal;
layout (location = 2) in vec2 v_in_uv;

layout (std140, binding = 0) uniform u_matrices
{
	mat4 view;
	mat4 projection;
};

uniform mat4 u_model;
uniform mat3 u_normal;

out vec3 f_in_position;
out vec3 f_in_normal;
out vec2 f_in_uv;

void main()
{
	f_in_position = vec3(u_model * vec4(v_in_position, 1.0));
	f_in_normal = u_normal * v_in_normal;
	f_in_uv = v_in_uv;
	gl_Position = projection * view * vec4(f_in_position, 1.0);
}


#version 450 core
layout (location = 0) in vec3 v_in_position;

uniform mat4 u_shadowMatrix;
uniform mat4 u_model;

void main()
{
	gl_Position = u_shadowMatrix * u_model * vec4(v_in_position, 1.0);
}
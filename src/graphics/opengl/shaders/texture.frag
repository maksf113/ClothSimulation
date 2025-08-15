#version 450 core
layout (location = 0) out vec4 out_color;

in vec2 f_texUV;

uniform sampler2D u_tex;

void main()
{
	out_color = texture(u_tex, f_texUV);
	//out_color = vec4(1.0, 0.0, 0.0, 1.0);
}
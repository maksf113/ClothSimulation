#version 450 core
layout (location = 0) in vec3 f_in_position;
layout (location = 1) in vec3 f_in_normal;
layout (location = 2) in vec2 f_in_uv;

layout (std140, binding = 0) uniform u_matrices
{
	mat4 view;
	mat4 projection;
};

uniform sampler2D texSampler;
uniform sampler2D shadowSampler;

uniform mat4 u_shadowMatrix;

uniform bool u_shadow;

uniform vec3 lightDir = normalize(vec3(0.0, -1.0, 0.5));

uniform float ambient = 0.7;

out vec4 f_out;

float shadow()
{
	if(u_shadow == false)
		return 0.0f;
	vec4 shadowClipPosition = u_shadowMatrix * vec4(f_in_position, 1.0);
	//adjust for perspective
	vec3 shadowNdcPosition = shadowClipPosition.xyz / shadowClipPosition.w;
	// transform [-1, 1] to [0, 1]
	vec3 shadowTexCoord = shadowNdcPosition * 0.5 + 0.5;

	float shadowDepth = texture(shadowSampler, shadowTexCoord.xy).r;
	float actualDepth = shadowTexCoord.z;
	//return shadowTexCoord.z;
	float bias = 0.005;
	if(actualDepth > shadowDepth + bias)
		return 1.0;
	return 0.0;
}

void main()
{
	vec4 texColor = texture(texSampler, f_in_uv);
	vec3 normal = normalize(f_in_normal);
	f_out = texColor * (ambient + (1.0 - shadow()) * max(dot(normal, -lightDir), dot(normal, lightDir)));
	f_out *= 0.8;
}
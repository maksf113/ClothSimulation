#pragma once
#include <memory>

#include <gl/glew.h>

#include "graphics/opengl/glError.hpp"
#include "graphics/opengl/Texture.hpp"
#include "graphics/opengl/FBO.hpp"
#include "graphics/opengl/UBO.hpp"
#include "graphics/opengl/Shader.hpp"

#include "models/Floor.hpp"
#include "models/Cloth.hpp"

#include "tools/InputRecievers.hpp"
#include "tools/InputManager.hpp"
#include "tools/GUI.hpp"

#include "Camera.hpp"

class Renderer
{
public:
	Renderer(uint32_t width, uint32_t height);
	~Renderer() = default;
	void draw(uint32_t windowWidth, uint32_t windowHeight, double dt);
	void processInput(const InputManager& im, double dt, bool cursorEnabled);
	ClothParams& getClothParams();
	bool& simulate();
private:
	Camera m_camera;
	Floor m_floor;
	Cloth m_cloth;
	UBO m_ubo;
	std::unique_ptr<Shader> m_diffuseShader;
	std::unique_ptr<Shader> m_shadowShader;
	std::unique_ptr<FBO> m_fbo;
	std::unique_ptr<FBO> m_shadowmap;
	std::unique_ptr<Texture> m_plankTexture;
	std::unique_ptr<Texture> m_clothTexture;

	std::unique_ptr<CudaProcessor> m_cudaProcessor;
	bool m_simulate = false;
};

Renderer::Renderer(uint32_t width, uint32_t height)
{
	GL(glClearColor(0.2, 0.2, 0.2, 0.0));
	GL(glEnable(GL_DEPTH_TEST));
	m_diffuseShader = std::make_unique<Shader>("opengl/shaders/diffuse.vert", "opengl/shaders/diffuse.frag");
	m_shadowShader = std::make_unique<Shader>("opengl/shaders/shadow.vert", "opengl/shaders/shadow.frag");
	m_fbo = std::make_unique<FBO>(width, height);
	m_fbo->attachColorTexture();
	m_fbo->attachDpethRenderbuffer();
	m_fbo->bind();
	GL(glClearColor(0.2, 0.2, 0.2, 0.0));
	GL(glEnable(GL_DEPTH_TEST));
	m_plankTexture = std::make_unique<Texture>("textures/planks.png");
	m_clothTexture = std::make_unique<Texture>("textures/tex2.jpg");
	m_shadowmap = std::make_unique<FBO>(1024, 1024);
	m_shadowmap->attachDepthTexture();

	m_cudaProcessor = std::make_unique<CudaProcessor>(m_cloth.positionVBO(), m_cloth.normalVBO(), m_cloth.particlesPerSide());
	m_cloth.setParams(m_cudaProcessor->params());
}

void Renderer::draw(uint32_t windowWidth, uint32_t windowHeight, double dt)
{
	if(m_simulate)
		m_cudaProcessor->process(static_cast<float>(dt) / 1000.0f);
	// model matrices
	glm::mat4 modelFloor(1.0f);
	glm::mat3 normalFloor = glm::transpose(glm::inverse(glm::mat3(modelFloor)));

	glm::vec3 clothPos = glm::vec3(-2.0f, 3.0f, 3.0f);
	glm::mat4 modelCloth = glm::translate(glm::mat4(1.0f), clothPos);
	glm::mat3 normalCloth = glm::transpose(glm::inverse(glm::mat3(modelCloth)));
	// view - projections
	glm::vec3 shadowPos = glm::vec3(0.0f, 10.0f, 0.0f) + clothPos;
	glm::mat4 shadowProjection = glm::ortho(-5.0f, 5.0f, -5.0f, 5.0f, 0.1f, 20.0f);
	glm::mat4 shadowView = glm::lookAt(shadowPos,
		clothPos, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 shadowMatrix = shadowProjection * shadowView;
	m_ubo.data(m_camera.view(), m_camera.projection(m_fbo->aspect()));
	m_ubo.bind();
	// shadow pass
	m_shadowmap->bind();
	m_shadowmap->viewport();
	GL(glEnable(GL_DEPTH_TEST));
	GL(glClear(GL_DEPTH_BUFFER_BIT));
	m_shadowShader->bind();
	m_shadowShader->setUniform("u_model", modelCloth);
	m_shadowShader->setUniform("u_shadowMatrix", shadowMatrix);
	m_cloth.bind();
	GL(glDrawElements(GL_TRIANGLES, m_cloth.elementCount(), GL_UNSIGNED_INT, 0));
	// main framebuffer
	m_fbo->bind();
	m_fbo->viewport();
	GL(glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT));
	
	m_diffuseShader->bind();
	m_diffuseShader->bindUniformBlock("u_matrices", 0);
	m_diffuseShader->setUniform("shadowSampler", 1);
	m_shadowmap->bindTexture(1);
	// cloth
	m_diffuseShader->setUniform("u_model", modelCloth);
	m_diffuseShader->setUniform("u_normal", normalCloth);
	m_diffuseShader->setUniform("u_shadow", false);
	m_clothTexture->bind(0);
	m_cloth.bind();
	GL(glDrawElements(GL_TRIANGLES, m_cloth.elementCount(), GL_UNSIGNED_INT, 0));
	// floor
	m_diffuseShader->setUniform("u_model", modelFloor);
	m_diffuseShader->setUniform("u_normal", normalFloor);
	m_diffuseShader->setUniform("u_shadow", true);
	m_diffuseShader->setUniform("u_shadowMatrix", shadowMatrix);
	m_plankTexture->bind(0);
	m_floor.bind();
	GL(glDrawElements(GL_TRIANGLES, m_floor.elementCount(), GL_UNSIGNED_INT, 0));
	
	// daw to window
	m_fbo->blit(windowWidth, windowHeight);
}

inline void Renderer::processInput(const InputManager& im, double dt, bool cursorEnabled)
{
	m_camera.processInput(im, dt, cursorEnabled);
}

inline ClothParams& Renderer::getClothParams()
{
	return m_cudaProcessor->params();
}

inline bool& Renderer::simulate()
{
	return m_simulate;
}

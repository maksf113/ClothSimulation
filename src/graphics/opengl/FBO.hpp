#pragma once
#include <memory>
#include <GL/glew.h>

#include "GLError.hpp"
#include "Texture.hpp"
#include "RBO.hpp"

#include "window/Window.hpp"

class FBO
{
public:
	FBO(uint32_t width, uint32_t height);
	~FBO();
	void bind() const;
	void unbind() const;
	void viewport();
	void attachColorTexture();
	void attachDepthTexture();
	void attachDpethRenderbuffer();
	void blit(uint32_t windowWidth, uint32_t windowHeight) const;
	uint32_t width() const;
	uint32_t height() const;
	float aspect() const;
	void bindTexture(int i) const;
private:
	uint32_t m_id;
	uint32_t m_width;
	uint32_t m_height;
	std::unique_ptr<RBO> m_rbo = nullptr;
	std::unique_ptr<Texture> m_texture = nullptr;
};

FBO::FBO(uint32_t width, uint32_t height) : m_width(width), m_height(height)
{
	GL(glGenFramebuffers(1, &m_id));
}

inline FBO::~FBO()
{
	GL(glDeleteFramebuffers(1, &m_id));
}

inline void FBO::bind() const
{
	GL(glBindFramebuffer(GL_FRAMEBUFFER, m_id));
}

inline void FBO::unbind() const
{
	GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

inline void FBO::viewport()
{
	glViewport(0, 0, m_width, m_height);
}

inline void FBO::attachColorTexture()
{
	m_texture = std::make_unique<Texture>(m_width, m_height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
	m_texture->data(nullptr);
	m_texture->wrapping(GL_CLAMP_TO_EDGE);
	bind();
	GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture->id(), 0));
}

inline void FBO::attachDepthTexture()
{
	m_texture = std::make_unique<Texture>(m_width, m_height, GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT, GL_FLOAT);
	m_texture->data(nullptr);
	m_texture->wrapping(GL_CLAMP_TO_EDGE);
	bind();
	GL(glDrawBuffer(GL_NONE));
	GL(glReadBuffer(GL_NONE));
	GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texture->id(), 0));
	unbind();
}

inline void FBO::attachDpethRenderbuffer()
{
	m_rbo = std::make_unique<RBO>(m_width, m_height);

	bind();
	GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_rbo->id()));
	unbind();
}

void FBO::blit(uint32_t windowWidth, uint32_t windowHeight) const
{
	GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, m_id));
	GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
	glBlitFramebuffer(0, 0, m_width, m_height, 0, 0, windowWidth, windowHeight,
		GL_COLOR_BUFFER_BIT, GL_LINEAR);
	GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, 0));
}

inline uint32_t FBO::width() const
{
	return m_width;
}

inline uint32_t FBO::height() const
{
	return m_height;
}

inline float FBO::aspect() const
{
	return static_cast<float>(m_width) / static_cast<float>(m_height);
}

inline void FBO::bindTexture(int i) const
{
	m_texture->bind(i);
}
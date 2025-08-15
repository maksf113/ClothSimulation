#pragma once
#include  <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glError.hpp"

class RBO
{
public:
	RBO();
	RBO(uint32_t width, uint32_t height);
	~RBO();
	void storage(uint32_t width, uint32_t height);
	void bind() const;
	void unbind() const;
	uint32_t id() const;
private:
	uint32_t m_id;
};

RBO::RBO()
{
	GL(glGenRenderbuffers(1, &m_id));
}

RBO::RBO(uint32_t width, uint32_t height)
{
	GL(glGenRenderbuffers(1, &m_id));
	bind();
	GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height));
	unbind();
}

RBO::~RBO()
{
	GL(glDeleteRenderbuffers(1, &m_id));
}

void RBO::storage(uint32_t width, uint32_t height)
{
	bind();
	GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height));
}

void RBO::bind() const
{
	GL(glBindRenderbuffer(GL_RENDERBUFFER, m_id));
}

void RBO::unbind() const
{
	GL(glBindRenderbuffer(GL_RENDERBUFFER, 0));
}

uint32_t RBO::id() const
{
	return m_id;
}

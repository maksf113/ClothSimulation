#pragma once
#include <glm/glm.hpp>
#include<glm/gtc/type_ptr.hpp>
#include <gl/glew.h>

class UBO
{
public:
	UBO();
	~UBO();
	void data(const glm::mat4& view, const glm::mat4& projection);
	void bind() const;
	void unbind() const;
private:
	uint32_t m_id;
};

UBO::UBO()
{
	GL(glGenBuffers(1, &m_id));
	GL(glBindBuffer(GL_UNIFORM_BUFFER, m_id));
	GL(glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), nullptr, GL_STATIC_DRAW));
	GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}

UBO::~UBO()
{
	GL(glDeleteBuffers(1, &m_id));
}

void UBO::data(const glm::mat4& view, const glm::mat4& projection)
{
	GL(glBindBuffer(GL_UNIFORM_BUFFER, m_id));
	GL(glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(view)));
	GL(glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(projection)));
	GL(glBindBufferRange(GL_UNIFORM_BUFFER, 0, m_id, 0, 2 * sizeof(glm::mat4)));
	GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}

inline void UBO::bind() const
{
	GL(glBindBuffer(GL_UNIFORM_BUFFER, m_id));
}

inline void UBO::unbind() const
{
	GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}

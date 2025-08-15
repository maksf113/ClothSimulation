#pragma once
#include <GL/glew.h>
#include <vector>
#include "GLError.hpp"

class VBO
{
private:
	uint32_t m_id;
	uint32_t m_target;
	uint32_t m_mode;
	size_t m_count = 0;
	size_t m_size = 0;
public:
	VBO(uint32_t target = GL_ARRAY_BUFFER, uint32_t mode = GL_STATIC_DRAW);
	template<typename T>
	VBO(const std::vector<T>& vertices, uint32_t target = GL_ARRAY_BUFFER, uint32_t mode = GL_STATIC_DRAW);
	~VBO();
	void bind() const;
	void unbind() const;
	template<typename T>
	void data(const std::vector<T>& vertices);
	uint32_t id() const;
	size_t size() const;
	size_t count() const;
};

VBO::VBO(uint32_t target, uint32_t mode) : m_target(target), m_mode(mode)
{
	GL(glGenBuffers(1, &m_id));
}

template<typename T>
inline VBO::VBO(const std::vector<T>& vertices, uint32_t target, uint32_t mode) : m_target(target), m_mode(mode)
{
	GL(glGenBuffers(1, &m_id));
	VBO::data(vertices);
}
inline VBO::~VBO()
{
	GL(glDeleteBuffers(1, &m_id));
}

inline void VBO::bind() const
{
	GL(glBindBuffer(m_target, m_id));
}
inline void VBO::unbind() const
{
	GL(glBindBuffer(m_target, 0));
}

inline uint32_t VBO::id() const
{
	return m_id;
}

inline size_t VBO::size() const
{
	return m_size;
}

inline size_t VBO::count() const
{
	return m_count;
}

template<typename T>
inline void VBO::data(const std::vector<T>& vertices)
{
	m_count = vertices.size();
	m_size = m_count * sizeof(T);
	bind();
	GL(glBufferData(m_target, GLsizeiptr(m_size), vertices.data(), m_mode));
	unbind();
}
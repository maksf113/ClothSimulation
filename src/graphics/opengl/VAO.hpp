#pragma once
#include "VBO.hpp"
#include "EBO.hpp"
#include "VBL.hpp"

class VAO
{
private:
	uint32_t m_id;
	uint32_t m_attributeCount = 0;
public:
	VAO();
	VAO(const VBO& vb, const VBL& layout, const EBO& eb);
	~VAO();
	void bind() const;
	void unbind() const;
	void addVertexBuffer(const VBO& vb, const VBL& layout);
	void addIndexBuffer(const EBO& eb);
};

VAO::VAO()
{
	GL(glGenVertexArrays(1, &m_id));
}
inline VAO::VAO(const VBO& vb, const VBL& layout, const EBO& eb)
{
	GL(glGenVertexArrays(1, &m_id));
	addVertexBuffer(vb, layout);
	addIndexBuffer(eb);
}
VAO::~VAO()
{
	GL(glDeleteVertexArrays(1, &m_id));
}

inline void VAO::bind() const
{
	GL(glBindVertexArray(m_id));
}

inline void VAO::unbind() const
{
	GL(glBindVertexArray(0));
}

inline void VAO::addVertexBuffer(const VBO& vb, const VBL& layout)
{
	bind();
	vb.bind();
	const auto& elements = layout.getElements();
	for (int i = 0; i < elements.size(); i++)
	{
		const auto& element = elements[i];
		GL(glEnableVertexAttribArray(i + m_attributeCount));
		GL(glVertexAttribPointer(i + m_attributeCount, element.count, element.type,
			element.normalized ? GL_TRUE : GL_FALSE, layout.getStride(), 
			(const void*)element.offset));
	}
	m_attributeCount += elements.size();
	unbind();
}

inline void VAO::addIndexBuffer(const EBO& ebo)
{
	bind();
	ebo.bind();
	unbind();
}
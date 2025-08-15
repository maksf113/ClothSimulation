#pragma once
#include <GL/glew.h>
#include <vector>
#include "GLError.hpp"

struct LayoutElement
{
	unsigned int type;
	unsigned int count;
	bool normalized;
	unsigned int offset;
	LayoutElement(unsigned int _type, unsigned int _count, bool _normalized, unsigned int _offset) :
		type(_type), count(_count), normalized(_normalized), offset(_offset) {}
};

class VBL
{
private:
	std::vector<LayoutElement> m_elements;
	unsigned int m_stride;
public:
	VBL() : m_stride(0) {};
	~VBL() {};
	template<typename T>
	void push(unsigned int count, bool normalized = false)
	{
		std::runtime_error(false);
	}
	const std::vector<LayoutElement>& getElements() const;
	const unsigned int getStride() const;
};

template<>
void VBL::push<float>(unsigned int count, bool normalized)
{
	m_elements.push_back({ GL_FLOAT, count, normalized, m_stride });
	m_stride += count * sizeof(float);
}

template<>
void VBL::push<unsigned int>(unsigned int count, bool normalized)
{
	m_elements.push_back(LayoutElement(GL_UNSIGNED_INT, count, normalized, m_stride));
	m_stride += count * sizeof(unsigned int);
}

template<>
void VBL::push<unsigned char>(unsigned int count, bool normalized)
{
	m_elements.push_back(LayoutElement(GL_UNSIGNED_BYTE, count, normalized, m_stride));
	m_stride += count * sizeof(unsigned char);
}

inline const std::vector<LayoutElement>& VBL::getElements() const
{
	return m_elements;
}

inline const unsigned int VBL::getStride() const
{
	return m_stride;
}
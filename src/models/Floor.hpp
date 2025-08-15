#pragma once
#include <vector>
#include <glm/glm.hpp>

#include "graphics/opengl/VBO.hpp"
#include "graphics/opengl/VBL.hpp"
#include "graphics/opengl/EBO.hpp"
#include "graphics/opengl/VAO.hpp"

class Floor
{
public: 
	Floor();
	~Floor() = default;
	void bind() const;
	uint32_t elementCount() const;
private:
	struct Vertex;
	std::vector<Vertex> m_vertices;
	std::vector<uint32_t> m_indices;
	std::unique_ptr<VBO> m_vbo;
	std::unique_ptr<VBL> m_vbl;
	std::unique_ptr<EBO> m_ebo;
	std::unique_ptr<VAO> m_vao;
	double m_scale = 10.0;
};

struct Floor::Vertex 
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
};

Floor::Floor()
{
	m_vertices =
	{ //               COORDINATES           /           NORMALS         /       TEXTURE COORDINATES    //
		Vertex{glm::vec3(-m_scale, 0.0f,   m_scale),  glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(-m_scale,  m_scale)},
		Vertex{glm::vec3(-m_scale, 0.0f,  -m_scale),  glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(-m_scale, -m_scale)},
		Vertex{glm::vec3( m_scale, 0.0f,  -m_scale),  glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2( m_scale, -m_scale)},
		Vertex{glm::vec3( m_scale, 0.0f,   m_scale),  glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2( m_scale,  m_scale)}
	};
	m_vbo = std::make_unique<VBO>(m_vertices);
	m_vbl = std::make_unique<VBL>();
	// push vertex attributes
	m_vbl->push<float>(3); // pos
	m_vbl->push<float>(3); // normal
	m_vbl->push<float>(2); // tex coord
	// Indices for vertices order
	m_indices =
	{
		0, 1, 2,
		0, 2, 3
	};
	m_ebo = std::make_unique<EBO>(m_indices);
	// VAO
	m_vao = std::make_unique<VAO>(*m_vbo, *m_vbl, *m_ebo);
}

inline void Floor::bind() const
{
	m_vao->bind();
}

inline uint32_t Floor::elementCount() const
{
	return m_indices.size();
}

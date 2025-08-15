#pragma once
#include <vector>
#include <glm/glm.hpp>

#include "graphics/opengl/VBO.hpp"
#include "graphics/opengl/VBL.hpp"
#include "graphics/opengl/EBO.hpp"
#include "graphics/opengl/VAO.hpp"

#include "CUDA/CudaProcessor.hpp"

#include "ClothParams.hpp"

class Cloth
{
public:
	Cloth();
	~Cloth() = default;
	void bind() const;
	uint32_t elementCount() const;
	int particlesPerSide() const;
	const std::shared_ptr<VBO>& positionVBO() const;
	const std::shared_ptr<VBO>& normalVBO() const;
	void setParams(ClothParams& params);
private:
	int N = 64; // particles per side

	std::vector<glm::vec3> m_position;
	std::vector<glm::vec3> m_normal;
	std::vector<glm::vec2> m_uv;

	std::vector<uint32_t> m_indices;
	
	std::shared_ptr<VBO> m_positionVBO;
	std::shared_ptr<VBO> m_normalVBO;
	std::unique_ptr<VBO> m_uvVBO;

	std::unique_ptr<EBO> m_ebo;

	std::unique_ptr<VAO> m_vao;

	float m_scale = 2.0f;
};

Cloth::Cloth()
{
	m_position.resize(N * N);
	m_normal.resize(N * N);
	m_uv.resize(N * N);
	for (size_t idx = 0; idx < N * N; idx++)
	{
		size_t i = idx % N;
		size_t j = idx / N;
		assert(j * N + i == idx);
		float x = static_cast<float>(i) / static_cast<float>(N - 1);
		float z = static_cast<float>(j) / static_cast<float>(N - 1);
		m_position[idx] = glm::vec3(x * m_scale, 0.0f, z * m_scale);
		m_normal[idx] = glm::vec3(0.0f, 1.0f, 0.0f);
		m_uv[idx] = glm::vec2(x, z);
	}
	m_positionVBO = std::make_shared<VBO>(m_position, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	VBL positionLayout;
	positionLayout.push<float>(3);
	m_normalVBO = std::make_shared<VBO>(m_normal, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	VBL normalLayout;
	normalLayout.push<float>(3);
	m_uvVBO = std::make_unique<VBO>(m_uv);
	VBL uvLayout;
	uvLayout.push<float>(2);
	// Indices for vertices
	m_indices.resize((N - 1) * (N - 1) * 2 * 3); // (N-1)^2 = #square tiles, * 2 for triangles, * 3 for vertex indices
	// helper function to flatten the vertex index
	auto flatten = [&](size_t i, size_t j) {return j * N + i; };
	for(size_t j = 0; j < N - 1; j++)
		for (size_t i = 0; i < N - 1; i++)
		{
			// first index for the current square tile (corresponding to left, bottom vertex) in m_indices
			size_t idx = (j * (N - 1) + i) * 6;
			// alternating diagonal of a square tile
			if ((i + j) % 2 == 0)
			{
				m_indices[idx++] = flatten(i, j);
				m_indices[idx++] = flatten(i + 1, j + 1);
				m_indices[idx++] = flatten(i, j + 1);

				m_indices[idx++] = flatten(i, j);
				m_indices[idx++] = flatten(i + 1, j);
				m_indices[idx++] = flatten(i + 1, j + 1);
			}
			else
			{
				m_indices[idx++] = flatten(i, j);
				m_indices[idx++] = flatten(i + 1, j);
				m_indices[idx++] = flatten(i, j + 1);

				m_indices[idx++] = flatten(i + 1, j);
				m_indices[idx++] = flatten(i + 1, j + 1);
				m_indices[idx++] = flatten(i, j + 1);
			}
		}
	m_ebo = std::make_unique<EBO>(m_indices);
	// VAO
	m_vao = std::make_unique<VAO>();
	m_vao->addVertexBuffer(*m_positionVBO, positionLayout);
	m_vao->addVertexBuffer(*m_normalVBO, normalLayout);
	m_vao->addVertexBuffer(*m_uvVBO, uvLayout);
	m_vao->addIndexBuffer(*m_ebo);
}

inline void Cloth::bind() const
{
	m_vao->bind();
}

inline uint32_t Cloth::elementCount() const
{
	return m_indices.size();
}

inline int Cloth::particlesPerSide() const
{
	return N;
}


inline const std::shared_ptr<VBO>& Cloth::positionVBO() const
{
	return m_positionVBO;
}

inline const std::shared_ptr<VBO>& Cloth::normalVBO() const
{
	return m_normalVBO;
}

inline void Cloth::setParams(ClothParams& params)
{
	params.mass = 0.01f;

	params.kStruct = 80.0f;
	params.kShear = 20.0f;
	params.kBend = 20.0f;

	params.kdStruct = 1.75f;
	params.kdShear = 1.00f;
	params.kdBend = 1.00f;

	params.structLength = glm::length(m_position[0] - m_position[1]);
	params.shearLength = glm::length(m_position[0] - m_position[N + 1]);
	params.bendLength = glm::length(m_position[0] - m_position[2]);
	params.damping = 0.001f;
	params.gravity = { 0.0f, -9.81f, 0.0f };
	params.wind = { 0.0f, 0.0f, 0.0f };
}

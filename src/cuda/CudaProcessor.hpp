#pragma once
#include <memory>
#include <cuda_gl_interop.h>

#include "kernel.hpp"

#include "graphics/opengl/VBO.hpp"

#include "ClothParams.hpp"

class CudaProcessor
{
public:
	CudaProcessor(const std::shared_ptr<VBO>& positionVBO, const std::shared_ptr<VBO>& normalVBO, int particlesPerSide);
	~CudaProcessor();
	void process(float dt);
	ClothParams& params();
private:
	void getDeviceInfo();
	void mapBuffers();
	void unmapBuffers();
private:
	const std::shared_ptr<VBO> m_positionVBO;
	const std::shared_ptr<VBO> m_normalVBO;
	cudaGraphicsResource* m_positionBuffer;
	cudaGraphicsResource* m_normalBuffer;
	float3* m_cudaMappedPosition = nullptr;
	float3* m_cudaMappedNormal = nullptr;
	float3* m_cudaPrevPosition = nullptr;

	int m_width;
	int m_height;
	dim3 m_blockSize;

	int m_iterations = 25;

	ClothParams m_params;
};

inline CudaProcessor::CudaProcessor(const std::shared_ptr<VBO>& positionVBO, const std::shared_ptr<VBO>& normalVBO, int particlesPerSide) : m_positionVBO(positionVBO), m_normalVBO(normalVBO)
{
	m_width = particlesPerSide;
	m_height = positionVBO->count() / particlesPerSide;

	getDeviceInfo();

	// register GL buffers with CUDA
	CUDA(cudaGraphicsGLRegisterBuffer(&m_positionBuffer, positionVBO->id(), cudaGraphicsRegisterFlagsNone));
	CUDA(cudaGraphicsGLRegisterBuffer(&m_normalBuffer, normalVBO->id(), cudaGraphicsRegisterFlagsNone));
	CUDA(cudaMalloc(&m_cudaPrevPosition, positionVBO->size()));
	// copy initial position to prevPosition
	mapBuffers();
	copy(m_cudaPrevPosition, m_cudaMappedPosition, m_width, m_height, m_blockSize);
	CUDA(cudaDeviceSynchronize());
	unmapBuffers();
}

inline CudaProcessor::~CudaProcessor()
{
	if (m_cudaMappedPosition != nullptr)
		CUDA(cudaGraphicsUnmapResources(1, &m_positionBuffer, 0));
	if (m_cudaMappedNormal!= nullptr)
		CUDA(cudaGraphicsUnmapResources(1, &m_normalBuffer, 0));
	CUDA(cudaGraphicsUnregisterResource(m_positionBuffer));
	CUDA(cudaGraphicsUnregisterResource(m_normalBuffer));
}

inline void CudaProcessor::process(float dt)
{
	mapBuffers();
	for (int i = 0; i < m_iterations; i++)
	{
		verlet(m_cudaMappedPosition, m_cudaPrevPosition, m_cudaMappedNormal, m_width, m_height, m_blockSize, m_params, dt / static_cast<float>(m_iterations));
		CUDA(cudaDeviceSynchronize());
	}
	unmapBuffers();
}

inline ClothParams& CudaProcessor::params()
{
	return m_params;
}

inline void CudaProcessor::getDeviceInfo()
{
	cudaDeviceProp deviceProp;
	CUDA(cudaGetDeviceProperties(&deviceProp, 0));
	int dim = 32;
	while (dim * dim > deviceProp.maxThreadsPerBlock)
		dim /= 2;
	while (dim > deviceProp.maxThreadsDim[0])
		dim /= 2;
	while (dim > deviceProp.maxThreadsDim[1])
		dim /= 2;
	m_blockSize = dim3(dim, dim);
}

inline void CudaProcessor::mapBuffers()
{
	cudaGraphicsResource* resources[] = { m_positionBuffer, m_normalBuffer };
	CUDA(cudaGraphicsMapResources(2, resources, 0));
	size_t  mappedSize;
	CUDA(cudaGraphicsResourceGetMappedPointer((void**) &m_cudaMappedPosition, &mappedSize, m_positionBuffer));
	assert(mappedSize == m_positionVBO->size());
	CUDA(cudaGraphicsResourceGetMappedPointer((void**) &m_cudaMappedNormal, &mappedSize, m_normalBuffer));
	assert(mappedSize == m_normalVBO->size());
}

inline void CudaProcessor::unmapBuffers()
{
	cudaGraphicsResource* resources[] = { m_positionBuffer, m_normalBuffer };
	CUDA(cudaGraphicsUnmapResources(2, resources, 0));
	m_cudaMappedPosition = nullptr;
	m_cudaMappedNormal = nullptr;
}

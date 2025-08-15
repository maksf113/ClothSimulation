#pragma once
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ClothParams.hpp"

#define CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void verlet(float3* pos, float3* prevPos, float3* normal, int width, int height, dim3 blockSize, ClothParams params, float dt);
void copy(float3* destination, float3* source, int width, int height, dim3 blockSize);

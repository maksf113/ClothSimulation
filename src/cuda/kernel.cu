#include <stdio.h>
#include <math.h>

#include "kernel.hpp"
#include "float3Arithmetic.hpp"



__device__
int flatten(int col, int row, int width)
{
    return col + row * width;
}

__device__
bool isValid(int col, int row, int width, int height)
{
    if (col < 0 || col >= width || row < 0 || row >= height)
        return false;
    return true;
}

__device__
int clip(int col, int width)
{
    if (col < 0)
        return 0;
    if (col >= width)
        return width - 1;
    return col;
}

__global__
void verletKernel(float3* pos, float3* prevPos, float3* normal, int width, int height, ClothParams params, float dt)
{
    // shared memory setup
    const int halo = 2; // for structural and bending
    const int sWidth = blockDim.x + 2 * halo;
    const int sHeight = blockDim.y + 2 * halo;

    extern __shared__ float3 s_data[];
    // pointers to block for each variable
    float3* s_pos = &s_data[0];
    float3* s_prevPos = &s_data[sWidth * sHeight];
    float3* s_norm = &s_data[sWidth * sHeight * 2];

    // grid - stride loading loop
    const int threadsInBlock = blockDim.x * blockDim.y;
    const int totalSharedElements = sWidth * sHeight;
    const int localIdx = threadIdx.x + blockDim.x * threadIdx.y;
    for (int sharedIdx = localIdx; sharedIdx < totalSharedElements; sharedIdx += threadsInBlock)
    {
        int sCol = sharedIdx % sWidth;
        int sRow = sharedIdx / sWidth;
        int globalCol = blockIdx.x * blockDim.x + sCol - halo;
        int globalRow = blockIdx.y * blockDim.y + sRow - halo;
        if (globalCol >= 0 && globalCol < width && globalRow >= 0 && globalRow < height)
        {
            int globalIdx = flatten(globalCol, globalRow, width);
            s_pos[sharedIdx] = pos[globalIdx];
            s_prevPos[sharedIdx] = prevPos[globalIdx];
            s_norm[sharedIdx] = normal[globalIdx];
        }
    }
    __syncthreads();
    if ((threadIdx.x + blockIdx.x * blockDim.x == 0 || threadIdx.x + blockIdx.x * blockDim.x == width - 1) && threadIdx.y + blockIdx.y * blockDim.y == height - 1)
        return;
    const int sCol = threadIdx.x + halo;
    const int sRow = threadIdx.y + halo;
    const int sharedIdx = flatten(sCol, sRow, sWidth);
    const int globalCol = threadIdx.x + blockIdx.x * blockDim.x;
    const int globalRow = threadIdx.y + blockIdx.y * blockDim.y;
    const int globalIdx = flatten(globalCol, globalRow, width);
    if (globalCol >= width || globalRow >= height)
        return;
    float3 force = { 0.0f, 0.0f, 0.0f };
    // gravity
    force += params.gravity * params.mass;
    // springs in subsequent code prefix "d" stands for "delta", prefix "n" for "neighbour"
    float k;
    float kd;
    float restLength;
    // structural
    int dColStruct[] = { 1, -1, 0, 0 };
    int dRowStruct[] = { 0, 0, 1, -1 };
    k = params.kStruct;
    kd = params.kdStruct;
    restLength = params.structLength;
    for (int i = 0; i < 4; i++)
    {
        int nCol = globalCol + dColStruct[i];
        int nRow = globalRow + dRowStruct[i];
        if (!isValid(nCol, nRow, width, height))
            continue;
        int sharedNeighIdx = flatten(sCol + dColStruct[i], sRow + dRowStruct[i], sWidth);
        // vector from current to neighbour
        float3 posDiff = s_pos[sharedNeighIdx] - s_pos[sharedIdx];
        float springLength = length(posDiff);
        float3 springDir = posDiff / springLength;
        force += k * springDir * (springLength - restLength) / restLength;
        // damping
        float3 relativeVelocity = (posDiff - (s_prevPos[sharedNeighIdx] - s_prevPos[sharedIdx])) / dt;
        force += kd * springDir * dot(relativeVelocity, springDir);

    }
    // shear
    int dColShear[] = { 1, 1, -1, -1 };
    int dRowShear[] = { 1, -1, 1, -1 };
    k = params.kShear;
    kd = params.kdShear;
    restLength = params.shearLength;
    for (int i = 0; i < 4; i++)
    {
        int nCol = globalCol + dColShear[i];
        int nRow = globalRow + dRowShear[i];
        if (!isValid(nCol, nRow, width, height))
            continue;
        int sharedNeighIdx = flatten(sCol + dColShear[i], sRow + dRowShear[i], sWidth);
        // vector from current to neighbour
        float3 posDiff = s_pos[sharedNeighIdx] - s_pos[sharedIdx];
        float springLength = length(posDiff);
        float3 springDir = posDiff / springLength;
        force += k * springDir * (springLength - restLength) / restLength;
        // damping
        float3 relativeVelocity = (posDiff - (s_prevPos[sharedNeighIdx] - s_prevPos[sharedIdx])) / dt;
        force += kd * springDir * dot(relativeVelocity, springDir);
    }
    // bending
    int dColBend[] = { 2, -2, 0, 0 };
    int dRowBend[] = { 0, 0, 2, -2 };
    k = params.kBend;
    kd = params.kdBend;
    restLength = params.bendLength;
    for (int i = 0; i < 4; i++)
    {
        int nCol = globalCol + dColBend[i];
        int nRow = globalRow + dRowBend[i];
        if (!isValid(nCol, nRow, width, height))
            continue;
        int sharedNeighIdx = flatten(sCol + dColBend[i], sRow + dRowBend[i], sWidth);
        // vector from current to neighbour
        float3 posDiff = s_pos[sharedNeighIdx] - s_pos[sharedIdx];
        float springLength = length(posDiff);
        float3 springDir = posDiff / springLength;
        force += k * springDir * (springLength - restLength) / restLength;
        // damping
        float3 relativeVelocity = (posDiff - (s_prevPos[sharedNeighIdx] - s_prevPos[sharedIdx])) / dt;
        force += kd * springDir * dot(relativeVelocity, springDir);
    }
    // normal calculation
    // horizontal
    int dColPlus = globalCol + 1 < width ? 1 : 0;
    int dColMinus = globalCol - 1 >= 0 ? 1 : 0;
    int sharedNeighIdxColPlus = flatten(sCol + dColPlus, sRow, sWidth);
    int sharedNeighIdxColMinus = flatten(sCol - dColMinus, sRow, sWidth);
    float3 horizontal = s_pos[sharedNeighIdxColPlus] - s_pos[sharedNeighIdxColMinus];
    // vertical
    int dRowPlus = globalRow + 1 < height ? 1 : 0;
    int dRowMinus = globalRow - 1 >= 0 ? 1 : 0;
    int sharedNeighIdxRowPlus = flatten(sCol, sRow + dRowPlus, sWidth);
    int sharedNeighIdxRowMinus = flatten(sCol, sRow - dRowMinus, sWidth);
    float3 vertical = s_pos[sharedNeighIdxRowPlus] - s_pos[sharedNeighIdxRowMinus];
    float3 areaVector = 0.25f * cross(vertical, horizontal);
    float3 normalVector = areaVector / (length(areaVector) + 1e-9);
    normal[globalIdx] = normalVector;
    // drag
    float3 relativeVel = params.wind - (s_pos[sharedIdx] - s_prevPos[sharedIdx]) / dt;
    float3 relativeVelDir = relativeVel / (length(relativeVel) + 1e-9);
    force += 1.2f * 0.5 * dot(relativeVel, relativeVel) * fabs(dot(areaVector, relativeVelDir)) * relativeVelDir;
    float3 prev = s_prevPos[sharedIdx];
    prevPos[globalIdx] = s_pos[sharedIdx];
    pos[globalIdx] = s_pos[sharedIdx] + (s_pos[sharedIdx] - prev) * (1.0f - params.damping) + force / params.mass * dt * dt;
}

void verlet(float3* pos, float3* prevPos, float3* normal, int width, int height, dim3 blockSize, ClothParams params, float dt)
{
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    const size_t halo = 2;
    size_t sharedMemSize = (blockSize.x + 2 * halo) * (blockSize.y + 2 * halo) * (sizeof(pos[0]) + sizeof(prevPos[0]) + sizeof(normal[0]));
    verletKernel << <gridSize, blockSize, sharedMemSize >> > (pos, prevPos, normal, width, height, params, dt);
}

__global__
void copyKernel(float3* destination, float3* source, int width, int height)
{
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= width || row >= height)
        return;
    const int idx = flatten(col, row, width);
    destination[idx] = source[idx];
}

void copy(float3* destination, float3* source, int width, int height, dim3 blockSize)
{
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    copyKernel << <gridSize, blockSize >> > (destination, source, width, height);
}
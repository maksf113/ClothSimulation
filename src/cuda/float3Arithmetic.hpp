#pragma once
#include <math.h>
#include "cuda_runtime.h"


__device__ __host__
inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__
inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__
inline float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__
inline float3 operator*(float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__
inline float3 operator/(const float3& a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ __host__
inline float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ __host__
inline float3& operator-=(float3& a, const float3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

__device__ __host__
inline float3& operator*=(float3& a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

__device__ __host__
inline float3& operator/=(float3& a, float s)
{
    a.x /= s; a.y /= s; a.z /= s;
    return a;
}

__device__ __host__
inline float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __host__
inline float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __host__
inline float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__
inline float length(const float3& a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
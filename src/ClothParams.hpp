#pragma once



struct ClothParams
{
    float mass;

    float kStruct;
    float kShear;
    float kBend;

    // spring damping
    float kdStruct;
    float kdShear;
    float kdBend;

    float structLength;
    float shearLength;
    float bendLength;

    float damping;

    float3 gravity;

    float3 wind;
};
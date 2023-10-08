#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <cuda_runtime.h>

// 引入std成员函数

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// 常量

__device__ const float infinity = std::numeric_limits<float>::infinity();
__device__ const float pi = 3.1415926535897932385f;

// 常用函数

inline float degrees_to_radians(float degrees)
{
    return degrees * pi / 180.0f;
}

__device__ inline float random_double(curandState *state)
{
    return curand_uniform(state);
}

__device__ inline float random_double(float min, float max, curandState *state)
{
    return min + (max - min) * curand_uniform(state);
}

// 常用头文件

#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif

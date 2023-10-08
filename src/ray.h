#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray
{
public:
    __host__ __device__ ray() {}

    __host__ __device__ ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }

    /**
     * @brief 返回射线上给定参数值处的点。
     * @param t 参数值，表示距离原点的距离
     * @return 射线上距离原点 t 单位的点
     */
    __host__ __device__ point3 at(double t) const
    {
        return orig + t * dir;
    }

private:
    point3 orig; // 射线的起始点
    vec3 dir;    // 射线的方向向量
};

#endif

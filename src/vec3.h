#ifndef VEC3_H
#define VEC3_H
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

using std::fabs;
using std::sqrt;

class vec3
{
public:
    float e[3];
    // 构造函数
    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}
    // 返回坐标
    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    // 重载运算符
    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float &operator[](int i) { return e[i]; }

    __host__ __device__ vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        /*
        vec3 a(1.0, 2.0, 3.0);
        vec3 b(4.0, 5.0, 6.0);
        a += b; // 使用+=运算符并更新a的值

        在这个示例中，a += b 实际上是 a.operator+=(b) 的缩写，其中 *this 表示 a，并且 a 的成员数据被更新为 a 和 b 的对应成员相加的结果。
        */
        return *this;
    }

    __host__ __device__ vec3 &operator*=(float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(float t)
    {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const
    {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    /**
     * @brief 判断向量是否接近零点
     */
    __host__ __device__ bool near_zero() const
    {
        auto s = 1e-8;
        return abs(e[0]) < s && abs(e[1]) < s && abs(e[2]) < s;
    }
};
// 随机采样 vec3
__device__ vec3 random(curandState *rand_state)
{
    return vec3(random_double(rand_state), random_double(rand_state), random_double(rand_state));
}
// 随机采样 vec3(min, max)
__device__ vec3 random(double min, double max, curandState *rand_state)
{
    return vec3(random_double(min, max, rand_state), random_double(min, max, rand_state),
                random_double(min, max, rand_state));
}

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions

//@brief 重载了输出运算符，用于将 vec3 向量的内容输出到输出流
inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
//@brief 重载了加法运算符，用于执行两个 vec3 向量的加法。
__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
//@brief 重载了减法运算符，用于执行两个 vec3 向量的减法。
__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
//@brief 逐个相乘
__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
//@brief 数乘
__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}
//@brief 数乘
__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
    return t * v;
}
//@brief 除法
__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
    return (1 / t) * v;
}

/**
 * @brief 向量点乘
 *
 * @param u
 * @param v
 * @return double
 */
__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

/**
 * @brief 向量叉乘
 *
 * @param u
 * @param v
 * @return vec3
 */
__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

/**
 * @brief 单位向量化
 *
 * @param v
 * @return vec3 单位向量
 */
__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

/**
 * @brief 返回xy坐标系中单位圆内的随机向量
 *
 * @return vec3
 */
__device__ inline vec3 random_in_unit_disk(curandState *rand_state)
{
    while (true)
    {
        auto p = vec3(random_double(-1, 1, rand_state), random_double(-1, 1, rand_state), 0);
        if (p.length_squared() < 1)
        {
            return p;
        }
    }
}

/**
 * @brief 返回单元球内的随机向量
 *
 * @return vec3
 */
__device__ inline vec3 random_in_unit_sphere(curandState *rand_state)
{
    while (true)
    {
        auto p = random(-1, 1, rand_state);
        if (p.length_squared() < 1)
            return p;
    }
}

/**
 * @brief 返回随机单位向量
 *
 * @return vec3
 */
__device__ inline vec3 random_unit_vector(curandState *rand_state)
{
    return unit_vector(random_in_unit_sphere(rand_state));
}

/**
 * @brief 返回位于给定法线方向上的随机半球上的随机向量
 *
 * @param normal 用于确定半球方向的法线
 * @return vec3 位于半球上的随机点,方向同法线
 */
__device__ inline vec3 random_on_hemisphere(const vec3 &normal,
                                            curandState *rand_state)
{
    vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(normal, on_unit_sphere) >= 0.0)
    {
        return on_unit_sphere;
    }
    else
    {
        return -on_unit_sphere;
    }
}

/**
 * @brief 向量反射
 *
 * @param v 入射向量
 * @param n 法线向量
 * @return vec3 反射向量
 */
__device__ inline vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

/**
 * @brief 向量折射
 *
 * @param uv 入射单位向量
 * @param n 法线单位向量
 * @param etai_over_etat 折射率之比（入射介质的折射率 / 折射介质的折射率）
 * @return vec3 折射向量
 */
__device__ inline vec3 refract(const vec3 &v, const vec3 &n, float etai_over_etat)
{
    float cos_theta = min(1.0, dot(-v, n));
    vec3 r_out_perp = etai_over_etat * (v + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif

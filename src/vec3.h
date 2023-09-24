#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

using std::sqrt;
using std::fabs;

class vec3 {
  public:
    double e[3];
    //构造函数
    vec3() : e{0,0,0} {}
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
    //返回坐标
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    //重载运算符
    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }
    
    vec3& operator+=(const vec3 &v) {
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

    vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3& operator/=(double t) {
        return *this *= 1/t;
    }

    double length() const {
        return sqrt(length_squared());
    }

    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    /**
     * @brief 判断向量是否接近零点
     */
    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    static vec3 random() {
        return vec3(random_double(), random_double(), random_double());
    }

    /**
     * @brief 返回坐标在指定区间内的随机向量
     *
     * @param min 最小坐标
     * @param max 最大坐标
     * @return vec3 随机向量
     */
    static vec3 random(double min, double max) {
        return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions

//@brief 重载了输出运算符，用于将 vec3 向量的内容输出到输出流
inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
//@brief 重载了加法运算符，用于执行两个 vec3 向量的加法。
inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
//@brief 重载了减法运算符，用于执行两个 vec3 向量的减法。
inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
//@brief 逐个相乘
inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
//@brief 数乘
inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
//@brief 数乘
inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}
//@brief 除法
inline vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

/**
 * @brief 向量点乘
 *
 * @param u
 * @param v
 * @return double
 */
inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

/**
 * @brief 向量叉乘
 *
 * @param u
 * @param v
 * @return vec3
 */
inline vec3 cross(const vec3 &u, const vec3 &v) {
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
inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

/**
 * @brief 返回xy坐标系中单位圆内的随机向量
 *
 * @return vec3
 */
inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1,1), random_double(-1,1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}

/**
 * @brief 返回单元球内的随机向量
 *
 * @return vec3
 */
inline vec3 random_in_unit_sphere() {
    while (true) {
        auto p = vec3::random(-1,1);
        if (p.length_squared() < 1)
            return p;
    }
}

/**
 * @brief 返回随机单位向量
 *
 * @return vec3
 */
inline vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

/**
 * @brief 返回位于给定法线方向上的随机半球上的随机向量
 *
 * @param normal 用于确定半球方向的法线
 * @return vec3 位于半球上的随机点,方向同法线
 */
inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

/**
 * @brief 向量反射
 *
 * @param v 入射向量
 * @param n 法线向量
 * @return vec3 反射向量
 */
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

/**
 * @brief 向量折射
 *
 * @param uv 入射单位向量
 * @param n 法线单位向量
 * @param etai_over_etat 折射率之比（入射介质的折射率 / 折射介质的折射率）
 * @return vec3 折射向量
 */
inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


#endif

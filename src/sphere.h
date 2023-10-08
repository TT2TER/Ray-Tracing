#ifndef SPHERE_H
#define SPHERE_H

#include "utils.h"
#include "material.h"
#include "hittable.h"
#include "vec3.h"


/**
 * @brief 表示球体的类，继承自 hittable
 *
 * 这个类用于表示一个球体，包括球心、半径和表面材料。
 */
class sphere : public hittable {
 public:
  ~sphere() {
    if (mat != nullptr) {
      delete mat;
    }
  }

  /**
   * @brief 构造函数，初始化球体
   *
   * @param _center 球心坐标
   * @param _radius 球半径
   * @param _material 表面材料指针
   */
  __device__ sphere(point3 _center, double _radius, material* _material)
      : center(_center), radius(_radius), mat(_material) {}

  /**
   * @brief 构造函数，初始化球体并指定颜色
   *
   * @param _center 球心坐标
   * @param _radius 球半径
   * @param c 球体颜色
   */
  __device__ sphere(point3 _center, double _radius, color c)
      : center(_center), radius(_radius) {
    mat = new lambertian(c);
  }

  /**
   * @brief 判断光线是否与球体相交
   *
   * @param r 光线
   * @param ray_t 光线有效范围
   * @param rec 存储相交信息的 hit_record 结构体
   * @return true 如果有合法的相交点
   * @return false 如果没有合法的相交点
   */
  __device__ bool hit(const ray& r, interval ray_t,
                      hit_record& rec) const override {
    // 计算光线与球的相交
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;

    float sqrtd = sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;

    if (!ray_t.surrounds(root)) {
      root = (-half_b + sqrtd) / a;
      if (!ray_t.surrounds(root)) {
        return false;
      }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat = mat;

    return true;
  }

  /**
   * @brief 获取对象自身的大小
   *
   * @return size_t 对象自身的大小（字节数）
   */
  __device__ virtual size_t self_size() const override { return sizeof(*this); }

 private:
  point3 center;   // 球心坐标
  double radius;   // 球半径
  material* mat = nullptr;  // 球的表面材料属性
};



#endif

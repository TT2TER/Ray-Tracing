#ifndef HITTABLE_H
#define HITTABLE_H

#include "utils.h"
#include "ray.h"

class material;


class hit_record
{
public:
    point3 p;
    vec3 normal;
    material *mat = nullptr;
    float t;
    bool front_face;

    __device__ void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        // 设置命中记录的法向量
        // 注意：参数 `outward_normal` 假定为单位长度
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? (outward_normal) : (-outward_normal);
    }
};

class hittable {
 public:
  __device__ hittable() = default;

  __device__ virtual bool hit(const ray& r, interval ray_t,
                              hit_record& rec) const = 0;
  __device__ virtual size_t self_size() const = 0;
};

#endif

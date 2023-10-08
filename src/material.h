#ifndef MATERIAL_H
#define MATERIAL_H

#include "utils.h"
#include "color.h"
#include "hittable_list.h"
#include "hittable.h"

class hit_record;
class material
{
public:
    virtual ~material() = default;

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec,
                                    color &attenuation, ray &scattered,
                                    curandState *rand_state) const = 0;
};

class lambertian : public material
{
public:
    __host__ __device__ lambertian(const color &a) : albedo(a) {}
    __device__ bool scatter(const ray &r_in, const hit_record &rec,
                            color &attenuation, ray &scattered,
                            curandState *rand_state) const override
    {
        // 检测散射方向是否有效 随机单位向量与法向量抵消则为无效
        vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);
        if (scatter_direction.near_zero())
        {
            scatter_direction = rec.normal;
        }
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

class metal : public material
{
public:
    __host__ __device__ metal(const color &a, double f)
        : albedo(a), fuzz(f < 1 ? f : 1) {}
    __device__ bool scatter(const ray &r_in, const hit_record &rec,
                            color &attenuation, ray &scattered,
                            curandState *rand_state) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_unit_vector(rand_state));

        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

private:
    color albedo;
    double fuzz;
};

class dielectric : public material
{
public:
    __host__ __device__ dielectric(float _etai_over_etat)
        : etai_over_etat(_etai_over_etat) {}

    __device__ bool scatter(const ray &r_in, const hit_record &rec,
                            color &attenuation, ray &scattered,
                            curandState *rand_state) const override
    {
        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio =
            rec.front_face ? (1.0f / etai_over_etat) : (etai_over_etat);
        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin(dot(rec.normal, -unit_direction), 1.0f);

        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        bool cannot_refraction = refraction_ratio * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refraction ||
            reflectance(cos_theta, refraction_ratio) > random_double(rand_state))
        {
            reflect(unit_direction, rec.normal);
        }
        else
        {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }
        scattered = ray(rec.p, direction);
        return true;
    }

private:
    float etai_over_etat;
    __device__ static float reflectance(float cosine, float ref_idx)
    {
        // 使用 Schlick's 逼近公式表示
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif

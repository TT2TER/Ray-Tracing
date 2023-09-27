#ifndef CAMERA_H
#define CAMERA_H

#include "utils.h"

#include "color.h"
#include "hittable.h"
#include "material.h"

#include <iostream>


class camera {
  public:
    double aspect_ratio      = 1.0;  // 图像宽高比
    int    image_width       = 100;  // 渲染图像宽度（以像素数为单位）
    int    samples_per_pixel = 10;   // 每个像素的随机采样数
    int    max_depth         = 10;   // 场景中光线反射的最大次数

    double vfov     = 90;              // 垂直视角（视野）
    point3 lookfrom = point3(0,0,-1);  // 相机坐标
    point3 lookat   = point3(0,0,0);   // 相机观察点坐标
    vec3   vup      = vec3(0,1,0);     // 相机相对“向上”方向

    double defocus_angle = 0;  // 通过每个像素的光线变化角度
    double focus_dist = 10;    // 从相机坐标到完美焦点平面的距离

    void render(const hittable& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; ++j) {
            std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; ++i) {
                color pixel_color(0,0,0);
                for (int sample = 0; sample < samples_per_pixel; ++sample) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }
                write_color(std::cout, pixel_color, samples_per_pixel);
            }
        }

        std::clog << "\rDone.                 \n";
    }

  private:
    int    image_height;    // 渲染图像高度
    point3 center;          // 相机中心
    point3 pixel00_loc;     // 像素 0, 0 的位置
    vec3   pixel_delta_u;   // 像素向右偏移量
    vec3   pixel_delta_v;   // 像素向下偏移像素
    vec3   u, v, w;         // 相机帧基向量
    vec3   defocus_disk_u;  // 散焦盘水平半径
    vec3   defocus_disk_v;  // 散焦盘垂直半径

    void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // 确定视口尺寸
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (static_cast<double>(image_width)/image_height);

        // 计算相机坐标系的 u,v,w 单位基向量
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // 计算视口水平方向和垂直方向的向量
        vec3 viewport_u = viewport_width * u;    // 视口水平方向的向量
        vec3 viewport_v = viewport_height * -v;  // 视口垂直方向的向量

        // 计算到下一个像素的水平和垂直增量向量
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // 计算左上角像素的位置
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // 计算相机散焦盘基向量
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j) const {
        // 获取位置 i,j 处像素的随机采样相机光线，源自
        // 相机散焦盘。

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square();

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    vec3 pixel_sample_square() const {
        // 返回原点像素周围正方形中的随机点
        auto px = -0.5 + random_double();
        auto py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    vec3 pixel_sample_disk(double radius) const {
        // 从原点像素周围给定半径的单位圆内采样
        auto p = radius * random_in_unit_disk();
        return (p[0] * pixel_delta_u) + (p[1] * pixel_delta_v);
    }

    point3 defocus_disk_sample() const {
        // 返回相机散焦盘中的随机点
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    color ray_color(const ray& r, int depth, const hittable& world) const {
        // 如果我们超出了光线反弹限制，则不再聚集光线
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth-1, world);
            return color(0,0,0);
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};


#endif

#ifndef CAMERA_H
#define CAMERA_H
#include "curand_kernel.h"
#include "utils.h"
#include "cuda_runtime.h"
#include "color.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "cuda.h"

#include <iostream>

/**
 * @brief 根据光线与场景的相交情况计算颜色
 *
 * @param r 输入的光线
 * @param world 指向场景对象的指针
 * @param scattered 散射后的光线
 * @param rand_state 随机数生成器状态
 *
 * @return vec3 表示计算得到的颜色
 *
 * 这个函数根据光线与场景中的物体相交情况来计算颜色。
 * 如果光线与物体相交，将计算散射后的光线，并返回相应的颜色。
 * 如果光线与物体没有相交，将返回天空盒子的背景颜色。
 */
__device__ vec3 ray_color(const ray &r, hittable_list **world, ray &scattered,
                          curandState *rand_state)
{
    hit_record rec;

    // 检查光线是否与场景中的物体相交
    if ((*world)->hit(r, interval(0.001f, infinity), rec))
    {
        color attenuation;

        // 调用材质对象的 scatter 函数，计算散射后的光线和颜色衰减
        if (rec.mat->scatter(r, rec, attenuation, scattered, rand_state))
        {
            return attenuation; // 返回散射后的颜色
        }
        else
        {
            scattered = ray(point3(0, 0, 0), vec3(0, 0, 0));
            return color(0, 0, 0); // 如果无散射，返回黑色
        }
    }

    // 没有相交时，根据光线的方向返回天空盒子的背景颜色
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    scattered = ray(point3(0, 0, 0), vec3(0, 0, 0));                   // 散射光线设置为无效
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0); // 返回天空盒子的颜色
}

/**
 * @brief 在一个圆盘上采样一个点
 *
 * @param center 圆盘的中心点
 * @param defocus_disk_u 圆盘的 u 轴方向向量
 * @param defocus_disk_v 圆盘的 v 轴方向向量
 * @param rand_state 随机数生成器状态
 *
 * @return point3 表示在圆盘上采样的点
 *
 * 这个函数用于在一个圆盘上进行随机采样，圆盘由中心点和两个轴方向向量定义。
 * 随机采样结果将位于圆盘内部，并返回采样的点坐标。
 */
__device__ point3 defocus_disk_sample(const vec3 &center,
                                      const vec3 &defocus_disk_u,
                                      const vec3 &defocus_disk_v,
                                      curandState *rand_state)
{
    auto p = random_in_unit_disk(rand_state); // 在单位圆盘上进行随机采样
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

/**
 * @brief GPU 在单位正方形内随机采样
 *
 * @return vec3
 */
__device__ vec3 pixel_sample_square(curandState *rand_state,
                                    const vec3 &pixel_delta_u,
                                    const vec3 &pixel_delta_v)
{
    auto px = -0.5 + curand_uniform(rand_state);
    auto py = -0.5 + curand_uniform(rand_state);

    return px * pixel_delta_u + py * pixel_delta_v;
}

/**
 * @brief 获取从摄像机到像素(i, j)的光线
 *
 * @param i 像素的水平坐标
 * @param j 像素的垂直坐标
 * @param rand_state 随机数生成器状态
 * @param center 摄像机位置
 * @param pixel00_loc 屏幕左上角像素的位置
 * @param defocus_angle 散焦角度（用于模拟焦散效果）
 * @param pixel_delta_u 水平像素间距
 * @param pixel_delta_v 垂直像素间距
 * @param defocus_disk_u 散焦盘的 u 轴方向向量
 * @param defocus_disk_v 散焦盘的 v 轴方向向量
 *
 * @return ray 表示从摄像机到像素的光线
 *
 * 这个函数用于获取从摄像机到屏幕上指定像素(i, j)的光线。
 * 光线可能经过焦点（模拟焦散效果）或者直接从摄像机射出。
 */
__device__ ray get_ray(int i, int j, curandState *rand_state,
                       const vec3 &center, const vec3 &pixel00_loc,
                       const float &defocus_angle, const vec3 &pixel_delta_u,
                       const vec3 &pixel_delta_v, const vec3 defocus_disk_u,
                       const vec3 defocus_disk_v)
{
    auto pixel_center = pixel00_loc + i * pixel_delta_u + j * pixel_delta_v;
    auto pixel_sample =
        pixel_center +
        pixel_sample_square(rand_state, pixel_delta_u, pixel_delta_v);
    auto ray_origin = (defocus_angle <= 0)
                          ? center
                          : defocus_disk_sample(center, defocus_disk_u,
                                                defocus_disk_v, rand_state);
    auto ray_direction = pixel_sample - ray_origin;
    return ray(ray_origin, ray_direction);
}

/**
 * @brief 渲染核函数，计算像素颜色并写入帧缓冲
 *
 * @param fb 帧缓冲，用于存储像素颜色
 * @param world 场景对象的指针
 * @param max_depth 最大反射深度
 * @param image_width 图像宽度
 * @param image_height 图像高度
 * @param samples_per_pixel 每像素采样次数
 * @param center 摄像机位置
 * @param pixel00_loc 屏幕左上角像素的位置
 * @param defocus_angle 散焦角度
 * @param pixel_delta_u 水平像素间距
 * @param pixel_delta_v 垂直像素间距
 * @param defocus_disk_u 散焦盘的 u 轴方向向量
 * @param defocus_disk_v 散焦盘的 v 轴方向向量
 * @param d_rand_state 设备上的随机数生成器状态数组
 * @param n 总共的线程数
 *
 * 此函数计算每个像素的颜色，考虑了光线追踪、散焦和反射等效果。
 * 计算结果将写入帧缓冲中，用于最终图像的渲染。
 */
__global__ void render_kernal(vec3 *fb, hittable_list **world, int max_depth,
                              int image_width, int image_height,
                              int samples_per_pixel, vec3 center,
                              vec3 pixel00_loc, float defocus_angle,
                              vec3 pixel_delta_u, vec3 pixel_delta_v,
                              vec3 defocus_disk_u, vec3 defocus_disk_v,
                              curandState *d_rand_state, int n)
{
    int id = getThreadId();
    if (id >= n)
        return;

    curandState *rand_state = d_rand_state + id;

    // 计算像素的索引
    int pixel_id = id / samples_per_pixel;

    int i = pixel_id % image_width;
    int j = pixel_id / image_width;

    ray r;
    ray scattered;
    color c(1, 1, 1);

    // 获取从摄像机到像素的光线
    r = get_ray(i, j, rand_state, center, pixel00_loc, defocus_angle,
                pixel_delta_u, pixel_delta_v, defocus_disk_u, defocus_disk_v);

    // 进行光线追踪，计算像素颜色
    for (int depth = 0; depth < max_depth; depth++)
    {
        c = c * ray_color(r, world, scattered, rand_state);
        r = scattered;
        if (r.direction().near_zero())
        {
            break;
        }
        if (depth == max_depth - 1)
        {
            c = color(0, 0, 0);
        }
    }

    // 使用原子操作将像素颜色写入帧缓冲
    atomicAdd(&fb[pixel_id].e[0], c.e[0]);
    atomicAdd(&fb[pixel_id].e[1], c.e[1]);
    atomicAdd(&fb[pixel_id].e[2], c.e[2]);
}

class camera
{
public:
    float aspect_ratio = 1.0;  // 图像宽高比
    int image_width = 100;      // 渲染图像宽度（以像素数为单位）
    int samples_per_pixel = 10; // 每个像素的随机采样数
    int max_depth = 10;         // 场景中光线反射的最大次数

    float vfov = 90;                   // 垂直视角（视野）
    point3 lookfrom = point3(0, 0, -1); // 相机坐标
    point3 lookat = point3(0, 0, 0);    // 相机观察点坐标
    vec3 vup = vec3(0, 1, 0);           // 相机相对“向上”方向

    float defocus_angle = 0; // 通过每个像素的光线变化角度
    float focus_dist = 10;   // 从相机坐标到完美焦点平面的距离

    __host__ void render(hittable_list **world)
    {
        initialize();
        int n = image_width * image_height * samples_per_pixel;
        dim3 grid_size((n + 127) / 128);
        dim3 block_size(128);

        render_kernal<<<grid_size, block_size>>>(
            fb, world, max_depth, image_width, image_height, samples_per_pixel,
            center, pixel00_loc, defocus_angle, pixel_delta_u, pixel_delta_v,
            defocus_disk_u, defocus_disk_v, d_rand_state, n);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        // 输出渲染结果
        write_color(fb, image_width, image_height, std::cout, samples_per_pixel);
    }

    ~camera()
    {
        if (fb != nullptr)
            checkCudaErrors(cudaFree(fb));
        if (d_rand_state != nullptr)
            checkCudaErrors(cudaFree(d_rand_state));
    }

private:
    int image_height;                    // 渲染图像高度
    point3 center;                       // 相机中心
    point3 pixel00_loc;                  // 像素 0, 0 的位置
    vec3 pixel_delta_u;                  // 像素向右偏移量
    vec3 pixel_delta_v;                  // 像素向下偏移像素
    vec3 u, v, w;                        // 相机帧基向量
    vec3 defocus_disk_u;                 // 散焦盘水平半径
    vec3 defocus_disk_v;                 // 散焦盘垂直半径
    size_t fb_size = 0;                  // 帧缓冲内存大小（存储图像像素的内存大小）
    vec3 *fb = nullptr;                  // 帧缓冲指针，用于存储图像像素颜色
    curandState *d_rand_state = nullptr; // 设备上的随机数生成器状态数组指针

    void initialize()
    {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // 确定视口尺寸
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width =
            viewport_height * (static_cast<float>(image_width) / image_height);

        // 计算相机坐标系的 u,v,w 单位基向量
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // 计算视口水平方向和垂直方向的向量
        vec3 viewport_u = viewport_width * u;   // 视口水平方向的向量
        vec3 viewport_v = viewport_height * -v; // 视口垂直方向的向量

        // 计算到下一个像素的水平和垂直增量向量
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // 计算左上角像素的位置
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // 计算相机散焦盘基向量
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
        // CUDA 相关

        // 计算像素总数
        int num_pixels = image_width * image_height;

        // 计算帧缓冲大小（以字节为单位）
        fb_size = num_pixels * sizeof(vec3);

        // 在GPU上分配帧缓冲的内存
        checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

        // CUDA 随机状态

        // 计算总的随机数生成器状态数
        int num_rand = num_pixels * samples_per_pixel;

        // 在GPU上分配随机数生成器状态数组的内存
        checkCudaErrors(cudaMallocManaged((void **)&d_rand_state,
                                          num_rand * sizeof(curandState)));

        // 设置CUDA线程块和网格大小
        dim3 grid_size = (num_rand + 127) / 128;
        dim3 block_size = 128;

        // 初始化随机数生成器状态数组的值
        initRandState<<<grid_size, block_size>>>(d_rand_state, num_rand);

        // 等待所有CUDA线程完成初始化
        checkCudaErrors(cudaDeviceSynchronize());
    }
};

#endif

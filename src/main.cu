#include <float.h>
#include <iostream>
#include <curand_kernel.h>
#include <time.h>
#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "interval.h"
#include "material.h"
#include "ray.h"
#include "utils.h"
#include "sphere.h"
#include "vec3.h"

__global__ void create_world(hittable_list **world, curandState *rand_state)
{
    int id = getThreadId();
    if (id > 0)
        return;
    curand_init(1984, id, 0, rand_state);

    *world = new hittable_list();

    lambertian *ground_material = new lambertian(color(0.5, 0.5, 0.5));

    (*world)->add(new sphere(point3(0, -1000, 0), 1000, ground_material));

    // 随机生成22*22个小球 半径为0.2

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            float choose_mat = random_double(rand_state);

            point3 center(a + 0.9f * random_double(rand_state), 0.2f,
                          b + 0.9f * random_double(rand_state));
            if ((center - point3(4, 0.2f, 0)).length() > 0.9f)
            { // 判断是否会遮挡金属材质的大球
                material *sphere_material;

                if (choose_mat < 0.8f)
                { // 0.8概率
                    // 漫反射材质
                    auto albedo = random(rand_state) * random(rand_state);

                    sphere_material = new lambertian(albedo);

                    (*world)->add(new sphere(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                { // 0.15概率
                    // 金属材质
                    auto albedo = random(0.5f, 1.0f, rand_state);
                    auto fuzz = random_double(0.0f, 0.5f, rand_state);

                    sphere_material = new metal(albedo, fuzz);
                    (*world)->add(new sphere(center, 0.2f, sphere_material));
                }
                else
                { // 0.5概率
                  // 玻璃材质
                    sphere_material = new dielectric(1.5);
                    (*world)->add(new sphere(center, 0.2f, sphere_material));
                }
            }
        }
    }

    // 固定生成3个大球 半径为1.0
    dielectric *material1 = new dielectric(1.5f);
    (*world)->add(new sphere(point3(0, 1, 0), 1.0f, material1));

    lambertian *material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
    (*world)->add(new sphere(point3(-4, 1, 0), 1.0, material2));

    metal *material3 = new metal(color(0.7, 0.6, 0.5), 0.0f);
    (*world)->add(new sphere(point3(4, 1, 0), 1.0, material3));
}

__global__ void free_world(hittable_list **world)
{
    (*world)->clear();
    delete *world;
}

int main()
{
    camera cam;
    std::clog << "Enter image width: ";
    std::cin >> cam.image_width;

    std::clog << "Enter samples per pixel: ";
    std::cin >> cam.samples_per_pixel;

    std::clog << "Enter max depth: ";
    std::cin >> cam.max_depth;
    clock_t start = clock(), stop;

    hittable_list **d_world = nullptr;

    // 声明一个指向 curandState 结构的指针 d_rand_state，用于存储随机数生成器状态
    curandState *d_rand_state;

    // 使用 cudaMallocManaged 在主机和设备之间分配内存，以存储 curandState 结构
    checkCudaErrors(
        cudaMallocManaged((void **)&d_rand_state, sizeof(curandState)));

    // 使用 cudaMalloc 分配内存，以便在设备上存储一个指向 hittable_list* 类型的指针 d_world
    cudaMalloc((void **)&d_world, 1 * sizeof(hittable_list *));

    // 调用create_world核函数以初始化场景（在设备上
    create_world<<<1, 1>>>(d_world, d_rand_state);

    // 使用 cudaDeviceSynchronize 等待设备上的所有操作完成
    checkCudaErrors(cudaDeviceSynchronize());

    /* 设置相机和输出图像的属性 */
    // camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    // cam.image_width = 800;
    // cam.samples_per_pixel = 100;
    // cam.max_depth = 50;

    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    cam.render(d_world);
    checkCudaErrors(cudaDeviceSynchronize());

    free_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(d_world);
    cudaFree(d_rand_state);
    stop = clock();
    std::clog << "Time: " << (double)(stop - start) / CLOCKS_PER_SEC << "s";
    return 0;
}

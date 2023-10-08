#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include "utils.h"
#include <iostream>

using color = vec3;

// 对线性分量应用伽玛校正
inline double linear_to_gamma(double linear_component)
{
    return sqrt(linear_component);
}

/**
 * @brief 输出结果
 *
 * @param fb 内存地址
 * @param nx image_width
 * @param ny image_height
 * @param out 输出目标流
 * @param samples_per_pixel
 */
void write_color(vec3 *fb, int nx, int ny, std::ostream &out,
                 int samples_per_pixel)
{
    out << "P3\n"
        << nx << " " << ny << "\n255\n";
    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            vec3 pixel_color = fb[pixel_index];
            float r = pixel_color.x();
            float g = pixel_color.y();
            float b = pixel_color.z();
            // 将颜色除以采样数
            auto scale = 1.0 / samples_per_pixel;
            r *= scale;
            g *= scale;
            b *= scale;

            // 对 gamma 2 应用 linear_to_gamma 变换
            r = linear_to_gamma(r);
            g = linear_to_gamma(g);
            b = linear_to_gamma(b);
            // 写入每个颜色分量的转换后的 [0,255] 值
            static const interval intensity(0.0, 0.999);

            out << static_cast<int>(255.999 * intensity.clamp(r)) << ' '
                << static_cast<int>(255.999 * intensity.clamp(g)) << ' '
                << static_cast<int>(255.999 * intensity.clamp(b)) << '\n';
        }
    }
}

#endif

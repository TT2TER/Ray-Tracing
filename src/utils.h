#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>


// 引入std成员函数

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// 常量

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// 常用函数

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {
    // 返回 [0,1) 中的随机实数
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // 返回 [min,max) 中的随机实数
    return min + (max-min)*random_double();
}

// 常用头文件

#include "interval.h"
#include "ray.h"
#include "vec3.h"


#endif

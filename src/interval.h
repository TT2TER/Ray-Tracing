#ifndef INTERVAL_H
#define INTERVAL_H

class interval
{
public:
    float min, max;

    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // 默认区间为空

    __host__ __device__ interval(float _min, float _max) : min(_min), max(_max) {}

    __host__ __device__ bool contains(float x) const { return min <= x && x <= max; }

    __host__ __device__ bool surrounds(float x) const { return min < x && x < max; }

    __host__ __device__ float clamp(float x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

    static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif

#ifndef CUDA_H_
#define CUDA_H_

#include <iostream>
#include <curand_kernel.h>

// 检查 CUDA 错误
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

// 检查 CUDA 错误并输出错误信息
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA错误 = " << static_cast<unsigned int>(result) << " 在 "
                  << file << ":" << line << " 中的 '" << func << "' 函数\n";
        // 在退出之前确保调用 CUDA 设备重置
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief 获取线程ID
 *
 * @return int 线程ID
 */
__device__ inline int getThreadId()
{
    int block_id = blockIdx.z * (gridDim.x * gridDim.y) +
                   blockIdx.y * (gridDim.x) + blockIdx.x;
    int thread_id = threadIdx.z * (blockDim.x * blockDim.y) +
                    threadIdx.y * (blockDim.x) + threadIdx.x +
                    block_id * (blockDim.x * blockDim.y * blockDim.z);
    return thread_id;
}

/**
 * @brief 初始化随机数生成器状态
 *
 * @param d_rand_state 指向设备上的随机数生成器状态数组的指针
 * @param n 随机数生成器状态数组的大小
 *
 * 此函数用于初始化在设备上使用的随机数生成器状态数组。
 * 随机数生成器状态数组包含了每个线程的随机数生成器状态。
 * 每个线程将使用其状态生成随机数。
 *
 * @note 需要在调用此函数之前分配和传递合适大小的 d_rand_state 数组。
 */
__global__ void initRandState(curandState *d_rand_state, int n)
{
    // 获取当前线程的唯一标识符
    int id = getThreadId();

    // 如果线程ID大于等于随机数生成器状态数组的大小，则不执行初始化
    if (id >= n)
        return;

    // 使用 curand_init 初始化线程的随机数生成器状态
    // 参数1：种子，用于初始化生成器
    // 参数2：序列号，用于初始化生成器
    // 参数3：偏移量，用于初始化生成器
    // 参数4：指向当前线程状态的指针
    curand_init(2077, id, 0, &d_rand_state[id]);
}

// 在CUDA核函数中，每个线程都会调用此函数以初始化其随机数生成器状态

#endif

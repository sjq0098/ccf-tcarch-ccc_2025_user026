#include "main.h"

// 基于 HIP 的数值稳定 softmax 实现：
// 1) 计算全局最大值 m（分块归约 -> 主机归并）
// 2) 计算 t_i = exp(x_i - m)，同时分块求和 S 的分块和
// 3) 归并得到 S 后，归一化 y_i = t_i / S

#include <math.h>
#include <algorithm>

// 每个 block 对输入做网格跨步遍历，先得到线程私有最大值，再在共享内存中做 block 内归约
__global__ void reduce_max_kernel(const float* __restrict__ input,
                                  float* __restrict__ block_max,
                                  int N) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int gridSize = blockDim.x * gridDim.x;

    float local_max = -FLT_MAX;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += gridSize) {
        float v = input[i];
        local_max = fmaxf(local_max, v);
    }
    shared[tid] = local_max;
    __syncthreads();

    // 归约到 shared[0]
    for (int stride = blockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = shared[0];
    }
}

// 计算 t_i = exp(x_i - m)，并对 t_i 做分块求和
__global__ void exp_and_block_sum_kernel(const float* __restrict__ input,
                                         float* __restrict__ tmp_t,
                                         float* __restrict__ block_sum,
                                         int N,
                                         float max_val) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int gridSize = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += gridSize) {
        float t = expf(input[i] - max_val);
        tmp_t[i] = t;
        local_sum += t;
    }

    shared[tid] = local_sum;
    __syncthreads();

    // 归约求和到 shared[0]
    for (int stride = blockSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum[blockIdx.x] = shared[0];
    }
}

// 归一化：y_i = t_i / S
__global__ void normalize_kernel(const float* __restrict__ tmp_t,
                                 float* __restrict__ output,
                                 int N,
                                 float sum_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        output[i] = tmp_t[i] / sum_val;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    const int blockSize = 256;
    int maxBlocksByN = (N + blockSize - 1) / blockSize;
    // 限制 blocks 上限，避免过多分块数组
    const int maxBlocksCap = 4096;
    int numBlocks = maxBlocksByN < maxBlocksCap ? maxBlocksByN : maxBlocksCap;
    if (numBlocks < 1) numBlocks = 1;

    // 设备内存分配
    float *d_input = nullptr, *d_tmp = nullptr, *d_output = nullptr;
    float *d_blockMax = nullptr, *d_blockSum = nullptr;

    hipMalloc(&d_input, sizeof(float) * static_cast<size_t>(N));
    hipMalloc(&d_tmp, sizeof(float) * static_cast<size_t>(N));
    hipMalloc(&d_output, sizeof(float) * static_cast<size_t>(N));
    hipMalloc(&d_blockMax, sizeof(float) * static_cast<size_t>(numBlocks));
    hipMalloc(&d_blockSum, sizeof(float) * static_cast<size_t>(numBlocks));

    // 拷贝输入到设备
    hipMemcpy(d_input, input, sizeof(float) * static_cast<size_t>(N), hipMemcpyHostToDevice);

    // 1) 分块求最大值
    size_t sharedBytes = static_cast<size_t>(blockSize) * sizeof(float);
    hipLaunchKernelGGL(reduce_max_kernel, dim3(numBlocks), dim3(blockSize), sharedBytes, 0, d_input, d_blockMax, N);
    hipDeviceSynchronize();

    // 将分块最大值拷回主机做最终归约
    std::vector<float> h_blockMax(static_cast<size_t>(numBlocks));
    hipMemcpy(h_blockMax.data(), d_blockMax, sizeof(float) * static_cast<size_t>(numBlocks), hipMemcpyDeviceToHost);
    float max_val = -FLT_MAX;
    for (int i = 0; i < numBlocks; ++i) max_val = std::max(max_val, h_blockMax[i]);

    // 2) 计算 t_i 和分块和
    hipLaunchKernelGGL(exp_and_block_sum_kernel, dim3(numBlocks), dim3(blockSize), sharedBytes, 0, d_input, d_tmp, d_blockSum, N, max_val);
    hipDeviceSynchronize();

    // 主机归并得到总和 S
    std::vector<float> h_blockSum(static_cast<size_t>(numBlocks));
    hipMemcpy(h_blockSum.data(), d_blockSum, sizeof(float) * static_cast<size_t>(numBlocks), hipMemcpyDeviceToHost);
    float sum_val = 0.0f;
    for (int i = 0; i < numBlocks; ++i) sum_val += h_blockSum[i];

    // 3) 归一化得到输出
    hipLaunchKernelGGL(normalize_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, d_tmp, d_output, N, sum_val);
    hipDeviceSynchronize();

    // 输出拷回主机
    hipMemcpy(output, d_output, sizeof(float) * static_cast<size_t>(N), hipMemcpyDeviceToHost);

    // 释放设备内存
    hipFree(d_input);
    hipFree(d_tmp);
    hipFree(d_output);
    hipFree(d_blockMax);
    hipFree(d_blockSum);
}
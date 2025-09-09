#include "main.h"

/*
Blocked inclusive scan (prefix sum) on GPU using hierarchical approach:
1) Per-block inclusive scan into output and write each block's total into blockSums
2) Recursively inclusive-scan blockSums, shift to exclusive offsets
3) Add exclusive block offsets to each element of its block in output
*/

__global__ void blockInclusiveScanKernel(const int* __restrict__ d_in,
                                         int* __restrict__ d_out,
                                         int* __restrict__ d_blockSums,
                                         int N) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int gid = blockStart + tid;

    int nInBlock = blockDim.x;
    int remaining = N - blockStart;
    if (remaining < nInBlock) nInBlock = max(remaining, 0);

    int x = 0;
    if (tid < nInBlock) {
        x = d_in[gid];
    }
    sdata[tid] = (tid < nInBlock) ? x : 0;
    __syncthreads();

    // Hillis-Steele inclusive scan within a block
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int addend = 0;
        if (tid < nInBlock && tid >= offset) {
            addend = sdata[tid - offset];
        }
        __syncthreads();
        if (tid < nInBlock) {
            sdata[tid] += addend;
        }
        __syncthreads();
    }

    if (tid < nInBlock) {
        d_out[gid] = sdata[tid];
    }

    if (d_blockSums && nInBlock > 0 && tid == nInBlock - 1) {
        d_blockSums[blockIdx.x] = sdata[tid];
    }
}

__global__ void addOffsetsKernel(int* __restrict__ d_data,
                                 const int* __restrict__ d_blockOffsets,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        int blockId = blockIdx.x;
        d_data[gid] += d_blockOffsets[blockId];
    }
}

__global__ void shiftRightExclusiveKernel(const int* __restrict__ d_inInclusive,
                                          int* __restrict__ d_outExclusive,
                                          int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        int val = (gid == 0) ? 0 : d_inInclusive[gid - 1];
        d_outExclusive[gid] = val;
    }
}

static void scanInclusiveDevice(const int* d_in, int* d_out, int N, int blockSize) {
    if (N <= 0) return;

    int numBlocks = (N + blockSize - 1) / blockSize;

    if (numBlocks == 1) {
        // Single-block inclusive scan; no block sums needed
        size_t shm = static_cast<size_t>(blockSize) * sizeof(int);
        dim3 grid(1), block(blockSize);
        blockInclusiveScanKernel<<<grid, block, shm>>>(d_in, d_out, nullptr, N);
        hipDeviceSynchronize();
        return;
    }

    // Multi-block case: per-block scan + scan block sums + add offsets
    int* d_blockSums = nullptr;
    hipMalloc(&d_blockSums, sizeof(int) * numBlocks);

    size_t shm = static_cast<size_t>(blockSize) * sizeof(int);
    dim3 grid(numBlocks), block(blockSize);
    blockInclusiveScanKernel<<<grid, block, shm>>>(d_in, d_out, d_blockSums, N);
    hipDeviceSynchronize();

    // Recursively inclusive-scan blockSums
    int* d_blockSumsInclusive = nullptr;
    hipMalloc(&d_blockSumsInclusive, sizeof(int) * numBlocks);
    scanInclusiveDevice(d_blockSums, d_blockSumsInclusive, numBlocks, blockSize);

    // Convert to exclusive offsets: offset[k] = inclusive[k-1], offset[0] = 0
    int* d_blockOffsets = nullptr;
    hipMalloc(&d_blockOffsets, sizeof(int) * numBlocks);
    {
        int shiftBlock = 256;
        int shiftGrid = (numBlocks + shiftBlock - 1) / shiftBlock;
        shiftRightExclusiveKernel<<<shiftGrid, shiftBlock>>>(d_blockSumsInclusive, d_blockOffsets, numBlocks);
    }
    hipDeviceSynchronize();

    // Add offsets to each element of its block
    addOffsetsKernel<<<grid, block>>>(d_out, d_blockOffsets, N);
    hipDeviceSynchronize();

    hipFree(d_blockSums);
    hipFree(d_blockSumsInclusive);
    hipFree(d_blockOffsets);
}

extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    const int BLOCK_SIZE = 1024; // tuned for common GPUs

    int *d_in = nullptr, *d_out = nullptr;
    hipMalloc(&d_in, sizeof(int) * N);
    hipMalloc(&d_out, sizeof(int) * N);

    hipMemcpy(d_in, input, sizeof(int) * N, hipMemcpyHostToDevice);

    scanInclusiveDevice(d_in, d_out, N, BLOCK_SIZE);

    hipMemcpy(output, d_out, sizeof(int) * N, hipMemcpyDeviceToHost);

    hipFree(d_in);
    hipFree(d_out);
}
#include "main.h"
#include <hip/hip_runtime.h>
#include <cstring>

// Block-inclusive scan kernel (Blelloch, using shared memory, pads to next pow2)
__global__ void blockInclusiveScanKernel(const int* __restrict__ d_in,
                                         int* __restrict__ d_out,
                                         int* __restrict__ d_blockSums,
                                         int N) {
    extern __shared__ int sdata[]; // will be allocated with nextPow2(blockSize) ints

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int gid_base = blockStart; // base for this block

    // number of valid items in this block
    int nInBlock = blockDim.x;
    int remaining = N - blockStart;
    if (remaining < nInBlock) nInBlock = remaining > 0 ? remaining : 0;

    // compute nPow2 (next power of two >= blockDim.x)
    int nPow2 = 1;
    while (nPow2 < blockDim.x) nPow2 <<= 1;

    // initialize shared memory (some threads may need to write multiple slots)
    for (int i = tid; i < nPow2; i += blockDim.x) {
        int srcIdx = gid_base + i;
        sdata[i] = (i < nInBlock && srcIdx < N) ? d_in[srcIdx] : 0;
    }
    __syncthreads();

    // Upsweep (reduce) phase
    for (int d = 1; d < nPow2; d <<= 1) {
        int stride = d << 1; // 2*d
        // each thread handles multiple positions spaced by blockDim.x
        for (int i = tid; i < nPow2 / stride; i += blockDim.x) {
            int idx = i * stride + stride - 1;
            int left = idx - d;
            // idx and left < nPow2 by construction
            sdata[idx] += sdata[left];
        }
        __syncthreads();
    }

    // set last element to 0 for downsweep (exclusive scan)
    if (tid == 0) {
        // sdata[nPow2-1] currently holds total sum
        // keep it for potential block sum retrieval later, but set to 0 for downsweep
        sdata[nPow2 - 1] = 0;
    }
    __syncthreads();

    // Downsweep phase
    for (int d = nPow2 >> 1; d >= 1; d >>= 1) {
        int stride = d << 1;
        for (int i = tid; i < nPow2 / stride; i += blockDim.x) {
            int idx = i * stride + stride - 1;
            int left = idx - d;
            int t = sdata[left];
            sdata[left] = sdata[idx];
            sdata[idx] += t;
        }
        __syncthreads();
    }

    // Now sdata[i] is the EXCLUSIVE prefix sum for element i.
    // inclusive = exclusive + original_value
    for (int i = tid; i < nInBlock; i += blockDim.x) {
        int gid = gid_base + i;
        int orig = (gid < N) ? d_in[gid] : 0;
        int inclusive = sdata[i] + orig;
        d_out[gid] = inclusive;

        // last valid element writes block sum
        if (d_blockSums && i == (nInBlock - 1)) {
            d_blockSums[blockIdx.x] = inclusive;
        }
    }
}

// Modified addOffsetsKernel: read previous-block inclusive sums directly
// d_blockSumsInclusive is the inclusive prefix sums of each block.
// For block b, offset = (b == 0 ? 0 : d_blockSumsInclusive[b - 1])
__global__ void addOffsetsKernel(int* __restrict__ d_data,
                                 const int* __restrict__ d_blockSumsInclusive,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        int b = blockIdx.x;
        int offset = 0;
        if (b > 0) {
            offset = d_blockSumsInclusive[b - 1];
        }
        d_data[gid] += offset;
    }
}

// Helper: next power of two on host
static inline int nextPow2Host(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// Recursively inclusive-scan device array (same signature as original)
static void scanInclusiveDevice(const int* d_in, int* d_out, int N, int blockSize) {
    if (N <= 0) return;

    int numBlocks = (N + blockSize - 1) / blockSize;

    // compute shared mem bytes for per-block kernel: nextPow2(blockSize) ints
    int nPow2 = nextPow2Host(blockSize);
    size_t shm = static_cast<size_t>(nPow2) * sizeof(int);

    if (numBlocks == 1) {
        // Single-block inclusive scan; no block sums needed
        dim3 grid(1), block(blockSize);
        // launch with shm sized to nPow2*sizeof(int)
        blockInclusiveScanKernel<<<grid, block, shm>>>(d_in, d_out, nullptr, N);
        return;
    }

    // Multi-block: per-block scan + scan block sums + add offsets
    int* d_blockSums = nullptr;
    hipMalloc(&d_blockSums, sizeof(int) * numBlocks);

    {
        dim3 grid(numBlocks), block(blockSize);
        blockInclusiveScanKernel<<<grid, block, shm>>>(d_in, d_out, d_blockSums, N);
    }

    // Recursively inclusive-scan blockSums
    int* d_blockSumsInclusive = nullptr;
    hipMalloc(&d_blockSumsInclusive, sizeof(int) * numBlocks);
    scanInclusiveDevice(d_blockSums, d_blockSumsInclusive, numBlocks, blockSize);

    // ----- REPLACED: no shift kernel, no d_blockOffsets malloc -----
    // Add offsets to each element of its block by directly reading d_blockSumsInclusive[blockId-1]
    {
        dim3 grid(numBlocks), block(blockSize);
        addOffsetsKernel<<<grid, block>>>(d_out, d_blockSumsInclusive, N);
    }

    hipFree(d_blockSums);
    hipFree(d_blockSumsInclusive);
}

// Host entry (unchanged)
extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    const int BLOCK_SIZE = 1024; // original default; you can tune this

    int *d_in = nullptr, *d_out = nullptr;
    hipMalloc(&d_in, sizeof(int) * N);
    hipMalloc(&d_out, sizeof(int) * N);

    hipMemcpy(d_in, input, sizeof(int) * N, hipMemcpyHostToDevice);

    scanInclusiveDevice(d_in, d_out, N, BLOCK_SIZE);

    hipMemcpy(output, d_out, sizeof(int) * N, hipMemcpyDeviceToHost);

    hipFree(d_in);
    hipFree(d_out);
}

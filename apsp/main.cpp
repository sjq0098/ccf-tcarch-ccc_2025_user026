// HIP-based blocked Floyd–Warshall APSP
#include "main.h"
#define INFI 1073741823

using namespace std;

static inline __device__ __host__ int min_with_inf_guard(int a, int b) {
    return (a < b) ? a : b;
}

static inline __device__ int add_with_inf_guard(int a, int b) {
    if (a == INFI || b == INFI) return INFI;
    int sum = a + b;
    // Since weights are bounded and INFI is large, overflow shouldn't occur,
    // but keep a defensive clamp just in case.
    if (sum < 0) return INFI;
    return sum;
}

// Phase 1: process pivot block (round, round)
__global__ void fw_phase1(int *dist, int v, int bsize, int round) {
    extern __shared__ int s[];
    int *pivot = s; // size: bsize*bsize

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i0 = round * bsize;
    int j0 = round * bsize;

    int gi = i0 + ty;
    int gj = j0 + tx;

    if (ty < bsize && tx < bsize) {
        int val = INFI;
        if (gi < v && gj < v) val = dist[gi * v + gj];
        pivot[ty * bsize + tx] = val;
    }
    __syncthreads();

    for (int k = 0; k < bsize; ++k) {
        int via = add_with_inf_guard(pivot[ty * bsize + k], pivot[k * bsize + tx]);
        int cur = pivot[ty * bsize + tx];
        if (via < cur) pivot[ty * bsize + tx] = via;
        __syncthreads();
    }

    if (ty < bsize && tx < bsize && gi < v && gj < v) {
        dist[gi * v + gj] = pivot[ty * bsize + tx];
    }
}

// Phase 2 (row): update (round, j) for all j != round
__global__ void fw_phase2_row(int *dist, int v, int bsize, int round, int numTiles) {
    extern __shared__ int s[];
    int *pivot = s;                              // b*b
    int *rowTile = s + bsize * bsize;            // b*b

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int jTile = blockIdx.x;
    if (jTile >= round) jTile += 1; // skip pivot tile
    if (jTile >= numTiles) return;

    int i0 = round * bsize;
    int j0 = jTile * bsize;

    // Load pivot tile
    int gi_p = i0 + ty;
    int gj_p = round * bsize + tx;
    int val_p = INFI;
    if (gi_p < v && gj_p < v) val_p = dist[gi_p * v + gj_p];
    pivot[ty * bsize + tx] = val_p;

    // Load row tile (round, jTile)
    int gi_r = i0 + ty;
    int gj_r = j0 + tx;
    int val_r = INFI;
    if (gi_r < v && gj_r < v) val_r = dist[gi_r * v + gj_r];
    rowTile[ty * bsize + tx] = val_r;
    __syncthreads();

    for (int k = 0; k < bsize; ++k) {
        int via = add_with_inf_guard(pivot[ty * bsize + k], rowTile[k * bsize + tx]);
        int cur = rowTile[ty * bsize + tx];
        if (via < cur) rowTile[ty * bsize + tx] = via;
        __syncthreads();
    }

    if (gi_r < v && gj_r < v) {
        dist[gi_r * v + gj_r] = rowTile[ty * bsize + tx];
    }
}

// Phase 2 (col): update (i, round) for all i != round
__global__ void fw_phase2_col(int *dist, int v, int bsize, int round, int numTiles) {
    extern __shared__ int s[];
    int *colTile = s;                            // b*b
    int *pivot = s + bsize * bsize;              // b*b

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int iTile = blockIdx.x;
    if (iTile >= round) iTile += 1; // skip pivot tile
    if (iTile >= numTiles) return;

    int i0 = iTile * bsize;
    int j0 = round * bsize;

    // Load column tile (iTile, round)
    int gi_c = i0 + ty;
    int gj_c = j0 + tx;
    int val_c = INFI;
    if (gi_c < v && gj_c < v) val_c = dist[gi_c * v + gj_c];
    colTile[ty * bsize + tx] = val_c;

    // Load pivot tile
    int gi_p = round * bsize + ty;
    int gj_p = j0 + tx;
    int val_p = INFI;
    if (gi_p < v && gj_p < v) val_p = dist[gi_p * v + gj_p];
    pivot[ty * bsize + tx] = val_p;
    __syncthreads();

    for (int k = 0; k < bsize; ++k) {
        int via = add_with_inf_guard(colTile[ty * bsize + k], pivot[k * bsize + tx]);
        int cur = colTile[ty * bsize + tx];
        if (via < cur) colTile[ty * bsize + tx] = via;
        __syncthreads();
    }

    if (gi_c < v && gj_c < v) {
        dist[gi_c * v + gj_c] = colTile[ty * bsize + tx];
    }
}

// Phase 3: update all remaining tiles (i, j) where i != round and j != round
__global__ void fw_phase3(int *dist, int v, int bsize, int round, int numTiles) {
    extern __shared__ int s[];
    int *rowTile = s;                            // b*b (i, round)
    int *colTile = s + bsize * bsize;            // b*b (round, j)

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int jTile = blockIdx.x;
    if (jTile >= round) jTile += 1;
    if (jTile >= numTiles) return;

    int iTile = blockIdx.y;
    if (iTile >= round) iTile += 1;
    if (iTile >= numTiles) return;

    int i0 = iTile * bsize;
    int j0 = jTile * bsize;
    int kRow0 = iTile * bsize;
    int kCol0 = jTile * bsize;

    // Load rowTile (iTile, round)
    int gi_r = i0 + ty;
    int gj_r = round * bsize + tx;
    int val_r = INFI;
    if (gi_r < v && gj_r < v) val_r = dist[gi_r * v + gj_r];
    rowTile[ty * bsize + tx] = val_r;

    // Load colTile (round, jTile)
    int gi_c = round * bsize + ty;
    int gj_c = j0 + tx;
    int val_c = INFI;
    if (gi_c < v && gj_c < v) val_c = dist[gi_c * v + gj_c];
    colTile[ty * bsize + tx] = val_c;
    __syncthreads();

    int gi = i0 + ty;
    int gj = j0 + tx;
    int cur = INFI;
    if (gi < v && gj < v) cur = dist[gi * v + gj];

    for (int k = 0; k < bsize; ++k) {
        int via = add_with_inf_guard(rowTile[ty * bsize + k], colTile[k * bsize + tx]);
        if (via < cur) cur = via;
    }

    if (gi < v && gj < v) {
        dist[gi * v + gj] = cur;
    }
}

int main(int argc, char* argv[]){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    istream *in = &cin;
    ifstream fin;
    if (argc >= 2) {
        fin.open(argv[1]);
        if (fin.is_open()) in = &fin;
    }

    int v,e;
    if(!(*in >> v >> e)){
        return 0;
    }

    const int B = 16; // block size

    vector<int> h_dist(static_cast<size_t>(v) * static_cast<size_t>(v), INFI);
    for (int i = 0; i < v; ++i) h_dist[static_cast<size_t>(i) * v + i] = 0;

    int src,dst,w;
    for(int i=0;i<e;i++){
        *in >> src >> dst >> w;
        h_dist[static_cast<size_t>(src) * v + dst] = w;
    }

    int *d_dist = nullptr;
    size_t bytes = static_cast<size_t>(v) * static_cast<size_t>(v) * sizeof(int);
    hipMalloc(&d_dist, bytes);
    hipMemcpy(d_dist, h_dist.data(), bytes, hipMemcpyHostToDevice);

    dim3 threads(B, B);
    int numTiles = (v + B - 1) / B;

    for (int round = 0; round < numTiles; ++round) {
        // Phase 1
        hipLaunchKernelGGL(fw_phase1, dim3(1), threads, B * B * sizeof(int), 0, d_dist, v, B, round);
        hipDeviceSynchronize();

        if (numTiles > 1) {
            // Phase 2 row and col
            int lineTiles = numTiles - 1;
            hipLaunchKernelGGL(fw_phase2_row, dim3(lineTiles), threads, 2 * B * B * sizeof(int), 0, d_dist, v, B, round, numTiles);
            hipLaunchKernelGGL(fw_phase2_col, dim3(lineTiles), threads, 2 * B * B * sizeof(int), 0, d_dist, v, B, round, numTiles);
            hipDeviceSynchronize();

            // Phase 3
            dim3 grid(lineTiles, lineTiles);
            hipLaunchKernelGGL(fw_phase3, grid, threads, 2 * B * B * sizeof(int), 0, d_dist, v, B, round, numTiles);
            hipDeviceSynchronize();
        }
    }

    hipMemcpy(h_dist.data(), d_dist, bytes, hipMemcpyDeviceToHost);
    hipFree(d_dist);

    for(int i=0;i<v;i++){
        for(int j=0;j<v;j++){
            if(j) cout << " ";
            cout << h_dist[static_cast<size_t>(i) * v + j];
        }
        cout << '\n';
    }
    return 0;
}
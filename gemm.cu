#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16
#define TILE_K 16

#define TILE_SIZE 16

// A : m * k
// B : k * n

__global__ void gemm(float *A, float *B, float *C, int m, int n, int k) {
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0;
    for(int i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
}

__global__ void gemm_shm(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float share_A[TILE_SIZE][TILE_SIZE];
    __shared__ float share_B[TILE_SIZE][TILE_SIZE];

    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0;

    // load A to shared mem
    for(int tile_k = 0; tile_k < k; tile_k += TILE_SIZE) {
        share_A[threadIdx.y][threadIdx.x] = A[row * k + (threadIdx.y + tile_k)];
        share_B[threadIdx.y][threadIdx.x] = B[(tile_k + threadIdx.x) * n + col];
        __syncthreads();
        float inner_sum = 0;
        for(int inner_k = 0; inner_k < TILE_SIZE; inner_k++) {
            inner_sum += share_A[threadIdx.y][inner_k] * share_B[inner_k][threadIdx.x];
        }
        sum += inner_sum;
        __syncthreads();
    }
    C[row * n + col] = sum;
}

void printMatrix(float *A, int m, int n) {
    for(int i = 0; i < m; i++) {
        printf("[");
        for(int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("]\n");
    }
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int m = 1024, n = 1024, k = 1024;
    int size_A = m * k * sizeof(float);
    int size_B = k * n * sizeof(float);
    int size_C = m * n * sizeof(float);

    A = (float *)malloc(size_A);
    B = (float *)malloc(size_B);
    C = (float *)malloc(size_C);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    for(int i = 0; i < m * k; i++) {
        A[i] = 1.0;
    }

    for(int i = 0; i < k * n; i++) {
        B[i] = 1.0;
    }

    for(int i = 0; i < m * n; i++) {
        C[i] = 0.0;
    }

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);

    dim3 grid(m / 16, n / 16);
    dim3 block(16, 16);

    gemm<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // printMatrix(C, m, n);
    return 0;
}
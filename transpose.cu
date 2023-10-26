#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void printMatrix(float *A, int m, int n)
{
    for (int i = 0; i < m; i++) {
        printf("[");
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("]\n");
    }
}

__global__ void transpose(float *in, float *out, int m, int n)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n)
        out[col * m + row] = in[row * n + col];
}


int main()
{
    float *in;
    float *out;
    float *d_in;
    float *d_out;
    int m = 32;
    int n = 32;

    int size_in = m * n * sizeof(float);
    int size_out = m * n * sizeof(float);

    in = (float *)malloc(size_in);
    out = (float *)malloc(size_out);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            in[i * n + j] = i;
        }
    }

    printMatrix(in, m, n);

    cudaMalloc((void **)&d_in, size_in);
    cudaMalloc((void **)&d_out, size_out);

    cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice);

    dim3 dimGrid((m + 15) / 16, (n + 15) / 16);
    dim3 dimBlock(16, 16);

    transpose<<<dimGrid, dimBlock>>>(d_in, d_out, m, n);

    cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost);
    printMatrix(out, n, m);
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

static constexpr int TILESIZE = 32;
static constexpr int ROWSIZE = 8;

void printMatrix(float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    printf("[");
    for (int j = 0; j < n; j++) {
      printf("%f ", A[i * n + j]);
    }
    printf("]\n");
  }
}

bool check(float *A, float *B, int size, float eps = 1e-2) {
  for (int i = 0; i < size; i++) {
    if (abs(A[i] - B[i]) > eps) {
      printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
      return false;
    }
  }
  return true;
}

__global__ void transpose(float *in, float *out, int m, int n) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < m && col < n)
    out[col * m + row] = in[row * n + col];
}

template <int TILE_SIZE,
          int ROW_SIZE> // blockDim.x == TILE_SIZE, blockDim.y == ROW_SIZE
__global__ void transpose_shared(float *in, float *out, int m, int n) {
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
    if (row + i < m && col < n) {
      tile[i + threadIdx.y][threadIdx.x] = in[(row + i) * n + col];
    }
  }

  __syncthreads();

  int out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  int out_col = blockIdx.y * TILE_SIZE + threadIdx.x;

  for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
    out[(out_row + i) * m + out_col] = tile[threadIdx.x][i + threadIdx.y];
  }
}

int main() {
  float *in;
  float *out;
  float *out_golden;
  float *d_in;
  float *d_out;
  int m = 32;
  int n = 32;

  int size_in = m * n * sizeof(float);
  int size_out = m * n * sizeof(float);

  in = (float *)malloc(size_in);
  out = (float *)malloc(size_out);
  out_golden = (float *)malloc(size_out);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      in[i * n + j] = i;
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      out_golden[i * m + j] = in[j * n + i];
    }
  }

  cudaMalloc((void **)&d_in, size_in);
  cudaMalloc((void **)&d_out, size_out);

  cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice);

  dim3 dimGrid((m + 15) / 16, (n + 15) / 16);
  dim3 dimBlock(16, 16);

  transpose<<<dimGrid, dimBlock>>>(d_in, d_out, m, n);

  // dim3 dimGrid_shared((m + TILESIZE - 1) / TILESIZE,
  //                     (n + TILESIZE - 1) / TILESIZE);
  // dim3 dimBlock_shared(TILESIZE, ROWSIZE);
  // transpose_shared<TILESIZE, ROWSIZE><<<dimGrid_shared, dimBlock_shared>>>(d_in, d_out, m, n);

  cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost);

  printf("Output:\n");

  printMatrix(out, n, m);

  printf("\nGolden:\n");

  printMatrix(out_golden, n, m);

  if (check(out, out_golden, m * n)) {
    printf("Success!\n");
  } else {
    printf("Failed!\n");
  }
}
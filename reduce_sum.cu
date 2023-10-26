#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

constexpr int threadsPerBlock = 1024;

__global__ void kernel1(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;  // tid号线程要负责的数组元素的位置
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();

    for(int s = 1; s < blockDim.x; s*=2){
        if(tid % (2*s) == 0 && i + s <N){     // 偶数线程work
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

__global__ void kernel2(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; 
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();

    for(int s = 1; s < blockDim.x; s*=2){
        int index = tid * 2 * s;       // 原来是每个线程对应一个位置，第一轮循环，只有0、2、4、6这些线程在执行，1、3、5线程闲置，同一个warp内有一半线程没有用上
        if((index + s) < blockDim.x && (blockIdx.x * blockDim.x + index + s) < N){   // 现在是tid号线程处理处理tid*2*s位置的任务，第一轮循环0123456线程都在线，warp利用率高
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

__global__ void kernel3(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; 
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){   // kernel2的访问share memory的方式，存在share memory bank conflit
        if(tid < s && i + s < N){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

int main()
{
    float *A = new float[1024];

    return 0;

}
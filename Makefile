

All: gemm transpose

gemm: gemm.cu
	nvcc -o gemm gemm.cu 

transpose: transpose.cu
	nvcc -o transpose transpose.cu
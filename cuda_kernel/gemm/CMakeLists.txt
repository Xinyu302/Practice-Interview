file(GLOB CUDA_SOURCES "*.cu")

add_executable(gemm_exec ${CUDA_SOURCES})

# target_compile_options(gemm_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>)
# set_target_properties(gemm_exec PROPERTIES CUDA_ARCHITECTURES "arch=sm_70")
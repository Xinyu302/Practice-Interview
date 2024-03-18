file(GLOB CUDA_SOURCES "*.cu")

add_executable(reduce_sum_exec ${CUDA_SOURCES})

# target_compile_options(reduce_sum_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>)
# set_target_properties(reduce_sum_exec PROPERTIES CUDA_ARCHITECTURES "arch=sm_70")
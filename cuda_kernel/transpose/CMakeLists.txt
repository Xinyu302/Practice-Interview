file(GLOB CUDA_SOURCES "*.cu")

add_executable(transpose_exec ${CUDA_SOURCES})

# target_compile_options(transpose_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>)
# set_target_properties(transpose_exec PROPERTIES CUDA_ARCHITECTURES "arch=sm_70")
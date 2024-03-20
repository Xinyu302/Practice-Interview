# 获取当前目录下的所有.cpp文件
file(GLOB SOURCES *.cpp)

# 获取当前目录名
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 创建可执行文件并链接所有.cpp文件
add_executable(${DIR_NAME} ${SOURCES})

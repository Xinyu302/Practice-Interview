# 获取当前目录下的所有子目录
file(GLOB subdirs LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)

# 添加所有子目录
foreach(subdir ${subdirs})
    if(IS_DIRECTORY ${subdir})
        add_subdirectory(${subdir})
    endif()
endforeach()
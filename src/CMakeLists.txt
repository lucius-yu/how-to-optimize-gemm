# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 项目信息
project (MMulti)

# set
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_VERBOSE_MAKEFILE ON)

# set C_FLAGS
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall -msse3 -lm")

# 查找目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
# aux_source_directory(. DIR_SRCS)

if(NOT NEW) 
    SET(NEW "MMult0")
    message(STATUS "NEW is not set. Use default value ${NEW}")
endif()

SET(DIR_SRCS "test_MMult.c" "${NEW}.c")

# 添加 math 子目录
add_subdirectory(utils)

# 指定生成目标
add_executable(test_MMult.x ${DIR_SRCS})

# 添加链接库
target_link_libraries(test_MMult.x utils)

# 启用测试
enable_testing()

# 测试程序是否成功运行
if(NOT OUT)
    SET(OUT "../data/${NEW}")
    message(STATUS "OUT is not set. Use default value ${OUT}")
endif()

add_test (NAME test_run COMMAND sh -c "test_MMult.x > ${OUT}.csv")

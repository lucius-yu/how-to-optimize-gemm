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
aux_source_directory(. DIR_SRCS)

# 添加 math 子目录
add_subdirectory(utils)

# 指定生成目标
add_executable(test_MMult.x ${DIR_SRCS})

# 添加链接库
target_link_libraries(test_MMult.x utils)

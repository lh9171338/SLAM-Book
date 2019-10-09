########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(useEigen)

########### 添加库
include_directories("/usr/include/eigen3/")

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
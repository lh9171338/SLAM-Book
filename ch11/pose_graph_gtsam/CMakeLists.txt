########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(pose_graph_gtsam)

########### 添加库
# eigen3
include_directories("/usr/include/eigen3/")

# gtsam
find_package(GTSAMCMakeTools)
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} gtsam)
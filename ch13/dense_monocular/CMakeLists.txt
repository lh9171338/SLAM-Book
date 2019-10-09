########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(dense_monocular)

########### 添加库
# opencv
find_package(OpenCV REQUIRED)

# Eigen
include_directories("/usr/include/eigen3")

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Sophus_LIBRARIES})

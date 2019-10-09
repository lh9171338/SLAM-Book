########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(dense_rgbd_pointcloud)

########### 添加库
# opencv
find_package(OpenCV REQUIRED)

# eigen
include_directories("/usr/include/eigen3/")

# pcl
find_package(PCL REQUIRED COMPONENT common io filters)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})
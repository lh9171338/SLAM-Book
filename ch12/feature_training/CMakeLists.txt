########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(feature_training)

########### 添加库
# opencv
find_package(OpenCV REQUIRED)

# dbow3
set(DBoW3_INCLUDE_DIRS "/usr/local/include")
set(DBoW3_LIBS "/usr/local/lib/libDBoW3.a")
include_directories(${DBoW3_INCLUDE_DIRS})

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${DBoW3_LIBS})
########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(curve_fitting_g2o)

########### 添加库
# opencv
find_package(OpenCV REQUIRED)

# eigen
include_directories("/usr/include/eigen3")

# g2o
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} g2o_core g2o_stuff)
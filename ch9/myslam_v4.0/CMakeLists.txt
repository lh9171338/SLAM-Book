########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

########### 设置项目名称
project(myslam_v4_0)

########### 添加库
# opencv
find_package(OpenCV REQUIRED)

# eigen3
include_directories("/usr/include/eigen3")

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
set(G2O_LIBRARIES g2o_core g2o_stuff g2o_types_sba)

########### 添加头文件路径
include_directories(include)

########### 添加源文件
aux_source_directory(src SRC_LIST)

########### 编译可执行文件
add_executable(${PROJECT_NAME} test/main.cpp ${SRC_LIST} include/g2otypes.h src/g2otypes.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${G2O_LIBRARIES})
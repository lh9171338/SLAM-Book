########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(pose_graph_g2o_SE3)

########### 添加库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# eigen3
include_directories("/usr/include/eigen3/")

# Cholmod
find_package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIR})

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
set(G2O_LIBRARIES g2o_core g2o_stuff g2o_types_slam3d)

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${G2O_LIBRARIES} ${CHOLMOD_LIBRARIES})
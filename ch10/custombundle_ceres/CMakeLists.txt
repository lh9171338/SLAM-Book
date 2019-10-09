########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(custombundle_ceres)

########### 添加库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# eigen3
include_directories("/usr/include/eigen3/")

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

########### 添加头文件路径
include_directories(include)
include_directories(include/tools)

########### 添加源文件
aux_source_directory(src SRC_LIST)

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${G2O_LIBS} ${CERES_LIBRARIES})
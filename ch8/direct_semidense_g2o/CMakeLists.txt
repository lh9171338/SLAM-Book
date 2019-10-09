########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(direct_semidense_g2o)

########### 添加库
# opencv
find_package(OpenCV REQUIRED)

# g2o
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O)
include_directories(${G2O_INCLUDE_DIRS})
set(G2O_LIBS g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension)

# eigen3
include_directories("/usr/include/eigen3")

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${G2O_LIBS})
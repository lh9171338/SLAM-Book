########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(myslam_v1_0)

########### 添加库
# opencv
find_package(OpenCV REQUIRED)

# eigen3
include_directories("/usr/include/eigen3")

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

########### 添加头文件路径
include_directories(include)

########### 添加源文件
aux_source_directory(src SRC_LIST)

########### 编译可执行文件
add_executable(${PROJECT_NAME} test/main.cpp ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Sophus_LIBRARIES})
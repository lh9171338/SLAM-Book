########### 设置版本
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

########### 设置项目名称
project(custombundle_g2o)

########### 添加库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# opencv
find_package(OpenCV REQUIRED)

# g2o
find_package(G2O)
include_directories(${G2O_INCLUDE_DIRS})
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear
        g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface
        g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense
        g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse)

# eigen3
include_directories("/usr/include/eigen3/")

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Cholmod
Find_Package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIR})

########### 添加头文件路径
include_directories(include)
include_directories(include/tools)

########### 添加源文件
aux_source_directory(src SRC_LIST)

########### 编译可执行文件
add_executable(${PROJECT_NAME} main.cpp ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${G2O_LIBS} ${CHOLMOD_LIBRARIES} ${CERES_LIBRARIES})
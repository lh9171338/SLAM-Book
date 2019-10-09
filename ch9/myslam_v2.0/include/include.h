//
// Created by lihao on 19-9-30.
//

#ifndef INCLUDE_H
#define INCLUDE_H

// define the commonly included file to avoid a long include list
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;

// for g2o
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

// for Sophus
#include <sophus/se3.h>
#include <sophus/so3.h>
using Sophus::SE3;
using Sophus::SO3;

// for cv
#include <opencv2/opencv.hpp>
using cv::Mat;
using namespace cv;

// std
#include <iostream>
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <set>
#include <unordered_map>
#include <map>
#include <boost/timer.hpp>
using namespace std;


#endif //INCLUDE_H

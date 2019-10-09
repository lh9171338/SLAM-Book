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

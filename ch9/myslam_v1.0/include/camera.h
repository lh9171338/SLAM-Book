//
// Created by lihao on 19-9-30.
//

#ifndef CAMERA_H
#define CAMERA_H

#include "include.h"
#include "config.h"

namespace myslam{

// Pinhole RGBD camera model
class Camera
{
public:
    typedef shared_ptr<Camera> Ptr;
    double m_fx, m_fy, m_cx, m_cy; // Camera intrinsics
    double m_depth_scale;

public:
    Camera();
    Camera(double fx, double fy, double cx, double cy, double depth_scale=0);

    // coordinate transform: world, camera, pixel
    Mat getIntrinsic();
    Vector3d world2camera(const Vector3d& p_w, const SE3& T_cw);
    Vector3d camera2world(const Vector3d& p_c, const SE3& T_cw);
    Vector2d camera2pixel(const Vector3d& p_c);
    Vector3d pixel2camera(const Vector2d& p_p, double depth=1);
    Vector3d pixel2world(const Vector2d& p_p, const SE3& T_cw, double depth=1);
    Vector2d world2pixel(const Vector3d& p_w, const SE3& T_cw);

};

}

#endif //CAMERA_H

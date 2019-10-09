//
// Created by lihao on 19-9-30.
//

#include "camera.h"

namespace myslam{

Camera::Camera()
{
    m_fx = Config::get<double>("fx");
    m_fy = Config::get<double>("fy");
    m_cx = Config::get<double>("cx");
    m_cy = Config::get<double>("cy");
    m_depth_scale = Config::get<double>("depth_scale");
}

Camera::Camera(double fx, double fy, double cx, double cy, double depth_scale)
    : m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy), m_depth_scale(depth_scale)
{

}

Mat Camera::getIntrinsic()
{
    Mat K = (Mat_<double>(3,3) <<
            m_fx, 0, m_cx,
            0, m_fy, m_cy,
            0, 0, 1
    );
    return K;
}

Vector3d Camera::world2camera(const Vector3d &p_w, const SE3 &T_cw)
{
    return T_cw * p_w;
}

Vector3d Camera::camera2world(const Vector3d &p_c, const SE3 &T_cw)
{
    return T_cw.inverse() * p_c;
}

Vector2d Camera::camera2pixel(const Vector3d &p_c)
{
    return Vector2d(
            m_fx * p_c[0] / p_c[2] + m_cx,
            m_fy * p_c[1] / p_c[2] + m_cy
    );
}

Vector3d Camera::pixel2camera(const Vector2d &p_p, double depth)
{
    return Vector3d(
            (p_p[0] - m_cx) * depth / m_fx,
            (p_p[1] - m_cy) * depth / m_fy,
            depth
    );
}

Vector2d Camera::world2pixel(const Vector3d &p_w, const SE3 &T_cw)
{
    return camera2pixel(world2camera(p_w, T_cw));
}

Vector3d Camera::pixel2world(const Vector2d &p_p, const SE3 &T_cw, double depth)
{
    return camera2world(pixel2camera(p_p, depth), T_cw);
}

}
//
// Created by lihao on 19-10-4.
//

#ifndef G2OTYPES_H
#define G2OTYPES_H

#include "include.h"
#include "camera.h"

namespace myslam{

struct ReprojectCost
{
private:
    Point3d m_p3d;
    Point2d m_p2d;

public:
    ReprojectCost(Point3d p3d, Point2d p2d) : m_p3d(p3d), m_p2d(p2d) {}

    template<typename T>
    bool operator()(const T* const intrinsic, const T* const extrinsic, T* residuals) const
    {
        const T* r = extrinsic;
        const T* t = &extrinsic[3];
        T p3d[3] = {T(m_p3d.x), T(m_p3d.y), T(m_p3d.z)};
        T p3d_trans[3];
        ceres::AngleAxisRotatePoint(r, p3d, p3d_trans);
        p3d_trans[0] += t[0];
        p3d_trans[1] += t[1];
        p3d_trans[2] += t[2];

        const T x = p3d_trans[0] / p3d_trans[2];
        const T y = p3d_trans[1] / p3d_trans[2];
        const T fx = intrinsic[0];
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(m_p2d.x);
        residuals[1] = v - T(m_p2d.y);

        return true;
    }
};

}


#endif //G2OTYPES_H

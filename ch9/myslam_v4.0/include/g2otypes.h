//
// Created by lihao on 19-10-4.
//

#ifndef G2OTYPES_H
#define G2OTYPES_H

#include "include.h"
#include "camera.h"

namespace myslam{

class EdgeProjectXYZRGBD : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void computeError();
    virtual void linearizeOplus();
    virtual bool read(istream& in){}
    virtual bool write(ostream& out) const {}
};

// only to optimize the pose, no point
class EdgeProjectXYZRGBDPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
private:
    Vector3d m_point;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectXYZRGBDPoseOnly(Vector3d point);

    // Error: measure = R*point+t
    virtual void computeError();
    virtual void linearizeOplus();

    virtual bool read(istream& in){}
    virtual bool write(ostream& out) const {}
};

class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap >
{
private:
    Vector3d m_point;
    Camera::Ptr m_ptrCamera;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectXYZ2UVPoseOnly(Vector3d point, Camera::Ptr ptrCamera);

    virtual void computeError();
    virtual void linearizeOplus();

    virtual bool read(istream& in){}
    virtual bool write(ostream& os) const {};
};

}


#endif //G2OTYPES_H

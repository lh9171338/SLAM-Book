//
// Created by lihao on 19-9-30.
//

#ifndef FRAME_H
#define FRAME_H

#include "include.h"
#include "camera.h"

namespace myslam{

class Frame
{
public:
    typedef shared_ptr<Frame> Ptr;
    unsigned long             m_id;             // id of this frame
    double                    m_timeStamp;      // when it is recorded
    SE3                       m_T_cw;          // transform from world to camera
    Camera::Ptr               m_ptrCamera;      // Pinhole RGBD Camera model
    Mat                       m_color, m_depth; // color and depth image

public: // data members
    Frame();
    Frame(unsigned long id=0, double timeStamp=0, Camera::Ptr ptrCamera=nullptr, Mat color=Mat(), Mat depth=Mat(), SE3 T_cw=SE3());

    // factory function
    static Frame::Ptr createFrame(double timeStamp=0, Camera::Ptr ptrCamera=nullptr, Mat color=Mat(), Mat depth=Mat());

    // find the depth in depth map
    double findDepth(const KeyPoint& kp);

    // Get Camera Center
    Vector3d getCamCenter();

    // check if a point is in this frame
    bool isInFrame(const Vector3d& p_w);
};

}


#endif //FRAME_H

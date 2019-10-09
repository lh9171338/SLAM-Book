//
// Created by lihao on 19-9-30.
//

#include "frame.h"

namespace myslam{

Frame::Frame()
    : m_id(0), m_timeStamp(0), m_ptrCamera(nullptr)
{

}

Frame::Frame(unsigned long id, double timeStamp, Camera::Ptr ptrCamera, Mat color, Mat depth, SE3 T_cw)
    : m_id(id), m_timeStamp(timeStamp), m_ptrCamera(ptrCamera), m_color(color), m_depth(depth), m_T_cw(T_cw)
{

}

Frame::Ptr Frame::createFrame(double timeStamp, Camera::Ptr ptrCamera, Mat color, Mat depth)
{
    static unsigned long factory_id = 0;
    return Frame::Ptr(new Frame(factory_id++, timeStamp, ptrCamera, color, depth));
}

double Frame::findDepth(const KeyPoint& kp)
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = m_depth.ptr<ushort>(y)[x];
    if (d != 0)
    {
        return double(d) / m_ptrCamera->m_depth_scale;
    }
    else
    {
        // check the nearby points
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, -1, 0, 1};
        for(int i=0; i<4; i++)
        {
            d = m_depth.ptr<ushort>(y+dy[i])[x+dx[i]];
            if(d != 0)
            {
                return double(d) / m_ptrCamera->m_depth_scale;
            }
        }
    }
    return -1.0;
}


Vector3d Frame::getCamCenter()
{
    return m_T_cw.inverse().translation();
}

bool Frame::isInFrame(const Vector3d& p_w)
{
    Vector3d p_c = m_ptrCamera->world2camera(p_w, m_T_cw);
    if(p_c[2] < 0)
        return false;
    Vector2d p_p = m_ptrCamera->world2pixel(p_w, m_T_cw);
    return p_p[0] >0 && p_p[0] < m_color.cols
          && p_p[1] >0 && p_p[1] < m_color.rows;
}

}
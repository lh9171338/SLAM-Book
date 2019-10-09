//
// Created by lihao on 19-9-30.
//

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "include.h"

namespace myslam{

class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long      m_id;        // ID
    Vector3d    m_position;         // Position in world
    Vector3d    m_norm;             // Normal of viewing direction
    Mat         m_descriptor_;      // Descriptor for matching
    int         m_observedTimes;    // being observed by feature matching algorithm.
    int         m_matchedTimes;     // being an inliner in pose estimation

public:
    MapPoint();
    MapPoint(unsigned long id, Vector3d position, Vector3d norm);

    // factory function
    static MapPoint::Ptr createMapPoint();
};

}

#endif //MAPPOINT_H

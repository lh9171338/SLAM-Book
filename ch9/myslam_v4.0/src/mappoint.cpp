//
// Created by lihao on 19-10-4.
//

#include "mappoint.h"

namespace myslam{

MapPoint::MapPoint()
    : m_id(0), m_position(Vector3d(0,0,0)), m_norm(Vector3d(0,0,0)), m_observedTimes(0), m_matchedTimes(0)
{

}

MapPoint::MapPoint(unsigned long id, Vector3d position, Vector3d norm, Mat descriptor)
    : m_id(id), m_position(position), m_norm(norm), m_descriptor(descriptor), m_observedTimes(0), m_matchedTimes(0)
{

}

MapPoint::Ptr MapPoint::createMapPoint(Vector3d position, Vector3d norm, Mat descriptor)
{
    static unsigned long factory_id = 0;
    return MapPoint::Ptr(
            new MapPoint(factory_id++, position, norm, descriptor)
    );
}

}
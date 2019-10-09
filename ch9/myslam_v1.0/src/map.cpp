//
// Created by lihao on 19-10-4.
//

#include "map.h"

namespace myslam{

void Map::insertKeyFrame(Frame::Ptr frame)
{
    if(m_keyFrames.find(frame->m_id) == m_keyFrames.end())
    {
        m_keyFrames.insert(make_pair(frame->m_id, frame));
    }
    else
    {
        m_keyFrames[frame->m_id] = frame;
    }
    cout <<"Key frame size = "<< m_keyFrames.size() << endl;
}

void Map::insertMapPoint(MapPoint::Ptr mapPoint)
{
    if(m_mapPoints.find(mapPoint->m_id) == m_mapPoints.end() )
    {
        m_mapPoints.insert( make_pair(mapPoint->m_id, mapPoint) );
    }
    else
    {
        m_mapPoints[mapPoint->m_id] = mapPoint;
    }
}

}
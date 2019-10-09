//
// Created by lihao on 19-10-4.
//

#ifndef MAP_H
#define MAP_H

#include "include.h"
#include "frame.h"
#include "mappoint.h"

namespace myslam{

class Map
{
public:
    typedef shared_ptr<Map> Ptr;
    unordered_map<unsigned long, MapPoint::Ptr>  m_mapPoints;        // all landmarks
    unordered_map<unsigned long, Frame::Ptr>     m_keyFrames;        // all key-frames

public:
    Map() {}

    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr mapPoint);
};

}


#endif //MAP_H

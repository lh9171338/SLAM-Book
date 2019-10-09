//
// Created by lihao on 19-10-4.
//

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "include.h"
#include "map.h"
#include "g2otypes.h"
#include "config.h"

namespace myslam{

class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState{
        INITIALIZING=-1,
        OK=0,
        LOST
    };

    VOState     m_state;     // current VO status
    Map::Ptr    m_map;       // map with all frames and map points
    Frame::Ptr  m_refFrame;       // reference frame
    Frame::Ptr  m_currFrame;      // current frame

    cv::Ptr<ORB> m_orb;  // orb detector and computer
    vector<MapPoint::Ptr> m_keypoints3d;    // keypoints in 3d
    vector<KeyPoint> m_currKeypoints;    // keypoints in current frame
    Mat m_refDescriptors;   // descriptor in reference frame
    Mat m_currDescriptors;  // descriptor in current frame
    vector<DMatch> m_matches;

    SE3 m_T_cr;  // the estimated pose of current frame
    int m_numInliers;        // number of inlier features in icp
    int m_numLost;           // number of lost times

    // parameters
    int m_num_of_features;   // number of features
    double m_scale_factor;   // scale in image pyramid
    int m_level_pyramid;     // number of pyramid levels
    float m_match_ratio;      // ratio for selecting  good matches
    int m_max_num_lost;      // max number of continuous lost times
    int m_min_inliers;       // minimum inliers

    double m_key_frame_min_rot;   // minimal rotation of two key-frames
    double m_key_frame_min_trans; // minimal translation of two key-frames
    double m_map_point_match_ratio; // remove map point ratio

public: // functions
    VisualOdometry();
    bool addFrame(Frame::Ptr frame);      // add a new frame

protected:
    // inner operation
    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void poseEstimationPnP();
    void optimizeMap();

    void addKeyFrame();
    void addMapPoints();
    bool checkEstimatedPose();
    bool checkKeyFrame();
    double getViewAngle(Frame::Ptr frame, MapPoint::Ptr point);
};

}

#endif //VISUALODOMETRY_H
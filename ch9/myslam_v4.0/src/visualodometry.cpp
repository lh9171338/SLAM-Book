//
// Created by lihao on 19-10-4.
//

#include "visualodometry.h"

namespace myslam{

VisualOdometry::VisualOdometry()
    : m_state(INITIALIZING), m_refFrame (nullptr), m_currFrame(nullptr), m_map(new Map), m_numLost(0), m_numInliers(0)
{
    m_num_of_features = Config::get<int>("number_of_features");
    m_scale_factor = Config::get<double>("scale_factor");
    m_level_pyramid = Config::get<int>("level_pyramid");
    m_match_ratio = Config::get<float>("match_ratio");
    m_max_num_lost = Config::get<float>("max_num_lost");
    m_min_inliers = Config::get<int> ("min_inliers");
    m_key_frame_min_rot = Config::get<double>("keyframe_rotation");
    m_key_frame_min_trans = Config::get<double>("keyframe_translation");
    m_map_point_match_ratio = Config::get<double>("map_point_match_ratio");
    m_orb = ORB::create(m_num_of_features, m_scale_factor, m_level_pyramid);
}

bool VisualOdometry::addFrame(Frame::Ptr frame)
{
    switch(m_state)
    {
        case INITIALIZING:
        {
            m_state = OK;
            m_currFrame = frame;
            extractKeyPoints();
            computeDescriptors();
            addKeyFrame();
            addMapPoints();
            break;
        }
        case OK:
        {
            m_currFrame = frame;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            if(checkEstimatedPose() == true) // a good estimation
            {
                m_currFrame->m_T_cw = m_T_cr;
                m_numLost = 0;
                optimizeMap();
                if(checkKeyFrame() == true) // is a key-frame
                {
                    addKeyFrame();
                }
            }
            else // bad estimation due to various reasons
            {
                m_currFrame->m_T_cw = m_refFrame->m_T_cw;
                m_numLost++;
                if(m_numLost > m_max_num_lost)
                {
                    m_state = LOST;
                }
                return false;
            }
            break;
        }
        case LOST:
        {
            cerr << "vo has lost." << endl;
            break;
        }
    }
    return true;
}

void VisualOdometry::extractKeyPoints()
{
    m_orb->detect(m_currFrame->m_color, m_currKeypoints);
}

void VisualOdometry::computeDescriptors()
{
    m_orb->compute(m_currFrame->m_color, m_currKeypoints, m_currDescriptors);
}

void VisualOdometry::featureMatching()
{
    m_keypoints3d.clear();
    m_refDescriptors = Mat();
    for(auto& mp: m_map->m_mapPoints)
    {
        MapPoint::Ptr& pt = mp.second;
        // check if pt in curr frame image
        if(m_refFrame->isInFrame(pt->m_position))
        {
            pt->m_observedTimes++;
            m_keypoints3d.push_back(pt);
            m_refDescriptors.push_back(pt->m_descriptor);
        }
    }

    // match reference descriptor and current descriptor, use OpenCV's brute force match
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(m_refDescriptors, m_currDescriptors, matches);
    // select the best matches
    float min_dist = min_element(
            matches.begin(), matches.end(),
            [](const DMatch& m1, const DMatch& m2)
            {
                return m1.distance < m2.distance;
            })->distance;

    m_matches.clear();
    for(DMatch& m : matches)
    {
        if(m.distance < max<float>(min_dist*m_match_ratio, 30.0))
        {
            m_matches.push_back(m);
            m_keypoints3d[m.queryIdx]->m_matchedTimes++;
        }
    }
    cout << "good matches: " << m_matches.size() << endl;
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<Point3f> pts3d;
    vector<Point2f> pts2d;

    for(DMatch m : m_matches)
    {
        Vector3d p_w = m_keypoints3d[m.queryIdx]->m_position;
        pts3d.push_back(Point3f(p_w[0], p_w[1], p_w[2]));
        pts2d.push_back(m_currKeypoints[m.trainIdx].pt);
    }

    Mat K = m_currFrame->m_ptrCamera->getIntrinsic();
    Mat rvec, tvec, inliers;
    solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
    m_numInliers = inliers.rows;
    cout << "pnp inliers: " << m_numInliers << endl;

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 2>> BlockSolver;
    typedef g2o::LinearSolverDense<BlockSolver::PoseMatrixType> LinearSolver;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate (g2o::SE3Quat(
            m_T_cr.rotation_matrix(),
            m_T_cr.translation()
    ) );
    optimizer.addVertex(pose);

    // edges
    for(int i=0; i<m_numInliers; i++)
    {
        int index = inliers.at<int>(i,0);
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly(
                Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z),
                m_currFrame->m_ptrCamera
        );
        edge->setVertex(0, pose);
        edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    m_T_cr = SE3(
            pose->estimate().rotation(),
            pose->estimate().translation()
    );
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if(m_numInliers < m_min_inliers)
    {
        cerr << "reject because inlier is too small: " << m_numInliers << endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = m_T_cr.log();
    if(d.norm() > 10.0)
    {
        cerr << "reject because motion is too large: " << d.norm() << endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = m_T_cr.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if(rot.norm() > m_key_frame_min_rot || trans.norm() > m_key_frame_min_trans)
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout << "adding a key-frame" << endl;
    m_map->insertKeyFrame(m_currFrame);
    m_refFrame = m_currFrame;
}

void VisualOdometry::addMapPoints()
{
    if(m_map->m_mapPoints.empty())
    {
        // first key-frame, add all 3d points into map
        int num_kpts = m_currKeypoints.size();
        for(int i=0; i<num_kpts; i++)
        {
            double d = m_currFrame->findDepth(m_currKeypoints[i]);
            if(d < 0)
                continue;
            Vector3d p_w = m_currFrame->m_ptrCamera->pixel2world(
                    Vector2d(m_currKeypoints[i].pt.x, m_currKeypoints[i].pt.y ), m_currFrame->m_T_cw, d
            );
            Vector3d n = p_w - m_currFrame->getCamCenter();
            n.normalize();
            MapPoint::Ptr mp = MapPoint::createMapPoint(p_w, n, m_currDescriptors.row(i).clone());
            m_map->insertMapPoint(mp);
        }
    }
    else
    {
        int num_kpts = m_currKeypoints.size();
        vector<bool> matched(num_kpts, false);
        for(auto m : m_matches)
            matched[m.trainIdx] = true;
        for(int i=0; i<num_kpts; i++)
        {
            if(matched[i] == true)
                continue;
            double d = m_currFrame->findDepth(m_currKeypoints[i]);
            if(d < 0)
                continue;
            Vector3d p_w = m_currFrame->m_ptrCamera->pixel2world(
                    Vector2d(m_currKeypoints[i].pt.x, m_currKeypoints[i].pt.y ), m_currFrame->m_T_cw, d
            );
            Vector3d n = p_w - m_currFrame->getCamCenter();
            n.normalize();
            MapPoint::Ptr mp = MapPoint::createMapPoint(p_w, n, m_currDescriptors.row(i).clone());
            m_map->insertMapPoint(mp);
        }
    }
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points
    for(auto iter = m_map->m_mapPoints.begin(); iter != m_map->m_mapPoints.end();)
    {
        if(!m_currFrame->isInFrame(iter->second->m_position))
        {
            iter = m_map->m_mapPoints.erase(iter);
            continue;
        }
        float match_ratio = float(iter->second->m_matchedTimes) / iter->second->m_observedTimes;
        if(match_ratio < m_map_point_match_ratio)
        {
            iter = m_map->m_mapPoints.erase(iter);
            continue;
        }
        double angle = getViewAngle(m_currFrame, iter->second);
        if(angle > M_PI/6.0)
        {
            iter = m_map->m_mapPoints.erase(iter);
            continue;
        }
        iter++;
    }
    if(m_numInliers < 200)
        addMapPoints();
    else if(m_numInliers > 500)
        m_map_point_match_ratio += 0.1;
    else
        m_map_point_match_ratio = Config::get<double>("map_point_match_ratio");
    cout <<"map points: " << m_map->m_mapPoints.size() << endl;
}

double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point)
{
    Vector3d n = point->m_position - frame->getCamCenter();
    n.normalize();
    return acos(n.transpose() * point->m_norm);
}

}
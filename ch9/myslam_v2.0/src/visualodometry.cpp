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
            addKeyFrame();
            // extract features from first frame
            extractKeyPoints();
            computeDescriptors();
            updateRefFrame();
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
                m_currFrame->m_T_cw = m_T_cr * m_refFrame->m_T_cw;  // T_cw = T_cr*T_rw
                m_numLost = 0;
                updateRefFrame();
                if(checkKeyFrame() == true) // is a key-frame
                {
                    addKeyFrame();
                }
            }
            else // bad estimation due to various reasons
            {
                m_currFrame->m_T_cw = m_T_cr;
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

void VisualOdometry::updateRefFrame()
{
    m_refKeypoints.assign(m_currKeypoints.begin(), m_currKeypoints.end());
    m_refDescriptors = m_currDescriptors.clone();
    m_refFrame = m_currFrame;
}

void VisualOdometry::featureMatching()
{
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
        double d = m_refFrame->findDepth(m_refKeypoints[m.queryIdx]);
        if(d > 0)
        {
            Vector3d p_c = m_refFrame->m_ptrCamera->pixel2camera(
                    Vector2d(m_refKeypoints[m.queryIdx].pt.x, m_refKeypoints[m.queryIdx].pt.y), d
            );
            pts3d.push_back(Point3f(p_c[0], p_c[1], p_c[2]));
            pts2d.push_back(m_currKeypoints[m.trainIdx].pt);
        }
    }

    Mat K = m_currFrame->m_ptrCamera->getIntrinsic();
    Mat rvec, tvec, inliers;
    solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
    m_numInliers = inliers.rows;
    cout << "pnp inliers: " << m_numInliers << endl;
    m_T_cr = SE3(
            SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
            Vector3d(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );
//    cout << "T: " << m_T_cr << endl;

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
//    cout << "T: " << m_T_cr << endl;
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
    if(d.norm() > 5.0)
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
}

}
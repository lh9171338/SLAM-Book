#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat& img_1,
        const Mat& img_2,
        vector<KeyPoint>& keypoints_1,
        vector<KeyPoint>& keypoints_2,
        vector<DMatch>& matches,
        int match_type = 0);

void bundleAdjustment(
        vector<Point3f> points_3d,
        vector<Point2f> points_2d,
        Mat& K,
        Mat& R,
        Mat& t,
        bool useRt = true);

// 像素坐标转相机归一化坐标
Point3d pixel2cam(const Point2d& p, const Mat& K);

double reprojectError(
        vector<Point3f> points_3d,
        vector<Point2f> points_2d,
        Mat& K,
        Mat& R,
        Mat& t
        );


int main(int argc, char** argv)
{
    //-- 读取图像
    Mat img_1 = imread("../1.png", -1);
    Mat img_2 = imread("../2.png", -1);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches, 1);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat d1 = imread("../1_depth.png", -1);       // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    double depth_scale = 5000.0;
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for(DMatch m:matches)
    {
        ushort d = d1.at<unsigned short>(int(keypoints_1[m.queryIdx].pt.y), int(keypoints_1[m.queryIdx].pt.x));
        if(d == 0)   // bad depth
            continue;
        float dd = d / depth_scale;
        Point3d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x*dd, p1.y*dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t.t() << endl;
    double error = 0;
    error = reprojectError(pts_3d, pts_2d, K, R, t);
    cout << "error=" << error << endl;

    bundleAdjustment(pts_3d, pts_2d, K, R, t, true);
    error = reprojectError(pts_3d, pts_2d, K, R, t);
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t.t() << endl;
    cout << "error="  << error << endl;
}

void find_feature_matches(
        const Mat& img_1,
        const Mat& img_2,
        vector<KeyPoint>& keypoints_1,
        vector<KeyPoint>& keypoints_2,
        vector<DMatch>& matches,
        int match_type)
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    //-- 第一步:检测特征点
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的描述子进行匹配
    double thresh = 30.0;
    double alpha = 0.6;
    vector<DMatch> all_matches;
    matches.clear();
    if(match_type == 0)
    {
        vector<vector<DMatch>> matchesList;
        matcher->knnMatch(descriptors_1, descriptors_2, matchesList, 2);
        for(int i = 0; i < matchesList.size(); i++)
        {
            all_matches.push_back(matchesList[i][0]);
            if(matchesList[i][0].distance <= matchesList[i][1].distance * alpha)
                matches.push_back(matchesList[i][0]);
        }
    }
    else
    {
        matcher->match(descriptors_1, descriptors_2, all_matches);
        double min_dist=10000, max_dist=0;
        //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
        for(int i = 0; i < descriptors_1.rows; i++)
        {
            double dist = all_matches[i].distance;
            if(dist < min_dist)
                min_dist = dist;
            if(dist > max_dist)
                max_dist = dist;
        }
        printf("-- Max dist : %f \n", max_dist);
        printf("-- Min dist : %f \n", min_dist);

        //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        for(int i = 0; i < descriptors_1.rows; i++)
        {
            if(all_matches[i].distance <= max(2*min_dist, thresh))
                matches.push_back(all_matches[i]);
        }
    }
}


Point3d pixel2cam(const Point2d& p, const Mat& K)
{
    Mat p_uv = Mat(Vec3d(p.x, p.y, 1.0));
    Mat p_cn = K.inv() * p_uv;
    return Point3d(p_cn.at<double>(0, 0), p_cn.at<double>(1, 0), p_cn.at<double>(2, 0));
}

double reprojectError(
        vector<Point3f> points_3d,
        vector<Point2f> points_2d,
        Mat& K,
        Mat& R,
        Mat& t)
{
    double error = 0;
    int num_pts = points_3d.size();
    for(int i=0;i<num_pts;i++)
    {
        Mat pc1(Vec3d(points_3d[i].x, points_3d[i].y, points_3d[i].z));
        Mat pc2 = R * pc1 + t;
        pc2 /= pc2.at<double>(2, 0);
        Mat puv_ = K * pc2;
        Vec2d puv(points_2d[i].x, points_2d[i].y);
        Vec2d puv2(puv_.at<double>(0, 0), puv_.at<double>(1, 0));
        error += norm(puv, puv2, NORM_L2);
    }
    error /= num_pts;
    return error;
}

void bundleAdjustment (
        vector<Point3f> points_3d,
        vector<Point2f> points_2d,
        Mat& K,
        Mat& R,
        Mat& t,
        bool useRt)
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3>> BlockSolver;
    typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> LinearSolver;

    // 梯度下降方法，从GN, LM, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
//    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(
//            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
//    g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(
//            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    Eigen::Matrix3d R_mat;
    Eigen::Vector3d t_vec;
    if(useRt)
    {
        R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
        t_vec = Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
    }
    else
    {
        R_mat = Eigen::Matrix3d::Identity();
        t_vec = Eigen::Vector3d(0, 0, 0);
    }

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, t_vec));
    optimizer.addVertex(pose);

    int index = 1;
    for(Point3f p:points_3d)   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex(point);
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters(
            K.at<double>(0, 0), Eigen::Vector2d (K.at<double>(0, 2), K.at<double>(1, 2)), 0
    );
    camera->setId(0);
    optimizer.addParameter(camera);

    // edges
    index = 1;
    for(Point2f p:points_2d)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index++)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    Eigen::Matrix4d T = Eigen::Isometry3d(pose->estimate()).matrix();
    R = (Mat_<double>(3, 3) <<
            T(0, 0), T(0, 1), T(0, 2),
            T(1, 0), T(1, 1), T(1, 2),
            T(2, 0), T(2, 1), T(2, 2)
    );
    t = (Mat_<double>(3, 1) <<
           T(0, 3), T(1, 3), T(2, 3)
    );
}

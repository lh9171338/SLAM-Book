#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat& img1,
        const Mat& img2,
        vector<KeyPoint>& keypoints1,
        vector<KeyPoint>& keypoints2,
        vector<DMatch>& matches,
        int match_type = 0);

void pose_estimation_3d3d(
        vector<Point3f>& pts1,
        vector<Point3f>& pts2,
        Mat& R,
        Mat& t);

void bundleAdjustment(
        vector<Point3f> pts1,
        vector<Point3f> pts2,
        Mat& R,
        Mat& t,
        bool useRt = true);

// 像素坐标转相机归一化坐标
Point3d pixel2cam(const Point2d& p, const Mat& K);

double reprojectError(
        vector<Point3f> pts1,
        vector<Point3f> pts2,
        Mat& R,
        Mat& t);

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
protected:
    Eigen::Vector3d _point;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d& point) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map(_point);
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = pose->estimate().map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = -z;
        _jacobianOplusXi(0, 2) = y;
        _jacobianOplusXi(0, 3) = -1;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = 0;

        _jacobianOplusXi(1, 0) = z;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(1, 2) = -x;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -1;
        _jacobianOplusXi(1, 5) = 0;

        _jacobianOplusXi(2, 0) = -y;
        _jacobianOplusXi(2, 1) = x;
        _jacobianOplusXi(2, 2) = 0;
        _jacobianOplusXi(2, 3) = 0;
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = -1;
    }
    virtual bool read(istream& /*is*/)
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
    virtual bool write(ostream& /*os*/) const
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
};

int main(int argc, char** argv)
{
    //-- 读取图像
    Mat img1 = imread("../1.png", -1);
    Mat img2 = imread("../2.png", -1);

    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches, 1);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat depth1 = imread("../1_depth.png", -1);       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread("../2_depth.png", -1);       // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1, pts2;
    double depth_scale = 5000.0;
    for(DMatch m:matches)
    {
        ushort d1 = depth1.at<unsigned short>(int(keypoints1[m.queryIdx].pt.y), int(keypoints1[m.queryIdx].pt.x));
        ushort d2 = depth2.at<unsigned short>(int(keypoints2[m.trainIdx].pt.y), int(keypoints2[m.trainIdx].pt.x));
        if(d1==0 || d2==0)   // bad depth
            continue;
        Point3d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
        Point3d p2 = pixel2cam(keypoints2[m.trainIdx].pt, K);
        float dd1 = float(d1) / depth_scale;
        float dd2 = float(d2) / depth_scale;
        pts1.push_back(Point3f(p1.x*dd1, p1.y*dd1, dd1));
        pts2.push_back(Point3f(p2.x*dd2, p2.y*dd2, dd2));
    }
    cout << "3d-3d pairs: " << pts1.size() << endl;

    Mat R, t;
    double error;
    pose_estimation_3d3d(pts1, pts2, R, t);
    error = reprojectError(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t.t() << endl;
    cout << "error ="  << error << endl << endl;

    bundleAdjustment(pts1, pts2, R, t, false);
    cout << "BA results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t.t() << endl;
    cout << "error ="  << error << endl << endl;
    // verify p2 = R*p1 + t
    for(int i=0; i<5; i++)
    {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout <<"(R*p1+t) = "<< (R * Mat(Point3d(pts1[i])) + t).t() << endl;
    }
}

void find_feature_matches(
        const Mat& img1,
        const Mat& img2,
        vector<KeyPoint>& keypoints1,
        vector<KeyPoint>& keypoints2,
        vector<DMatch>& matches,
        int match_type)
{
    //-- 初始化
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    //-- 第一步:检测特征点
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    //-- 第二步:根据描述子
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    //-- 第三步:对两幅图像中的描述子进行匹配
    double thresh = 30.0;
    double alpha = 0.6;
    vector<DMatch> all_matches;
    matches.clear();
    if(match_type == 0)
    {
        vector<vector<DMatch>> matchesList;
        matcher->knnMatch(descriptors1, descriptors2, matchesList, 2);
        for(int i = 0; i < matchesList.size(); i++)
        {
            all_matches.push_back(matchesList[i][0]);
            if(matchesList[i][0].distance <= matchesList[i][1].distance * alpha)
                matches.push_back(matchesList[i][0]);
        }
    }
    else
    {
        matcher->match(descriptors1, descriptors2, all_matches);
        double min_dist=10000, max_dist=0;
        //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
        for(int i = 0; i < descriptors1.rows; i++)
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
        for(int i = 0; i < descriptors1.rows; i++)
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
        vector<Point3f> pts1,
        vector<Point3f> pts2,
        Mat& R,
        Mat& t)
{
    double error = 0;
    int num_pts = pts1.size();
    for(int i=0;i<num_pts;i++)
    {
        Mat p1(Vec3d(pts1[i].x, pts1[i].y, pts1[i].z));
        Mat p2(Vec3d(pts2[i].x, pts2[i].y, pts2[i].z));
        Mat p2_ = R * p1 + t;
        error += norm(p2_, p2, NORM_L2);
    }
    error /= num_pts;
    return error;
}

void pose_estimation_3d3d(
        vector<Point3f>& pts1,
        vector<Point3f>& pts2,
        Mat& R,
        Mat& t)
{
    // 计算质心
    Point3d p1, p2;
    int num_pts = pts1.size();
    for(int i=0;i<num_pts;i++)
    {
        p1 += Point3d(pts1[i]);
        p2 += Point3d(pts2[i]);
    }
    p1 = Point3d(Vec3d(p1)/num_pts);
    p2 = Point3d(Vec3d(p2)/num_pts);
    // 去质心
    vector<Point3d> q1(num_pts), q2(num_pts);
    for(int i=0;i<num_pts;i++)
    {
        q1[i] = Point3d(pts1[i]) - p1;
        q2[i] = Point3d(pts2[i]) - p2;
    }

    // 计算 q2*q1^T
    Mat W = Mat(3, 3, CV_64FC1, Scalar::all(0));
    for(int i=0;i<num_pts;i++)
    {
        W += Mat(q2[i]) * Mat(q1[i]).t();
    }

    // SVD
    Mat S, U, VT;
    SVDecomp(W, S, U, VT, SVD::FULL_UV);

    // 计算R和t
    R = U * VT;
    t = Mat(p2) - R * Mat(p1);
}

void bundleAdjustment(
        vector<Point3f> pts1,
        vector<Point3f> pts2,
        Mat& R,
        Mat& t,
        bool useRt)
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > BlockSolver;
    typedef g2o::LinearSolverEigen<BlockSolver::PoseMatrixType> LinearSolver;

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
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

    // edges
    int num_pts = pts1.size();
    for(int i=0; i<num_pts; i++)
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
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

struct ReprojectCost
{
private:
    Point3d p1_, p2_;

public:
    ReprojectCost(Point3d p1, Point3d p2) : p1_(p1), p2_(p2){}
    template<typename T>
    bool operator()(const T* const extrinsic, T* residuals) const
    {
        const T* r = extrinsic;
        const T* t = &extrinsic[3];

        T p1[3] = {T(p1_.x), T(p1_.y), T(p1_.z)};
        T p1_trans[3];
        ceres::AngleAxisRotatePoint(r, p1, p1_trans);
        p1_trans[0] += t[0];
        p1_trans[1] += t[1];
        p1_trans[2] += t[2];

        residuals[0] = p1_trans[0] - T(p2_.x);
        residuals[1] = p1_trans[1] - T(p2_.y);
        residuals[2] = p1_trans[2] - T(p2_.z);

        return true;
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
    Mat extrinsic(6, 1, CV_64FC1);
    Mat R_(3, 3, CV_64FC1);
    Mat t_(3, 1, CV_64FC1);
    if(useRt)
    {
        R_= R.clone();
        t_ = t.clone();
    }
    else
    {
        R_ = Mat::eye(3, 3, CV_64FC1);
        t_ = Mat::zeros(3, 1, CV_64FC1);
    }
    Mat r(3, 1, CV_64FC1);
    Rodrigues(R_, r);
    r.copyTo(extrinsic.rowRange(0, 3));
    t_.copyTo(extrinsic.rowRange(3, 6));

    ceres::Problem problem;
    problem.AddParameterBlock(extrinsic.ptr<double>(), 6);
    int num_pts = pts1.size();
    for(int i=0;i<num_pts;i++)
    {
        Point3d p1 = Point3d(pts1[i]);
        Point3d p2 = Point3d(pts2[i]);

        ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<ReprojectCost, 3, 6>(new ReprojectCost(p1, p2));
        problem.AddResidualBlock(
                cost,
                NULL,
                extrinsic.ptr<double>());
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.preconditioner_type = ceres::JACOBI;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if(!summary.IsSolutionUsable())
    {
        cout << "Bundle Adjustment failed" << endl;
    }
    else
    {
        cout << summary.BriefReport() << endl;
        cout << "Bundle Adjustment statistics (approximated RMSE):" << endl;
        cout << " Initial RMSE: " << sqrt(summary.initial_cost / summary.num_residuals) << endl;
        cout << " Final RMSE: " << sqrt(summary.final_cost / summary.num_residuals) << endl;
        cout << " Time (s): " << summary.total_time_in_seconds << endl;
    }

    r = extrinsic.rowRange(0, 3).clone();
    t = extrinsic.rowRange(3, 6).clone();
    Rodrigues(r, R);
}
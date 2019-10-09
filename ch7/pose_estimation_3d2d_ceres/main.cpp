#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
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


struct ReprojectCost
{
private:
    Point2d p2_;

public:
    ReprojectCost(Point2d p2) : p2_(p2) {}
    template<typename T>
    bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const p1, T* residuals) const
    {
        const T* r = extrinsic;
        const T* t = &extrinsic[3];
        T p[3];
        ceres::AngleAxisRotatePoint(r, p1, p);
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];

        const T x = p[0] / p[2];
        const T y = p[1] / p[2];
        const T fx = intrinsic[0];
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(p2_.x);
        residuals[1] = v - T(p2_.y);

        return true;
    }
};



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
    Mat intrinsic = (Mat_<double>(4, 1) <<
            K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)
    );

    ceres::Problem problem;
    problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // [fx, fy, cx, cy]
    problem.SetParameterBlockConstant(intrinsic.ptr<double>());
    int num_pts = points_3d.size();
    vector<Point3d> pts_3d(num_pts);
    for(int i=0;i<num_pts;i++)
    {
        pts_3d[i] = Point3d(points_3d[i]);
//        problem.AddParameterBlock(&(pts_3d[i].x), 3);
//        problem.SetParameterBlockConstant(&(pts_3d[i].x));

        ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(Point2d(points_2d[i])));
        problem.AddResidualBlock(
                cost,
                NULL,
                intrinsic.ptr<double>(),
                extrinsic.ptr<double>(),
                &(pts_3d[i].x));
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
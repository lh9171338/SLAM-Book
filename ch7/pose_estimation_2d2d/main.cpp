#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches(
        const Mat& img_1,
        const Mat& img_2,
        vector<KeyPoint>& keypoints_1,
        vector<KeyPoint>& keypoints_2,
        vector<DMatch>& matches,
        int match_type = 0);

void pose_estimation_2d2d(
        vector<KeyPoint> keypoints_1,
        vector<KeyPoint> keypoints_2,
        vector<DMatch> matches,
        Mat& K,
        Mat& R,
        Mat& t);

void triangulation(
        vector<KeyPoint>& keypoint_1,
        vector<KeyPoint>& keypoint_2,
        vector< DMatch >& matches,
        Mat& K,
        Mat& R,
        Mat& t,
        vector<Point3d>& points);

// 像素坐标转相机归一化坐标
Point3d pixel2cam(const Point2d& p, const Mat& K);

int main(int argc, char** argv)
{
    //-- 读取图像
    Mat img_1 = imread("../1.png");
    Mat img_2 = imread("../2.png");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    //-- 估计两张图像间运动
    Mat R, t;
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, K, R, t);

    //-- 验证对极约束
    Mat t_x = (Mat_<double>(3, 3) <<
                                  0,                      -t.at<double>(2, 0),     t.at<double>(1, 0),
            t.at<double>(2, 0),      0,                      -t.at<double> (0, 0),
            -t.at<double>(1, 0),     t.at<double>(0, 0),      0);
    double error = 0;
    for(DMatch m: matches)
    {
        Point3d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point3d pt2 = pixel2cam(keypoints_2[ m.trainIdx ].pt, K);
        Mat d = Mat(pt2).t() * t_x * R * Mat(pt1);
        error += abs(d.at<double>(0, 0));
    }
    error /= matches.size();
    cout << "average epipolar constraint error= " << error << endl;

    //-- 三角化
    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, K, R, t, points);

    //-- 验证三角化点与特征点的重投影关系
    for(int i=0; i<5; i++)
    {
        Point3d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx ].pt, K);
        Point3d pt1_cam_3d(points[i].x/points[i].z, points[i].y/points[i].z, 1);

        cout << "point in the first camera frame: " << pt1_cam << endl;
        cout << "point projected from 3D " << pt1_cam_3d << endl;

        // 第二个图
        Point3d pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        Mat pt2_trans = R * Mat(points[i]) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);
        Point3d pt2_cam_3d(pt2_trans.at<double>(0, 0), pt2_trans.at<double>(1, 0), pt2_trans.at<double>(2, 0));
        cout << "point in the second camera frame: " << pt2_cam << endl;
        cout << "point reprojected from second frame: " << pt2_trans.t() << endl;
        cout << endl;
    }

    return 0;
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


void pose_estimation_2d2d(
        vector<KeyPoint> keypoints_1,
        vector<KeyPoint> keypoints_2,
        vector< DMatch > matches,
        Mat& K,
        Mat& R,
        Mat& t)
{
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
    Point2d principal_point(325.1, 249.7);	//相机光心, TUM dataset标定值
    double focal_length = 521;			//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

    //-- 验证E=t^R*scale
    Mat t_x = (Mat_<double>(3, 3) <<
                                  0,                      -t.at<double>(2, 0),     t.at<double>(1, 0),
            t.at<double>(2, 0),      0,                      -t.at<double> (0, 0),
            -t.at<double>(1, 0),     t.at<double>(0, 0),      0);
    Mat t_xR = t_x * R;
    Mat scale(3, 3, CV_64FC1);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            scale.at<double>(i, j) = essential_matrix.at<double>(i, j) / t_xR.at<double>(i, j);
        }
    }

    cout << "t^R/E" << endl << scale << endl;
}

void triangulation(
        vector<KeyPoint>& keypoint_1,
        vector<KeyPoint>& keypoint_2,
        vector<DMatch>& matches,
        Mat& K,
        Mat& R,
        Mat& t,
        vector<Point3d>& points)
{
    Mat T1 = (Mat_<float>(3, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    vector<Point2f> pts_1, pts_2;
    for(DMatch m:matches)
    {
        // 将像素坐标转换至相机坐标
        Point3d p1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        Point3d p2 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        pts_1.push_back(Point2f(p1.x, p1.y));
        pts_2.push_back(Point2f(p2.x, p2.y));
    }

    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // 转换成非齐次坐标
    for(int i=0; i<pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}
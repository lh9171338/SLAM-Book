#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <chrono>

using namespace std;
using namespace cv;


int main ( int argc, char** argv )
{
    //-- 读取图像
    Mat img_1 = imread("../1.png");
    Mat img_2 = imread("../2.png");
    int feature_type = 0;
    bool match_type = 0;
    double thresh = 30.0;
    double alpha = 0.6;


    //-- 初始化
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;
    Ptr<DescriptorMatcher> matcher;

    if(feature_type == 0)
    {
        // ORB
        detector = ORB::create();
        descriptor = ORB::create();
        matcher  = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    }
    else if(feature_type == 1)
    {
        // SIFT
         detector = xfeatures2d::SIFT::create();
         descriptor = xfeatures2d::SIFT::create();
         matcher  = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    }
    else
    {
        // SURF
        detector = xfeatures2d::SURF::create();
        descriptor = xfeatures2d::SURF::create();
        matcher  = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    //-- 第一步:检测特征点
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:计算描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "time cost = " << time_used.count() << " seconds. " << endl;

    //-- 第三步:对两幅图像中的描述子进行匹配

    vector<DMatch> all_matches;
    vector<DMatch> good_matches;
    if(match_type == 0)
    {
        vector<vector<DMatch>> matchesList;
        matcher->knnMatch(descriptors_1, descriptors_2, matchesList, 2);
        for(int i = 0; i < matches.size(); i++)
        {
            all_matches.push_back(matchesList[i][0]);
            if(matchesList[i][0].distance <= matchesList[i][1].distance * alpha)
                good_matches.push_back(matchesList[i][0]);
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
                good_matches.push_back(all_matches[i]);
        }
    }

    //-- 第四步:绘制匹配结果
    Mat img_allmatch;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, all_matches, img_allmatch);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    namedWindow("所有匹配点对", 0);
    imshow("所有匹配点对", img_allmatch);
    namedWindow("优化后匹配点对", 0);
    imshow("优化后匹配点对", img_goodmatch);
    printf("-- matches : %lu \n", all_matches.size());
    printf("-- good_matches : %lu \n", good_matches.size());
    waitKey(0);

    return 0;
}

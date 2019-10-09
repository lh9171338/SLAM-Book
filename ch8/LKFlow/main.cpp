#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main( int argc, char** argv )
{
    string path_to_dataset = "../../data/rgbd_dataset_freiburg1_desk";
    string associate_file = path_to_dataset + "/associate.txt";
    int image_num = 100;

    ifstream fin(associate_file);
    if(!fin)
    {
        cerr << "I cann't find associate.txt!" << endl;
        return 1;
    }

    string rgb_file, depth_file, time_rgb, time_depth;
    vector<Point2f> keypoints, last_keypoints;
    Mat color, depth, last_color;
    bool init_flag = false;
    for(int index=0;index<image_num;index++)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = imread(path_to_dataset + "/" + rgb_file);
        depth = imread(path_to_dataset + "/" + depth_file, -1 );
        if(color.data == nullptr || depth.data == nullptr)
            continue;
        if(!init_flag)
        {
            init_flag = true;
            // 对第一帧提取FAST特征点
            vector<KeyPoint> kps;
            Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
            detector->detect(color, kps);
            for(auto kp:kps)
                keypoints.push_back(kp.pt);
        }
        else
        {
            // 对其他帧用LK跟踪特征点
            vector<unsigned char> status;
            vector<float> error;
            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            calcOpticalFlowPyrLK(last_color, color, last_keypoints, keypoints, status, error);
            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
            cout << "LK Flow " << index + 1 << ":" << endl;
            cout << "\tuse time：" << time_used.count() << " seconds." << endl;
            // 把跟丢的点删掉
            int i = 0;
            for(auto iter=keypoints.begin();iter!=keypoints.end();i++)
            {
                if(status[i] == 0)
                    iter = keypoints.erase(iter);
                else
                    iter++;
            }
            cout << "\ttracked keypoints: " << keypoints.size() << endl;
            if(keypoints.size() == 0)
            {
                cout << "all keypoints are lost." << endl;
                break;
            }
            // 画出 keypoints
            Mat img_show = color.clone();
            for(auto kp:keypoints)
                circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
            namedWindow("corners", 0);
            imshow("corners", img_show);
            waitKey(10);
        }
        last_keypoints = keypoints;
        last_color = color;
    }

    return 0;
}

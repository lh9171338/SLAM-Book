#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "DBoW3/DBoW3.h"

using namespace cv;
using namespace std;

/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main(int argc, char** argv)
{
    // read the image
    string dataset_dir = argv[1];
    cout << "reading images... " << endl;
    vector<Mat> images;
    for(int i=0; i<10; i++)
    {
        string path = dataset_dir+"/"+to_string(i+1)+".png";
        images.push_back(imread(path));
    }
    // detect ORB features
    cout << "detecting ORB features ... " << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for(Mat& image:images)
    {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // create vocabulary
    cout << "creating vocabulary ... " << endl;
    DBoW3::Vocabulary vocab(10, 5, DBoW3::TF_IDF, DBoW3::L2_NORM);
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    vocab.save("../../vocabulary.yml");
    cout << "done" << endl;

    return 0;
}
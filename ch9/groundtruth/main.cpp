// -------------- test the visual odometry -------------
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    string filename = "../../data/rgbd_dataset_freiburg1_desk/groundtruth.txt";
    string savefile = "../groundtruth.png";
    int time = 10;
    bool init_flag = false;

    ifstream fin(filename);
    if(!fin)
    {
        cout << "fail to open file" << endl;
        return -1;
    }

    viz::Viz3d vis("Groundtruth");
    viz::WCoordinateSystem world_frame(0.5);
    viz::WCameraPosition camera(0.2);
    vis.showWidget("World", world_frame);
    vis.showWidget("Camera", camera);
    vector<Affine3d> path;

    string str;
    while(getline(fin, str))
    {
        double data[8];
        if(sscanf(str.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
                &data[0], &data[1], &data[2], &data[3],
                &data[4], &data[5], &data[6], &data[7]) > 0)
        {
            Eigen::Vector3d t(data[1], data[2], data[3]); // tx, ty, tz
            Eigen::Quaterniond q(data[7], data[4], data[5], data[6]); // qw, qx, qy, qz
            Eigen::Isometry3d T(q);
            T.pretranslate(t);
//            cout << "T: " << T.matrix() << endl;
            T.inverse();
            Affine3d pose(
                    Affine3d::Mat3(
                            T(0,0), T(0,1), T(0,2),
                            T(1,0), T(1,1), T(1,2),
                            T(2,0), T(2,1), T(2,2)
                    ),
                    Affine3d::Vec3(
                            T(0,3), T(1,3), T(2,3)
                    )
            );
            path.push_back(pose);
            viz::WTrajectory trajectory(path, viz::WTrajectory::PATH, 1.0, viz::Color::blue());
            vis.showWidget("trajectory", trajectory);
            if(!init_flag)
            {
                init_flag = true;
                vis.setWidgetPose("World", pose);
            }
            vis.setWidgetPose("Camera", pose);
            vis.spinOnce(time, false);
        }
    }
    vis.saveScreenshot(savefile);

    return 0;
}
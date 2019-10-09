// -------------- test the visual odometry -------------
#include <iostream>
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "visualodometry.h"

int main(int argc, char** argv)
{
    string paramFile = "../config/default.yaml";

    myslam::Config::setParameterFile(paramFile);
    string dataset_dir = myslam::Config::get<string>("dataset_dir");
    string savefile = myslam::Config::get<string>("savefile");
    cout << "dataset: " << dataset_dir << endl;
    ifstream fin(dataset_dir+"/associate.txt");
    if(!fin)
    {
        cout << "please generate the associate file called associate.txt!" << endl;
        return -1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while(!fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir+"/"+rgb_file);
        depth_files.push_back(dataset_dir+"/"+depth_file);

        if(fin.good() == false)
            break;
    }

    // visualization
    viz::Viz3d vis("Visual Odometry");
    viz::WCoordinateSystem world_coor(0.5);
    viz::WCameraPosition camera_coor(0.5);
    Point3d cam_pos(0,-1.0,-0.5), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    Affine3d cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    // vo
    int num_files = rgb_files.size();
    cout << "read total " << num_files << " entries" <<endl;
    myslam::Camera::Ptr camera(new myslam::Camera());
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry());
    vector<Affine3d> path;
    for(int i=0; i<num_files; i++)
    {
        Mat color = imread(rgb_files[i]);
        Mat depth = imread(depth_files[i], -1);
        if(color.data == nullptr || depth.data == nullptr)
            break;
        myslam::Frame::Ptr ptrFrame = myslam::Frame::createFrame(rgb_times[i], camera, color, depth);
        boost::timer timer;
        vo->addFrame(ptrFrame);
        cout << "VO costs time: " << timer.elapsed() << endl;

        if(vo->m_state == myslam::VisualOdometry::LOST)
            break;
        SE3 Twc = ptrFrame->m_T_cw.inverse();

        // show the map and the camera pose
        cv::Affine3d pose(
                cv::Affine3d::Mat3(
                        Twc.rotation_matrix()(0,0), Twc.rotation_matrix()(0,1), Twc.rotation_matrix()(0,2),
                        Twc.rotation_matrix()(1,0), Twc.rotation_matrix()(1,1), Twc.rotation_matrix()(1,2),
                        Twc.rotation_matrix()(2,0), Twc.rotation_matrix()(2,1), Twc.rotation_matrix()(2,2)
                ),
                cv::Affine3d::Vec3(
                        Twc.translation()(0,0), Twc.translation()(1,0), Twc.translation()(2,0)
                )
        );

        imshow("image", color);
        waitKey(10);
        path.push_back(pose);
        viz::WTrajectory trajectory(path, viz::WTrajectory::PATH, 1.0, viz::Color::blue());
        vis.showWidget("trajectory", trajectory);
        vis.setWidgetPose("Camera", pose);
        vis.spinOnce(1, false);
    }
    vis.saveScreenshot(savefile);

    return 0;
}
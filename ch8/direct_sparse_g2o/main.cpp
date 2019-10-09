#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace cv;

/********************************************
 * 本节演示了RGBD上的稀疏直接法
 ********************************************/

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement
{
    Measurement(Eigen::Vector3d point_world, double gray_scale)
        : _point_world(point_world), _gray_scale(gray_scale) {}
    Eigen::Vector3d _point_world;
    double _gray_scale;
};

inline Eigen::Vector3d project2Dto3D(Eigen::Vector2d p_2d, double d, double fx, double fy, double cx, double cy, double scale)
{
    double u = p_2d[0];
    double v = p_2d[1];
    double z = d /scale;
    double x = z * (u - cx) / fx;
    double y = z * (v - cy) / fy;
    return Eigen::Vector3d(x, y, z);
}

inline Eigen::Vector2d project3Dto2D(Eigen::Vector3d p_3d, double fx, double fy, double cx, double cy)
{
    double x = p_3d[0];
    double y = p_3d[1];
    double z = p_3d[2];
    double u = fx * x/ z + cx;
    double v = fy * y/ z + cy;
    return Eigen::Vector2d(u, v);
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参； 输出：相机位姿
void poseEstimationDirect(
        vector<Measurement>& measurements,
        Mat& gray,
        Eigen::Matrix3d& K,
        Eigen::Isometry3d& T);


// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge< 1, double, g2o::VertexSE3Expmap>
{
private:
    Eigen::Vector3d _point_world;   // 3D point in world frame
    double _fx = 0, _fy = 0, _cx = 0, _cy = 0; // Camera intrinsics
    Mat _image;    // reference image

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect(){}

    EdgeSE3ProjectDirect(Eigen::Vector3d point_world, Eigen::Matrix3d K, Mat& image)
            : _point_world(point_world), _fx(K(0, 0)), _fy(K(1, 1)), _cx(K(0, 1)), _cy(K(0, 2)), _image(image){}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d point_trans = pose->estimate().map(_point_world);
        double x = point_trans[0] / point_trans[2];
        double y = point_trans[1] / point_trans[2];
        double u = _fx * x  + _cx;
        double v = _fy * y  + _cy;
        // check x,y is in the image
        if((u-4) < 0 || (u+4) > _image.cols || (v-4) < 0 || (v+4) > _image.rows)
        {
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
        else
        {
            _error(0, 0) = getPixelValue(u, v) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus()
    {
        if (level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d point_trans = pose->estimate().map(_point_world);   // q in book

        double invz = 1.0 / point_trans[2];
        double invz_2 = invz * invz;
        double x = point_trans[0] * invz;
        double y = point_trans[1] * invz;
        double u = _fx * x + _cx;
        double v = _fy * y + _cy;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai(0, 0) = - x * y * invz_2 * _fx;
        jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * _fx;
        jacobian_uv_ksai(0, 2) = -y * invz * _fx;
        jacobian_uv_ksai(0, 3) = invz * _fx;
        jacobian_uv_ksai(0, 4) = 0;
        jacobian_uv_ksai(0, 5) = -x * invz_2 * _fx;

        jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * _fx;
        jacobian_uv_ksai(1, 1) = x * y * invz_2 * _fy;
        jacobian_uv_ksai(1, 2) = x * invz * _fy;
        jacobian_uv_ksai(1, 3) = 0;
        jacobian_uv_ksai(1, 4) = invz * _fy;
        jacobian_uv_ksai(1, 5) = -y * invz_2 * _fy;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv(0, 0) = (getPixelValue(u+1, v) - getPixelValue (u-1, v)) / 2;
        jacobian_pixel_uv(0, 1) = (getPixelValue(u, v+1) - getPixelValue (u, v-1)) / 2;

        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read(istream& in){}
    virtual bool write(ostream& out) const {}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline double getPixelValue(double x, double y)
    {
        uchar* data = &_image.data[int(y) * _image.step + int(x)];
        double xx = x - floor(x);
        double yy = y - floor(y);
        return double(
                (1 - xx) * (1 - yy) * data[0] +
                xx * (1 - yy) * data[1] +
                (1 - xx) * yy * data[_image.step] +
                xx * yy * data[_image.step+1]);
    }
};

int main(int argc, char** argv)
{
    string path_to_dataset = "../../data/rgbd_dataset_freiburg1_desk";
    string associate_file = path_to_dataset + "/associate.txt";
    int image_num = 10;

    ifstream fin(associate_file);
    if(!fin)
    {
        cerr << "I cann't find associate.txt!" << endl;
        return 1;
    }

    string rgb_file, depth_file, time_rgb, time_depth;
    Mat color, prev_color, depth, gray;
    vector<Measurement> measurements;
    // 相机内参
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depth_scale = 1000.0;
    Eigen::Matrix3d K;
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    // 我们以第一个图像为参考，对后续图像和参考图像做直接法
    bool init_flag = false;
    srand((unsigned int) time(0));
    for(int index=0;index<image_num;index++)
    {
        cout << "*********** loop " << index+1 << " ************" << endl;
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = imread(path_to_dataset + "/" + rgb_file );
        depth = imread(path_to_dataset + "/" + depth_file, -1);
        if(color.data == nullptr || depth.data == nullptr)
            continue;
        cvtColor(color, gray, COLOR_BGR2GRAY);
        if(!init_flag)
        {
            init_flag = true;
            // 对第一帧提取FAST特征点
            vector<KeyPoint> keypoints;
            Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
            detector->detect(color, keypoints);
            for(auto kp:keypoints)
            {
                // 去掉邻近边缘处的点
                if(kp.pt.x < 20 || kp.pt.y < 20 || (kp.pt.x+20) > color.cols || (kp.pt.y+20) > color.rows)
                    continue;
                double d = (double)depth.at<ushort>(cvRound(kp.pt.y), cvRound(kp.pt.x));
                if(d == 0)
                    continue;
                Eigen::Vector2d p_2d(kp.pt.x, kp.pt.y);
                Eigen::Vector3d p_3d = project2Dto3D(p_2d, d, fx, fy, cx, cy, depth_scale);
                double grayscale = float(gray.at<uchar>(cvRound(kp.pt.y), cvRound(kp.pt.x)));
                measurements.push_back(Measurement(p_3d, grayscale));
            }
        }
        else
        {
            // 使用直接法计算相机运动
            poseEstimationDirect(measurements, gray, K, T);
            cout << "T=" << T.matrix() << endl;

            // plot the feature points
            Mat img_show(color.rows*2, color.cols, CV_8UC3);
            prev_color.copyTo(img_show(Rect(0, 0, color.cols, color.rows)));
            color.copyTo(img_show(Rect(0, color.rows, color.cols, color.rows)));
            for(auto m:measurements)
            {
                if(rand() > RAND_MAX/5)
                    continue;
                Eigen::Vector3d p1 = m._point_world;
                Eigen::Vector2d pixel_prev = project3Dto2D(p1, fx, fy, cx, cy);
                Eigen::Vector3d p2 = T * p1;
                Eigen::Vector2d pixel_now = project3Dto2D(p2, fx, fy, cx, cy);
                if(pixel_now(0, 0) < 0 || pixel_now(0, 0) >= color.cols || pixel_now(1, 0) < 0 || pixel_now(1, 0) >= color.rows)
                    continue;

                float b = 255 * float(rand()) / RAND_MAX;
                float g = 255 * float(rand()) / RAND_MAX;
                float r = 255 * float(rand()) / RAND_MAX;
                circle(img_show, Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 8, Scalar(b, g, r), 2);
                circle(img_show, Point2d(pixel_now(0, 0), pixel_now(1, 0)+color.rows), 8, Scalar(b, g, r), 2);
                line(img_show, Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), Point2d(pixel_now(0, 0), pixel_now(1, 0)+color.rows), Scalar(b, g, r), 1);
            }
            namedWindow("result", 0);
            imshow("result", img_show);
            waitKey(0);
        }
        prev_color = color.clone();
    }

    return 0;
}

void poseEstimationDirect(
        vector<Measurement>& measurements,
        Mat& gray,
        Eigen::Matrix3d& K,
        Eigen::Isometry3d& T)
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 1>> BlockSolver;
    typedef g2o::LinearSolverDense<BlockSolver::PoseMatrixType> LinearSolver;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // Vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(T.rotation(), T.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // Edge
    for(auto m: measurements)
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(m._point_world, K, gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(m._gray_scale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.optimize(20);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "direct method costs time: " << time_used.count() << " seconds." << endl;

    T = pose->estimate();
}
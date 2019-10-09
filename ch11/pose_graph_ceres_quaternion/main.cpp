#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


using namespace std;
using namespace cv;


/************************************************
 * 本程序演示如何用ceres进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用旋转和平移向量表达位姿图，节点和边的方式为自定义
 * **********************************************/
typedef Eigen::Matrix<double,6,6> Matrix6d;

struct Pose3d{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d m_t;
    Eigen::Quaterniond m_q;

    Pose3d() {}
    Pose3d(Eigen::Vector3d t, Eigen::Quaterniond q) : m_t(t), m_q(q) {}
};

struct Vertex{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int m_id;
    Pose3d m_pose;

    Vertex() {}
    Vertex(int id, Pose3d pose) : m_id(id), m_pose(pose) {}

    // The name of the data type in the g2o file format.
    static std::string name() {
        return "VERTEX_SE3:QUAT";
    }
};

struct Edge{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int m_id1;
    int m_id2;
    Pose3d m_pose;
    Matrix6d m_information;

    Edge() {}
    Edge(int id1, int id2, Pose3d pose, Matrix6d information)
        : m_id1(id1), m_id2(id2), m_pose(pose), m_information(information) {}

    // The name of the data type in the g2o file format.
    static std::string name() {
        return "EDGE_SE3:QUAT";
    }
};

struct PoseGraphCost
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Eigen::Vector3d m_tij;
    Eigen::Quaterniond m_qij;
    Matrix6d m_information;

public:
    PoseGraphCost(Eigen::Vector3d tij, Eigen::Quaterniond qij, Matrix6d information)
    : m_tij(tij), m_qij(qij), m_information(information) {}
    template<typename T>
    bool operator()(const T* const ti, const T* const qi, const T* const tj, const T* const qj, T* residuals) const
    {
        // Ti
        const Eigen::Matrix<T,3,1> m_ti(ti);
        const Eigen::Quaternion<T> m_qi(qi);

        // Tj
        const Eigen::Matrix<T,3,1> m_tj(tj);
        const Eigen::Quaternion<T> m_qj(qj);
        const Eigen::Transform<T,3,Eigen::Isometry> m_Tj;

        // Ti^{-1} Tj
        const Eigen::Quaternion<T> m_qi_inverse = m_qi.conjugate();
        const Eigen::Matrix<T,3,1> tij = m_qi_inverse * (m_tj - m_ti);
        const Eigen::Quaternion<T> qij = m_qi_inverse * m_qj;

        // residuals
        // [ position         ]   [ delta_t          ]
        // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
        const Eigen::Matrix<T,3,1> delta_t = tij - m_tij.template cast<T>();
        const Eigen::Quaternion<T> delta_q = m_qij.template cast<T>() * qij.conjugate();

        Eigen::Map<Eigen::Matrix<T, 6, 1>> m_residuals(residuals);
        m_residuals.template block<3,1>(0, 0) = delta_t;
        m_residuals.template block<3,1>(3, 0) = T(2.0) * delta_q.vec();

        // Scale the residuals by the measurement uncertainty.
        m_residuals.applyOnTheLeft(m_information.template cast<T>());

        return true;
    }
};


int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cerr << "Usage: pose_graph_ceres_quaternion sphere.g2o result.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if(!fin)
    {
        cerr << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }
    vector<Vertex> Vertexs;
    vector<Edge> Edges;
    int vertexCnt = 0;
    int edgeCnt = 0;
    while(!fin.eof())
    {
        string name;
        fin >> name;
        if(name == Vertex::name())
        {
            // 顶点
            vertexCnt++;
            Vertex v;
            fin >> v.m_id >> v.m_pose.m_t[0] >> v.m_pose.m_t[1] >> v.m_pose.m_t[2]
                >> v.m_pose.m_q.x() >> v.m_pose.m_q.y() >> v.m_pose.m_q.z() >> v.m_pose.m_q.w();
            Vertexs.push_back(v);
        }
        else if(name == Edge::name())
        {
            // 边
            edgeCnt++;

            Edge e;
            fin >> e.m_id1 >> e.m_id2 >> e.m_pose.m_t[0] >> e.m_pose.m_t[1] >> e.m_pose.m_t[2]
                >> e.m_pose.m_q.x() >> e.m_pose.m_q.y() >> e.m_pose.m_q.z() >> e.m_pose.m_q.w();

            Matrix6d information;
            for(int i=0; i<information.rows(); i++)
            {
                for(int j=i; j<information.cols(); j++)
                {
                    fin >> information(i,j);
                    if(i != j)
                        information(j,i) = information(i,j);
                }
            }
            e.m_information = information;
            Edges.push_back(e);
        }
        if(!fin.good())
            break;
    }
    fin.close();

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "prepare optimizing ..." << endl;
    ceres::Problem problem;
    ceres::LocalParameterization* quaternion_local_parameterization =
            new ceres::EigenQuaternionParameterization;

    for(int i=0;i<edgeCnt;i++)
    {
        Edge e = Edges[i];

        ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<PoseGraphCost, 6, 3, 4, 3, 4>(new PoseGraphCost(
                        e.m_pose.m_t, e.m_pose.m_q, e.m_information.llt().matrixL()));

        problem.AddResidualBlock(
                cost,
                NULL,
                Vertexs[e.m_id1].m_pose.m_t.data(),
                Vertexs[e.m_id1].m_pose.m_q.coeffs().data(),
                Vertexs[e.m_id2].m_pose.m_t.data(),
                Vertexs[e.m_id2].m_pose.m_q.coeffs().data()
        );

        problem.SetParameterization(Vertexs[e.m_id1].m_pose.m_q.coeffs().data(),
                                     quaternion_local_parameterization);
        problem.SetParameterization(Vertexs[e.m_id2].m_pose.m_q.coeffs().data(),
                                     quaternion_local_parameterization);
    }
    problem.SetParameterBlockConstant(Vertexs[0].m_pose.m_t.data());
    problem.SetParameterBlockConstant(Vertexs[0].m_pose.m_q.coeffs().data());

    cout <<"calling optimizing ..." << endl;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 20;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << "saving optimization results ..." << endl;
    // 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现
    // 伪装成 SE3 顶点和边，让 g2o_viewer 可以认出
    ofstream fout(argv[2]);
    for(int i=0;i<vertexCnt;i++)
    {
        Vertex v = Vertexs[i];
        fout << Vertex::name() << " " << v.m_id << " "
             << v.m_pose.m_t[0] << " " << v.m_pose.m_t[1] << " " << v.m_pose.m_t[2] << " "
             << v.m_pose.m_q.x() << " " << v.m_pose.m_q.y() << " " << v.m_pose.m_q.z() << " " << v.m_pose.m_q.w() << " " << endl;
    }
    for(int i=0;i<edgeCnt;i++)
    {
        Edge e = Edges[i];
        fout << Edge::name() << " " << e.m_id1 << " " << e.m_id2 << " "
             << e.m_pose.m_t[0] << " " << e.m_pose.m_t[1] << " " << e.m_pose.m_t[2] << " "
             << e.m_pose.m_q.x() << " " << e.m_pose.m_q.y() << " " << e.m_pose.m_q.z() << " " << e.m_pose.m_q.w() << " " << endl;

        Matrix6d information = e.m_information;
        for(int i=0; i<information.rows(); i++)
        {
            for(int j=i; j<information.cols(); j++)
            {
                fout << information(i,j) << " ";
            }
        }
        fout << endl;
    }
    fout.close();

    return 0;
}
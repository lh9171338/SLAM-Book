#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>


using namespace std;
using namespace cv;

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() // 重置
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double* update) // 更新
    {
        _estimate += Eigen::Vector3d(update);
    }
    virtual bool read(istream& /*is*/)
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
    virtual bool write(ostream& /*os*/) const
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge() {}
    // 计算曲线模型误差
    void computeError()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double x = _measurement(0);
        double y = _measurement(1);
        _error(0) = y - exp(abc(0) * x * x + abc(1) * x + abc(2));
    }
    virtual bool read(istream& /*is*/)
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
    virtual bool write(ostream& /*os*/) const
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

};

int main(int argc, char** argv)
{
    double a = 1.0, b = 2.0, c = 1.0;           // 真实参数值
    int N = 100;                                // 数据点
    double w_sigma = 1.0;                       // 噪声Sigma值
    RNG rng;                                    // OpenCV随机数产生器
    double abc[3] = {0, 0, 0};                  // abc参数的估计值

    vector<double> x_data, y_data;              // 数据

    cout << "generating data: " << endl;
    for(int i=0; i<N; i++)
    {
        double x = i / 100.0;
        double y = exp(a * x * x + b * x + c) + rng.gaussian(w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
        cout << x_data[i] << " " << y_data[i] << endl;
    }

    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3, 1>> BlockSolver; // 每个误差项优化变量维度为3，误差值维度为1
    typedef g2o::LinearSolverDense<BlockSolver::PoseMatrixType> LinearSolver; // 线性方程求解器

    // 梯度下降方法，从GN, LM, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
//    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(
//            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
//    g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(
//            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // 往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);

    // 往图中增加边
    for(int i=0; i<N; i++)
    {
        CurveFittingEdge* edge = new CurveFittingEdge();
        edge->setVertex(0, v);                // 设置连接的顶点
        edge->setMeasurement(Eigen::Vector2d(x_data[i], y_data[i]));      // 观测数值
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()*1/(w_sigma*w_sigma)); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge);
    }

    // 执行优化
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}

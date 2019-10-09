#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

using namespace std;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 * **********************************************/

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cerr << "Usage: pose_graph_g2o_SE3 sphere.g2o result.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if(!fin)
    {
        cerr << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }


    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 6>> BlockSolver;
    typedef g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType> LinearSolver;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    while(!fin.eof())
    {
        string name;
        fin >> name;
        if(name == "VERTEX_SE3:QUAT")
        {
            // SE3 顶点
            g2o::VertexSE3* v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if(index == 0)
                v->setFixed(true);
        }
        else if (name=="EDGE_SE3:QUAT")
        {
            // SE3-SE3 边
            g2o::EdgeSE3* e = new g2o::EdgeSE3();
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2;
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edgeCnt++;
        }
        if(!fin.good())
            break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "prepare optimizing ..." << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    cout << "calling optimizing ..." << endl;
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;
    optimizer.save(argv[2]);

    return 0;
}
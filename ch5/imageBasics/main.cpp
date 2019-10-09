#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    string filename = "../ubuntu.png";
    Mat image;
    image = imread(filename); //cv::imread函数读取指定路径下的图像
    // 判断图像文件是否正确读取
    if(image.empty()) //数据不存在,可能是文件不存在
    {
        cerr << "文件" << filename << "不存在." << endl;
        return 0;
    }

    // 文件顺利读取, 首先输出一些基本信息
    cout << "图像宽为" << image.cols << ",高为" << image.rows << ",通道数为" << image.channels() << endl;
    namedWindow("image", 0);
    imshow("image", image);      // 用cv::imshow显示图像
    waitKey(0);                  // 暂停程序,等待一个按键输入
    // 判断image的类型
    if(image.type() != CV_8UC1 && image.type() != CV_8UC3)
    {
        // 图像类型不符合要求
        cout << "请输入一张彩色图或灰度图." << endl;
        return 0;
    }

    // 遍历图像, 请注意以下遍历方式亦可使用于随机像素访问
    // 使用 std::chrono 来给算法计时
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();
    for(size_t y=0; y<rows; y++)
    {
        // 用cv::Mat::ptr获得图像的行指针
        unsigned char* row_ptr = image.ptr<unsigned char>(y);  // row_ptr是第y行的头指针
        for(size_t x=0; x<cols; x++)
        {
            // 访问位于 x,y 处的像素
            unsigned char* data_ptr = &row_ptr[x*channels]; // data_ptr 指向待访问的像素数据
            // 输出该像素的每个通道,如果是灰度图就只有一个通道
            for(int c=0; c<channels;c++)
            {
                unsigned char data = data_ptr[c]; // data为I(x,y)第c个通道的值
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "遍历图像用时：" << time_used.count() << " 秒。" << endl;

    // 关于 cv::Mat 的拷贝
    // 直接赋值并不会拷贝数据
    Mat image_another = image;
    // 修改 image_another 会导致 image 发生变化
    image_another(Rect(0, 0, 100, 100)).setTo(0); // 将左上角100*100的块置零
    namedWindow("image", 0);
    imshow("image", image);
    waitKey(0);

    // 使用clone函数来拷贝数据
    Mat image_clone = image.clone();
    image_clone(Rect(0, 0, 100, 100)).setTo(255);
    namedWindow("image", 0);
    imshow("image", image);
    namedWindow("image_clone", 0);
    imshow("image_clone", image_clone);
    waitKey(0);

    // 对于图像还有很多基本的操作,如剪切,旋转,缩放等,限于篇幅就不一一介绍了,请参看OpenCV官方文档查询每个函数的调用方法.
    destroyAllWindows();

    return 0;
}
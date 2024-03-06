#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

// 高斯滤波函数
// src: 输入图像
// kernelSize: 高斯核大小，通常为奇数
// sigma: 高斯核的标准差
cv::Mat gaussianBlur(const cv::Mat& src, int kernelSize, double sigma) 
{
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(kernelSize, kernelSize), sigma);
    return blurred;
}

// Sobel算子计算梯度幅值和方向
// src: 输入图像
// magnitude: 输出的梯度幅值矩阵
// angle: 输出的梯度方向矩阵，单位为度
void sobelGradient(const cv::Mat& src, cv::Mat& magnitude, cv::Mat& angle) 
{
    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(src, grad_y, CV_32F, 0, 1, 3);
    cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);
}

// 非极大值抑制函数
// magnitude: 梯度幅值矩阵
// angle: 梯度方向矩阵
cv::Mat nonMaxSuppression(const cv::Mat& magnitude, const cv::Mat& angle) 
{
    cv::Mat suppressed = cv::Mat::zeros(magnitude.size(), CV_32F);

    for (int y = 1; y < magnitude.rows - 1; ++y) 
    {
        for (int x = 1; x < magnitude.cols - 1; ++x) 
        {
            float angleDeg = angle.at<float>(y, x);
            float mag = magnitude.at<float>(y, x);

            float mag1 = 0, mag2 = 0;

            // 根据梯度方向确定相邻像素点的位置
            if ((angleDeg >= 0 && angleDeg < 22.5) || (angleDeg >= 157.5 && angleDeg <= 180) ||
                (angleDeg >= -180 && angleDeg < -157.5) || (angleDeg >= -22.5 && angleDeg < 0)) 
            {
                mag1 = magnitude.at<float>(y, x - 1);
                mag2 = magnitude.at<float>(y, x + 1);
            }
            else if ((angleDeg >= 22.5 && angleDeg < 67.5) || (angleDeg >= -157.5 && angleDeg < -112.5)) 
            {
                mag1 = magnitude.at<float>(y + 1, x - 1);
                mag2 = magnitude.at<float>(y - 1, x + 1);
            }
            else if ((angleDeg >= 67.5 && angleDeg < 112.5) || (angleDeg >= -112.5 && angleDeg < -67.5)) 
            {
                mag1 = magnitude.at<float>(y - 1, x);
                mag2 = magnitude.at<float>(y + 1, x);
            }
            else if ((angleDeg >= 112.5 && angleDeg < 157.5) || (angleDeg >= -67.5 && angleDeg < -22.5)) 
            {
                mag1 = magnitude.at<float>(y - 1, x - 1);
                mag2 = magnitude.at<float>(y + 1, x + 1);
            }

            // 仅保留局部最大值点作为边缘候选
            if (mag >= mag1 && mag >= mag2) 
            {
                suppressed.at<float>(y, x) = mag;
            }
            else 
            {
                suppressed.at<float>(y, x) = 0;
            }
        }
    }

    return suppressed;
}

// 双阈值检测函数
// suppressed: 非极大值抑制后的图像
// lowThresh: 低阈值
// highThresh: 高阈值
cv::Mat doubleThreshold(const cv::Mat& suppressed, float lowThresh, float highThresh) 
{
    cv::Mat edges = cv::Mat::zeros(suppressed.size(), CV_8U);

    for (int y = 0; y < suppressed.rows; ++y) 
    {
        for (int x = 0; x < suppressed.cols; ++x) 
        {
            float val = suppressed.at<float>(y, x);
            if (val >= highThresh) 
            {
                edges.at<uchar>(y, x) = 255; // 强边缘
            }
            else if (val >= lowThresh) 
            {
                edges.at<uchar>(y, x) = 100; // 弱边缘
            }
        }
    }

    // 边缘跟踪：将与强边缘相连的弱边缘标记为强边缘
    for (int y = 1; y < edges.rows - 1; ++y) 
    {
        for (int x = 1; x < edges.cols - 1; ++x) 
        {
            if (edges.at<uchar>(y, x) == 100) 
            {
                if (edges.at<uchar>(y - 1, x - 1) == 255 || edges.at<uchar>(y - 1, x) == 255 ||
                    edges.at<uchar>(y - 1, x + 1) == 255 || edges.at<uchar>(y, x - 1) == 255 ||
                    edges.at<uchar>(y, x + 1) == 255 || edges.at<uchar>(y + 1, x - 1) == 255 ||
                    edges.at<uchar>(y + 1, x) == 255 || edges.at<uchar>(y + 1, x + 1) == 255) 
                {
                    edges.at<uchar>(y, x) = 255;
                }
                else 
                {
                    edges.at<uchar>(y, x) = 0;
                }
            }
        }
    }

    return edges;
}

int main() 
{
    cv::Mat src = cv::imread("C:\\Users\\HJUN\\Desktop\\naruto.jpg");
    if (src.empty()) 
    {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 高斯滤波
    cv::Mat blurred = gaussianBlur(gray, 5, 1.5);

    // Sobel算子计算梯度
    cv::Mat magnitude, angle;
    sobelGradient(blurred, magnitude, angle);

    // 非极大值抑制
    cv::Mat suppressed = nonMaxSuppression(magnitude, angle);

    // 双阈值检测
    cv::Mat edges = doubleThreshold(suppressed, 50, 150);

    // 显示各个阶段的图像
    cv::imshow("原始图像", src);
    cv::imshow("灰度图", gray);
    cv::imshow("高斯平滑", blurred);
    cv::imshow("Sobel梯度", magnitude);
    cv::imshow("非极大值抑制", suppressed);
    cv::imshow("Canny边缘检测结果", edges);
    cv::waitKey(0);

    return 0;
}

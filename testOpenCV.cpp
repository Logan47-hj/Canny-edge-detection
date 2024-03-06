#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

// ��˹�˲�����
// src: ����ͼ��
// kernelSize: ��˹�˴�С��ͨ��Ϊ����
// sigma: ��˹�˵ı�׼��
cv::Mat gaussianBlur(const cv::Mat& src, int kernelSize, double sigma) 
{
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(kernelSize, kernelSize), sigma);
    return blurred;
}

// Sobel���Ӽ����ݶȷ�ֵ�ͷ���
// src: ����ͼ��
// magnitude: ������ݶȷ�ֵ����
// angle: ������ݶȷ�����󣬵�λΪ��
void sobelGradient(const cv::Mat& src, cv::Mat& magnitude, cv::Mat& angle) 
{
    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(src, grad_y, CV_32F, 0, 1, 3);
    cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);
}

// �Ǽ���ֵ���ƺ���
// magnitude: �ݶȷ�ֵ����
// angle: �ݶȷ������
cv::Mat nonMaxSuppression(const cv::Mat& magnitude, const cv::Mat& angle) {
    cv::Mat suppressed = cv::Mat::zeros(magnitude.size(), CV_32F);

    for (int y = 1; y < magnitude.rows - 1; ++y) 
    {
        for (int x = 1; x < magnitude.cols - 1; ++x) 
        {
            float angleDeg = angle.at<float>(y, x);
            float mag = magnitude.at<float>(y, x);

            float mag1 = 0, mag2 = 0;

            // �����ݶȷ���ȷ���������ص��λ��
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

            // �������ֲ����ֵ����Ϊ��Ե��ѡ
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

// ˫��ֵ��⺯��
// suppressed: �Ǽ���ֵ���ƺ��ͼ��
// lowThresh: ����ֵ
// highThresh: ����ֵ
cv::Mat doubleThreshold(const cv::Mat& suppressed, float lowThresh, float highThresh) 
{
    cv::Mat edges = cv::Mat::zeros(suppressed.size(), CV_8U);

    for (int y = 0; y < suppressed.rows; ++y) 
    {
        for (int x = 0; x < suppressed.cols; ++x) {
            float val = suppressed.at<float>(y, x);
            if (val >= highThresh) {
                edges.at<uchar>(y, x) = 255; // ǿ��Ե
            }
            else if (val >= lowThresh) {
                edges.at<uchar>(y, x) = 100; // ����Ե
            }
        }
    }

    // ��Ե���٣�����ǿ��Ե����������Ե���Ϊǿ��Ե
    for (int y = 1; y < edges.rows - 1; ++y) 
    {
        for (int x = 1; x < edges.cols - 1; ++x) {
            if (edges.at<uchar>(y, x) == 100) {
                if (edges.at<uchar>(y - 1, x - 1) == 255 || edges.at<uchar>(y - 1, x) == 255 ||
                    edges.at<uchar>(y - 1, x + 1) == 255 || edges.at<uchar>(y, x - 1) == 255 ||
                    edges.at<uchar>(y, x + 1) == 255 || edges.at<uchar>(y + 1, x - 1) == 255 ||
                    edges.at<uchar>(y + 1, x) == 255 || edges.at<uchar>(y + 1, x + 1) == 255) {
                    edges.at<uchar>(y, x) = 255;
                }
                else {
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

    // ת��Ϊ�Ҷ�ͼ
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // ��˹�˲�
    cv::Mat blurred = gaussianBlur(gray, 5, 1.5);

    // Sobel���Ӽ����ݶ�
    cv::Mat magnitude, angle;
    sobelGradient(blurred, magnitude, angle);

    // �Ǽ���ֵ����
    cv::Mat suppressed = nonMaxSuppression(magnitude, angle);

    // ˫��ֵ���
    cv::Mat edges = doubleThreshold(suppressed, 50, 150);

    // ��ʾ�����׶ε�ͼ��
    cv::imshow("ԭʼͼ��", src);
    cv::imshow("�Ҷ�ͼ", gray);
    cv::imshow("��˹ƽ��", blurred);
    cv::imshow("Sobel�ݶ�", magnitude);
    cv::imshow("�Ǽ���ֵ����", suppressed);
    cv::imshow("Canny��Ե�����", edges);
    cv::waitKey(0);

    return 0;
}

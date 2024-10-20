#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;

// Функция для сложения двух изображений
void mergeImages(const cv::Mat& imgA, const cv::Mat& imgB, cv::Mat& outputImg) {
    for (int row = 0; row < imgA.rows; ++row) {
        for (int col = 0; col < imgA.cols; ++col) {
            for (int channel = 0; channel < 3; ++channel) {
                outputImg.at<cv::Vec3b>(row, col)[channel] = cv::saturate_cast<uchar>(
                    imgA.at<cv::Vec3b>(row, col)[channel] + imgB.at<cv::Vec3b>(row, col)[channel]);
            }
        }
    }
}

// Функция для вычитания одного изображения из другого
void subtractImages(const cv::Mat& imgA, const cv::Mat& imgB, cv::Mat& outputImg) {
    for (int row = 0; row < imgA.rows; ++row) {
        for (int col = 0; col < imgA.cols; ++col) {
            for (int channel = 0; channel < 3; ++channel) {
                outputImg.at<cv::Vec3b>(row, col)[channel] = cv::saturate_cast<uchar>(
                    imgA.at<cv::Vec3b>(row, col)[channel] - imgB.at<cv::Vec3b>(row, col)[channel]);
            }
        }
    }
}

int main() {
    // Загрузка изображений
    cv::Mat firstImage = cv::imread("Lenna.png");
    cv::Mat secondImage = cv::imread("Lenna_gs.png");
    
    // Создание матрицы для хранения результата
    cv::Mat resultImage(firstImage.size(), firstImage.type());

    // Измерение времени для сложения изображений
    auto startTime = chrono::high_resolution_clock::now();
    mergeImages(firstImage, secondImage, resultImage);
    auto endTime = chrono::high_resolution_clock::now();
    
    // Сохранение результата сложения
    cv::imwrite("merged_image.jpg", resultImage);
    cout << "Time taken to merge images: " 
         << chrono::duration_cast<chrono::microseconds>(endTime - startTime).count() 
         << " microseconds" << endl;

    // Измерение времени для вычитания изображений
    startTime = chrono::high_resolution_clock::now();
    subtractImages(firstImage, secondImage, resultImage);
    endTime = chrono::high_resolution_clock::now();
    
    // Сохранение результата вычитания
    cv::imwrite("subtracted_image.jpg", resultImage);
    cout << "Time taken to subtract images: " 
         << chrono::duration_cast<chrono::microseconds>(endTime - startTime).count() 
         << " microseconds" << endl;

    return 0;
}
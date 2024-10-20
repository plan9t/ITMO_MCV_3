#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;

// Функция для объединения двух изображений с использованием NEON
void mergeImagesNeon(const cv::Mat& source1, const cv::Mat& source2, cv::Mat& outputImage) {
    for (int row = 0; row < source1.rows; ++row) {
        for (int col = 0; col < source1.cols; col += 8) { 
            uint8x8_t r1 = vld1_u8(&source1.at<cv::Vec3b>(row, col)[0]); 
            uint8x8_t g1 = vld1_u8(&source1.at<cv::Vec3b>(row, col)[1]); 
            uint8x8_t b1 = vld1_u8(&source1.at<cv::Vec3b>(row, col)[2]); 

            uint8x8_t r2 = vld1_u8(&source2.at<cv::Vec3b>(row, col)[0]);
            uint8x8_t g2 = vld1_u8(&source2.at<cv::Vec3b>(row, col)[1]);
            uint8x8_t b2 = vld1_u8(&source2.at<cv::Vec3b>(row, col)[2]);

            // Сложение
            uint8x8_t rOut = vqadd_u8(r1, r2);
            uint8x8_t gOut = vqadd_u8(g1, g2);
            uint8x8_t bOut = vqadd_u8(b1, b2);

            vst1_u8(&outputImage.at<cv::Vec3b>(row, col)[0], rOut);
            vst1_u8(&outputImage.at<cv::Vec3b>(row, col)[1], gOut);
            vst1_u8(&outputImage.at<cv::Vec3b>(row, col)[2], bOut);
        }
    }
}

// Функция для вычитания одного изображения из другого с использованием NEON
void removeImagesNeon(const cv::Mat& source1, const cv::Mat& source2, cv::Mat& outputImage) {
    for (int row = 0; row < source1.rows; ++row) {
        for (int col = 0; col < source1.cols; col += 8) { 
            uint8x8_t r1 = vld1_u8(&source1.at<cv::Vec3b>(row, col)[0]); 
            uint8x8_t g1 = vld1_u8(&source1.at<cv::Vec3b>(row, col)[1]); 
            uint8x8_t b1 = vld1_u8(&source1.at<cv::Vec3b>(row, col)[2]); 

            uint8x8_t r2 = vld1_u8(&source2.at<cv::Vec3b>(row, col)[0]);
            uint8x8_t g2 = vld1_u8(&source2.at<cv::Vec3b>(row, col)[1]);
            uint8x8_t b2 = vld1_u8(&source2.at<cv::Vec3b>(row, col)[2]);

            // Вычитание
            uint8x8_t rOut = vqsub_u8(r1, r2);
            uint8x8_t gOut = vqsub_u8(g1, g2);
            uint8x8_t bOut = vqsub_u8(b1, b2);

            vst1_u8(&outputImage.at<cv::Vec3b>(row, col)[0], rOut);
            vst1_u8(&outputImage.at<cv::Vec3b>(row, col)[1], gOut);
            vst1_u8(&outputImage.at<cv::Vec3b>(row, col)[2], bOut);
        }
    }
}

int main() {
    // Чтение изображений из файлов
    cv::Mat imgA = cv::imread("Lenna.png");
    cv::Mat imgB = cv::imread("Lenna_gs.png");
    
    // Создание матрицы для хранения результата
    cv::Mat outputImg(imgA.size(), imgA.type());

    // Измерение времени для объединения изображений
    auto startMergeTime = chrono::high_resolution_clock::now();
    mergeImagesNeon(imgA, imgB, outputImg);
    auto endMergeTime = chrono::high_resolution_clock::now();
    
    // Сохранение результата объединения
    cv::imwrite("merged_image_neon.jpg", outputImg);
    cout << "Time taken to merge images with NEON: " 
         << chrono::duration_cast<chrono::microseconds>(endMergeTime - startMergeTime).count() 
         << " microseconds" << endl;

    // Измерение времени для вычитания изображений
    auto startRemoveTime = chrono::high_resolution_clock::now();
    removeImagesNeon(imgA, imgB, outputImg);
    auto endRemoveTime = chrono::high_resolution_clock::now();
    
    // Сохранение результата вычитания
    cv::imwrite("removed_image_neon.jpg", outputImg);
    cout << "Time taken to remove images with NEON: " 
         << chrono::duration_cast<chrono::microseconds>(endRemoveTime - startRemoveTime).count() 
         << " microseconds" << endl;

    return 0;
}
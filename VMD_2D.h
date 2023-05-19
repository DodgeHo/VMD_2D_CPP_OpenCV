#pragma once


#include <opencv2/opencv.hpp>
#include <complex>
#include <iostream>
#include <string>
using namespace cv;

enum ShiftType {
	FFT_SHIFT,   // equivalent to Matlab's fftshift
	IFFT_SHIFT   // equivalent to Matlab's ifftshift
};

void VMD_2D
(std::vector<Mat>& u, std::vector<Mat>& u_hat, std::vector<std::vector<cv::Point2d>>& omega,
	Mat& signal, const double alpha, const double tau,
	const int K, const int DC, const int init, const double tol, const double eps);

Mat fft2(const Mat& signal);
Mat ifft2(const Mat& signal);
Mat fftshift(const Mat& in_mat);
Mat ifftshift(const Mat& in_mat);
Mat real(const Mat& in_mat);
Mat getHilbertMask(const Mat& signal);


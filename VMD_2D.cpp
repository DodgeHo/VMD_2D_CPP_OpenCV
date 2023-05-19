/*
<VMD_CPP: C++ implementation of Variational Mode Decomposition using OpenCV.>
Copyright (C) <2023>  <Lang He: asdsay@gmail.com>
Mozilla Public License v. 2.0.
*/

#include "VMD_2D.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iostream>
using namespace std;
using namespace cv;


int main() {
	// use a image as signal2D to simulation the procedure.
	String str = "F:\\VMD_2D_cpp1\\Sample.bmp";
	Mat image = imread(str);
	Mat signal2D;
	cvtColor(image, signal2D, COLOR_BGR2GRAY);
	imshow("test_opencv_srtup", signal2D);

	// initial some input parameters
	const int K = 5, DC = 1, init = 1;
	const double alpha = 5000.0, tau = 0.25, tol = K * 1e-6, eps = 2.2204e-16;
	std::vector<Mat> u, u_hat;
	std::vector<std::vector<cv::Point2d>> omega;

	// run VMD
	VMD_2D(u, u_hat, omega, signal2D, alpha, tau, K, DC, init, tol, eps);

	// out results of the K modes of signals.
	for (int k = 0; k < K; k++) {
		Mat eachSignal = u[k];
		string filename = "DecomResult_" + std::to_string(k) + ".bmp";
		imwrite(filename, eachSignal);
		imshow(filename, eachSignal);
	}
	waitKey(0);

	return 0;
}

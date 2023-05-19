#include "VMD_2D.h"
#include <iostream>
using namespace cv;
using namespace std;
#define M_PI acos(-1)
#define MATRIX_TYPE CV_64FC2

void checkValid(cv::Mat& matrix) {
	if (!cv::checkRange(matrix))
		throw std::runtime_error("矩阵中包含 NaN 值");

	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			cv::Vec2d value = matrix.at<cv::Vec2d>(i, j);
			double channel1 = value[0]; // 第一个通道的值
			double channel2 = value[1]; // 第二个通道的值
			if (std::isnan(channel1) || std::isnan(channel2)) {
				throw std::runtime_error("矩阵中包含 NaN 值");
			}
		}
	}
}

void VMD_2D
(std::vector<Mat>& u, std::vector<Mat>& u_hat, std::vector<std::vector<cv::Point2d>>& omega,
	Mat& signal, const double alpha, const double tau,
	const int K, const int DC, const int init, const double tol, const double eps) {
	/* ---------------------

	Output:
	-------
	u - the collection of decomposed modes (std::vector<Mat>)
	u_hat - spectra of the modes (std::vector<Mat>)
	omega - estimated mode center - frequencies (std::vector<std::vector<cv::Point2d>>)
	-------
	Input:
	-------
	signal - the time domain signal2D(2D matrix) to be decomposed
	alpha - the balancing parameter of the data - fidelity constraint
	tau - time - step of the dual ascent(pick 0 for noise - slack)
	K - the number of modes to be recovered
	DC - true if the first mode is putand kept at DC(0 - freq)
	init - 0 = all omegas start at 0
			1 = all omegas start initialized randomly
	tol - tolerance of convergence criterion; typically around 1e-7
	*/

	// ----------Preparations
	// Resolution of image
	int Hy = signal.rows;
	int Hx = signal.cols;

	// Spectral Domain discretization
	Mat X = Mat::zeros(Hy, Hx, MATRIX_TYPE);
	Mat Y = Mat::zeros(Hy, Hx, MATRIX_TYPE);
	for (int i = 0; i < Hy; i++)
		for (int j = 0; j < Hx; j++) {
			X.at<double>(i, j) = double(j + 1) / Hx;
			Y.at<double>(i, j) = double(i + 1) / Hy;
		}
	double fx = 1.0 / Hx;
	double fy = 1.0 / Hy;
	Mat freqs_1 = X - 0.5 - fx;
	Mat freqs_2 = Y - 0.5 - fy;

	const int N = 3000; // the maximum number of iterations

	// For future generalizations: alpha might be individual for each mode
	std::vector<double> Alpha(K, alpha);

	// Construct f and f_hat
	Mat f_hat = fftshift(fft2(signal));
	f_hat.convertTo(f_hat, MATRIX_TYPE);

	// Storage matrices for (Fourier) modes. All iterations are not recorded.
	u_hat.resize(K, Mat::zeros(Hy, Hx, MATRIX_TYPE));
	std::vector<Mat> u_hat_old = u_hat;
	Mat sum_uk = Mat::zeros(Hy, Hx, MATRIX_TYPE);

	// Storage matrices for (Fourier) Lagrange multiplier.
	Mat mu_hat = Mat::zeros(Hy, Hx, MATRIX_TYPE);

	// N iterations at most, 2 spatial coordinates, K clusters
	omega.resize(N, std::vector<cv::Point2d>(K, cv::Point2d(0, 0)));

	// Initialization of omega_k
	if (init == 0) {
		// spread omegas radially
		// if DC, keep first mode at 0,0
		int maxK = DC ? K - 1 : K;
		for (int k = DC; k < maxK; ++k) {
			omega[0][k].x = 0.25 * std::cos(M_PI * k / maxK);
			omega[0][k].y = 0.25 * std::sin(M_PI * k / maxK);
		}
	}
	else {
		// Case 1: random on half-plane
		for (int k = 0; k < K; ++k) {
			omega[0][k].x = static_cast<double>(rand()) / RAND_MAX - 0.5;
			omega[0][k].y = static_cast<double>(rand()) / RAND_MAX / 2;
		}
		// DC component (if expected)
		if (DC) {
			omega[0][0].x = 0;
			omega[0][0].y = 0;
		}
	}


	// Main loop for iterative updates
	// Stopping criteria tolerances
	double uDiff = tol + eps;
	double omegaDiff = tol + eps;

	// first run
	int n = 0;

	// run until convergence or max number of iterations
	while ((uDiff > tol || omegaDiff > tol) && n < N){

		// first things first
		int k = 0;

		// compute the halfplane mask for the 2D "analytic signal"
		Mat HilbertMask = getHilbertMask(freqs_1.mul(omega[n][k].x) + freqs_2.mul(omega[n][k].y));
		
		// update first mode accumulator
		sum_uk = u_hat.back() + sum_uk - u_hat[k];
		// update first mode's spectrum through wiener filter (on half plane)
		Mat T = (f_hat - sum_uk - mu_hat / 2.0);
		T.convertTo(T, MATRIX_TYPE);
		u_hat[k] = (T.mul(HilbertMask)) /
			(1 + 
				Alpha[k] * (freqs_1 - omega[n][k].x).mul(freqs_1 - omega[n][k].x) + 
				(freqs_2 - omega[n][k].y).mul(freqs_2 - omega[n][k].y)
			);
		// update first mode's central frequency as spectral center of gravity
		if (!DC) {
			Mat u_hat_abs_sq;
			cv::pow(cv::abs(u_hat[k]), 2, u_hat_abs_sq); // abs(u_hat).^2
			omega[n + 1][k].x = cv::sum(freqs_1.mul(u_hat_abs_sq))[0] / cv::sum(u_hat_abs_sq)[0];
			omega[n + 1][k].y = cv::sum(freqs_2.mul(u_hat_abs_sq))[0] / cv::sum(u_hat_abs_sq)[0];

			// keep omegas on same halfplane
			if (omega[n + 1][k].y < 0) {
				omega[n + 1][k] = - omega[n + 1][k];
			}
		}

		// recover full spectrum from analytic signal
		u_hat[k] = fftshift(fft2(real(ifft2(ifftshift(u_hat[k])))));
		checkValid(u_hat[k]);
		// work on other modes
		for (k = 1; k < K; k++) {
			// recompute Hilbert mask
			HilbertMask = getHilbertMask(freqs_1.mul(omega[n][k].x) + freqs_2.mul(omega[n][k].y));

			// update accumulator
			sum_uk = u_hat[k-1] + sum_uk - u_hat[k];

			// update signal frequencies
			u_hat[k] = ((f_hat - sum_uk - mu_hat / 2.0).mul(HilbertMask)) /
				(1 +
					Alpha[k] * ((freqs_1 - omega[n][k].x).mul(freqs_1 - omega[n][k].x) +
					(freqs_2 - omega[n][k].y).mul(freqs_2 - omega[n][k].y))
				);
			
			// update signal frequencies
			Mat u_hat_abs_sq = cv::Mat::zeros(1, 1, CV_64FC1);
			cv::pow(cv::abs(u_hat[k]), 2, u_hat_abs_sq); // abs(u_hat).^2
			omega[n + 1][k].x = cv::sum(freqs_1.mul(u_hat_abs_sq))[0] / cv::sum(u_hat_abs_sq)[0];
			omega[n + 1][k].y = cv::sum(freqs_2.mul(u_hat_abs_sq))[0] / cv::sum(u_hat_abs_sq)[0];

			// keep omegas on same halfplane
			if (omega[n + 1][k].y < 0) {
				omega[n + 1][k] = -omega[n + 1][k];
			}


			// recover full spectrum from analytic signal
			u_hat[k] = fftshift(fft2(real(ifft2(ifftshift(u_hat[k])))));

			checkValid(u_hat[k]);
		}

		// Gradient ascent for augmented Lagrangian
		Mat sum_u_hat(Hy, Hx, MATRIX_TYPE);
		for (Mat eachMat : u_hat)
			sum_u_hat += eachMat;
		mu_hat += tau * (sum_u_hat - f_hat);

		// increment iteration counter
		n++;

		// convergence?
		double uDiff = eps;
		double omegaDiff = eps;

		for (int k = 0; k < K; k++) {
			cv::Point2d p = omega[n][k] - omega[n - 1][k];
			omegaDiff += p.x*p.x + p.y*p.y;


			checkValid(u_hat_old[k]);
			Mat diff = (u_hat[k] - u_hat_old[k]);

			/*
			for (int j = 0; j < 10; j++) {
				cv::Vec2d value = diff.at<cv::Vec2d>(0, j);
				std::cout << value[0] << " + " << value[0] << "i ";
			}
			std::cout << std::endl;
			*/

			double s = cv::sum(diff.mul(diff))[0];
			uDiff += s / (Hx * Hy);
		}

		uDiff = std::abs(uDiff);

		for (int k = 0; k < K; k++)
			u_hat_old[k] = u_hat[k];

		std::cout << n << " time; uDiff: " <<uDiff<< " ; omegaDiff: " << omegaDiff << std::endl;
	}

	// Signal Reconstruction
	// Inverse Fourier Transform to compute (spatial) modes
	u.resize(K, Mat::zeros(Hy, Hx, MATRIX_TYPE));
	for (int k = 0; k < K; k++) {
		u[k] = ifft2(fftshift(u_hat[k]));
	}

	// Should the omega-history be returned, or just the final results?
	//std::vector<std::vector<cv::Point2d>> omega_final(omega.begin(), omega.begin() + n);

	return;
}

#pragma region Ancillary Functions 
cv::Mat fft2(const cv::Mat& signal) {
	cv::Mat complexResult = signal;
	cv::Mat complexSignal = signal;
	complexSignal.convertTo(complexSignal, CV_64FC2);
	cv::dft(complexSignal, complexResult, cv::DFT_COMPLEX_OUTPUT);
	return complexResult;
}

cv::Mat ifft2(const cv::Mat& signal) {
	cv::Mat complexResult = signal;
	cv::idft(signal, complexResult, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
	return complexResult;
}

cv::Mat fftshift(const cv::Mat& in_mat) {
	cv::Mat shifted = in_mat;
	int cx = in_mat.cols / 2;
	int cy = in_mat.rows / 2;
	cv::Mat q0(in_mat, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(in_mat, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(in_mat, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(in_mat, cv::Rect(cx, cy, cx, cy));
	q0.copyTo(shifted);
	q3.copyTo(q0);
	shifted.copyTo(q3);
	q1.copyTo(shifted);
	q2.copyTo(q1);
	shifted.copyTo(q2);
	return in_mat;
}

cv::Mat ifftshift(const cv::Mat& in_mat) {
	cv::Mat shifted = in_mat;
	int cx = in_mat.cols / 2;
	int cy = in_mat.rows / 2;

	// when size is odd, for ifftshift, shift center point to left
	if (in_mat.cols % 2 != 0) cx++;
	if (in_mat.rows % 2 != 0) cy++;

	cv::Mat q0(in_mat, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(in_mat, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(in_mat, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(in_mat, cv::Rect(cx, cy, cx, cy));
	q3.copyTo(shifted);
	q0.copyTo(q3);
	shifted.copyTo(q0);
	q2.copyTo(shifted);
	q1.copyTo(q2);
	shifted.copyTo(q1);
	return in_mat;
}

cv::Mat real(const cv::Mat& in_mat) {
	cv::Mat result;
	cv::Mat channels[2];
	cv::split(in_mat, channels);
	result = channels[0];
	return result;
}



Mat getHilbertMask(const Mat& signal) {
	Mat posiHilbertMask = ((signal) > 0);
	Mat zeroHilbertMask = ((signal) == 0);
	Mat HilbertMask = posiHilbertMask.mul(2) + zeroHilbertMask;
	HilbertMask.convertTo(HilbertMask, MATRIX_TYPE);
	if (!cv::checkRange(HilbertMask))
		abort();
	return HilbertMask;
}
#pragma endregion


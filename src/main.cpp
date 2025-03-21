#include <opencv2/opencv.hpp>
#include "OpenCLDeviceSelector.h"
#include "cppkernels/SADCostKernel.h"
#include "cppkernels/SADKernel.h"
#include "cppkernels/HorizontalAggregationKernel.h"
#include "cppkernels/ComputeBestDisparityKernel.h"

#include "OpenCVHelper.h"

#include "OpenCLManager.h"
#include <iostream>
#include <fstream>
#include <sstream>




int main() {


	int maxDisparity = 64;
	int halfWindowSize = 2;

	OpenCLDeviceSelector selector;
	if (!selector.selectBestDevice()) {
		std::cerr << "Failed to select OpenCL device!" << std::endl;
		return 1;
	}
	selector.printDeviceInfo();

	OpenCLManager manager;
	if (!manager.initialize()) {
		std::cerr << "Failed to initialize OpenCL manager!" << std::endl;
		return 1;
	}


	// FPS counter
	int frameCounter = 0; 
	auto start = std::chrono::high_resolution_clock::now();

	std::string root = "C:\\Users\\bmorel\\Desktop\\stream\\captures\\capture500_500";
	std::string leftPath = root + "\\" + "output_30_left.png";
	std::string rightPath = root + "\\" + "output_30_right.png";
	cv::Mat left = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
	cv::Mat right = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
	cv::Mat disparity = cv::Mat(left.rows, left.cols, CV_16U);
	int width = left.cols;
	int height = left.rows;
	auto leftBuffer = createOpenCLBufferFromMat(left, manager.getContext(), manager.getCommandQueue(), false);
	auto rightBuffer = createOpenCLBufferFromMat(right, manager.getContext(), manager.getCommandQueue(), false);
	auto costBuffer = createCostsOpenCLBuffer(width * height * maxDisparity, manager.getContext(), manager.getCommandQueue());
	auto aggregatedBuffer = createCostsOpenCLBuffer(width * height * maxDisparity, manager.getContext(), manager.getCommandQueue());
	auto disparityBuffer = createOpenCLBufferFromMat(disparity, manager.getContext(), manager.getCommandQueue(), true);

	SADCostKernel costKernel(manager);
	HorizontalAggregationKernel horizontalAggregationKernel(manager);
	ComputeBestDisparityKernel bestDisparityKernel(manager);

	// create opencv window with sliders
	int P1 = 100;
	int P2 = 1000;
	cv::namedWindow("parameters", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("P1", "parameters", &P1, 800);
	cv::createTrackbar("P2", "parameters", &P2, 2000);

	while (true) {
		 

		// update trackbar values 
		P1 = cv::getTrackbarPos("P1", "parameters");
		P2 = cv::getTrackbarPos("P2", "parameters"); 

		costKernel.setArguments(leftBuffer, rightBuffer, costBuffer, width, height, halfWindowSize, maxDisparity);
		costKernel.runKernel(left.cols * left.rows);
		  
		horizontalAggregationKernel.setArguments(costBuffer, aggregatedBuffer, width, height, maxDisparity, P1, P2);
		horizontalAggregationKernel.runKernel(left.rows);
		

		bestDisparityKernel.setArguments(aggregatedBuffer, disparityBuffer, width, height, maxDisparity);
		bestDisparityKernel.runKernel(left.cols * left.rows);

		fillMatFromOpenCLBuffer(disparity, disparityBuffer, manager.getContext(), manager.getCommandQueue());
		  
		  
		bool debug = true;  

		if (debug) {  
			cv::Mat disparityU8;
			disparity.convertTo(disparityU8, CV_8U, 1.0f / 16.0f);
			// convert disparity to colored map for visualization
			 
			cv::Mat normalizedDisparity;
			cv::normalize(disparityU8, normalizedDisparity, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::Mat disparityColor; 
			cv::applyColorMap(normalizedDisparity, disparityColor, cv::COLORMAP_JET);
			cv::imshow("disparity color", disparityColor);
			//cv::imshow("left", left);
			cv::waitKey(1);  // Wait 30 ms (or any appropriate delay)
		}


		// update FPS and print it after 1sec
		frameCounter++;
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		if (duration > 1000) {
			std::cout << "FPS: " << frameCounter << std::endl;
			frameCounter = 0;
			start = std::chrono::high_resolution_clock::now();
		}
	}
	 
    return 0;
}

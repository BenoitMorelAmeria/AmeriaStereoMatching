#pragma once

#include "Kernel.h"

class HorizontalAggregationKernel : public Kernel {
public:
	HorizontalAggregationKernel(OpenCLManager& manager);
	bool setArguments(cl_mem costBuffer, 
		cl_mem aggregatedCostBuffer, 
		int width, 
		int height, 
		int maxDisparity,
		int P1,
		int P2);
	virtual bool runKernel(size_t globalSize);  // Run the kernel

	int _width;
	int _height;
};
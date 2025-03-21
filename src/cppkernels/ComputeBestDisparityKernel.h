#pragma once


#include "Kernel.h"

class ComputeBestDisparityKernel : public Kernel {
public:
	ComputeBestDisparityKernel(OpenCLManager& manager);
	bool setArguments(cl_mem aggregatedCosts,
		cl_mem disparityBuffer,
		int width,
		int height,
		int maxDisparity);

};
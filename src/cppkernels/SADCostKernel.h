#pragma once

#include "Kernel.h"


class SADCostKernel: public Kernel {
public:
    SADCostKernel(OpenCLManager& manager);
    bool setArguments(cl_mem leftImageBuffer, 
        cl_mem rightImageBuffer,
        cl_mem outputCostFunctionBuffer, 
        int width,
        int height,
		int halfWindowSize,
		int disparityRange); 
};

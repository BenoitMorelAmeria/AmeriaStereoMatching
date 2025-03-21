#pragma once

#include <CL/cl.h>
#include <iostream>
#include "Kernel.h"


class SADKernel: public Kernel {
public:
    SADKernel(OpenCLManager& manager);
    bool setArguments(cl_mem leftImageBuffer, 
        cl_mem rightImageBuffer,
        cl_mem disparityBuffer,
        int width,
        int height,
        int maxDisparity,
        int windowSize);
};

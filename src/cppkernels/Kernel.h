#pragma once

#include <CL/cl.h>
#include <iostream>
#include "../OpenCLProgram.h"

class OpenCLManager;

class Kernel {
public:
    Kernel(OpenCLManager& manager, const std::string& kernelPath, const std::string& kernelName);
    virtual ~Kernel() {}
    virtual bool runKernel(size_t globalSize);  // Run the kernel

protected:
    OpenCLManager& manager;
    OpenCLProgram program;
    cl_kernel kernel;
};

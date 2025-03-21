#include "SADKernel.h"



SADKernel::SADKernel(OpenCLManager& manager):
	Kernel(manager, "kernels.cl", "computeSAD")
{
}

bool SADKernel::setArguments(cl_mem leftImageBuffer, 
    cl_mem rightImageBuffer,
    cl_mem disparityBuffer, 
    int width,
    int height,
	int maxDisparity,
    int windowSize) {
    cl_int err;

    // Set the kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &leftImageBuffer);    
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &rightImageBuffer);   
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &disparityBuffer);    
    err |= clSetKernelArg(kernel, 3, sizeof(int), &width);            
    err |= clSetKernelArg(kernel, 4, sizeof(int), &height);             
    err |= clSetKernelArg(kernel, 5, sizeof(int), &maxDisparity);            
    err |= clSetKernelArg(kernel, 6, sizeof(int), &windowSize);            
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments" << std::endl;
        return false;
    } 

    return true;
}

#include "ComputeBestDisparityKernel.h"


ComputeBestDisparityKernel::ComputeBestDisparityKernel(OpenCLManager& manager) :
	Kernel(manager, "kernels.cl", "computeBestDisparity")
{
}

bool ComputeBestDisparityKernel::setArguments(cl_mem aggregatedCosts,
	cl_mem disparityBuffer,
	int width,
	int height,
	int maxDisparity)
{
	cl_int err;
	// Set the kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aggregatedCosts);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &disparityBuffer);
	err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &maxDisparity);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to set kernel arguments" << std::endl;
		return false;
	}
	return true;
}
#include "ComputeBestDisparityKernel.h"


ComputeBestDisparityKernel::ComputeBestDisparityKernel(OpenCLManager& manager) :
	Kernel(manager, "kernels.cl", "computeBestDisparity")
{
}

bool ComputeBestDisparityKernel::setArguments(cl_mem aggregatedCosts,
	cl_mem disparityBuffer,
	int width,
	int height,
	int maxDisparity,
	float uniquenessRatio)
{
	cl_int err;
	// Set the kernel arguments
	int i = 0;
	err = clSetKernelArg(kernel, i++, sizeof(cl_mem), &aggregatedCosts);
	err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &disparityBuffer);
	err |= clSetKernelArg(kernel, i++, sizeof(int), &width);
	err |= clSetKernelArg(kernel, i++, sizeof(int), &height);
	err |= clSetKernelArg(kernel, i++, sizeof(int), &maxDisparity);
	err |= clSetKernelArg(kernel, i++, sizeof(float), &uniquenessRatio);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to set kernel arguments" << std::endl;
		return false;
	}
	return true;
}
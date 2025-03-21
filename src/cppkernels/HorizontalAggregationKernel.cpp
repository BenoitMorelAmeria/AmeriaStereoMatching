#include "HorizontalAggregationKernel.h"

HorizontalAggregationKernel::HorizontalAggregationKernel(OpenCLManager& manager) :
	Kernel(manager, "kernels.cl", "horizontalAggregation")
{
}

bool HorizontalAggregationKernel::setArguments(cl_mem costBuffer, 
	cl_mem aggregatedCostBuffer,
	int width,
	int height,
	int maxDisparity,
	int P1,
	int P2)
{
	_height = height;
	_width = width;
	cl_int err;
	// Set the kernel arguments
	int i = 0;
	err = clSetKernelArg(kernel, i++, sizeof(cl_mem), &costBuffer);
	err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &aggregatedCostBuffer);
	err |= clSetKernelArg(kernel, i++, sizeof(int), &width);
	err |= clSetKernelArg(kernel, i++, sizeof(int), &height);
	err |= clSetKernelArg(kernel, i++, sizeof(int), &maxDisparity);
	float p1 = P1;
	float p2 = P2;
	err |= clSetKernelArg(kernel, i++, sizeof(float), &p1);
	err |= clSetKernelArg(kernel, i++, sizeof(float), &p2);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to set kernel arguments" << std::endl;
		return false;
	}
	return true;
}

bool HorizontalAggregationKernel::runKernel(size_t globalSize)
{
#ifndef DISABLE_KERNEL
	cl_int err;
	// 4. Define the global and local work sizes
	size_t globalWorkSize[2] = { (size_t)(_width), (size_t)(_height) };  // Global size
	size_t localWorkSize[2] = { 16, 1 };  // Local size for each workgroup

	// Enqueue the kernel for execution
	err = clEnqueueNDRangeKernel(manager.getCommandQueue(), kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);

	if (err != CL_SUCCESS) {
		std::cerr << "Failed to enqueue kern el " << err << std::endl;
		return false;
	}

	// Wait for the kernel to finish executing
	clFinish(manager.getCommandQueue());
	return true;
	
#else

	return Kernel::runKernel(globalSize);
#endif	
}

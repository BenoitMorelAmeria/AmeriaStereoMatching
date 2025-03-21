#include "Kernel.h"
#include "../OpenCLManager.h"

Kernel::Kernel(OpenCLManager& manager, const std::string& kernelPath, const std::string& kernelName) :
	manager(manager),
	program(manager)
{
	cl_int err;
	program.loadAndBuildProgram(kernelPath);
	kernel = clCreateKernel(program.getProgram(), kernelName.c_str(), &err);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to create kernel: " << kernelName << std::endl;
	}
}


bool Kernel::runKernel(size_t globalSize) {
	cl_int err;
	// Enqueue the kernel for execution
	err = clEnqueueNDRangeKernel(manager.getCommandQueue(), kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to enqueue kernel" << std::endl;
		return false;
	}
	// Wait for the kernel to finish executing
	clFinish(manager.getCommandQueue());
	return true;
}
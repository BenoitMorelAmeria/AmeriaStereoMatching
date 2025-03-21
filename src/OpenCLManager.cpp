#include "OpenCLManager.h"
#include <fstream>
#include <sstream>



OpenCLManager::OpenCLManager() : context(nullptr), commandQueue(nullptr), device(nullptr) {}

OpenCLManager::~OpenCLManager() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (context) clReleaseContext(context);
}

bool OpenCLManager::initialize() {
    cl_int err;

    // Step 1: Select platform and device
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform" << std::endl;
        return false;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device" << std::endl;
        return false;
    }

    // Step 2: Create OpenCL context and command queue
    return createContextAndQueue();
}

bool OpenCLManager::createContextAndQueue() {
    cl_int err;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }

    commandQueue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue" << std::endl;
        return false;
    }

    return true;
}




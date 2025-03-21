#include "OpenCLProgram.h"


OpenCLProgram::OpenCLProgram(OpenCLManager& openCLManager)
    : openCLManager(openCLManager), program(nullptr) {
}

OpenCLProgram::~OpenCLProgram() {
    if (program) {
        clReleaseProgram(program);
    }
}

bool OpenCLProgram::loadAndBuildProgram(const std::string& programFile) {
    cl_int err;

    // Load OpenCL source code from file
    std::ifstream file(programFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << programFile << std::endl;
        return false;
    }
    std::string sourceCode((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* source = sourceCode.c_str();

    // Create program from source
    program = clCreateProgramWithSource(openCLManager.getContext(), 1, &source, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program from source" << std::endl;
        return false;
    }

    // Build program
	auto device = openCLManager.getDevice();
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program" << std::endl;
        return false;
    }

    return true;
}
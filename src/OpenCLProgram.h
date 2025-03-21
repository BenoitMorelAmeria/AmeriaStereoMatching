#pragma once

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include "OpenCLManager.h"

class OpenCLProgram {
public:
    OpenCLProgram(OpenCLManager& openCLManager);
    ~OpenCLProgram();

    bool loadAndBuildProgram(const std::string& programFile);
    cl_program getProgram() const { return program; }

private:
    OpenCLManager& openCLManager;
    cl_program program;
};
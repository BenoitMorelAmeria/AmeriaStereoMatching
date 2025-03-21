#pragma once

#include <CL/cl.h>
#include <iostream>
#include <vector>

class OpenCLManager {
public:
    OpenCLManager();
    ~OpenCLManager();

    bool initialize();                       // Initializes OpenCL context and queue
    cl_command_queue getCommandQueue() const { return commandQueue; }
    cl_context getContext() const { return context; }
    cl_device_id getDevice() const { return device; }

private:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;

    bool createContextAndQueue();            // Create context and queue
};

#ifndef OPENCL_DEVICE_SELECTOR_H
#define OPENCL_DEVICE_SELECTOR_H

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>

class OpenCLDeviceSelector {
public:
    OpenCLDeviceSelector();
    ~OpenCLDeviceSelector();

    bool selectBestDevice();
    void printDeviceInfo() const;

    cl_device_id getDevice() const { return selectedDevice; }
    cl_platform_id getPlatform() const { return selectedPlatform; }

private:
    cl_platform_id selectedPlatform = nullptr;
    cl_device_id selectedDevice = nullptr;

    std::vector<cl_platform_id> getPlatforms() const;
    std::vector<cl_device_id> getDevices(cl_platform_id platform, cl_device_type type) const;
};

#endif // OPENCL_DEVICE_SELECTOR_H

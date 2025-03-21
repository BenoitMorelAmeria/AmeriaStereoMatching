#include "OpenCLDeviceSelector.h"

OpenCLDeviceSelector::OpenCLDeviceSelector() {}

OpenCLDeviceSelector::~OpenCLDeviceSelector() {}

// Fetch all OpenCL platforms available
std::vector<cl_platform_id> OpenCLDeviceSelector::getPlatforms() const {
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    return platforms;
}

// Fetch all GPU devices available for a given platform
std::vector<cl_device_id> OpenCLDeviceSelector::getDevices(cl_platform_id platform, cl_device_type type) const {
    cl_uint numDevices = 0;
    clGetDeviceIDs(platform, type, 0, nullptr, &numDevices);

    if (numDevices == 0) return {};

    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, type, numDevices, devices.data(), nullptr);

    return devices;
}

// Select the best OpenCL GPU (prioritizing NVIDIA)
bool OpenCLDeviceSelector::selectBestDevice() {
    auto platforms = getPlatforms();
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return false;
    }

    // Prioritize NVIDIA platform
    for (auto platform : platforms) {
        char name[1024];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, nullptr);

        if (std::string(name).find("NVIDIA") != std::string::npos) {
            selectedPlatform = platform;
            break;
        }
    }

    // If NVIDIA is not found, fall back to first available platform
    if (!selectedPlatform) {
        std::cout << "NVIDIA platform not found. Using the first available platform." << std::endl;
        selectedPlatform = platforms[0];
    }

    // Get all GPU devices for the selected platform
    auto devices = getDevices(selectedPlatform, CL_DEVICE_TYPE_GPU);
    if (devices.empty()) {
        std::cerr << "No GPU devices found on the selected platform!" << std::endl;
        return false;
    }

    selectedDevice = devices[0]; // Choose first GPU
    return true;
}

// Print device details
void OpenCLDeviceSelector::printDeviceInfo() const {
    if (!selectedDevice) {
        std::cerr << "No device selected!" << std::endl;
        return;
    }

    char buffer[1024];
    cl_ulong memorySize;
    cl_uint computeUnits;
    size_t maxWorkGroupSize;

    clGetDeviceInfo(selectedDevice, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    std::cout << "Using Device: " << buffer << std::endl;

    clGetDeviceInfo(selectedDevice, CL_DEVICE_VERSION, sizeof(buffer), buffer, nullptr);
    std::cout << "  OpenCL Version: " << buffer << std::endl;

    clGetDeviceInfo(selectedDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memorySize), &memorySize, nullptr);
    std::cout << "  Global Memory: " << (memorySize / (1024 * 1024)) << " MB" << std::endl;

    clGetDeviceInfo(selectedDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
    std::cout << "  Compute Units: " << computeUnits << std::endl;

    clGetDeviceInfo(selectedDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    std::cout << "  Max Work Group Size: " << maxWorkGroupSize << std::endl;


}

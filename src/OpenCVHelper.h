#pragma once

#include <CL/cl.h>
#include <opencv2/opencv.hpp>

cl_mem createCostsOpenCLBuffer(size_t bufferSize, cl_context context, cl_command_queue queue)
{
    cl_int err;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize * sizeof(float), nullptr, &err);
    // print size in MB
    if (err != CL_SUCCESS || buffer == nullptr) {
        std::cerr << "Error: Failed to create OpenCL buffer! (Error code: " << err << ")" << std::endl;
        return nullptr;
    }
    return buffer;
}

cl_mem createOpenCLBufferFromMat(const cv::Mat& mat, cl_context context, cl_command_queue queue, bool write) {
    if (mat.empty()) {
        std::cerr << "Error: Input cv::Mat is empty!" << std::endl;
        return nullptr;
    }   
    size_t bufferSize = mat.total() * mat.elemSize();

     
    cl_int err;
    cl_mem buffer = nullptr;
	cl_mem_flags flags = write ? CL_MEM_WRITE_ONLY: (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR );
	uchar* data = write ? nullptr : mat.data;
    // Handle the type based on whether it's an input (e.g., grayscale 8-bit) or output (e.g., disparity map, unsigned short)
    if (mat.depth() == CV_8U) {
        buffer = clCreateBuffer(context, flags,
            mat.total() * mat.elemSize(), data, &err);
    } 
    else if (mat.depth() == CV_16U) {

        buffer = clCreateBuffer(context, flags,
            mat.total() * mat.elemSize(), data, &err);
    }
    else {
		std::cerr << "Error: Unsupported image type!" << std::endl;
		return nullptr;
    }

    if (err != CL_SUCCESS || buffer == nullptr) {
        std::cerr << "Error: Failed to create OpenCL buffer! (Error code: " << err << ")" << std::endl;
        return nullptr;
    }
    return  buffer;
}

void fillMatFromOpenCLBuffer(cv::Mat& mat, cl_mem buffer, cl_context context, cl_command_queue queue) {
    // Check if the input cv::Mat is empty, or the OpenCL buffer is null
    if (mat.empty()) {
        std::cerr << "Error: Output cv::Mat is empty!" << std::endl;
        return;
    }

    if (buffer == nullptr) {
        std::cerr << "Error: Input OpenCL buffer is null!" << std::endl;
        return;
    }

    // Read the buffer data into the cv::Mat
    cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, mat.total() * mat.elemSize(),
        mat.data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to read OpenCL buffer into cv::Mat!" << std::endl;
        return;
    }

    // Wait for the reading operation to finish (blocking read)
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to finish OpenCL buffer read operation!" << std::endl;
        return;
    }
}
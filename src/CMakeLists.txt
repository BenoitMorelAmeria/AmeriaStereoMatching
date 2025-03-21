# CMakeList.txt : CMake project for AmeriaStereoMatching, include source and define
# project specific logic here.
#


# Base include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/external/OpenCV
    ${CMAKE_SOURCE_DIR}/external/OpenCL/include
)


link_directories(
	${CMAKE_SOURCE_DIR}/external/OpenCV
	${CMAKE_SOURCE_DIR}/external/OpenCL
)


link_libraries(
	opencv_world4100.lib
	OpenCL.lib
)



# Add source to this project's executable.
add_executable (AmeriaStereoMatching "main.cpp" "../backup/OpenCLStereoMatcher.h" "../backup/main (2).cpp" "OpenCLDeviceSelector.cpp" "OpenCLDeviceSelector.h" "OpenCLManager.cpp" "OpenCLManager.h" "cppkernels/SADKernel.h" "OpenCLProgram.cpp" "OpenCLProgram.h" "cppkernels/SADKernel.cpp" "OpenCVHelper.h" "cppkernels/SADCostKernel.cpp" "cppkernels/SADCostKernel.h" "cppkernels/HorizontalAggregationKernel.cpp" "cppkernels/HorizontalAggregationKernel.h" "cppkernels/Kernel.h" "cppkernels/Kernel.cpp" "cppkernels/ComputeBestDisparityKernel.cpp" "cppkernels/ComputeBestDisparityKernel.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET AmeriaStereoMatching PROPERTY CXX_STANDARD 20)
endif()

if(CMAKE_BUILD_ENVIRONMENT STREQUAL "Visual studio Code")
	set(CMAKE_LOCAL_BUILD_PATH "${CMAKE_BINARY_DIR}/$<CONFIG>")
elseif(CMAKE_BUILD_ENVIRONMENT STREQUAL "CMake")
	set(CMAKE_LOCAL_BUILD_PATH "${CMAKE_BINARY_DIR}/$<CONFIG>")
else() #Assume it's Visual Studio
	set(CMAKE_BUILD_ENVIRONMENT "Visual studio")
	set(CMAKE_LOCAL_BUILD_PATH "${CMAKE_BINARY_DIR}")
endif()
message("Configure for ${CMAKE_BUILD_ENVIRONMENT}. Build dir is: ${CMAKE_LOCAL_BUILD_PATH}")


add_custom_command(TARGET AmeriaStereoMatching POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/external/OpenCV/opencv_world4100.dll ${CMAKE_LOCAL_BUILD_PATH}/src/opencv_world4100.dll
)

add_custom_command(TARGET AmeriaStereoMatching POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/src/kernels.cl ${CMAKE_LOCAL_BUILD_PATH}/src/kernels.cl
)
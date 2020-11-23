#pragma once
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <map>


cl_platform_id *getPlatforms(cl_uint &numPlatforms) {
	cl_int status;

	// Get number of platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if ((status != CL_SUCCESS) || (numPlatforms <= 0)) {
		std::cerr << "Filed toto find any OpenCL platform." << std::endl;
	}

	cl_platform_id *platformsID = new cl_platform_id[numPlatforms];

	// Get platforms
	status = clGetPlatformIDs(numPlatforms, platformsID, NULL);
	if (status != CL_SUCCESS) {
		std::cerr << "Filed toto find any OpenCL platform." << std::endl;
	}

	std::cout << "Number of platforms is " << numPlatforms << ":" << std::endl;
	return platformsID;
}

void DisplayPlatformInfo(cl_platform_id id) {
	size_t info_size;
	char *info = nullptr;
	cl_int status;

	std::map <std::string, cl_platform_info> infoMap = {
		{ "Platform Name",  CL_PLATFORM_NAME },
	{ "Platform Vendor", CL_PLATFORM_VENDOR },
	{ "Platform Version", CL_PLATFORM_VERSION } };


	for (auto it = infoMap.begin(); it != infoMap.end(); ++it) {
		status = clGetPlatformInfo(id, it->second, 0, NULL, &info_size);
		if (status != CL_SUCCESS) {
			std::cerr << "\t" << "Filed to get " << it->first << "." << std::endl;
		}
		info = new char[info_size];
		status = clGetPlatformInfo(id, it->second, info_size, info, NULL);
		if (status != CL_SUCCESS) {
			std::cerr << "\t" << "Filed to get " << it->first << "." << std::endl;
		}
		std::cout << "\t" << it->first << ": " << info << std::endl;
		delete[] info;
		info = nullptr;
	}
}


cl_device_id *getDevices(cl_uint &numDevices, cl_platform_id platform_id) {
	cl_int status;

	// Get number of platforms
	status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if ((status != CL_SUCCESS) || (numDevices < 1)) {
		std::cerr << "No GPU device found for platform." << std::endl;
	}

	cl_device_id *devicesID = new cl_device_id[numDevices];

	// Get platforms
	status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, numDevices, devicesID, NULL);
	if (status != CL_SUCCESS) {
		std::cerr << "No GPU device found for platform." << std::endl;
	}

	std::cout << "Number of devices is " << numDevices << ":" << std::endl;
	return devicesID;
}


void DisplayDeviceInfo(cl::Device &device) {

	std::cout << "\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
	std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
	std::cout << "\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
	std::cout << "\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
	std::cout << "\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
	std::cout << "\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
	std::cout << "\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
	std::cout << "\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
	std::cout << "\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
	std::cout << "--------------------------------------------------------------------------------\n" << std::endl;
}


void OpenCL_Info() {
	std::vector<cl::Platform> platforms;  
    cl::Platform::get(&platforms);  

    int platform_id = 0;
    int device_id = 0;

        cl::Platform platform(platforms[0]);

        std::vector<cl::Device> devices;  

        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

            cl::Device device(devices[0]);

            std::cout << "\tUse GPU device: " << std::endl;
            std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;  
            std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
            std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;  
            std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\t\tDevice Max Work Item: "
                      << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0]
                      << " x " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1]
                      << " x " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[2]
                      << std::endl;
            std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
            std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
            std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
            std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
		std::cout << "--------------------------------------------------------------------------------" << std::endl;

}

#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <chrono>
#include <CL/cl.hpp>
#include "traits.h"
#include "settings.h"
#include "operations.hpp"

template <typename T, typename Th>
void fdn_gpu(T *Y, const int num_platf, const int num_dev, const T *S, const T *W, const int *D, const Th *Df, const Th *H)
{

#ifdef QT_DEBUG
    const char fileKernMatMul[] = "./src/kernels/shift_matrix_mult.cl";
    const char fileKernConv[]   = "./src/kernels/convolution.cl";
    const char fileKernMulW[]   = "./src/kernels/mult_w.cl";
#else
    const char fileKernMatMul[] = "./kernels/shift_matrix_mult.cl";
    const char fileKernConv[]   = "./kernels/convolution.cl";
    const char fileKernMulW[]   = "./kernels/mult_w.cl";
#endif
    // Host/device data structures
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    // Identify a platform
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cerr << " No platforms found!" << std::endl;
        exit(1);
    }
    cl::Platform default_platform = platforms.at(num_platf);

    // Get all available devices for num_platf platform
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0) {
        std::cerr << " No GPU devices found!" << std::endl;
        exit(1);
    }
    cl::Device default_device = devices.at(num_dev);

    // Setup a context for num_dev device
    cl::Context context({ default_device });

    cl_command_queue_properties Prop = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    // Setup a command queue for device
    cl::CommandQueue queue(context, default_device, Prop);

    // Load files from sources
#ifdef QT_DEBUG
    auto settings = fileRead("./src/settings.h");
#else
    auto settings = fileRead("settings.h");
#endif
    auto srcMatMul = fileRead(fileKernMatMul);
    auto srcConv = fileRead(fileKernConv);
    auto srcMulW = fileRead(fileKernMulW);

    // Push sources code into vector sources
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(settings.c_str(), settings.size()));
    sources.push_back(std::make_pair(srcMatMul.c_str(), srcMatMul.size()));
    sources.push_back(std::make_pair(srcConv.c_str(), srcConv.size()));
    sources.push_back(std::make_pair(srcMulW.c_str(), srcMulW.size()));

    // Set options for building
    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName() << " -D Th=" << dtype_traits<Th>::getName();
    if (iscplx<T>()) {
        options << " -D CPLX=1";
    } else {
        options << " -D CPLX=0";
    }

    // Build OpenCL program
    cl::Program program(context, sources);
    cl_int build_status = program.build({default_device}, options.str().c_str());
    if (build_status != CL_SUCCESS) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
                  << "\n" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(default_device) << std::endl;
        exit(1);
    }

    // Make Kernel
    cl::Kernel kernelMatMul(program, "shift_matrix_mult");
    cl::Kernel kernelConv(program, "convolve");
    cl::Kernel kernelMulW(program, "mult_w");

    size_t result;
    kernelMatMul.getWorkGroupInfo<decltype(result)>(default_device, CL_KERNEL_WORK_GROUP_SIZE, &result);
    std::cout << "local" <<  result << std::endl;

#ifdef POL2
    //----------------------------------------------
    cl::Kernel kernelMatMul1(program, "shift_matrix_mult");
    cl::Kernel kernelConv1(program, "convolve");
    //----------------------------------------------
#endif
    // Create buffers on the device
    cl::Buffer buffer_S (context, CL_MEM_WRITE_ONLY, sizeof(T)*POLAR*CHP*NP); // Signal
    cl::Buffer buffer_W (context, CL_MEM_READ_WRITE, sizeof(T)*DNP*CHP); // Coefficients
    cl::Buffer buffer_Del(context, CL_MEM_WRITE_ONLY, sizeof(Th)*DNP*CHP); // Fractional delay
    cl::Buffer buffer_D (context, CL_MEM_WRITE_ONLY, sizeof(int)*DNP*CHP); // Integer delay
    cl::Buffer buffer_SW(context, CL_MEM_READ_WRITE, sizeof(T)*POLAR*DNP*NP); // Result of matrix multiplication
    cl::Buffer buffer_H (context, CL_MEM_WRITE_ONLY, sizeof(Th)*NPOLINOM*NF); // Coefficients FIR
    cl::Buffer buffer_Y (context, CL_MEM_READ_WRITE, sizeof(T)*POLAR*DNP*(NP+NF-1)); // Output DN

//    std::cout << std::endl << "NP " << NP << std::endl;

    // Write arrays to the device
    queue.enqueueWriteBuffer(buffer_S, CL_TRUE, 0, sizeof(T)*POLAR*CHP*NP, S);
    queue.enqueueWriteBuffer(buffer_W, CL_TRUE, 0, sizeof(T)*DNP*CHP, W);
    queue.enqueueWriteBuffer(buffer_D, CL_TRUE, 0, sizeof(int)*DNP*CHP, D);
    queue.enqueueWriteBuffer(buffer_Del, CL_TRUE, 0, sizeof(Th)*DNP*CHP, Df);
    queue.enqueueWriteBuffer(buffer_H, CL_TRUE, 0, sizeof(Th)*NPOLINOM*NF, H);
    queue.enqueueWriteBuffer(buffer_Y, CL_TRUE, 0, sizeof(T)*POLAR*DNP*(NP+NF-1), Y);

    // Set arguments for kernel kernelMatMul
    kernelMatMul.setArg(0, buffer_S);
    kernelMatMul.setArg(1, buffer_D);
    kernelMatMul.setArg(2, buffer_W);
    kernelMatMul.setArg(3, buffer_SW);
    // Set arguments for kernel kernelConv
    kernelConv.setArg(0, buffer_Y);
    kernelConv.setArg(1, buffer_SW);
    kernelConv.setArg(2, buffer_H);
#ifdef POL2
    //-----------------------------------------
    // Set arguments for kernel kernelMatMul
    kernelMatMul1.setArg(0, buffer_S);
    kernelMatMul1.setArg(1, buffer_D);
    kernelMatMul1.setArg(2, buffer_W);
    kernelMatMul1.setArg(3, buffer_SW);
    // Set arguments for kernel kernelConv
    kernelConv1.setArg(0, buffer_Y);
    kernelConv1.setArg(1, buffer_SW);
    kernelConv1.setArg(2, buffer_H);
    //-----------------------------------------
#endif
    // Set arguments for kernel kernelMulW
    kernelMulW.setArg(0, buffer_W);
    kernelMulW.setArg(1, buffer_Del);

    // Configure global and local size of NDRange
#if RECTANGLE_TILE == 1
    cl::NDRange globalMatMul(DNP, NP);
    cl::NDRange localMatMul(TS_DN, TS_N);
#endif
#if RECTANGLE_TILE_WITH_REGISTER == 1
    cl::NDRange globalMatMul(DNP, NP/WPT_N);
    cl::NDRange localMatMul(TS_DN, TS_N/WPT_N);
#endif
#if QUADRATE_TILE == 1
     cl::NDRange globalMatMul(DNP, NP);
    cl::NDRange localMatMul(TS, TS);
#endif
    cl::NDRange globalConv(DIVUP((NP+NF-1), THREADS)*THREADS, DNP);
    cl::NDRange localConv(THREADS, 1);
    cl::NDRange globalMulW(DNP*CHP, 1);
    cl::NDRange localMulW(1, 1);

    // Create events
    cl::Event k_eventsMul;
    cl::Event k_eventsConv;
#ifdef POL2
    //-------------------------
    cl::Event k_eventsMul1;
    cl::Event k_eventsConv1;
    //--------------------------
#endif
    cl::Event k_eventsW;

#if TIME_TEST == 1
    int count_run = 0;
    int sum_time = 0;
    int worst_time = 0;
    std::cout<< "\tRun program on GPU:" << std::endl;
    std::cout<< "--------------------------------------------------------------------------------" << std::endl;
    while(count_run < 21)
    {
        // Time start
        std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
        count_run++;
#endif
        // Loop polinomial
        for(int m=0;m<NPOLINOM;m++)
        {
#ifdef POL2
            //----------------------------------
            // Set number polinim for kernel kernelConv
            kernelConv.setArg(4, m);
            kernelConv1.setArg(4, m);
            // Set polarization for kernels
            kernelMatMul.setArg(4, 0);
            kernelConv.setArg(3, 0);
            kernelMatMul1.setArg(4, 1);
            kernelConv1.setArg(3, 1);

            // Run kernel kernelMatMul
            queue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, globalMatMul, localMatMul, NULL, &k_eventsMul);
            queue.enqueueNDRangeKernel(kernelMatMul1, cl::NullRange, globalMatMul, localMatMul, NULL, &k_eventsMul1);

            // Run kernel kernelConv
            queue.enqueueNDRangeKernel(kernelConv, cl::NullRange, globalConv, localConv, NULL, &k_eventsConv);
            queue.enqueueNDRangeKernel(kernelConv1, cl::NullRange, globalConv, localConv, NULL, &k_eventsConv1);
            k_eventsConv.wait();
            k_eventsConv1.wait();
#else
            //----------------------------------

            // Set number polinim for kernel kernelConv
            kernelConv.setArg(4, m);

            // Loop polarization
            for (int p = 0; p < POLAR; p++)
            {
                // Set polarization for kernels
                kernelMatMul.setArg(4, p);
                kernelConv.setArg(3, p);

                // Run kernel kernelMatMul
                queue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, globalMatMul, localMatMul, NULL, &k_eventsMul);
                k_eventsMul.wait();
std::chrono::system_clock::time_point start_time_conv = std::chrono::system_clock::now();
                // Run kernel kernelConv
                queue.enqueueNDRangeKernel(kernelConv, cl::NullRange, globalConv, localConv, NULL, &k_eventsConv);
                k_eventsConv.wait();
std::chrono::system_clock::time_point end_time_conv = std::chrono::system_clock::now();
auto elapsed_conv = std::chrono::duration_cast<std::chrono::microseconds>(end_time_conv-start_time_conv);
std::cout<< "Time convolution: " << elapsed_conv.count() << " mks." << std::endl;
            }
#endif
            // Run kernel kernelMulW
            queue.enqueueNDRangeKernel(kernelMulW, cl::NullRange, globalMulW, localMulW, NULL, &k_eventsW);
            k_eventsW.wait();
        }

        // Wait finish
        queue.finish();
#if TIME_TEST == 1
        // Time end
        std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
        // Computation time
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time);
        std::cout<< "\t\tTime: " << elapsed.count() << " mks." << std::endl;
        if(count_run > 1)
        {
            sum_time += elapsed.count();
            if(worst_time < elapsed.count())
                worst_time = elapsed.count();
        }
    }

    int average_time = sum_time/(count_run);
    std::cout<< "-------------------------------------------------------------------------------" << std::endl;
    std::cout<< "\tAverage program execution time on GPU: " << average_time << " mks." << std::endl << std::endl;
    std::cout<< "\tWorst program execution time on GPU: " << worst_time << " mks." << std::endl;
    std::cout<< "-------------------------------------------------------------------------------" << std::endl;
#else

    // Read output DN
    queue.enqueueReadBuffer(buffer_Y, CL_TRUE, 0, sizeof(T)*POLAR*DNP*(NP+NF-1), Y);

#endif
}

#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <chrono>
#include <CL/cl.hpp>
#include "traits.h"
#include "settings.h"
#include "operations.hpp"

size_t read_program(char *&program_buffer,const char *kernel_file)
{
    FILE *program_handle;
    size_t program_size;

    /* Load kernel */
    /* Read program file and place content into buffer */
    program_handle = fopen(kernel_file, "r");
    if(program_handle == NULL) {
       perror("Couldn't find the program file");
       exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    return program_size;
}

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


    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_program Program;

    char platform_name[128];
    char device_name[128];

    cl_kernel kernelConv;
    cl_kernel kernelMulW;
    cl_kernel kernelMatMul;

    // Setup OpenCL environment.
    err = clGetPlatformIDs( 1, &platform, NULL );

    size_t ret_param_size = 0;
    err = clGetPlatformInfo( platform, CL_PLATFORM_NAME,
            sizeof(platform_name), platform_name,
            &ret_param_size );
//    printf("Platform found: %s\n", platform_name);

    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL );

    err = clGetDeviceInfo( device, CL_DEVICE_NAME,
            sizeof(device_name), device_name,
            &ret_param_size );
    printf("Device found on the above platform: %s\n", device_name);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    char *program_buffer;
    char *settings_buffer;
    char *conv_buffer;
    char *sh_matr_mult_buffer;
    char *prog_buff_mult_w;

    size_t settings = read_program(settings_buffer, "settings.h");
    size_t size_conv = read_program(conv_buffer, fileKernConv);
    size_t size_sh_matr_mult = read_program(sh_matr_mult_buffer, fileKernMatMul);
    size_t size_mult_w = read_program(prog_buff_mult_w, fileKernMulW);

    // Size program buffer
    size_t program_size = settings + size_conv + size_mult_w + size_sh_matr_mult + 1;

    program_buffer = (char*) malloc(program_size);
    // Write program buffer
    strcpy( program_buffer, settings_buffer );
    strcat( program_buffer, sh_matr_mult_buffer );
    strcat( program_buffer, conv_buffer );
    strcat( program_buffer, prog_buff_mult_w );

    Program = clCreateProgramWithSource(ctx, 1,
       (const char**)&program_buffer, &program_size, &err);

    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    char def[200];
    int res = snprintf(def, sizeof(def),
                       "-D T=float2 -D Th=float "
                       "-D CPLX=1 "
                       );

    // Build the program executable
    err = clBuildProgram(Program, 0, NULL, def, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("building program failed\n");
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            size_t log_size;
            clGetProgramBuildInfo( Program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size );
            char *log = (char *) malloc(log_size);
            clGetProgramBuildInfo( Program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL );
            printf("%s\n", log);
        }
    }

    // Create kerrnels.
    kernelConv = clCreateKernel( Program, "convolve", &err );
    if (err != CL_SUCCESS) {
            std::cout << std::endl << "HOST-Error: Failed to create kernelConv" << err << std::endl;
            throw std::exception();
    }
    kernelMatMul = clCreateKernel( Program, "shift_matrix_mult", &err );
    if (err != CL_SUCCESS) {
            std::cout << std::endl << "HOST-Error: Failed to create kernelMatMul" << err << std::endl;
            throw std::exception();
    }
    kernelMulW = clCreateKernel( Program, "mult_w", &err );
    if (err != CL_SUCCESS) {
            std::cout << std::endl << "HOST-Error: Failed to create kernelMulW" << err << std::endl;
            throw std::exception();
    }

#ifdef POL2
    //----------------------------------------------
    cl::Kernel kernelMatMul1(program, "shift_matrix_mult");
    cl::Kernel kernelConv1(program, "convolve");
    //----------------------------------------------
#endif
    cl_mem buffer_S = clCreateBuffer( ctx, CL_MEM_WRITE_ONLY, sizeof(T)*POLAR*CHP*NP, NULL, &err ); // Signal
    cl_mem buffer_W = clCreateBuffer( ctx, CL_MEM_READ_WRITE, sizeof(T)*DNP*CHP, NULL, &err ); // Coefficients
    cl_mem buffer_Del = clCreateBuffer( ctx, CL_MEM_WRITE_ONLY, sizeof(Th)*DNP*CHP, NULL, &err ); // Fractional delay
    cl_mem buffer_D = clCreateBuffer( ctx, CL_MEM_WRITE_ONLY, sizeof(int)*DNP*CHP, NULL, &err ); // Integer delay
    cl_mem buffer_SW = clCreateBuffer( ctx, CL_MEM_READ_WRITE, sizeof(T)*POLAR*DNP*NP, NULL, &err ); // Result of matrix multiplication
    cl_mem buffer_H = clCreateBuffer( ctx, CL_MEM_WRITE_ONLY, sizeof(Th)*NPOLINOM*NF, NULL, &err ); // Coefficients FIR
    cl_mem buffer_Y = clCreateBuffer( ctx, CL_MEM_READ_WRITE, sizeof(T)*POLAR*DNP*(NP+NF-1), NULL, &err ); // Output DN

    err  = clEnqueueWriteBuffer( queue, buffer_S, CL_TRUE, 0, sizeof(T)*POLAR*CHP*NP, S, 0, NULL, NULL );
    err |= clEnqueueWriteBuffer( queue, buffer_W, CL_TRUE, 0, sizeof(T)*DNP*CHP, W, 0, NULL, NULL );
    err |= clEnqueueWriteBuffer( queue, buffer_D, CL_TRUE, 0, sizeof(int)*DNP*CHP, D, 0, NULL, NULL );
    err |= clEnqueueWriteBuffer( queue, buffer_Del, CL_TRUE, 0, sizeof(Th)*DNP*CHP, Df, 0, NULL, NULL );
    err |= clEnqueueWriteBuffer( queue, buffer_H, CL_TRUE, 0, sizeof(Th)*NPOLINOM*NF, H, 0, NULL, NULL );
    err |= clEnqueueWriteBuffer( queue, buffer_Y, CL_TRUE, 0, sizeof(T)*POLAR*DNP*(NP+NF-1), Y, 0, NULL, NULL );

    err  = clSetKernelArg( kernelMatMul, 0, sizeof(buffer_S), &buffer_S );
    err |= clSetKernelArg( kernelMatMul, 1, sizeof(buffer_D), &buffer_D );
    err |= clSetKernelArg( kernelMatMul, 2, sizeof(buffer_W), &buffer_W );
    err |= clSetKernelArg( kernelMatMul, 3, sizeof(buffer_SW), &buffer_SW );

    err |= clSetKernelArg( kernelConv, 0, sizeof(buffer_Y), &buffer_Y );
    err |= clSetKernelArg( kernelConv, 1, sizeof(buffer_SW), &buffer_SW );
    err |= clSetKernelArg( kernelConv, 2, sizeof(buffer_H), &buffer_H );

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

    err |= clSetKernelArg( kernelMulW, 0, sizeof(buffer_W), &buffer_W );
    err |= clSetKernelArg( kernelMulW, 1, sizeof(buffer_Del), &buffer_Del );

    // Configure global and local size of NDRange
#if RECTANGLE_TILE == 1
    // Configure global and local size for K_shift_matrix_mult
    size_t globalMatMul[2];
    size_t localMatMul[2];
    globalMatMul[1] = NP;
    globalMatMul[0] = DNP;
    localMatMul[1]  = TS_N;
    localMatMul[0]  = TS_DN;

    std::cout << "NP " << NP << ", DNP " << DNP << ", TS_N " << TS_N << ", TS_DN " << TS_DN << std::endl;

#endif
#if RECTANGLE_TILE_WITH_REGISTER == 1
    cl::NDRange globalMatMul(DNP, NP/WPT_N);
    cl::NDRange localMatMul(TS_DN, TS_N/WPT_N);
#endif
#if QUADRATE_TILE == 1
     cl::NDRange globalMatMul(DNP, NP);
    cl::NDRange localMatMul(TS, TS);
#endif
    // Configure global and local size for K_convolve
    size_t globalConv[2];
    size_t localConv[2];
    globalConv[1] = DNP;
    globalConv[0] = DIVUP((NP+NF-1), THREADS)*THREADS;
    localConv[1]  = 1;
    localConv[0]  = THREADS;
    // Configure global and local size for K_mult_w
    size_t globalMulW[2];
    size_t localMulW[2];
    globalMulW[1] = 1;
    globalMulW[0] = DNP*CHP;
    localMulW[1]  = 1;
    localMulW[0]  = 1;


    static size_t local;
    err = clGetKernelWorkGroupInfo(kernelMatMul, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);

    std::cout << "local " << local << std::endl;

#ifdef POL2
    //-------------------------
    cl::Event k_eventsMul1;
    cl::Event k_eventsConv1;
    //--------------------------
#endif

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
            err  = clSetKernelArg( kernelConv, 4, sizeof(int), &m );
            if(err != CL_SUCCESS)
            {
                std::cerr << "HOST-Error: Failed set argument for kernel kernelConv 4. Err=" << err << std::endl;
                throw std::exception();
            }

            for (int p = 0; p < POLAR; p++)
            {
                err  = clSetKernelArg( kernelMatMul, 4, sizeof(int), &p );
                err |= clSetKernelArg( kernelConv, 3, sizeof(int), &p );
                if(err != CL_SUCCESS)
                {
                    std::cerr << "HOST-Error: Failed set argument for kernel kernelMatMul, kernelConv. Err=" << err << std::endl;
                    throw std::exception();
                }

                // Run K_shift_matrix_mult
                err = clEnqueueNDRangeKernel( queue, kernelMatMul, 2, NULL, globalMatMul, localMatMul, 0, NULL, NULL );
                if(err != CL_SUCCESS)
                {
                    std::cerr << "HOST-Error: Failed to run K_shift_matrix_mult. Err=" << err << std::endl;
                    throw std::exception();
                }
                // Run K_convolve
                err = clEnqueueNDRangeKernel( queue, kernelConv, 2, NULL, globalConv, localConv, 0, NULL, NULL );
                if(err != CL_SUCCESS)
                {
                    std::cerr << "HOST-Error: Failed to run K_convolve. Err=" << err << std::endl;
                    throw std::exception();
                }
            }

#endif
            // Run K_mult_w
            err = clEnqueueNDRangeKernel( queue, kernelMulW, 2, NULL, globalMulW, localMulW, 0, NULL, NULL );
            if(err != CL_SUCCESS)
            {
                std::cerr << "HOST-Error: Failed to run K_mult_w. Err=" << err << std::endl;
                throw std::exception();
            }
        }
        err = clFinish( queue );

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

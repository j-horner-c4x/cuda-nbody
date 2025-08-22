/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header
// files, please refer the CUDA examples for examples of the needed CUDA
// headers, which may change depending on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}
#endif

template <typename T> void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char* errorMessage, const char* file, const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file,
                line,
                errorMessage,
                static_cast<int>(err),
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char* errorMessage, const char* file, const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file,
                line,
                errorMessage,
                static_cast<int>(err),
                cudaGetErrorString(err));
    }
}
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;    // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},
                                       {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {0xa0, 128}, {0xa1, 128}, {0xc0, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf("MapSMtoCores for SM %d.%d is undefined."
           "  Default to use %d Cores/SM\n",
           major,
           minor,
           nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the GPU Arch name)
    typedef struct {
        int SM;    // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        const char* name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {
        {0x30, "Kepler"}, {0x32, "Kepler"}, {0x35, "Kepler"}, {0x37, "Kepler"}, {0x50, "Maxwell"}, {0x52, "Maxwell"}, {0x53, "Maxwell"}, {0x60, "Pascal"},    {0x61, "Pascal"},    {0x62, "Pascal"},    {0x70, "Volta"},
        {0x72, "Xavier"}, {0x75, "Turing"}, {0x80, "Ampere"}, {0x86, "Ampere"}, {0x87, "Ampere"},  {0x89, "Ada"},     {0x90, "Hopper"},  {0xa0, "Blackwell"}, {0xa1, "Blackwell"}, {0xc0, "Blackwell"}, {-1, "Graphics Device"}};

    int index = 0;

    while (nGpuArchNameSM[index].SM != -1) {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchNameSM[index].name;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf("MapSMtoArchName for SM %d.%d is undefined."
           "  Default to use %s\n",
           major,
           minor,
           nGpuArchNameSM[index - 1].name);
    return nGpuArchNameSM[index - 1].name;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID) {
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr,
                "gpuDeviceInit() CUDA error: "
                "no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0) {
        devID = 0;
    }

    if (devID > device_count - 1) {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_count);
        fprintf(stderr,
                ">> gpuDeviceInit (-device=%d) is not a valid"
                " GPU device. <<\n",
                devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    if (computeMode == cudaComputeModeProhibited) {
        fprintf(stderr,
                "Error: device is running in <Compute Mode "
                "Prohibited>, no threads can use cudaSetDevice().\n");
        return -1;
    }

    if (major < 1) {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName(major, minor));

    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device    = 0;
    int device_count       = 0;
    int devices_prohibited = 0;

    uint64_t max_compute_perf = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() CUDA error:"
                " no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count) {
        int computeMode = -1, major = 0, minor = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != cudaComputeModeProhibited) {
            if (major == 9999 && minor == 9999) {
                sm_per_multiproc = 1;
            } else {
                sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
            }
            int multiProcessorCount = 0, clockRate = 0;
            checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
            cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
            if (result != cudaSuccess) {
                // If cudaDevAttrClockRate attribute is not supported we
                // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
                if (result == cudaErrorInvalidValue) {
                    clockRate = 1;
                } else {
                    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__, static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
                    exit(EXIT_FAILURE);
                }
            }
            uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

            if (compute_perf > max_compute_perf) {
                max_compute_perf = compute_perf;
                max_perf_device  = current_device;
            }
        } else {
            devices_prohibited++;
        }

        ++current_device;
    }

    if (devices_prohibited == device_count) {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() CUDA error:"
                " all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }

    return max_perf_device;
}

#endif

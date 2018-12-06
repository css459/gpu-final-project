//
// Created by cole on 11/8/18.
//

#include <cuda.h>
#include <stdio.h>
#include "cuda_util.h"

// Method to check for CUDA errors
#define cudaCheckError(err) {                                                                    \
    if (err != cudaSuccess) {                                                                    \
        fprintf(stderr,"[ ERR ] CUDA: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                                                                 \
    }                                                                                            \
}

int get_num_cuda_devices() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    cudaCheckError(err);

    if (device_count < 1) {
        printf("[ ERR ] No CUDA devices, aborting.\n");
        exit(1);
    }

    return device_count;
}

size_t get_global_memory_size_for_device(int device_number) {
    cudaDeviceProp dev;
    cudaError_t err = cudaGetDeviceProperties(&dev, 0);
    cudaCheckError(err);

    return dev.totalGlobalMem;

}
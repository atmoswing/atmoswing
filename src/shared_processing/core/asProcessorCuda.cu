/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

// Disable some MSVC warnings
#ifdef _MSC_VER
#pragma warning( disable : 4244 ) // C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data
#pragma warning( disable : 4267 ) // C4267: conversion from 'size_t' to 'int', possible loss of data
#endif

#include "asProcessorCuda.cuh"
#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#define _TIME_CUDA false

// The number of threads per block should be a multiple of 32 threads, because this provides optimal computing
// efficiency and facilitates coalescing.
static const int maxBlockSize = 1024;

__device__
void warpReduce64(volatile float *shared, int tid)
{
    shared[tid] += shared[tid + 32];
    shared[tid] += shared[tid + 16];
    shared[tid] += shared[tid + 8];
    shared[tid] += shared[tid + 4];
    shared[tid] += shared[tid + 2];
    shared[tid] += shared[tid + 1];
}

__device__
void warpReduce32(volatile float *shared, int tid)
{
    shared[tid] += shared[tid + 16];
    shared[tid] += shared[tid + 8];
    shared[tid] += shared[tid + 4];
    shared[tid] += shared[tid + 2];
    shared[tid] += shared[tid + 1];
}

__global__
void processS1grads(int blockSize, long candNb, int ptsNb, const float *data, const long *idxTarg, const long *idxArch, float w, float *out)
{
    const long blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;

    if (blockId < candNb) {
        int bs = blockSize;
        int nPts = ptsNb;
        long iCand = blockId;
        int iPt = threadIdx.x;

        extern __shared__ float mem[];

        float *diff = mem;
        float *amax = &diff[bs];

        // Process differences and get abs max
        if (iPt < nPts) {

            // Lookup data value
            float xi = data[idxTarg[iCand] * nPts + iPt];
            float yi = data[idxArch[iCand] * nPts + iPt];

            float diffi = xi - yi;
            float amaxi = fabs(xi);
            if (fabs(yi) > amaxi) {
                amaxi = fabs(yi);
            }

            diff[iPt] = fabs(diffi);
            amax[iPt] = amaxi;
        } else {
            // Set rest of the block to 0
            diff[iPt] = 0;
            amax[iPt] = 0;
        }
        __syncthreads();

        // Process sum reduction
        for (unsigned int stride = bs / 2; stride > 32; stride /= 2) {
            if (iPt < stride) {
                diff[iPt] += diff[iPt + stride];
                amax[iPt] += amax[iPt + stride];
            }
            __syncthreads();
        }
        if (bs >= 64) {
            if (iPt < 32) {
                warpReduce64(diff, iPt);
                warpReduce64(amax, iPt);
            }
        } else {
            if (iPt < 16) {
                warpReduce32(diff, iPt);
                warpReduce32(amax, iPt);
            }
        }

        // Process final score
        if (iPt == 0) {
            float res = 0;

            if (amax[0] == 0) {
                res = 200;
            } else {
                res = 100.0f * (diff[0] / amax[0]);
            }
            *(out + iCand) += res * w;
        }
    }
}

bool asProcessorCuda::ProcessCriteria(std::vector<std::vector<float *>> &data, std::vector<int> &indicesTarg,
                                      std::vector<std::vector<int>> &indicesArch,
                                      std::vector<std::vector<float>> &resultingCriteria,
                                      std::vector<int> &nbCandidates, std::vector<int> &colsNb,
                                      std::vector<int> &rowsNb, std::vector<float> &weights,
                                      std::vector<CudaCriteria> &criteria)
{
    int ptorsNb = weights.size();

#if _TIME_CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;
#endif

    // Sizes
    long candNb = 0;
    std::vector<long> indexStart(nbCandidates.size() + 1);
    for (int i = 0; i < nbCandidates.size(); i++) {
        indexStart[i] = candNb;
        candNb += nbCandidates[i];
    }
    indexStart[nbCandidates.size()] = candNb;

    // Alloc space for indices
#if _TIME_CUDA
    cudaEventRecord(start);
#endif
    long *hIdxTarg, *dIdxTarg;
    hIdxTarg = (long *)malloc(candNb * sizeof(long));
    checkCudaErrors(cudaMalloc((void **)&dIdxTarg, candNb * sizeof(long)));
    long *hIdxArch, *dIdxArch;
    hIdxArch = (long *)malloc(candNb * sizeof(long));
    checkCudaErrors(cudaMalloc((void **)&dIdxArch, candNb * sizeof(long)));
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to allocate IdxTarg and IdxArch:    %f\n", milliseconds);
#endif

#if _TIME_CUDA
    cudaEventRecord(start);
#endif
    for (int i = 0; i < indicesTarg.size(); i++) {
        for (int j = 0; j < nbCandidates[i]; j++) {
            hIdxArch[indexStart[i] + j] = indicesArch[i][j];
            hIdxTarg[indexStart[i] + j] = indicesTarg[i];
        }
    }
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to initialize IdxTarg and IdxArch:  %f\n", milliseconds);
#endif

    // Copy to device
#if _TIME_CUDA
    cudaEventRecord(start);
#endif
    checkCudaErrors(cudaMemcpy(dIdxTarg, hIdxTarg, candNb * sizeof(long), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dIdxArch, hIdxArch, candNb * sizeof(long), cudaMemcpyHostToDevice));
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to copy IdxTarg and IdxArch:        %f\n", milliseconds);
#endif

    // Alloc space for results
#if _TIME_CUDA
    cudaEventRecord(start);
#endif
    float *hRes, *dRes;
    hRes = (float *)malloc(candNb * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dRes, candNb * sizeof(float)));
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to allocate dRes:                   %f\n", milliseconds);
#endif

    // Init resulting array to 0s
#if _TIME_CUDA
    cudaEventRecord(start);
#endif
    checkCudaErrors(cudaMemset(dRes, 0, candNb * sizeof(float)));
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to memset dRes:                     %f\n", milliseconds);
#endif

    // Get max predictor size
    long maxDataSize = 0;
    for (int iPtor = 0; iPtor < ptorsNb; iPtor++) {
        int ptsNb = colsNb[iPtor] * rowsNb[iPtor];
        long dataSize = data[iPtor].size() * ptsNb;
        if (dataSize > maxDataSize) {
            maxDataSize = dataSize;
        }
    }

    // Alloc space for data
#if _TIME_CUDA
    cudaEventRecord(start);
#endif
    float *hData, *dData;
    hData = (float *)malloc(maxDataSize * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dData, maxDataSize * sizeof(float)));
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to allocate dData:                  %f\n", milliseconds);
#endif

    // Loop over all predictors
    for (int iPtor = 0; iPtor < ptorsNb; iPtor++) {

        int ptsNb = colsNb[iPtor] * rowsNb[iPtor];
        float weight = weights[iPtor];
        long dataSize = data[iPtor].size() * ptsNb;

        // Copy data in the new arrays
#if _TIME_CUDA
        cudaEventRecord(start);
#endif
        for (int iDay = 0; iDay < data[iPtor].size(); iDay++) {
            for (int iPt = 0; iPt < ptsNb; iPt++) {
                hData[iDay * ptsNb + iPt] = data[iPtor][iDay][iPt];
            }
        }
#if _TIME_CUDA
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("time to initialize hData:                %f\n", milliseconds);
#endif

        // Copy the data to the device
#if _TIME_CUDA
        cudaEventRecord(start);
#endif
        checkCudaErrors(cudaMemcpy(dData, hData, dataSize * sizeof(float), cudaMemcpyHostToDevice));
#if _TIME_CUDA
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("time to copy data:                       %f\n", milliseconds);
#endif

        // Reduction only allowed on 1 block yet
        if (ptsNb > maxBlockSize) {
            printf("Using more than 1 gpu block (too much data points)\n");
            return false;
        }

        // Define block size (must be multiple of 32) and blocks nb
        int blockSize = (int)ceil(ptsNb / 32.0) * 32;
        int blocksNbXY = ceil(std::cbrt(candNb));
        int blocksNbZ = ceil((double)candNb / (blocksNbXY * blocksNbXY));
        dim3 blocksNb3D(blocksNbXY, blocksNbXY, blocksNbZ);

        // Launch kernel
#if _TIME_CUDA
        cudaEventRecord(start);
#endif
        switch (criteria[iPtor]) {
            case S1grads:
                // 3rd <<< >>> argument is for the dynamically allocated shared memory
                processS1grads<<<blocksNb3D, blockSize, 2*blockSize*sizeof(float)>>>(blockSize, candNb, ptsNb, dData, dIdxTarg, dIdxArch, weight, dRes);
                break;
            default:
                printf("Criteria not yet implemented on GPU.");
                return false;
        }

        // Check for any errors launching the kernel
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaDeviceSynchronize());
#if _TIME_CUDA
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("time for kernel:                         %f\n", milliseconds);
#endif
    }

    // Copy the resulting array to the device
#if _TIME_CUDA
    cudaEventRecord(start);
#endif
    checkCudaErrors(cudaMemcpy(hRes, dRes, candNb * sizeof(float), cudaMemcpyDeviceToHost));
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to copy results:                    %f\n", milliseconds);
#endif

    // Set the criteria values in the vector container
    for (int i = 0; i < nbCandidates.size(); i++) {
        std::vector<float> tmpCrit(nbCandidates[i]);

        for (int j = 0; j < nbCandidates[i]; j++) {
            tmpCrit[j] = hRes[indexStart[i] + j];
        }
        resultingCriteria[i] = tmpCrit;
    }

    free(hData);
    checkCudaErrors(cudaFree(dData));
    free(hRes);
    checkCudaErrors(cudaFree(dRes));
    free(hIdxTarg);
    checkCudaErrors(cudaFree(dIdxTarg));
    free(hIdxArch);
    checkCudaErrors(cudaFree(dIdxArch));

    return true;
}

bool asProcessorCuda::SelectBestDevice()
{
    cudaError_t cudaStatus;
    bool showDeviceName = false;

    // Count the devices
    int devicesCount = 0;
    cudaStatus = cudaGetDeviceCount(&devicesCount);
    if (cudaStatus != cudaSuccess) {
        if (cudaStatus == cudaErrorNoDevice) {
            printf("cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\n");
            return false;
        } else if (cudaStatus == cudaErrorInsufficientDriver) {
            printf("cudaGetDeviceCount failed! No driver can be loaded to determine if any device exists.\n");
            return false;
        }

        printf("cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\n");
        return false;
    }

    // Get some info on the devices
    int bestDevice = 0;
    int memSize = 0;
    struct cudaDeviceProp deviceProps;
    for (int i_dev = 0; i_dev < devicesCount; i_dev++) {
        checkCudaErrors(cudaGetDeviceProperties(&deviceProps, i_dev));
        if (showDeviceName) {
            printf("CUDA device [%s]\n", deviceProps.name);
        }

        // Compare memory
        if (deviceProps.totalGlobalMem > memSize) {
            memSize = deviceProps.totalGlobalMem;
            bestDevice = i_dev;
        }
    }

    // Select the best device
    checkCudaErrors(cudaSetDevice(bestDevice));

    return true;
}

float *asProcessorCuda::MallocCudaData(int n)
{
    float *data;
    checkCudaErrors(cudaMallocManaged(&data, n * sizeof(float)));

    return data;
}

void asProcessorCuda::FreeCudaData(float *data)
{
    checkCudaErrors(cudaFree(data));
}

void asProcessorCuda::DeviceSynchronize()
{
    checkCudaErrors(cudaDeviceSynchronize());
}

void asProcessorCuda::DeviceReset()
{
    cudaDeviceReset();
}


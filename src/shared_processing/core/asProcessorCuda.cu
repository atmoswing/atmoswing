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
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

// The number of threads per block should be a multiple of 32 threads, because this provides optimal computing
// efficiency and facilitates coalescing.
static const int blockSize = 1024;

__global__
void criteriaS1grads(int n, const float *x, const float *y, float w, float *out)
{
    // Only on a single block for now
    int idx = threadIdx.x;

    __shared__ float diff[blockSize];
    __shared__ float amax[blockSize];

    // Process differences and get abs max
    if (idx < n) {
        float xi = x[idx];
        float yi = y[idx];

        float diffi = xi - yi;
        float amaxi = fabs(xi);
        if (fabs(yi) > amaxi) {
            amaxi = fabs(yi);
        }

        diff[idx] = fabs(diffi);
        amax[idx] = amaxi;
    }
    __syncthreads();

    // Set rest of the block to 0
    if (idx >= n) {
        diff[idx] = 0;
        amax[idx] = 0;
    }
    __syncthreads();

    // Process sum reduction
    for (int size = blockSize / 2; size > 0; size /= 2) {
        if (idx < size) {
            diff[idx] += diff[idx + size];
            amax[idx] += amax[idx + size];
        }
        __syncthreads();
    }

    // Process final score
    if (idx == 0) {
        float res = 0;

        if (amax[0] == 0) {
            res = 200;
        } else {
            res = 100.0f * (diff[0] / amax[0]);
        }
        *out += res * w;
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

    // Sizes
    int candNb = 0;
    std::vector<int> indexStart(nbCandidates.size() + 1);
    for (int i = 0; i < nbCandidates.size(); i++) {
        indexStart[i] = candNb;
        candNb += nbCandidates[i];
    }
    indexStart[nbCandidates.size()] = candNb;

    std::vector<int> arrIndicesTarg(candNb);
    std::vector<int> arrIndicesArch(candNb);

    for (int i = 0; i < indicesTarg.size(); i++) {
        for (int j = 0; j < nbCandidates[i]; j++) {
            arrIndicesArch[indexStart[i] + j] = indicesArch[i][j];
            arrIndicesTarg[indexStart[i] + j] = indicesTarg[i];
        }
    }

    // Alloc space for results
    float *hRes, *dRes;
    hRes = (float*)malloc(candNb * sizeof(float));
    checkCudaErrors(cudaMalloc((void **) &dRes, candNb * sizeof(float)));

    for (int i = 0; i < candNb; ++i) {
        hRes[i] = 0;
    }

    // Copy the resulting array to the device
    checkCudaErrors(cudaMemcpy(dRes, hRes, candNb * sizeof(float), cudaMemcpyHostToDevice));

    for (int iPtor = 0; iPtor < ptorsNb; iPtor++) {

        int ptsNb = colsNb[iPtor] * rowsNb[iPtor];
        float weight = weights[iPtor];
        int dataSize = data[iPtor].size() * ptsNb;

        // Alloc space for data
        float *hData, *dData;
        hData = (float*)malloc(dataSize * sizeof(float));
        checkCudaErrors(cudaMalloc((void **) &dData, dataSize * sizeof(float)));


        // Copy data in the new arrays
        for (int iDay = 0; iDay < data[iPtor].size(); iDay++) {
            for (int iPt = 0; iPt < ptsNb; iPt++) {
                hData[iDay * ptsNb + iPt] = data[iPtor][iDay][iPt];
            }
        }

        // Copy the data to the device
        checkCudaErrors(cudaMemcpy(dData, hData, dataSize * sizeof(float), cudaMemcpyHostToDevice));

        // Launch kernel on GPU
        int blocksNb = (ptsNb + blockSize - 1) / blockSize;

        if (blocksNb > 1) {
            printf("blocksNb > 1\n");
            return false;
        }

        for (int iCand = 0; iCand < candNb; iCand += 1) {

            int targIndex = arrIndicesTarg[iCand] * ptsNb;
            int archIndex = arrIndicesArch[iCand] * ptsNb;

            switch (criteria[iPtor]) {
            case S1grads:
                criteriaS1grads<<<blocksNb, blockSize>>>(ptsNb, dData + targIndex, dData + archIndex, weight, dRes + iCand);
                break;
            default:
                printf("Criteria not yet implemented on GPU.");
                return false;
            }

            // Check for any errors launching the kernel
            checkCudaErrors(cudaGetLastError());
        }

        checkCudaErrors(cudaDeviceSynchronize());

        free(hData);
        checkCudaErrors(cudaFree(dData));
    }

    // Copy the resulting array to the device
    checkCudaErrors(cudaMemcpy(hRes, dRes, candNb * sizeof(float), cudaMemcpyDeviceToHost));

    // Set the criteria values in the vector container
    for (int i = 0; i < nbCandidates.size(); i++) {
        std::vector<float> tmpCrit(nbCandidates[i]);

        for (int j = 0; j < nbCandidates[i]; j++) {
            tmpCrit[j] = hRes[indexStart[i] + j];
        }
        resultingCriteria[i] = tmpCrit;
    }

    free(hRes);
    checkCudaErrors(cudaFree(dRes));

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


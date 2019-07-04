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


// From https://riptutorial.com/cuda/example/22456/single-block-parallel-reduction-for-commutative-operator
__global__
void sumSingleBlock(int n, const float *a, float *out)
{
    int idx = threadIdx.x;
    float sum = 0;
    for (int i = idx; i < n; i += blockSize)
        sum += a[i];
    __shared__ float r[blockSize];
    r[idx] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) {
        if (idx < size)
            r[idx] += r[idx + size];
        __syncthreads();
    }
    if (idx == 0)
        *out = r[0];
}

// From https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__
void diff(int n, const float *x, const float *y, float *r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        r[i] = x[i] - y[i];
    }
}

__global__
void maxAbs(int n, const float *x, const float *y, float *r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        r[i] = fabs(x[i]);
        if (fabs(y[i]) > r[i]) {
            r[i] = fabs(y[i]);
        }
    }
}

__global__
void criteriaS1grads(int n, const float *x, const float *y, float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float diff[blockSize];
    __shared__ float amax[blockSize];

    for (int i = idx; i < n; i += stride) {
        float xi = x[i];
        float yi = y[i];

        float diffi = xi - yi;
        float amaxi = fabs(xi);
        if (fabs(yi) > amaxi) {
            amaxi = fabs(yi);
        }

        diff[i] = diffi;
        amax[i] = amaxi;
    }
    __syncthreads();

    float sumDiff = 0;
    float sumMax = 0;
    for (int i = idx; i < n; i += blockSize) {
        sumDiff += fabs(diff[i]);
        sumMax += amax[i];
    }

    __shared__ float rDiff[blockSize];
    __shared__ float rMax[blockSize];
    rDiff[idx] = sumDiff;
    rMax[idx] = sumMax;
    __syncthreads();

    for (int size = blockSize / 2; size > 0; size /= 2) {
        if (idx < size) {
            rDiff[idx] +=rDiff[idx + size];
            rMax[idx] +=rMax[idx + size];
        }
        __syncthreads();
    }
    if (idx == 0) {
        *out = 100.0f * (rDiff[0] / rMax[0]);

        if (rMax[0] == 0) {
            if (rDiff[0] == 0) {
                *out = 0;
            } else {
                *out = 200;
            }
        }
    }
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

bool asProcessorCuda::ProcessS1grads(float *out, const float *refData, const float *evalData, int rowsNb, int colsNb)
{
    int n = rowsNb * colsNb;

 

    int blocksNb = (n + blockSize - 1) / blockSize;

    if (blocksNb > 1) {
        printf("blocksNb > 1\n");
        return false;
    }


    bool m_checkNaNs = false;


    // Note here that the actual gradient data do not fill the entire data blocks,
    // but the rest being 0-filled, we can simplify the sum calculation !

    if (!m_checkNaNs) {
        criteriaS1grads<<<blocksNb, blockSize>>>(n, refData, evalData, out);
    } else {
        /*
        a2f refDataCorr = (!evalData.isNaN() && !refData.isNaN()).select(refData, 0);
        a2f evalDataCorr = (!evalData.isNaN() && !refData.isNaN()).select(evalData, 0);

        dividend = ((refDataCorr - evalDataCorr).abs()).sum();
        divisor = (refDataCorr.abs().max(evalDataCorr.abs())).sum();*/
    }

    return true;
}


__global__
void allPredictorsCriteriaS1grads(float *criteria, const float *data, const int *indicesTarg,
                                  const int *indicesArch, const cudaPredictorsDataPropStruct dataProp,
                                  const int n_cand, const int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    int stride = blockDim.x * gridDim.x;

    for (int iCand = idx; iCand < n_cand; iCand += stride) {

        float criterion;
        int targIndexBase = indicesTarg[iCand] * dataProp.totPtsNb;
        int archIndexBase = indicesArch[iCand] * dataProp.totPtsNb;

        for (int iPtor = 0; iPtor < dataProp.ptorsNb; iPtor++) {
            int targIndex = targIndexBase + dataProp.indexStart[iPtor];
            int archIndex = archIndexBase + dataProp.indexStart[iPtor];



            int blocksNb = 1;

            __shared__ float r;

            switch (dataProp.criteria[iPtor]) {
                case S1grads:
                    criteriaS1grads< < <blocksNb, blockSize> > >(dataProp.ptsNb[iPtor], &data[targIndex], &data[archIndex], &r);
                default:
                    printf("Incorrect criteria provided.");
            }
            cudaDeviceSynchronize();

            criterion += dataProp.weights[iPtor] * r;
        }

        criteria[iCand] = criterion;
    }

}

bool asProcessorCuda::ProcessCriteria(std::vector<std::vector<float *>> &data, std::vector<int> &indicesTarg,
                                      std::vector<std::vector<int>> &indicesArch,
                                      std::vector<std::vector<float>> &resultingCriteria,
                                      std::vector<int> &nbCandidates, const cudaPredictorsDataPropStruct &struc)
{

    // Sizes
    int nbCandidatesSum = 0;
    std::vector<int> indexStart(nbCandidates.size() + 1);
    for (int i = 0; i < nbCandidates.size(); i++) {
        indexStart[i] = nbCandidatesSum;
        nbCandidatesSum += nbCandidates[i];
    }
    indexStart[nbCandidates.size()] = nbCandidatesSum;

    // Blocks of threads
    int n_cand = nbCandidatesSum;
    int blocksNb = (n_cand + blockSize - 1) / blockSize;


#if USE_STREAMS
    // Create streams
    const int nStreams = 4; // no need to change
    // rowsNbPerStream must be dividable by nStreams and blockSize
    int rowsNbPerStream = ceil(float(nbCandidatesSum) / float(nStreams * blockSize)) * blockSize;
    // Streams
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&stream[i]);
    }
#endif

    // Data pointers
    float *arrData, *arrCriteria;
    int *arrIndicesTarg, *arrIndicesArch;

    // Alloc space for data
    checkCudaErrors(cudaMallocManaged(&arrData, data.size() * struc.totPtsNb * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&arrCriteria, nbCandidatesSum * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&arrIndicesTarg, nbCandidatesSum * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&arrIndicesArch, nbCandidatesSum * sizeof(int)));

    // Copy data in the new arrays
    for (int iDay = 0; iDay < data.size(); iDay++) {
        for (int iPtor = 0; iPtor < struc.ptorsNb; iPtor++) {
            for (int iPt = 0; iPt < struc.ptsNb[iPtor]; iPt++) {
                arrData[iDay * struc.totPtsNb + struc.indexStart[iPtor] + iPt] = data[iDay][iPtor][iPt];
            }
        }
    }

    for (int i = 0; i < indicesTarg.size(); i++) {
        for (int j = 0; j < nbCandidates[i]; j++) {
            arrIndicesArch[indexStart[i] + j] = indicesArch[i][j];
            arrIndicesTarg[indexStart[i] + j] = indicesTarg[i];
        }
    }

    // Launch kernel on GPU
#if USE_STREAMS
    for (int i = 0; i < nStreams; i++) {
        int offset = i * rowsNbPerStream;
        blocksNb = rowsNbPerStream / blockSize;
        gpuPredictorCriteriaS1grads<<<blocksNb, blockSize, 0, stream[i]>>>(arrCriteria, arrData, arrIndicesTarg, arrIndicesArch, struc, n_cand, offset);
    }
#else
    allPredictorsCriteriaS1grads< < <blocksNb, blockSize> > >(arrCriteria, arrData, arrIndicesTarg, arrIndicesArch, struc, n_cand, 0);
#endif

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Set the criteria values in the vector container
    for (int i = 0; i < nbCandidates.size(); i++) {
        std::vector<float> tmpCrit(nbCandidates[i]);

        for (int j = 0; j < nbCandidates[i]; j++) {
            tmpCrit[j] = arrCriteria[indexStart[i] + j];
        }
        resultingCriteria[i] = tmpCrit;
    }

    // Cleanup

#if USE_STREAMS
    for (int i = 0; i< nStreams; i++) {
        cudaStreamDestroy(stream[i]);
    }
#endif

    cudaFree(arrData);
    cudaFree(arrCriteria);
    cudaFree(arrIndicesTarg);
    cudaFree(arrIndicesArch);

    return false;
}


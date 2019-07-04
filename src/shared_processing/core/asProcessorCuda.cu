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
void gpuPredictorCriteriaS1grads(float *criteria, const float *data, const int *indicesTarg,
                                 const int *indicesArch, const int *indexStart,
                                 const cudaPredictorsDataPropStruct dataProp, const int n_targ,
                                 const int n_cand, const int offset)
{

#if USE_STREAMS
    int i_cand = offset + threadIdx.x + blockIdx.x * blockDim.x;
    if (i_cand < n_cand) {
        // Find the target index
        float meanNbCand = float(n_cand) / float(n_targ);
        int i_targ = floorf(float(i_cand) / meanNbCand);

        if (i_targ < 0) {
            i_targ = 0;
        }

        if (i_targ >= n_targ) {
            i_targ = n_targ - 1;
        }

        // Check and correct
        if (i_cand < indexStart[i_targ]) {
            while (i_cand < indexStart[i_targ]) {
                i_targ--;

                if (i_targ < 0) {
                    printf("Device error: The target index is < 0 : i_targ = %d.\n", i_targ);
                    criteria[i_cand] = -9999;
                    return;
                }
            }
        }
        if (i_cand >= indexStart[i_targ + 1]) // safe
        {
            while (i_cand >= indexStart[i_targ + 1]) {
                i_targ++;

                if (i_targ >= n_targ) {
                    printf("Device error: The target index is >= n_targ : i_targ = %d (n_targ = %d)\n", i_targ, n_targ);
                    criteria[i_cand] = -9999;
                    return;
                }
            }
        }

        float criterion = 0;

        int targIndexBase = indicesTarg[i_targ] * dataProp.totPtsNb;
        int archIndexBase = indicesArch[i_cand] * dataProp.totPtsNb;

        for (int iPtor = 0; iPtor < dataProp.ptorsNb; iPtor++) {
            float dividend = 0, divisor = 0;
            int targIndex = targIndexBase + dataProp.indexStart[iPtor];
            int archIndex = archIndexBase + dataProp.indexStart[iPtor];

            for (int i = 0; i < dataProp.ptsNb[iPtor]; i++) {
                dividend += fabsf(data[targIndex] - data[archIndex]);
                divisor += fmaxf(fabsf(data[targIndex]), fabsf(data[archIndex]));

                targIndex++;
                archIndex++;
            }

            criterion += dataProp.weights[iPtor] * 100.0f * (dividend / divisor);
        }

        criteria[i_cand] = criterion;
    }

#else

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i_cand = index; i_cand < n_cand; i_cand += stride) {

        // Find the target index
        float meanNbCand = float(n_cand) / float(n_targ);
        int i_targ = (int)floorf(float(i_cand) / meanNbCand);

        if (i_targ < 0) {
            i_targ = 0;
        }

        if (i_targ >= n_targ) {
            i_targ = n_targ - 1;
        }

        // Check and correct
        if (i_cand < indexStart[i_targ]) {
            while (i_cand < indexStart[i_targ]) {
                i_targ--;
                if (i_targ < 0) {
                    printf("Device error: The target index is < 0 : i_targ = %d.\n", i_targ);
                    criteria[i_cand] = 9999;
                    return;
                }
            }
        }
        if (i_cand >= indexStart[i_targ + 1]) // safe
        {
            while (i_cand >= indexStart[i_targ + 1]) {
                i_targ++;
                if (i_targ >= n_targ) {
                    printf("Device error: The target index is >= n_targ : i_targ = %d (n_targ = %d)\n", i_targ, n_targ);
                    criteria[i_cand] = 9999;
                    return;
                }
            }
        }

        float criterion = 0;
        int targIndexBase = indicesTarg[i_targ] * dataProp.totPtsNb;
        int archIndexBase = indicesArch[i_cand] * dataProp.totPtsNb;

        for (int iPtor = 0; iPtor < dataProp.ptorsNb; iPtor++) {
            float dividend = 0, divisor = 0;
            int targIndex = targIndexBase + dataProp.indexStart[iPtor];
            int archIndex = archIndexBase + dataProp.indexStart[iPtor];

            for (int i = 0; i < dataProp.ptsNb[iPtor]; i++) {
                dividend += fabsf(data[targIndex] - data[archIndex]);
                divisor += fmaxf(fabsf(data[targIndex]), fabsf(data[archIndex]));

                targIndex++;
                archIndex++;
            }

            criterion += dataProp.weights[iPtor] * 100.0f * (dividend / divisor);
        }

        criteria[i_cand] = criterion;
    }

#endif
}


bool asProcessorCuda::ProcessCriteria(std::vector <std::vector<float *>> &data,
                                      std::vector<int> &indicesTarg,
                                      std::vector <std::vector<int>> &indicesArch,
                                      std::vector <std::vector<float>> &resultingCriteria,
                                      std::vector<int> &nbArchCandidates,
                                      std::vector<int> &colsNb, std::vector<int> &rowsNb,
                                      std::vector<float> &weights)
{

    // Get the data structure
    cudaPredictorsDataPropStruct struc;
    struc.ptorsNb = (int) weights.size();
    if (struc.ptorsNb > STRUCT_MAX_SIZE) {
        printf("The number of predictors is > %d. Please adapt the source code in asProcessorCuda::ProcessCriteria.\n",
               STRUCT_MAX_SIZE);
        return false;
    }

    struc.totPtsNb = 0;

    for (int iPtor = 0; iPtor < struc.ptorsNb; iPtor++) {
        struc.rowsNb[iPtor] = rowsNb[iPtor];
        struc.colsNb[iPtor] = colsNb[iPtor];
        struc.weights[iPtor] = weights[iPtor];
        struc.ptsNb[iPtor] = colsNb[iPtor] * rowsNb[iPtor];
        struc.indexStart[iPtor] = struc.totPtsNb;
        struc.totPtsNb += colsNb[iPtor] * rowsNb[iPtor];
    }

    // Sizes
    int nbArchCandidatesSum = 0;
    std::vector<int> indexStart(nbArchCandidates.size() + 1);
    for (int i = 0; i < nbArchCandidates.size(); i++) {
        indexStart[i] = nbArchCandidatesSum;
        nbArchCandidatesSum += nbArchCandidates[i];
    }
    indexStart[nbArchCandidates.size()] = nbArchCandidatesSum;

    // Blocks of threads
    int n_targ = nbArchCandidates.size();
    int n_cand = nbArchCandidatesSum;
    // The number of threads per block should be a multiple of 32 threads, because this provides optimal computing efficiency and facilitates coalescing.
    const int threadsPerBlock = 512; // no need to change
    int blocksNb = (n_cand + threadsPerBlock - 1) / threadsPerBlock;


#if USE_STREAMS
    // Create streams
    const int nStreams = 4; // no need to change
    // rowsNbPerStream must be dividable by nStreams and threadsPerBlock
    int rowsNbPerStream = ceil(float(nbArchCandidatesSum) / float(nStreams * threadsPerBlock)) * threadsPerBlock;
    // Streams
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&stream[i]);
    }
#endif

    // Data pointers
    float *arrData, *arrCriteria;
    int *arrIndicesTarg, *arrIndicesArch, *arrIndexStart;

    // Alloc space for data
    checkCudaErrors(cudaMallocManaged(&arrData, data.size() * struc.totPtsNb * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&arrCriteria, nbArchCandidatesSum * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&arrIndicesTarg, nbArchCandidates.size() * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&arrIndicesArch, nbArchCandidatesSum * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&arrIndexStart, (nbArchCandidates.size() + 1) * sizeof(int)));

    // Copy data in the new arrays
    for (int iDay = 0; iDay < data.size(); iDay++) {
        for (int iPtor = 0; iPtor < struc.ptorsNb; iPtor++) {
            for (int iPt = 0; iPt < struc.ptsNb[iPtor]; iPt++) {
                arrData[iDay * struc.totPtsNb + struc.indexStart[iPtor] + iPt] = data[iDay][iPtor][iPt];
            }
        }
    }

    for (int i = 0; i < nbArchCandidates.size(); i++) {
        for (int j = 0; j < nbArchCandidates[i]; j++) {
            arrIndicesArch[indexStart[i] + j] = indicesArch[i][j];
        }
    }

    for (int i = 0; i < indicesTarg.size(); i++) {
        arrIndicesTarg[i] = indicesTarg[i];
    }

    for (int i = 0; i < indexStart.size(); i++) {
        arrIndexStart[i] = indexStart[i];
    }

    // Launch kernel on GPU
#if USE_STREAMS
    for (int i = 0; i < nStreams; i++) {
        int offset = i * rowsNbPerStream;
        blocksNb = rowsNbPerStream / threadsPerBlock;
        gpuPredictorCriteriaS1grads<<<blocksNb, threadsPerBlock, 0, stream[i]>>>(arrCriteria, arrData, arrIndicesTarg, arrIndicesArch, arrIndexStart, struc, n_targ, n_cand, offset);
    }
#else
    gpuPredictorCriteriaS1grads<<<blocksNb, threadsPerBlock>>>(arrCriteria, arrData, arrIndicesTarg, arrIndicesArch, arrIndexStart, struc, n_targ, n_cand, 0);
#endif

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Set the criteria values in the vector container
    for (int i = 0; i < nbArchCandidates.size(); i++) {
        std::vector<float> tmpCrit(nbArchCandidates[i]);

        for (int j = 0; j < nbArchCandidates[i]; j++) {
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
    cudaFree(arrIndexStart);

    return false;
}


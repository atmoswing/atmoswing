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
                    //printf("Device error: The target index is < 0 : i_targ = %d.\n", i_targ);
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
                    //printf("Device error: The target index is >= n_targ : i_targ = %d (n_targ = %d)\n", i_targ, n_targ);
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
    // Error var
    cudaError_t cudaStatus;
    bool hasError = false;

    if (!SelectBestDevice()) {
        return false;
    }

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
    const int threadsPerBlock = 1024; // no need to change
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
    cudaStatus = cudaMallocManaged(&arrData, data.size() * struc.totPtsNb * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for the data!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMallocManaged(&arrCriteria, nbArchCandidatesSum * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for arrCriteria!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMallocManaged(&arrIndicesTarg, nbArchCandidates.size() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for arrIndicesTarg!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMallocManaged(&arrIndicesArch, nbArchCandidatesSum * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for arrIndicesArch!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMallocManaged(&arrIndexStart, (nbArchCandidates.size() + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for arrIndexStart!\n");
        hasError = true;
        goto cleanup;
    }

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
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(cudaStatus));
        hasError = true;
        goto cleanup;
    }

    // Set the criteria values in the vector container
    for (int i = 0; i < nbArchCandidates.size(); i++) {
        std::vector<float> tmpCrit(nbArchCandidates[i]);

        for (int j = 0; j < nbArchCandidates[i]; j++) {
            tmpCrit[j] = arrCriteria[indexStart[i] + j];
        }
        resultingCriteria[i] = tmpCrit;
    }

    // Cleanup
    cleanup:

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

    return !hasError;
}


bool asProcessorCuda::SelectBestDevice()
{
    cudaError_t cudaStatus;

    // Count the devices
    int devicesCount = 0;
    cudaStatus = cudaGetDeviceCount(&devicesCount);
    if (cudaStatus != cudaSuccess) {
        if (cudaStatus == cudaErrorNoDevice) {
            fprintf(stderr, "cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\n");
            return false;
        } else if (cudaStatus == cudaErrorInsufficientDriver) {
            fprintf(stderr, "cudaGetDeviceCount failed! No driver can be loaded to determine if any device exists.\n");
            return false;
        }

        fprintf(stderr, "cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\n");
        return false;
    }

    // Get some info on the devices
    int bestDevice = 0;
    int memSize = 0;
    struct cudaDeviceProp deviceProp;
    for (int i_dev = 0; i_dev < devicesCount; i_dev++) {
        cudaStatus = cudaGetDeviceProperties(&deviceProp, i_dev);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed!\n");
            return false;
        }

        // Compare memory
        if (deviceProp.totalGlobalMem > memSize) {
            memSize = deviceProp.totalGlobalMem;
            bestDevice = i_dev;
        }
    }

    // Select the best device
    cudaStatus = cudaSetDevice(bestDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        return false;
    }

    return true;
}

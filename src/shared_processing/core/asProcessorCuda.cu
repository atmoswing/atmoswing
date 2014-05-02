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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */

// Disable some MSVC warnings
#ifdef _MSC_VER
    #pragma warning( disable : 4244 ) // C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data
    #pragma warning( disable : 4267 ) // C4267: conversion from 'size_t' to 'int', possible loss of data
#endif


#include "asProcessorCuda.cuh"

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void gpuPredictorCriteriaS1grads(float *criteria,
                                            const float *data,
                                            const int *indicesTarg,
                                            const int *indicesArch,
                                            const int *indexStart,
                                            const cudaPredictorsDataPropStruct dataProp,
                                            int n, int offset)
{
    int i_targ = offset + threadIdx.x + blockIdx.x * blockDim.x;
    if (i_targ < n)
    {
        int baseIndex = indexStart[i_targ];
        int nbCandidates = indexStart[i_targ+1] - indexStart[i_targ]; // Safe: indexStart has size of n+1
        int targIndexBase = indicesTarg[i_targ] * dataProp.totPtsNb;

        for (int i_cand=0; i_cand<nbCandidates; i_cand++)
        {
            float criterion = 0;
            int archIndexBase = indicesArch[baseIndex + i_cand] * dataProp.totPtsNb;

            for (int i_ptor=0; i_ptor<dataProp.ptorsNb; i_ptor++)
            {
                float dividend = 0, divisor = 0;
				int targIndex = targIndexBase + dataProp.indexStart[i_ptor];
				int archIndex = archIndexBase + dataProp.indexStart[i_ptor];
				
                for (int i=0; i<dataProp.ptsNb[i_ptor]; i++)
                {
					dividend += abs(data[targIndex] - data[archIndex]);
                    divisor += max(abs(data[targIndex]), abs(data[archIndex]));

					targIndex++;
					archIndex++;
                }

                criterion += dataProp.weights[i_ptor] * 100.0f * (dividend / divisor);
            }

			criteria[baseIndex + i_cand] = criterion;
        }
    }
}


bool asProcessorCuda::ProcessCriteria(std::vector < std::vector < float* > > &data,
                                      std::vector < int > &indicesTarg,
                                      std::vector < std::vector < int > > &indicesArch,
                                      std::vector < std::vector < float > > &resultingCriteria,
                                      std::vector < int > &lengths,
                                      std::vector < int > &colsNb,
                                      std::vector < int > &rowsNb,
                                      std::vector < float > &weights)
{
    // Error var
    cudaError_t cudaStatus;
    bool hasError = false;

    // Count the devices
    int devicesCount = 0;
    cudaStatus = cudaGetDeviceCount(&devicesCount);
    if (cudaStatus != cudaSuccess) {
        if (cudaStatus == cudaErrorNoDevice)
        {
            fprintf(stderr, "cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\n");
            return false;
        }
        else if (cudaStatus == cudaErrorInsufficientDriver)
        {
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
    for (int i_dev=0; i_dev<devicesCount; i_dev++)
    {
        cudaStatus = cudaGetDeviceProperties(&deviceProp, i_dev);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed!\n");
            return false;
        }

        // Compare memory
        if (deviceProp.totalGlobalMem>memSize)
        {
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

    // Get the data structure
    cudaPredictorsDataPropStruct dataProp;
    dataProp.ptorsNb = (int)weights.size();
    if (dataProp.ptorsNb>STRUCT_MAX_SIZE)
    {
        printf("The number of predictors is > %d. Please adapt the source code in asProcessorCuda::ProcessCriteria.\n", STRUCT_MAX_SIZE);
        return false;
    }

    dataProp.totPtsNb = 0;

    for (int i_ptor=0; i_ptor<dataProp.ptorsNb; i_ptor++)
    {
        dataProp.rowsNb[i_ptor] = rowsNb[i_ptor];
        dataProp.colsNb[i_ptor] = colsNb[i_ptor];
        dataProp.weights[i_ptor] = weights[i_ptor];
        dataProp.ptsNb[i_ptor] = colsNb[i_ptor] * rowsNb[i_ptor];
        dataProp.indexStart[i_ptor] = dataProp.totPtsNb;
        dataProp.totPtsNb += colsNb[i_ptor] * rowsNb[i_ptor];
    }

    // Sizes
    int lengthsSum = 0;
    std::vector < int > indexStart(lengths.size()+1);
    for (int i_len=0; i_len<lengths.size(); i_len++)
    {
        indexStart[i_len] = lengthsSum;
        lengthsSum += lengths[i_len];
    }
    indexStart[lengths.size()] = lengthsSum;
    int sizeData = dataProp.totPtsNb * data.size() * sizeof(float);
    int sizeCriteria = lengthsSum * sizeof(float);
    int sizeIndicesTarg = lengths.size() * sizeof(int);
    int sizeIndicesArch = lengthsSum * sizeof(int);
    int sizeIndexStart = (lengths.size() + 1) * sizeof(int); // + 1 relative to lengths

    // Create streams
    const int nStreams = 20; // 20
    const int threadsPerBlock = 8; // 8
	// rowsNbPerStream must be dividable by nStreams and threadsPerBlock
    int rowsNbPerStream = ceil(float(lengths.size()) / float(nStreams*threadsPerBlock)) * threadsPerBlock;
	// Streams
    cudaStream_t stream[nStreams];
    for (int i=0; i<nStreams; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    // Host and device pointers
    float *arrCriteria, *arrData;
    int *arrIndicesTarg = &indicesTarg[0];
    int *arrIndicesArch;
    int *arrIndexStart = &indexStart[0];
    float *devData, *devCriteria;
    int *devIndicesTarg, *devIndicesArch, *devIndexStart;

	// Alloc space for host copies of data

    #if USE_PINNED_MEM

        // See http://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/

        cudaStatus = cudaHostAlloc((void**)&arrData, data.size() * dataProp.totPtsNb * sizeof(float),  cudaHostAllocDefault);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed for the data!\n");
            hasError = true;
            goto cleanup;
        }

        cudaStatus = cudaHostAlloc((void**)&arrIndicesArch, lengthsSum * sizeof(int), cudaHostAllocDefault);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed for arrIndicesArch!\n");
            hasError = true;
            goto cleanup;
        }
		
        cudaStatus = cudaHostAlloc((void**)&arrCriteria, lengthsSum * sizeof(float), cudaHostAllocDefault);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed for arrCriteria!\n");
            hasError = true;
            goto cleanup;
        }

    #else // USE_PINNED_MEM

        arrData = new float[data.size() * dataProp.totPtsNb];
        arrIndicesArch = new int[lengthsSum];
		arrCriteria = new float[lengthsSum];

    #endif // USE_PINNED_MEM

    // Copy data in the new arrays
    for (int i_day=0; i_day<data.size(); i_day++)
    {
        for (int i_ptor=0; i_ptor<dataProp.ptorsNb; i_ptor++)
        {
            for (int i_pt=0; i_pt<dataProp.ptsNb[i_ptor]; i_pt++)
            {
                arrData[i_day * dataProp.totPtsNb + dataProp.indexStart[i_ptor] + i_pt] = data[i_day][i_ptor][i_pt];
            }
            //std::copy(vvpArchData[i_day][i_ptor], vvpArchData[i_day][i_ptor] + dataProp.indexEnd[i_ptor], arrArchData + i_day*dataProp.totPtsNb + dataProp.indexStart[i_ptor]); -> fails
        }
    }

    for (int i_len=0; i_len<lengths.size(); i_len++)
    {
        for (int j_len=0; j_len<lengths[i_len]; j_len++)
        {
            arrIndicesArch[indexStart[i_len] + j_len] = indicesArch[i_len][j_len];
        }
    }

    // Alloc space for device copies of data
    cudaStatus = cudaMalloc(&devData, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the data!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMalloc(&devCriteria, sizeCriteria);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the criteria!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMalloc(&devIndicesTarg, sizeIndicesTarg);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the target indices!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMalloc(&devIndicesArch, sizeIndicesArch);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the archive indices!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMalloc(&devIndexStart, sizeIndexStart);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the archive indices!\n");
        hasError = true;
        goto cleanup;
    }

    /*
     * Asynchronous memcpy and processing. See:
     * https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
     * http://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/
     */

	// No async for the indices as they don't use pinned memory
    cudaStatus = cudaMemcpy(devIndicesTarg, arrIndicesTarg, sizeIndicesTarg, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the target indices!\n");
        hasError = true;
        goto cleanup;
    }

    cudaStatus = cudaMemcpy(devIndexStart, arrIndexStart, sizeIndexStart, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the start indices!\n");
        hasError = true;
        goto cleanup;
    }

	// For the data, create its own stream
	cudaStream_t streamData;
    cudaStreamCreate(&streamData);
    cudaStatus = cudaMemcpyAsync(devData, arrData, sizeData, cudaMemcpyHostToDevice, streamData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the data!\n");
        hasError = true;
        goto cleanup;
    }

    // Copy archive indices to device
    for (int i=0; i<nStreams; i++)
    {
        int offset = indexStart[i*rowsNbPerStream];
        int length = 0;
        if (i<nStreams-1)
        {
            length = indexStart[(i+1)*rowsNbPerStream] - offset;
        }
        else
        {
            length = lengthsSum - offset; // Last slice
        }
        int streamBytes = length*sizeof(int);

        cudaStatus = cudaMemcpyAsync(&devIndicesArch[offset], &arrIndicesArch[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync failed for the archive data (stream %d/%d)!\n", i, nStreams);
            hasError = true;
            goto cleanup;
        }
    }

	// Make sure the indices and the data are copied
	cudaStatus = cudaStreamSynchronize(streamData);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed on streamData !\n");
        hasError = true;
        goto cleanup;
    }

    // Launch kernel on GPU
    for (int i=0; i<nStreams; i++)
    {
        int offset = i*rowsNbPerStream;
        int blocksNb = rowsNbPerStream/threadsPerBlock;
        int totSize = lengths.size();
        gpuPredictorCriteriaS1grads<<<blocksNb,threadsPerBlock, 0, stream[i]>>>(devCriteria, devData, devIndicesTarg, devIndicesArch, devIndexStart, dataProp, totSize, offset);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        hasError = true;
        goto cleanup;
    }

    // Copy results back to host
    for (int i=0; i<nStreams; i++)
    {
        int offset = indexStart[i*rowsNbPerStream];
        int length = 0;
        if (i<nStreams-1)
        {
            length = indexStart[(i+1)*rowsNbPerStream] - offset;
        }
        else
        {
            length = lengthsSum - offset; // Last slice
        }
        int streamBytes = length*sizeof(float);

        cudaStatus = cudaMemcpyAsync(&arrCriteria[offset], &devCriteria[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync failed for the results (stream %d/%d)!\n", i, nStreams);
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
            hasError = true;
            goto cleanup;
        }
    }

	cudaDeviceSynchronize();

    // Set the criteria values in the vector container
    for (int i_len=0; i_len<lengths.size(); i_len++)
    {
        std::vector < float > tmpCrit(lengths[i_len]);

        for (int j_len=0; j_len<lengths[i_len]; j_len++)
        {
            tmpCrit[j_len] = arrCriteria[indexStart[i_len] + j_len];
        }
        resultingCriteria[i_len] = tmpCrit;
    }

    // Cleanup
    cleanup:
    cudaFree(devData);
    cudaFree(devCriteria);
    cudaFree(devIndicesTarg);
    cudaFree(devIndicesArch);
    cudaFree(devIndexStart);

    #if USE_PINNED_MEM
        cudaFreeHost(arrData);
        cudaFreeHost(arrIndicesArch);
        cudaFreeHost(arrCriteria);
    #else
        delete[] arrData;
        delete[] arrIndicesArch;
		delete[] arrCriteria;
    #endif // USE_PINNED_MEM

    for (int i=0; i<nStreams; i++)
    {
        cudaStreamDestroy(stream[i]);
    }
	cudaStreamDestroy(streamData);

    if (hasError) return false;

    return true;
}

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

#if USE_THRUST
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/transform.h>
    #include <thrust/for_each.h>
    #include <thrust/fill.h>
    #include <thrust/iterator/zip_iterator.h>
#else // USE_THRUST
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#endif // USE_THRUST


#if USE_THRUST

struct gpuPredictorCriteriaS1grads
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // 0: targData, 1: archData, 2: dividend, 3: divisor
        // Dividend
        thrust::get<2>(t) = abs(thrust::get<0>(t)-thrust::get<1>(t));
        // Divisor
        thrust::get<3>(t) = thrust::max(abs(thrust::get<0>(t)), abs(thrust::get<1>(t)));
    }
};

struct gpuAddToCriteriaS1grads
{
    const float weight;
    gpuAddToCriteriaS1grads(float _weight) : weight(_weight) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // 0: reducedDividend, 1: reducedDivisor, 2: resultingCriteria
        thrust::get<2>(t) += weight*100.0f*(thrust::get<0>(t)/thrust::get<1>(t));
    }
};

#else // USE_THRUST

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
            criteria[baseIndex + i_cand] = 0;
            int archIndexBase = indicesArch[baseIndex + i_cand] * dataProp.totPtsNb;

            for (int i_ptor=0; i_ptor<dataProp.ptorsNb; i_ptor++)
            {
                float dividend = 0, divisor = 0;
				
                for (int i=0; i<dataProp.rowsNb[i_ptor]; i++)
                {
					/*
                    for (int j=0; j<dataProp.colsNb[i_ptor]; j++)
                    {
                        int subindex = dataProp.indexStart[i_ptor] + i * dataProp.colsNb[i_ptor] + j;

                        dividend += abs(data[targIndexBase + subindex] - data[archIndexBase + subindex]);
                        divisor += max(abs(data[targIndexBase + subindex]), abs(data[archIndexBase + subindex]));
                    }*/

					int subindex = dataProp.indexStart[i_ptor] + i;

					dividend += abs(data[targIndexBase + subindex] - data[archIndexBase + subindex]);
                    divisor += max(abs(data[targIndexBase + subindex]), abs(data[archIndexBase + subindex]));
                }

                criteria[baseIndex + i_cand] += dataProp.weights[i_ptor] * 100.0f * (dividend / divisor);
            }
        }
    }
}

#endif // USE_THRUST

bool asProcessorCuda::ProcessCriteria(std::vector < std::vector < float* > > &data,
                                      std::vector < int > &indicesTarg,
                                      std::vector < std::vector < int > > &indicesArch,
                                      std::vector < std::vector < float > > &resultingCriteria,
                                      std::vector < int > &lengths,
                                      std::vector < int > &colsNb,
                                      std::vector < int > &rowsNb,
                                      std::vector < float > &weights)
{

    #if USE_THRUST

    // Allocate storage
    thrust::device_vector<float> resultingCriteria(size, 0);
    thrust::device_vector<float> reducedDivisor(size);
    thrust::device_vector<float> reducedDividend(size);
    thrust::device_vector<int> reducedKeys(size);

    // Number of predictors
    int ptorsNb = (int)weights.size();

    // Loop over every predictor
    for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
    {
        // Number of points
        int ptsNb = colsNb[i_ptor]*rowsNb[i_ptor];

        // Allocate storage
        thrust::host_vector<float> hostTargData(size*ptsNb);
        thrust::host_vector<float> hostArchData(size*ptsNb);
        thrust::device_vector<float> devTargData(size*ptsNb);
        thrust::device_vector<float> devArchData(size*ptsNb);
        thrust::device_vector<float> devDividend(size*ptsNb);
        thrust::device_vector<float> devDivisor(size*ptsNb);
        thrust::device_vector<int> keys(size*ptsNb);

        // Populate host vectors (to do only 1 copy to the device)
        for (int i_day=0; i_day<size; i_day++)
        {
            int destinationIndex = i_day*ptsNb;
            thrust::copy(vpTargData[i_ptor], vpTargData[i_ptor]+ptsNb, hostTargData.begin()+destinationIndex);
            thrust::copy(vvpArchData[i_day][i_ptor], vvpArchData[i_day][i_ptor]+ptsNb, hostArchData.begin()+destinationIndex);
            thrust::fill(keys.begin()+destinationIndex, keys.begin()+destinationIndex+ptsNb, i_day);
        }

        // Copy data to device
        devTargData = hostTargData;
        devArchData = hostArchData;

        // Process dividend and divisor
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(devTargData.begin(), devArchData.begin(), devDividend.begin(), devDivisor.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(devTargData.end(), devArchData.end(), devDividend.end(), devDivisor.end())),
                         gpuPredictorCriteriaS1grads());

        // Proceed to reduction
        thrust::reduce_by_key(keys.begin(), keys.end(), devDivisor.begin(), reducedKeys.begin(), reducedDivisor.begin());
        thrust::reduce_by_key(keys.begin(), keys.end(), devDividend.begin(), reducedKeys.begin(), reducedDividend.begin());

        // Add to the resulting criteria
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(reducedDividend.begin(), reducedDivisor.begin(), resultingCriteria.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(reducedDividend.end(), reducedDivisor.end(), resultingCriteria.end())),
                         gpuAddToCriteriaS1grads(weights[i_ptor]));
    }

    // Copy to the final container
    thrust::copy(resultingCriteria.begin(), resultingCriteria.end(), criteriaValues.begin());


    #else // USE_THRUST

    // Error var
    cudaError_t cudaStatus;
    bool hasError = false;

	// Set flags
	//cudaSetDeviceFlags( cudaDeviceMapHost );

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

        //printf("device %d: %d Mb memory\n", i_dev, int(deviceProp.totalGlobalMem/1048576));

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

    // Get the meta data
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
        dataProp.indexEnd[i_ptor] = dataProp.totPtsNb + dataProp.ptsNb[i_ptor] - 1;
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
    int sizeIndexStart = lengths.size() * sizeof(int);

    // Create streams
    const int nStreams = 20; // 20
    const int threadsPerBlock = 8; // 8
    int streamSizeIndices = ceil((float)lengths.size()/(float)nStreams);
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
		/*
        cudaStatus = cudaHostAlloc((void**)&arrCriteria, lengthsSum * sizeof(float), cudaHostAllocDefault);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed for arrCriteria!\n");
            hasError = true;
            goto cleanup;
        }*/
		arrCriteria = new float[lengthsSum];

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


    // For target indices and lengths, no need of async due to its small size
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

	// Do not use async for the predictor data, as the first request can access any data
    cudaStatus = cudaMemcpy(devData, arrData, sizeData, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the data!\n");
        hasError = true;
        goto cleanup;
    }

    // Copy archive indices to device
    for (int i=0; i<nStreams; i++)
    {
        int offset = indexStart[i*streamSizeIndices];
        int length = 0;
        if (i<nStreams-1)
        {
            length = indexStart[(i+1)*streamSizeIndices] - indexStart[i*streamSizeIndices];
        }
        else
        {
            length = lengthsSum - indexStart[i*streamSizeIndices];
        }
        int streamBytes = length*sizeof(int);

        cudaStatus = cudaMemcpyAsync(&devIndicesArch[offset], &arrIndicesArch[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync failed for the archive data (stream %d/%d)!\n", i, nStreams);
            hasError = true;
            goto cleanup;
        }
    }
	
    // Launch kernel on GPU
    for (int i=0; i<nStreams; i++)
    {
        int offset = i*streamSizeIndices;
        int blocksNb = 1+streamSizeIndices/threadsPerBlock;
        int totSize = lengths.size();
        gpuPredictorCriteriaS1grads<<<blocksNb,threadsPerBlock, 0, stream[i]>>>(devCriteria, devData, devIndicesTarg, devIndicesArch, devIndexStart, dataProp, totSize, offset);
    }
/*	int totSize = lengths.size();
	gpuPredictorCriteriaS1grads<<<totSize/threadsPerBlock,threadsPerBlock>>>(devCriteria, devData, devIndicesTarg, devIndicesArch, devIndexStart, dataProp, totSize, 0);
	cudaDeviceSynchronize();*/

    // Check for any errors launching the kernel
	//cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        hasError = true;
        goto cleanup;
    }

    // Copy results back to host
    for (int i=0; i<nStreams; i++)
    {
        int offset = indexStart[i*streamSizeIndices];
        int length = 0;
        if (i<nStreams-1)
        {
            length = indexStart[(i+1)*streamSizeIndices] - indexStart[i*streamSizeIndices];
        }
        else
        {
            length = lengthsSum - indexStart[i*streamSizeIndices];
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
        //cudaFreeHost(arrCriteria);
		delete[] arrCriteria;
    #else
        delete[] arrData;
        delete[] arrIndicesArch;
		delete[] arrCriteria;
    #endif // USE_PINNED_MEM

    for (int i=0; i<nStreams; i++)
    {
        cudaStreamDestroy(stream[i]);
    }

    if (hasError) return false;

    #endif // USE_THRUST

    return true;
}

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

#define USE_THRUST 1
#define DO_PROFILE 1

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
                                            const float *targData,
                                            const float *archData,
                                            const cudaPredictorsMetaDataStruct metaData,
                                            int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        criteria[index] = 0;

        for (int i_ptor=0; i_ptor<metaData.ptorsNb; i_ptor++)
        {
            float dividend = 0, divisor = 0;

            for (int i=0; i<metaData.rowsNb[i_ptor]; i++)
            {
                for (int j=0; j<metaData.colsNb[i_ptor]; j++)
                {
                    int subindex = metaData.indexStart[i_ptor]+i*metaData.colsNb[i_ptor]+j;

                    dividend += abs(targData[subindex]-archData[index*metaData.totPtsNb+subindex]);
                    divisor += max(abs(targData[subindex]),abs(archData[index*metaData.totPtsNb+subindex]));
                }
            }

            criteria[index] += metaData.weights[i_ptor]*100.0f*(dividend/divisor);
        }
    }
}

#endif // USE_THRUST

bool asProcessorCuda::ProcessCriteria(std::vector < float* > &vpTargData,
                                      std::vector < std::vector < float* > > &vvpArchData,
                                      std::vector < float > &criteriaValues,
                                      int size,
                                      std::vector < int > &colsNb,
                                      std::vector < int > &rowsNb,
                                      std::vector < float > &weights)
{

    #if USE_THRUST
    
    #if DO_PROFILE
        clock_t start, stop;
        float time;
        start = clock();
    #endif //DO_PROFILE

    // Allocate storage
    thrust::device_vector<float> resultingCriteria(size, 0);
    thrust::device_vector<float> reducedDivisor(size);
    thrust::device_vector<float> reducedDividend(size);
    thrust::device_vector<int> reducedKeys(size);

    #if DO_PROFILE
        stop = clock();   
        time = (float)(stop-start)/CLOCKS_PER_SEC*1000;
        fprintf(stderr, "First storage allocation: %f ms\n", time);
    #endif //DO_PROFILE

    // Number of predictors
    int ptorsNb = (int)weights.size();

    // Loop over every predictor
    for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
    {
        // Number of points
        int ptsNb = colsNb[i_ptor]*rowsNb[i_ptor];

        #if DO_PROFILE
            start = clock();
        #endif //DO_PROFILE

        // Allocate storage
        thrust::host_vector<float> hostTargData(size*ptsNb);
        thrust::host_vector<float> hostArchData(size*ptsNb);
        thrust::device_vector<float> devTargData(size*ptsNb);
        thrust::device_vector<float> devArchData(size*ptsNb);
        thrust::device_vector<float> devDividend(size*ptsNb);
        thrust::device_vector<float> devDivisor(size*ptsNb);
        thrust::device_vector<int> keys(size*ptsNb);

        #if DO_PROFILE
            stop = clock();   
            time = (float)(stop-start)/CLOCKS_PER_SEC*1000;
            fprintf(stderr, "Predictor storage allocation: %f ms\n", time);

            start = clock();
        #endif //DO_PROFILE

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

        #if DO_PROFILE
            stop = clock();   
            time = (float)(stop-start)/CLOCKS_PER_SEC*1000;
            fprintf(stderr, "Data copy: %f ms\n", time);

            start = clock();
        #endif //DO_PROFILE

        // Process dividend and divisor
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(devTargData.begin(), devArchData.begin(), devDividend.begin(), devDivisor.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(devTargData.end(), devArchData.end(), devDividend.end(), devDivisor.end())),
                         gpuPredictorCriteriaS1grads());

        #if DO_PROFILE
            stop = clock();   
            time = (float)(stop-start)/CLOCKS_PER_SEC*1000;
            fprintf(stderr, "Dividend and divisor calculation: %f ms\n", time);

            start = clock();
        #endif //DO_PROFILE

        // Proceed to reduction
        /*
        for (int i_day=0; i_day<size; i_day++)
        {
            int indexStart = i_day*ptsNb;
            int indexEnd = indexStart+ptsNb;
            reducedDivisor[i_day] = thrust::reduce(devDivisor.begin()+indexStart, devDivisor.begin()+indexEnd);
            reducedDividend[i_day] = thrust::reduce(devDividend.begin()+indexStart, devDividend.begin()+indexEnd);
        }*/
        thrust::reduce_by_key(keys.begin(), keys.end(), devDivisor.begin(), reducedKeys.begin(), reducedDivisor.begin());
        thrust::reduce_by_key(keys.begin(), keys.end(), devDividend.begin(), reducedKeys.begin(), reducedDividend.begin());

        #if DO_PROFILE
            stop = clock();   
            time = (float)(stop-start)/CLOCKS_PER_SEC*1000;
            fprintf(stderr, "Reduction: %f ms\n", time);

            start = clock();
        #endif //DO_PROFILE

        // Add to the resulting criteria
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(reducedDividend.begin(), reducedDivisor.begin(), resultingCriteria.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(reducedDividend.end(), reducedDivisor.end(), resultingCriteria.end())),
                         gpuAddToCriteriaS1grads(weights[i_ptor]));

        #if DO_PROFILE
            stop = clock();   
            time = (float)(stop-start)/CLOCKS_PER_SEC*1000;
            fprintf(stderr, "Final merging: %f ms\n", time);

            std::cout << "Press ENTER to continue... " << std::flush;
            std::cin.ignore( std::numeric_limits <std::streamsize> ::max(), '\n' );
        #endif //DO_PROFILE
    }

    // Copy to the final container
    thrust::copy(resultingCriteria.begin(), resultingCriteria.end(), criteriaValues.begin());


    #else // USE_THRUST

    // Error var
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return false;
    }

    // Get the meta data
    cudaPredictorsMetaDataStruct metaData;
    metaData.ptorsNb = (int)weights.size();
    if (metaData.ptorsNb>20)
    {
        printf("The number of predictors is >20. Please adapt the source code in asProcessorCuda::ProcessCriteria.");
        return false;
    }

    metaData.totPtsNb = 0;

    for (int i_ptor=0; i_ptor<metaData.ptorsNb; i_ptor++)
    {
        metaData.rowsNb[i_ptor] = rowsNb[i_ptor];
        metaData.colsNb[i_ptor] = colsNb[i_ptor];

        metaData.weights[i_ptor] = weights[i_ptor];
        metaData.ptsNb[i_ptor] = colsNb[i_ptor]*rowsNb[i_ptor];
        metaData.indexStart[i_ptor] = metaData.totPtsNb;
        metaData.indexEnd[i_ptor] = metaData.totPtsNb+metaData.ptsNb[i_ptor]-1;
        metaData.totPtsNb += colsNb[i_ptor]*rowsNb[i_ptor];

    }

    // Device copies of data
    float *devTargData, *devArchData, *devCriteriaValues;

    // Get data as arrays
    float* arrCriteriaValues = &criteriaValues[0];
    float* arrTargData;
    arrTargData = new float[metaData.totPtsNb];
    for (int i_ptor=0; i_ptor<metaData.ptorsNb; i_ptor++)
    {
        for (int i_pt=0; i_pt<metaData.ptsNb[i_ptor]; i_pt++)
        {
            arrTargData[metaData.indexStart[i_ptor] + i_pt] = vpTargData[i_ptor][i_pt];
        }
        //std::copy(vpTargData[i_ptor], vpTargData[i_ptor] + metaData.indexEnd[i_ptor], arrTargData + metaData.indexStart[i_ptor]); -> fails
    }
    float* arrArchData;
    arrArchData = new float[size*metaData.totPtsNb];
    for (int i_day=0; i_day<size; i_day++)
    {
        for (int i_ptor=0; i_ptor<metaData.ptorsNb; i_ptor++)
        {
            for (int i_pt=0; i_pt<metaData.ptsNb[i_ptor]; i_pt++)
            {
                arrArchData[i_day*metaData.totPtsNb + metaData.indexStart[i_ptor] + i_pt] = vvpArchData[i_day][i_ptor][i_pt];
            }
            //std::copy(vvpArchData[i_day][i_ptor], vvpArchData[i_day][i_ptor] + metaData.indexEnd[i_ptor], arrArchData + i_day*metaData.totPtsNb + metaData.indexStart[i_ptor]); -> fails
        }
    }

    // Alloc space for device copies of data
    int sizeTargData = metaData.totPtsNb*sizeof(float);
    cudaStatus = cudaMalloc(&devTargData, sizeTargData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the target data!");
        delete[] arrTargData;
        cudaFree(devTargData);
        return false;
    }

    int sizeArchData = size*metaData.totPtsNb*sizeof(float);
    cudaStatus = cudaMalloc(&devArchData, sizeArchData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the archive data!");
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        return false;
    }

    int sizeCriteriaValues = size*sizeof(float);
    cudaStatus = cudaMalloc(&devCriteriaValues, sizeCriteriaValues);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for the criteria!");
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        cudaFree(devCriteriaValues);
        return false;
    }

    // Copy inputs to device
    cudaStatus = cudaMemcpy(devCriteriaValues, arrCriteriaValues, sizeCriteriaValues, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the criteria!");
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        cudaFree(devCriteriaValues);
        return false;
    }

    cudaStatus = cudaMemcpy(devTargData, arrTargData, sizeTargData, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the target data!");
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        cudaFree(devCriteriaValues);
        return false;
    }

    cudaStatus = cudaMemcpy(devArchData, arrArchData, sizeArchData, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the archive data!");
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        cudaFree(devCriteriaValues);
        return false;
    }

    // Launch kernel on GPU
    int threadsPerBlock = 512;
    int blocksNb = 1+size/threadsPerBlock;
    gpuPredictorCriteriaS1grads<<<blocksNb,threadsPerBlock>>>(devCriteriaValues, devTargData, devArchData, metaData, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        cudaFree(devCriteriaValues);
        return false;
    }

    // cudaDeviceSynchronize waits for the kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        cudaFree(devCriteriaValues);
        return false;
    }

    // Copy result back to host
    cudaStatus = cudaMemcpy(arrCriteriaValues, devCriteriaValues, sizeCriteriaValues, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for the results!");
        delete[] arrTargData;
        delete[] arrArchData;
        cudaFree(devTargData);
        cudaFree(devArchData);
        cudaFree(devCriteriaValues);
        return false;
    }

    // Cleanup
    cudaFree(devCriteriaValues);
    cudaFree(devTargData);
    cudaFree(devArchData);
    delete[] arrTargData;
    delete[] arrArchData;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }*/

    #endif // USE_THRUST

    return true;
}

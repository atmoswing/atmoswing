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


#include "asProcessorCuda.cuh"

#define USE_THRUST 1

#if USE_THRUST
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/transform.h>
    #include <thrust/for_each.h>
    #include <thrust/iterator/zip_iterator.h>
#else // USE_THRUST
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
    #include <stdio.h>
#endif // USE_THRUST



#include <iostream>



#if USE_THRUST
/*
struct gpuPredictorCriteriaS1grads
{
    const float a;

    gpuPredictorCriteriaS1grads(float _a) : a(_a) {}

    template <typename Tuple>

    __host__ __device__
        float operator()(Tuple t) 
        { 
            // D[i] = A[i] + B[i] * C[i];
            thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
        }
};
*/
struct arbitrary_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
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

    // allocate storage
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D(5);

    // initialize input vectors
    A[0] = 3;  B[0] = 6;  C[0] = 2; 
    A[1] = 4;  B[1] = 7;  C[1] = 5; 
    A[2] = 0;  B[2] = 2;  C[2] = 7; 
    A[3] = 8;  B[3] = 1;  C[3] = 4; 
    A[4] = 2;  B[4] = 8;  C[4] = 3; 

    // apply the transformation
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D.end())),
                     arbitrary_functor());

    // print the output
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl;



    // H has storage for 4 integers
    //thrust::host_vector<int> H(4);



    //thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), gpuPredictorCriteriaS1grads(A));

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

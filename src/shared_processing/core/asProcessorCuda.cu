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
/*
#ifdef _MSC_VER
    #pragma warning( disable : 4267 ) // C4267: 'var' : conversion from 'size_t' to 'type'
    #pragma warning( disable : 4244 ) // C4244: 'initializing' : conversion from 'unsigned __int64' to 'unsigned int'
#endif
*/
#include "asProcessorCuda.h"

#define THREADS_PER_BLOCK 512

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

            // update the pointer to point to the beginning of the next row
            // See http://stackoverflow.com/questions/5029920/how-to-use-2d-arrays-in-cuda/9974989#9974989
    // float* rowData = (float*)(((char*)d_array) + (row * pitch)); 

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

bool asProcessorCuda::ProcessCriteria(std::vector < float* > &vpTargData, 
                                      std::vector < std::vector < float* > > &vvpArchData, 
                                      std::vector < float > &criteriaValues, 
                                      int size, 
                                      std::vector < int > &colsNb, 
                                      std::vector < int > &rowsNb, 
                                      std::vector < float > &weights)
{
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




    /*

    !!!!!!!!!!
    Use structure: http://choorucode.com/2011/03/04/cuda-structures-as-kernel-parameters/
    http://stackoverflow.com/questions/12211241/cuda-how-to-implement-dynamic-array-of-struct-in-cuda-kernel

    !!!!!!!!!!!!!

    */

    // Device copies of data
    float *devTargData, *devArchData, *devCriteriaValues;
    
    // Get data as arrays
    float* arrCriteriaValues = &criteriaValues[0];
    float* arrTargData = vpTargData[0];
    float* arrArchData = new float[size*metaData.totPtsNb];
    for (int i=0; i<size; i++)
    {
        std::copy(vvpArchData[i][0], vvpArchData[i][0] + metaData.totPtsNb, arrArchData + i*metaData.totPtsNb);
    }

    // The pitch value assigned by cudaMallocPitch  
    // (which ensures correct data structure alignment) 
    // See http://stackoverflow.com/questions/5029920/how-to-use-2d-arrays-in-cuda/9974989#9974989
//    size_t pitch;

    // Alloc space for device copies of data
    int sizeTargData = metaData.totPtsNb*sizeof(float);
    cudaMalloc(&devTargData, sizeTargData);
    int sizeArchData = size*metaData.totPtsNb*sizeof(float);
//    cudaMallocPitch(&devArchData, &pitch, metaData.totPtsNb * sizeof(float), size);
    cudaMalloc(&devArchData, sizeArchData);
    int sizeCriteriaValues = size*sizeof(float);
    cudaMalloc(&devCriteriaValues, sizeCriteriaValues);

    // Copy inputs to device
    cudaMemcpy(devCriteriaValues, arrCriteriaValues, sizeCriteriaValues, cudaMemcpyHostToDevice);
    cudaMemcpy(devTargData, arrTargData, sizeTargData, cudaMemcpyHostToDevice);

    // Launch kernel on GPU
    gpuPredictorCriteriaS1grads<<<size/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(devCriteriaValues, devTargData, devArchData, metaData, size);

    // Copy result back to host
    cudaMemcpy(arrCriteriaValues, devCriteriaValues, sizeCriteriaValues, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(devCriteriaValues);
    cudaFree(devTargData);
    cudaFree(devArchData);
    delete[](arrArchData);

	return true;
}

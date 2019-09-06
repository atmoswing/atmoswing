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

#define FULL_MASK 0xffffffff
#define _TIME_CUDA true

// The number of threads per block should be a multiple of 32 threads, because this provides optimal computing
// efficiency and facilitates coalescing.
static const int blockSize = 64; // must be 64 <= blockSize <= 1024

cudaStream_t *g_streams = new cudaStream_t[nStreams];

#if _TIME_CUDA
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
#endif

// From https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
__inline__ __device__
float warpReduceSum(float val)
{
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);

    return val;
}

__global__
void processS1grads(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg, const int *idxArch,
    float w, float *out, int offset)
{
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        extern __shared__ float mem[];
        float *diff = mem;
        float *amax = &diff[blockSize];

        float rdiff = 0;
        float rmax = 0;

        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops-1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                float xi = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                float yi = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];

                diff[threadId] = fabsf(xi - yi);
                amax[threadId] = fmaxf(fabsf(xi), fabsf(yi));
            } else {
                // Set rest of the block to 0
                diff[threadId] = 0;
                amax[threadId] = 0;
            }
            __syncthreads();

            // Process sum reduction
            for (unsigned int stride = blockSize / 2; stride >= 32; stride /= 2) {
                if (threadId < stride) {
                    diff[threadId] += diff[threadId + stride];
                    amax[threadId] += amax[threadId + stride];
                }
                __syncthreads();
            }

            float ldiff = diff[threadId];
            float lamax = amax[threadId];
            __syncthreads();

            if (threadId < 32) {
                ldiff = warpReduceSum(ldiff);
                lamax = warpReduceSum(lamax);
            }
            __syncthreads();

            if (threadId == 0) {
                rdiff += ldiff;
                rmax += lamax;
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            if (rmax == 0) {
                *(out + offset + blockId) += 200.0f * w;
            } else {
                *(out + offset + blockId) += 100.0f * (rdiff / rmax) * w;
            }
        }
    }
}

bool asProcessorCuda::ProcessCriteria(const float *dData, std::vector<long> ptorStart, int indexTarg, const int *indicesArch,
                                      float *dRes, int nbCandidates, std::vector<int> &colsNb, std::vector<int> &rowsNb,
                                      std::vector<float> &weights, std::vector<CudaCriteria> &criteria, int streamId, int offset)
{
    for (int iPtor = 0; iPtor < ptorStart.size(); iPtor++) {
        int ptsNb = colsNb[iPtor] * rowsNb[iPtor];

        // Define block size (must be multiple of 32) and blocks nb
        int blocksNbXY = ceil(std::cbrt(nbCandidates));
        int blocksNbZ = ceil((double)nbCandidates / (blocksNbXY * blocksNbXY));
        dim3 blocksNb3D(blocksNbXY, blocksNbXY, blocksNbZ);

        // Launch kernel
        switch (criteria[iPtor]) {
            case S1grads:
                // 3rd <<< >>> argument is for the dynamically allocated shared memory
                processS1grads<<<blocksNb3D, blockSize, 2 * blockSize * sizeof(float), g_streams[streamId]>>>
                     (dData, ptorStart[iPtor], nbCandidates, ptsNb, indexTarg, indicesArch, weights[iPtor], dRes, offset);
                break;
            default:
                printf("Criteria not yet implemented on GPU.");
                return false;
        }
    }

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

void asProcessorCuda::InitStreams()
{
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&(g_streams[i]));
}

void asProcessorCuda::DestroyStreams()
{
    for (int i = 0; i < nStreams; i++)
        cudaStreamDestroy(g_streams[i]);
}

void asProcessorCuda::CudaMalloc(int *&data, int length)
{
    checkCudaErrors(cudaMalloc((void **)&data, length * sizeof(int)));
}

void asProcessorCuda::CudaMalloc(float *&data, long length)
{
    checkCudaErrors(cudaMalloc((void **)&data, length * sizeof(float)));
}

void asProcessorCuda::CudaMemset0(float *data, long length)
{
    checkCudaErrors(cudaMemset(data, 0, length * sizeof(float)));
}

void asProcessorCuda::CudaMemset0Async(float *data, long length, int streamId)
{
    checkCudaErrors(cudaMemsetAsync(data, 0, length * sizeof(float), g_streams[streamId]));
}

void asProcessorCuda::CudaMemCopyToDevice(int *devData, int *hostData, int length)
{
    checkCudaErrors(cudaMemcpy(devData, hostData, length * sizeof(int), cudaMemcpyHostToDevice));
}

void asProcessorCuda::CudaMemCopyToDeviceAsync(int *devData, int *hostData, int length, int streamId)
{
    checkCudaErrors(cudaMemcpyAsync(devData, hostData, length * sizeof(int), cudaMemcpyHostToDevice, g_streams[streamId]));
}

void asProcessorCuda::CudaMemCopyToDevice(float *devData, float *hostData, long length)
{
    checkCudaErrors(cudaMemcpy(devData, hostData, length * sizeof(float), cudaMemcpyHostToDevice));
}

void asProcessorCuda::CudaMemCopyFromDevice(int *hostData, int *devData, int length)
{
    checkCudaErrors(cudaMemcpy(hostData, devData, length * sizeof(int), cudaMemcpyDeviceToHost));
}

void asProcessorCuda::CudaMemCopyFromDeviceAsync(int *hostData, int *devData, int length, int streamId)
{
    checkCudaErrors(cudaMemcpyAsync(hostData, devData, length * sizeof(int), cudaMemcpyDeviceToHost, g_streams[streamId]));
}

void asProcessorCuda::CudaMemCopyFromDevice(float *hostData, float *devData, long length)
{
    checkCudaErrors(cudaMemcpy(hostData, devData, length * sizeof(float), cudaMemcpyDeviceToHost));
}

void asProcessorCuda::CudaMemCopyFromDeviceAsync(float *hostData, float *devData, long length, int streamId)
{
    checkCudaErrors(cudaMemcpyAsync(hostData, devData, length * sizeof(float), cudaMemcpyDeviceToHost, g_streams[streamId]));
}

void asProcessorCuda::CudaFree(int *data)
{
    checkCudaErrors(cudaFree(data));
}

void asProcessorCuda::CudaFree(float *data)
{
    checkCudaErrors(cudaFree(data));
}

void asProcessorCuda::CudaGetLastError()
{
    checkCudaErrors(cudaGetLastError());
}

void asProcessorCuda::DeviceSynchronize()
{
    checkCudaErrors(cudaDeviceSynchronize());
}

void asProcessorCuda::StreamSynchronize(int streamId)
{
    checkCudaErrors(cudaStreamSynchronize(g_streams[streamId]));
}

void asProcessorCuda::DeviceReset()
{
    cudaDeviceReset();
}

void asProcessorCuda::StartTimer()
{
#if _TIME_CUDA
    cudaEventRecord(start);
#endif
}

void asProcessorCuda::EndTimer(const std::string &task)
{
#if _TIME_CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time to %s: %f\n", task.c_str(), milliseconds);
#endif
}

void asProcessorCuda::CudaLaunchHostFuncStoring(CudaCallbackParams *cbParams, int streamId)
{
    checkCudaErrors(cudaStreamAddCallback(g_streams[streamId], postprocessCallback, cbParams, 0));
}

void CUDART_CB asProcessorCuda::postprocessCallback(cudaStream_t stream, cudaError_t status, void *data)
{
    // Check status of GPU after stream operations are done
    checkCudaErrors(status);

    auto *cbParams = (CudaCallbackParams*) data;

    int analogsNb = cbParams->analogsNb;
    int offset = cbParams->offset;

    // Sort and store results
    int counter = 0;

    // Containers for daily results
    Eigen::Array<float, Eigen::Dynamic, 1> scoreArrayOneDay(analogsNb);
    Eigen::Array<float, Eigen::Dynamic, 1> dateArrayOneDay(analogsNb);
    scoreArrayOneDay.fill(std::numeric_limits<float>::quiet_NaN());
    dateArrayOneDay.fill(std::numeric_limits<float>::quiet_NaN());

    for (int iDateArch = 0; iDateArch < cbParams->nbCand; iDateArch++) {
        float analogDate = cbParams->currentDates[offset + iDateArch];
        float score = cbParams->hRes[offset + iDateArch];

        // Check if the array is already full
        if (counter > analogsNb - 1) {
            if (score < scoreArrayOneDay[analogsNb - 1]) {
                ArraysInsert<float>(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                               &dateArrayOneDay[analogsNb - 1], cbParams->isAsc, score, analogDate);
            }
        } else if (counter < analogsNb - 1) {
            // Add score and date to the vectors
            scoreArrayOneDay[counter] = score;
            dateArrayOneDay[counter] = analogDate;
        } else if (counter == analogsNb - 1) {
            // Add score and date to the vectors
            scoreArrayOneDay[counter] = score;
            dateArrayOneDay[counter] = analogDate;

            // Sort both scores and dates arrays
            SortArrays<float>(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                              &dateArrayOneDay[analogsNb - 1], cbParams->isAsc);
        }

        counter++;
    }

    if (counter < analogsNb) {
        printf("There is not enough available data to satisfy the number of analogs.");
    }

    for (int i = 0; i < analogsNb; ++i) {
        *(cbParams->finalAnalogsCriteria + i) = scoreArrayOneDay(i);
        *(cbParams->finalAnalogsDates + i) = dateArrayOneDay(i);
    }

    delete(cbParams);
}

template<class T>
void asProcessorCuda::ArraysInsert(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd, bool isAsc,
                                   const T valRef, const T valOther)
{
    // Check the other array length
    auto vlength = (int) (pArrRefEnd - pArrRefStart);

    // Index where to insert the new element
    int iNext = 0;

    // Check order
    if (isAsc) {
        iNext = FindCeil<float>(pArrRefStart, pArrRefEnd, valRef);
        if (iNext == -1) {
            iNext = 0;
        }
    } else {
        iNext = FindCeil<float>(pArrRefStart, pArrRefEnd, valRef);
        if (iNext == -1) {
            iNext = 0;
        }
    }

    // Swap next elements
    for (int i = vlength - 1; i >= iNext; i--) // Minus 1 because we overwrite the last element by the previous one
    {
        pArrRefStart[i + 1] = pArrRefStart[i];
        pArrOtherStart[i + 1] = pArrOtherStart[i];
    }

    // Insert new element
    pArrRefStart[iNext] = valRef;
    pArrOtherStart[iNext] = valOther;
}

template<class T>
int asProcessorCuda::FindCeil(const T *pArrStart, const T *pArrEnd, const T targetValue)
{
    T *pFirst = nullptr, *pMid = nullptr, *pLast = nullptr;
    int vlength;

    // Initialize first and last variables.
    pFirst = (T *) pArrStart;
    pLast = (T *) pArrEnd;

    double tolerance = 0;

    // Check array order
    if (*pLast > *pFirst) {
        // Check that the value is within the array
        if (targetValue - tolerance > *pLast || targetValue + tolerance < *pFirst) {
            return -1;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = (int) (pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pFirst = pMid + 1;
            } else if (targetValue + tolerance < *pMid) {
                pLast = pMid - 1;
            } else {
                // Return found index
                return int(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pLast - pArrStart < 0) {
            pLast = (T *) pArrStart;
        } else if (pLast - pArrEnd > 0) {
            pLast = (T *) pArrEnd;
        }

        // If the value was not found, return ceil value
        return int(pLast - pArrStart + 1);
    } else if (*pLast < *pFirst) {
        // Check that the value is within the array
        if (targetValue + tolerance < *pLast || targetValue - tolerance > *pFirst) {
            return -1;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = (int) (pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pLast = pMid - 1;
            } else if (targetValue + tolerance < *pMid) {
                pFirst = pMid + 1;
            } else {
                // Return found index
                return int(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pFirst - pArrStart < 0) {
            pFirst = (T *) pArrStart;
        } else if (pFirst - pArrEnd > 0) {
            pFirst = (T *) pArrEnd;
        }

        // If the value was not found, return ceil value
        return int(pFirst - pArrStart - 1);
    } else {
        if (pLast - pFirst == 0) {
            if (std::abs(*pFirst - targetValue) <= tolerance) {
                return 0; // Value corresponds
            } else {
                return -1;
            }
        }

        if (std::abs(*pFirst - targetValue) <= tolerance) {
            return 0; // Value corresponds
        } else {
            return -1;
        }
    }
}

template<class T>
bool asProcessorCuda::SortArrays(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd, bool isAsc)
{
    // Check the other array length
    auto vlength = (int) (pArrRefEnd - pArrRefStart);
    auto ovlength = (int) (pArrOtherEnd - pArrOtherStart);

    if (vlength > 0 && vlength == ovlength) {
        int low = 0, high = vlength;
        QuickSortMulti<T>(pArrRefStart, pArrOtherStart, low, high, isAsc);
    } else if (vlength != ovlength) {
        printf("The dimension of the two arrays are not equal.");
        return false;
    } else if (vlength == 0) {
        return true;
    } else {
        printf("The array has a negative size...");
        return false;
    }
    return true;
}

template<class T>
void asProcessorCuda::QuickSortMulti(T *pArrRef, T *pArrOther, const int low, const int high, bool isAsc)
{
    int L, R;
    T pivot, tmp;

    R = high;
    L = low;

    pivot = pArrRef[(low + high) / 2];

    do {
        if (isAsc) {
            while (pArrRef[L] < pivot)
                L++;
            while (pArrRef[R] > pivot)
                R--;
        } else {
            while (pArrRef[L] > pivot)
                L++;
            while (pArrRef[R] < pivot)
                R--;
        }

        if (R >= L) {
            if (R != L) {
                // Reference array
                tmp = pArrRef[R];
                pArrRef[R] = pArrRef[L];
                pArrRef[L] = tmp;
                // Other array
                tmp = pArrOther[R];
                pArrOther[R] = pArrOther[L];
                pArrOther[L] = tmp;
            }

            R--;
            L++;
        }
    } while (L <= R);

    if (low < R)
        QuickSortMulti<T>(pArrRef, pArrOther, low, R, isAsc);
    if (L < high)
        QuickSortMulti<T>(pArrRef, pArrOther, L, high, isAsc);

}
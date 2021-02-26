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
#pragma warning(disable : 4244)  // C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data
#pragma warning(disable : 4267)  // C4267: conversion from 'size_t' to 'int', possible loss of data
#endif

#include <stdio.h>

#include <cmath>

#include "asProcessorCuda.cuh"

#define FULL_MASK 0xffffffff

// The number of threads per block should be a multiple of 32 threads, because this provides optimal computing
// efficiency and facilitates coalescing.
static const int blockSize = 64;  // must be 64 <= blockSize <= 1024

// From https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) val += __shfl_down_sync(FULL_MASK, val, offset);

    return val;
}

__global__ void processSgrads(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg,
                              const int *idxArch, float w, float *out, int offset) {
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        __shared__ float mem[2 * blockSize / 32];
        float *sdiff = mem;
        float *smax = &sdiff[blockSize / 32];

        float rdiff = 0;
        float rmax = 0;

        float diff = 0;
        float amax = 0;

        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                float xi = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                float yi = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];

                diff = fabsf(xi - yi);
                amax = fmaxf(fabsf(xi), fabsf(yi));
            } else {
                diff = 0;
                amax = 0;
            }
            __syncthreads();

            // Reduction
            diff = warpReduceSum(diff);
            amax = warpReduceSum(amax);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                sdiff[idx] = diff;
                smax[idx] = amax;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rdiff += diff;
                rmax += amax;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rdiff += sdiff[j];
                    rmax += smax[j];
                }
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            if (rmax == 0) {
                out[offset + blockId] += 200.0f * w;
            } else {
                out[offset + blockId] += 100.0f * (rdiff / rmax) * w;
            }
        }
    }
}

__global__ void processMD(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg, const int *idxArch,
                          float w, float *out, int offset) {
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        __shared__ float sdiff[blockSize / 32];

        float rdiff = 0;
        float diff = 0;

        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                float xi = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                float yi = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];

                diff = fabsf(xi - yi);
            } else {
                diff = 0;
            }
            __syncthreads();

            // Reduction
            diff = warpReduceSum(diff);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                sdiff[idx] = diff;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rdiff += diff;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rdiff += sdiff[j];
                }
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            out[offset + blockId] += (rdiff / float(ptsNbtot)) * w;
        }
    }
}

__global__ void processRMSE(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg,
                            const int *idxArch, float w, float *out, int offset) {
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        __shared__ float sdiff[blockSize / 32];

        float rdiff = 0;
        float diff = 0;

        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                float xi = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                float yi = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];

                diff = xi - yi;
                diff = diff * diff;
            } else {
                diff = 0;
            }
            __syncthreads();

            // Reduction
            diff = warpReduceSum(diff);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                sdiff[idx] = diff;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rdiff += diff;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rdiff += sdiff[j];
                }
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            out[offset + blockId] += std::sqrt(rdiff / float(ptsNbtot)) * w;
        }
    }
}

__global__ void processRSE(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg, const int *idxArch,
                           float w, float *out, int offset) {
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        __shared__ float sdiff[blockSize / 32];

        float rdiff = 0;
        float diff = 0;

        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                float xi = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                float yi = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];

                diff = xi - yi;
                diff = diff * diff;
            } else {
                diff = 0;
            }
            __syncthreads();

            // Reduction
            diff = warpReduceSum(diff);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                sdiff[idx] = diff;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rdiff += diff;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rdiff += sdiff[j];
                }
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            out[offset + blockId] += std::sqrt(rdiff) * w;
        }
    }
}

__global__ void processSAD(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg, const int *idxArch,
                           float w, float *out, int offset) {
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        __shared__ float sdiff[blockSize / 32];

        float rdiff = 0;
        float diff = 0;

        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                float xi = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                float yi = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];

                diff = fabsf(xi - yi);
            } else {
                diff = 0;
            }
            __syncthreads();

            // Reduction
            diff = warpReduceSum(diff);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                sdiff[idx] = diff;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rdiff += diff;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rdiff += sdiff[j];
                }
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            out[offset + blockId] += rdiff * w;
        }
    }
}

__global__ void processDMV(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg, const int *idxArch,
                           float w, float *out, int offset) {
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        __shared__ float mem[2 * blockSize / 32];
        float *ssumX = mem;
        float *ssumY = &ssumX[blockSize / 32];

        float rsumX = 0;
        float rsumY = 0;

        float sumX = 0;
        float sumY = 0;

        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                sumX = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                sumY = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];
            } else {
                sumX = 0;
                sumY = 0;
            }
            __syncthreads();

            // Reduction
            sumX = warpReduceSum(sumX);
            sumY = warpReduceSum(sumY);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                ssumX[idx] = sumX;
                ssumY[idx] = sumY;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rsumX += sumX;
                rsumY += sumY;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rsumX += ssumX[j];
                    rsumY += ssumY[j];
                }
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            out[offset + blockId] += std::fabs(rsumX / float(ptsNbtot) - rsumY / float(ptsNbtot)) * w;
        }
    }
}

__global__ void processDSD(const float *data, long ptorStart, int candNb, int ptsNbtot, int idxTarg, const int *idxArch,
                           float w, float *out, int offset) {
    const int blockId = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.x;

    if (blockId < candNb) {
        int iTarg = idxTarg;
        int iArch = idxArch[offset + blockId];

        __shared__ float mem[2 * blockSize / 32 + 2];
        float *smemX = mem;
        float *smemY = &smemX[blockSize / 32];
        float *meanX = &smemY[blockSize / 32];
        float *meanY = &meanX[1];

        float rvarX = 0;
        float rvarY = 0;

        float varX = 0;
        float varY = 0;

        // First loop: process the mean
        int nLoops = ceil(double(ptsNbtot) / blockSize);
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                varX = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId];
                varY = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId];
            } else {
                varX = 0;
                varY = 0;
            }
            __syncthreads();

            // Reduction
            varX = warpReduceSum(varX);
            varY = warpReduceSum(varY);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                smemX[idx] = varX;
                smemY[idx] = varY;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rvarX += varX;
                rvarY += varY;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rvarX += smemX[j];
                    rvarY += smemY[j];
                }
            }
        }
        __syncthreads();

        // Process the mean
        if (threadId == 0) {
            *meanX = rvarX / float(ptsNbtot);
            *meanY = rvarY / float(ptsNbtot);
        }
        __syncthreads();

        // Second loop: process the std dev
        rvarX = 0;
        rvarY = 0;
        for (int i = 0; i < nLoops; ++i) {
            int nPts = blockSize;
            if (i == nLoops - 1) {
                nPts = ptsNbtot - (i * blockSize);
            }

            // Process differences and get abs max
            if (threadId < nPts) {
                // Lookup data value
                varX = data[ptorStart + iTarg * ptsNbtot + i * blockSize + threadId] - *meanX;
                varY = data[ptorStart + iArch * ptsNbtot + i * blockSize + threadId] - *meanY;
                varX *= varX;
                varY *= varY;
            } else {
                varX = 0;
                varY = 0;
            }
            __syncthreads();

            // Reduction
            varX = warpReduceSum(varX);
            varY = warpReduceSum(varY);

            __syncthreads();

            // Store in shared memory
            if (threadId > 0 && threadId % 32 == 0) {
                int idx = threadId / 32;
                smemX[idx] = varX;
                smemY[idx] = varY;
            }
            __syncthreads();

            // Final sum
            if (threadId == 0) {
                rvarX += varX;
                rvarY += varY;
                for (int j = 1; j < blockSize / 32; ++j) {
                    rvarX += smemX[j];
                    rvarY += smemY[j];
                }
            }
        }
        __syncthreads();

        // Process final score
        if (threadId == 0) {
            float refStdDev = std::sqrt(rvarX / float(ptsNbtot - 1));
            float evalStdDev = std::sqrt(rvarY / float(ptsNbtot - 1));

            out[offset + blockId] += std::fabs(refStdDev - evalStdDev) * w;
        }
    }
}

bool asProcessorCuda::ProcessCriteria(const float *dData, std::vector<long> ptorStart, int indexTarg,
                                      const int *indicesArch, float *dRes, int nbCandidates, std::vector<int> &colsNb,
                                      std::vector<int> &rowsNb, std::vector<float> &weights,
                                      std::vector<CudaCriteria> &criteria, cudaStream_t &stream, int offset) {
    for (int iPtor = 0; iPtor < ptorStart.size(); iPtor++) {
        int ptsNb = colsNb[iPtor] * rowsNb[iPtor];

        // Define block size (must be multiple of 32) and blocks nb
        int blocksNbXY = ceil(std::cbrt(nbCandidates));
        int blocksNbZ = ceil((double)nbCandidates / (blocksNbXY * blocksNbXY));
        dim3 blocksNb3D(blocksNbXY, blocksNbXY, blocksNbZ);

        // Launch kernel
        switch (criteria[iPtor]) {
            case S0:
            case S1grads:
            case S2grads:
                // Valid for S0, S1, and S2 as the gradients were processed beforehand
                processSgrads<<<blocksNb3D, blockSize, 0, stream>>>(dData, ptorStart[iPtor], nbCandidates, ptsNb,
                                                                    indexTarg, indicesArch, weights[iPtor], dRes, offset);
                break;
            case MD:
                processMD<<<blocksNb3D, blockSize, 0, stream>>>(dData, ptorStart[iPtor], nbCandidates, ptsNb, indexTarg,
                                                                indicesArch, weights[iPtor], dRes, offset);
                break;
            case RMSE:
                processRMSE<<<blocksNb3D, blockSize, 0, stream>>>(dData, ptorStart[iPtor], nbCandidates, ptsNb,
                                                                  indexTarg, indicesArch, weights[iPtor], dRes, offset);
                break;
            case RSE:
                processRSE<<<blocksNb3D, blockSize, 0, stream>>>(dData, ptorStart[iPtor], nbCandidates, ptsNb,
                                                                 indexTarg, indicesArch, weights[iPtor], dRes, offset);
                break;
            case SAD:
                processSAD<<<blocksNb3D, blockSize, 0, stream>>>(dData, ptorStart[iPtor], nbCandidates, ptsNb,
                                                                 indexTarg, indicesArch, weights[iPtor], dRes, offset);
                break;
            case DMV:
                processDMV<<<blocksNb3D, blockSize, 0, stream>>>(dData, ptorStart[iPtor], nbCandidates, ptsNb,
                                                                 indexTarg, indicesArch, weights[iPtor], dRes, offset);
                break;
            case DSD:
                processDSD<<<blocksNb3D, blockSize, 0, stream>>>(dData, ptorStart[iPtor], nbCandidates, ptsNb,
                                                                 indexTarg, indicesArch, weights[iPtor], dRes, offset);
                break;
            default:
                printf("Criteria not yet implemented on GPU.");
                return false;
        }
    }

    return true;
}

bool asProcessorCuda::SelectBestDevice() {
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

int asProcessorCuda::GetDeviceCount() {
    cudaError_t cudaStatus;

    // Count the devices
    int devicesCount = 0;
    cudaStatus = cudaGetDeviceCount(&devicesCount);
    if (cudaStatus != cudaSuccess) {
        if (cudaStatus == cudaErrorNoDevice) {
            printf("cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\n");
            return 0;
        } else if (cudaStatus == cudaErrorInsufficientDriver) {
            printf("cudaGetDeviceCount failed! No driver can be loaded to determine if any device exists.\n");
            return 0;
        }

        printf("cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?\n");
        return 0;
    }

    return devicesCount;
}

void asProcessorCuda::SetDevice(int device) {
    checkCudaErrors(cudaSetDevice(device));
}

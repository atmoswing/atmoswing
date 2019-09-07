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

#ifndef AS_PROCESSOR_CUDA_H
#define AS_PROCESSOR_CUDA_H

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#ifndef EIGEN_NO_DEBUG
#   define EIGEN_NO_DEBUG
#endif

// Modules and Header files: http://eigen.tuxfamily.org/dox-3.0/QuickRefPage.html#QuickRef_Headers
#include <Eigen/StdVector>
#include <Eigen/Core>

#include <vector>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int nStreams = 20;

enum CudaCriteria
{
    S0grads = 0,
    S1grads = 1,
    S2grads = 2,
    MD = 3,
    RMSE = 4,
    RSE = 5,
    SAD = 6,
    DMV = 7,
    DSD = 8,
};

class asProcessorCuda
{
public:
    static bool ProcessCriteria(const float *dData, std::vector<long> ptorStart, int indicesTarg, const int *indicesArch,
                                float *dRes, int nbCandidates, std::vector<int> &colsNb, std::vector<int> &rowsNb,
                                std::vector<float> &weights, std::vector<CudaCriteria> &criteria, int streamId, int offset);

    static bool SelectBestDevice();

    static void InitStreams();

    static void DestroyStreams();

    static void CudaMalloc(int *&data, int length);

    static void CudaMalloc(float *&data, long length);

    static void CudaMemset0(float *data, long length);

    static void CudaMemset0Async(float *data, long length, int streamId);

    static void CudaMemCopyToDevice(int *devData, int *hostData, int length);

    static void CudaMemCopyToDeviceAsync(int *devData, int *hostData, int length, int streamId);

    static void CudaMemCopyToDevice(float *devData, float *hostData, long length);

    static void CudaMemCopyFromDevice(int *hostData, int *devData, int length);

    static void CudaMemCopyFromDeviceAsync(int *hostData, int *devData, int length, int streamId);

    static void CudaMemCopyFromDevice(float *hostData, float *devData, long length);

    static void CudaMemCopyFromDeviceAsync(float *hostData, float *devData, long length, int streamId);

    static void CudaFree(int *data);

    static void CudaFree(float *data);

    static void CudaGetLastError();

    static void DeviceSynchronize();

    static void StreamSynchronize(int streamId);

    static void DeviceReset();

protected:

private:

};


extern cudaStream_t *g_streams;

#endif

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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <helper_cuda.h>

#include <vector>

const int nStreams = 4;  // must be a multiple of 2!

enum CudaCriteria {
    S0 = 0,
    S1grads = 1,
    S2grads = 2,
    MD = 3,
    RMSE = 4,
    RSE = 5,
    SAD = 6,
    DMV = 7,
    DSD = 8,
};

class asProcessorCuda {
  public:
    static bool ProcessCriteria(const float *dData, std::vector<long> ptorStart, int indicesTarg,
                                const int *indicesArch, float *dRes, int nbCandidates, std::vector<int> &colsNb,
                                std::vector<int> &rowsNb, std::vector<float> &weights,
                                std::vector<CudaCriteria> &criteria, cudaStream_t &stream, int offset);

    static bool SelectBestDevice();

    static int GetDeviceCount();

    static void SetDevice(int device);

  protected:
  private:
};

#endif

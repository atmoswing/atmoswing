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

#define STRUCT_MAX_SIZE 12
#define USE_STREAMS 0

#include <vector>

struct cudaPredictorsDataPropStruct
{
    int ptorsNb;
    int rowsNb[STRUCT_MAX_SIZE];
    int colsNb[STRUCT_MAX_SIZE];
    int ptsNb[STRUCT_MAX_SIZE];
    int totPtsNb;
    int indexStart[STRUCT_MAX_SIZE];
    float weights[STRUCT_MAX_SIZE];
};

class asProcessorCuda
{
public:
    static bool SelectBestDevice();

    static float *MallocCudaData(int n);

    static void FreeCudaData(float *data);

    static void DeviceSynchronize();

    static void DeviceReset();

    static bool ProcessS1grads(float *out, const float *refData, const float *evalData, int rowsNb, int colsNb);

    static bool ProcessCriteria(std::vector <std::vector<float *>> &data,
                                std::vector<int> &indicesTarg,
                                std::vector <std::vector<int>> &indicesArch,
                                std::vector <std::vector<float>> &resultingCriteria,
                                std::vector<int> &nbArchCandidates,
                                std::vector<int> &colsNb, std::vector<int> &rowsNb,
                                std::vector<float> &weights);

protected:

private:
};

#endif

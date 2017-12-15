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
 */

#ifndef ASPROCESSORCUDA_H
#define ASPROCESSORCUDA_H

#define USE_PINNED_MEM 1

#define STRUCT_MAX_SIZE 12

#include <vector>

#if USE_THRUST == 0
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
#endif

class asProcessorCuda
{
public:
    static bool ProcessCriteria(std::vector <std::vector<float *>> &data, std::vector<int> &indicesTarg,
                                std::vector <std::vector<int>> &indicesArch,
                                std::vector <std::vector<float>> &resultingCriteria, std::vector<int> &lengths,
                                std::vector<int> &colsNb, std::vector<int> &rowsNb, std::vector<float> &weights);

protected:

private:
};

#endif

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
 
#ifndef ASPROCESSORCUDA_H
#define ASPROCESSORCUDA_H

#include <vector>


struct cudaPredictorsMetaDataStruct {
    int ptorsNb;
    int rowsNb[20];
    int colsNb[20];
    int ptsNb[20];
    int totPtsNb;
    int indexStart[20];
    int indexEnd[20];
    float weights[20];
};

class asProcessorCuda
{
public:

    static bool ProcessCriteria(std::vector < float* > &vpTargData, 
                                std::vector < std::vector < float* > > &vvpArchData, 
                                std::vector < float > &criteriaValues, 
                                int size, 
                                std::vector < int > &colsNb, 
                                std::vector < int > &rowsNb, 
                                std::vector < float > &weights);

protected:
private:
};

#endif
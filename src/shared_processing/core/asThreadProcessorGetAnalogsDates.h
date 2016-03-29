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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASTHREADPROCESSORGETANALOGSDATES_H
#define ASTHREADPROCESSORGETANALOGSDATES_H

#include <asThread.h>
#include <asParameters.h>
#include <asIncludes.h>

class asDataPredictor;

class asPredictorCriteria;

class asTimeArray;


class asThreadProcessorGetAnalogsDates
        : public asThread
{
public:
    asThreadProcessorGetAnalogsDates(std::vector<asDataPredictor *> predictorsArchive,
                                     std::vector<asDataPredictor *> predictorsTarget, asTimeArray *timeArrayArchiveData,
                                     asTimeArray *timeArrayArchiveSelection, asTimeArray *timeArrayTargetData,
                                     asTimeArray *timeArrayTargetSelection, std::vector<asPredictorCriteria *> criteria,
                                     asParameters &params, int step, VpArray2DFloat &vRefData,
                                     VpArray2DFloat &vEvalData, Array1DInt &vRowsNb, Array1DInt &vColsNb, int start,
                                     int end, Array2DFloat *finalAnalogsCriteria, Array2DFloat *finalAnalogsDates,
                                     bool *containsNaNs);

    virtual ~asThreadProcessorGetAnalogsDates();

    virtual ExitCode Entry();

protected:

private:
    std::vector<asDataPredictor *> m_pPredictorsArchive;
    std::vector<asDataPredictor *> m_pPredictorsTarget;
    asTimeArray *m_pTimeArrayArchiveData;
    asTimeArray *m_pTimeArrayArchiveSelection;
    asTimeArray *m_pTimeArrayTargetData;
    asTimeArray *m_pTimeArrayTargetSelection;
    std::vector<asPredictorCriteria *> m_criteria;
    asParameters m_params;
    int m_step;
    VpArray2DFloat m_vTargData;
    VpArray2DFloat m_vArchData;
    Array1DInt m_vRowsNb;
    Array1DInt m_vColsNb;
    int m_start;
    int m_End;
    Array2DFloat *m_pFinalAnalogsCriteria;
    Array2DFloat *m_pFinalAnalogsDates;
    bool *m_pContainsNaNs;

};

#endif // ASTHREADPROCESSORGETANALOGSDATES_H

/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASTHREADPROCESSORGETANALOGSDATES_H
#define ASTHREADPROCESSORGETANALOGSDATES_H

#include <asThread.h>
#include <asParameters.h>
#include <asIncludes.h>

class asDataPredictor;
class asPredictorCriteria;
class asTimeArray;


class asThreadProcessorGetAnalogsDates: public asThread
{
public:
    /** Default constructor */
    asThreadProcessorGetAnalogsDates(std::vector < asDataPredictor >* predictorsArchive,
                                     std::vector < asDataPredictor >* predictorsTarget,
                                     asTimeArray* timeArrayArchiveData,
                                     asTimeArray* timeArrayArchiveSelection,
                                     asTimeArray* timeArrayTargetData,
                                     asTimeArray* timeArrayTargetSelection,
                                     std::vector < asPredictorCriteria* > criteria,
                                     asParameters &params,
                                     int step,
                                     VpArray2DFloat &vRefData,
                                     VpArray2DFloat &vEvalData,
                                     Array1DInt &vRowsNb,
                                     Array1DInt &vColsNb,
                                     int start,
                                     int end,
                                     Array2DFloat* finalAnalogsCriteria,
                                     Array2DFloat* finalAnalogsDates,
                                     bool* containsNaNs);
    /** Default destructor */
    virtual ~asThreadProcessorGetAnalogsDates();

    virtual ExitCode Entry();


protected:
private:
    std::vector < asDataPredictor >* m_pPredictorsArchive;
    std::vector < asDataPredictor >* m_pPredictorsTarget;
    asTimeArray* m_pTimeArrayArchiveData;
    asTimeArray* m_pTimeArrayArchiveSelection;
    asTimeArray* m_pTimeArrayTargetData;
    asTimeArray* m_pTimeArrayTargetSelection;
    std::vector < asPredictorCriteria* > m_Criteria;
    asParameters m_Params;
    int m_Step;
    VpArray2DFloat m_vTargData;
    VpArray2DFloat m_vArchData;
    Array1DInt m_vRowsNb;
    Array1DInt m_vColsNb;
    int m_Start;
    int m_End;
    Array2DFloat* m_pFinalAnalogsCriteria;
    Array2DFloat* m_pFinalAnalogsDates;
    bool* m_pContainsNaNs;

};

#endif // ASTHREADPROCESSORGETANALOGSDATES_H

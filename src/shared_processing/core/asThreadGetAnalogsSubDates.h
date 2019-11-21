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

#ifndef AS_THREAD_PROCESSOR_GET_ANALOGS_SUB_DATES_H
#define AS_THREAD_PROCESSOR_GET_ANALOGS_SUB_DATES_H

#include <asIncludes.h>
#include <asParameters.h>
#include <asThread.h>

class asPredictor;

class asCriteria;

class asTimeArray;

class asThreadGetAnalogsSubDates : public asThread {
   public:
    asThreadGetAnalogsSubDates(std::vector<asPredictor *> predictorsArchive,
                               std::vector<asPredictor *> predictorsTarget, asTimeArray *timeArrayArchiveData,
                               asTimeArray *timeArrayTargetData, a1f *timeTargetSelection,
                               std::vector<asCriteria *> criteria, asParameters *params, int step, vpa2f &vRefData,
                               vpa2f &vEvalData, a1i &vRowsNb, a1i &vColsNb, int start, int end,
                               a2f *finalAnalogsCriteria, a2f *finalAnalogsDates, a2f *previousAnalogsDates,
                               bool *containsNaNs, bool *success);

    virtual ~asThreadGetAnalogsSubDates();

    virtual ExitCode Entry();

   protected:
   private:
    std::vector<asPredictor *> m_pPredictorsArchive;
    std::vector<asPredictor *> m_pPredictorsTarget;
    asTimeArray *m_pTimeArrayArchiveData;
    asTimeArray *m_pTimeArrayTargetData;
    a1f *m_pTimeTargetSelection;
    std::vector<asCriteria *> m_criteria;
    asParameters *m_params;
    int m_step;
    vpa2f m_vTargData;
    vpa2f m_vArchData;
    a1i m_vRowsNb;
    a1i m_vColsNb;
    int m_start;
    int m_end;
    a2f *m_pFinalAnalogsCriteria;
    a2f *m_pFinalAnalogsDates;
    a2f *m_pPreviousAnalogsDates;
    bool *m_pContainsNaNs;
    bool *m_success;
};

#endif

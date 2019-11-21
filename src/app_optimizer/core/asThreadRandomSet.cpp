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

#include "asThreadRandomSet.h"

asThreadRandomSet::asThreadRandomSet(asMethodOptimizerRandomSet *optimizer, asParametersOptimization *params,
                                     float *finalScoreCalib, vf *scoreClimatology)
    : asThread(asThread::MethodOptimizerRandomSet),
      m_optimizer(optimizer),
      m_params(params),
      m_finalScoreCalib(finalScoreCalib),
      m_scoreClimatology(scoreClimatology) {}

asThreadRandomSet::~asThreadRandomSet() {}

wxThread::ExitCode asThreadRandomSet::Entry() {
    // Create results objects. Needs to be in a critical section because of access to the config pointer.
    asResultsDates anaDates;
    asResultsDates anaDatesPrevious;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    // Set the climatology score value
    if (m_scoreClimatology->size() != 0) {
        wxLogVerbose(_("Process score of the climatology"));
        m_optimizer->SetScoreClimatology(*m_scoreClimatology);
    }

    // Process every step one after the other
    int stepsNb = m_params->GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!m_optimizer->GetAnalogsDates(anaDates, m_params, iStep, containsNaNs)) {
                wxLogError(_("Failed processing the analogs dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        } else {
            if (!m_optimizer->GetAnalogsSubDates(anaDates, m_params, anaDatesPrevious, iStep, containsNaNs)) {
                wxLogError(_("Failed processing the analogs sub dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return NULL;
        }
    }
    if (!m_optimizer->GetAnalogsValues(anaValues, m_params, anaDates, stepsNb - 1)) {
        wxLogError(_("Failed processing the analogs values"));
        return NULL;
    }
    if (!m_optimizer->GetAnalogsScores(anaScores, m_params, anaValues, stepsNb - 1)) {
        wxLogError(_("Failed processing the scores"));
        return NULL;
    }
    if (!m_optimizer->GetAnalogsTotalScore(anaScoreFinal, m_params, anaScores, stepsNb - 1)) {
        wxLogError(_("Failed processing the total score"));
        return NULL;
    }
    *m_finalScoreCalib = anaScoreFinal.GetScore();

    if (m_scoreClimatology->size() == 0) {
        *m_scoreClimatology = m_optimizer->GetScoreClimatology();
    }

    return 0;
}

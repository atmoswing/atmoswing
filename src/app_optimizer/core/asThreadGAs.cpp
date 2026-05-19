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

#include "asThreadGAs.h"

#ifdef USE_CUDA
#include "asProcessorCuda.cuh"
#endif

asThreadGAs::asThreadGAs(asMethodOptimizerGAs* optimizer, asParametersOptimization* params, float* finalScoreCalib,
                         vf* scoreClimatology)
    : asThread(asThread::MethodOptimizerGAs),
      _optimizer(optimizer),
      _params(params),
      _finalScoreCalib(finalScoreCalib),
      _scoreClimatology(scoreClimatology) {}

asThreadGAs::~asThreadGAs() {}

wxThread::ExitCode asThreadGAs::Entry() {
    // Create results objects. Needs to be in a critical section because of access to the config pointer.
    asResultsDates anaDates;
    asResultsDates anaDatesPrevious;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    *_finalScoreCalib = NAN;

    // Set the climatology score value
    if (!_scoreClimatology->empty()) {
        _optimizer->SetScoreClimatology(*_scoreClimatology);
    }

    // Process every step one after the other
    int stepsNb = _params->GetStepsNb();

#ifdef USE_CUDA
    asProcessorCuda::SetDevice(_device);
#endif

    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!_optimizer->GetAnalogsDates(anaDates, _params, iStep, containsNaNs)) {
                wxLogError(_("Failed processing the analogs dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        } else {
            if (!_optimizer->GetAnalogsSubDates(anaDates, _params, anaDatesPrevious, iStep, containsNaNs)) {
                wxLogError(_("Failed processing the analogs sub dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return NULL;
        }
        if (anaDates.GetTargetDates().size() == 0 || anaDates.GetAnalogsDates().size() == 0 ||
            anaDates.GetAnalogsCriteria().size() == 0) {
            wxLogError(_("The asResultsDates object is empty in asThreadGAs."));
            return NULL;
        }
    }
    if (!_optimizer->GetAnalogsValues(anaValues, _params, anaDates, stepsNb - 1)) {
        wxLogError(_("Failed processing the analogs values"));
        return NULL;
    }
    if (!_optimizer->GetAnalogsScores(anaScores, _params, anaValues, stepsNb - 1)) {
        wxLogError(_("Failed processing the scores"));
        return NULL;
    }
    if (!_optimizer->GetAnalogsTotalScore(anaScoreFinal, _params, anaScores, stepsNb - 1)) {
        wxLogError(_("Failed processing the total score"));
        return NULL;
    }
    *_finalScoreCalib = anaScoreFinal.GetScore();

    if (_scoreClimatology->empty()) {
        *_scoreClimatology = _optimizer->GetScoreClimatology();
    }

    return 0;
}

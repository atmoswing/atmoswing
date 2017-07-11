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

#include "asMethodOptimizer.h"

asMethodOptimizer::asMethodOptimizer()
        : asMethodCalibrator()
{
    m_iterator = 0;
    m_optimizerStage = asINITIALIZATION;
    m_skipNext = false;
    m_isOver = false;
    m_paramsNb = 0;

    // Seeds the random generator
    asTools::InitRandom();
}

asMethodOptimizer::~asMethodOptimizer()
{
    //dtor
}

bool asMethodOptimizer::SaveDetails(asParametersOptimization &params)
{
    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params.GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb - 1))
        return false;

    anaDates.SetSubFolder("calibration");
    anaDates.Save();
    anaValues.SetSubFolder("calibration");
    anaValues.Save();
    anaScores.SetSubFolder("calibration");
    anaScores.Save();

    return true;
}

bool asMethodOptimizer::Validate(asParametersOptimization &params)
{
    if (!params.HasValidationPeriod()) {
        wxLogWarning("The parameters have no validation period !");
        return false;
    }

    m_validationMode = true;

    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params.GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb - 1))
        return false;

    anaDates.SetSubFolder("validation");
    anaDates.Save();
    anaValues.SetSubFolder("validation");
    anaValues.Save();
    anaScores.SetSubFolder("validation");
    anaScores.Save();

    m_scoreValid = anaScoreFinal.GetForecastScore();

    m_validationMode = false;

    return true;
}

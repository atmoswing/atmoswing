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

#include "asMethodCalibratorSingle.h"

asMethodCalibratorSingle::asMethodCalibratorSingle()
        : asMethodCalibrator()
{

}

asMethodCalibratorSingle::~asMethodCalibratorSingle()
{

}

bool asMethodCalibratorSingle::Calibrate(asParametersCalibration &params)
{
    // Check that we really handle a single case
    bool checkSizes = true;
    wxString errorField = wxEmptyString;
    if (params.GetTimeArrayAnalogsIntervalDaysVector().size() > 1) {
        checkSizes = false;
        errorField.Append("IntervalDays, ");
    }
    if (params.GetForecastScoreNameVector().size() > 1) {
        checkSizes = false;
        errorField.Append("ForecastScoreName, ");
    }
    if (params.GetForecastScoreTimeArrayModeVector().size() > 1) {
        checkSizes = false;
        errorField.Append("ForecastScoreTimeArrayMode, ");
    }
    if (params.GetForecastScoreTimeArrayDateVector().size() > 1) {
        checkSizes = false;
        errorField.Append("ForecastScoreTimeArrayDate ,");
    }
    if (params.GetForecastScoreTimeArrayIntervalDaysVector().size() > 1) {
        checkSizes = false;
        errorField.Append("ForecastScoreTimeArrayIntervalDays, ");
    }
    if (params.GetForecastScorePostprocessDupliExpVector().size() > 1) {
        checkSizes = false;
        errorField.Append("ForecastScorePostprocessDupliExp, ");
    }

    for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
        if (params.GetAnalogsNumberVector(iStep).size() > 1) {
            checkSizes = false;
            errorField.Append(wxString::Format("analogsNumber (step %d), ", iStep));
        }
        for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
            if (params.NeedsPreprocessing(iStep, iPtor)) {
                for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
                    if (params.GetPreprocessDataIdVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(
                                wxString::Format("preprocessDataId (step %d, predictor %d, preprocess %d), ", iStep,
                                                 iPtor, iPre));
                    }
                    if (params.GetPreprocessLevelVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(
                                wxString::Format("PreprocessLevel (step %d, predictor %d, preprocess %d), ", iStep,
                                                 iPtor, iPre));
                    }
                    if (params.GetPreprocessTimeHoursVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(
                                wxString::Format("preprocessTimeHours (step %d, predictor %d, preprocess %d), ", iStep,
                                                 iPtor, iPre));
                    }
                }
            }

            // Do the other ones anyway
            if (params.GetPredictorDataIdVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorDataId (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorLevelVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorLevel (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorTimeHoursVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(
                        wxString::Format("PredictorTimeHours (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorXminVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorXmin (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorXptsnbVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorXptsnb (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorYminVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorYmin (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorYptsnbVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorYptsnb (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorCriteriaVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorCriteria (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorWeightVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorWeight (step %d, predictor %d), ", iStep, iPtor));
            }

            if (params.NeedsPreprocessing(iStep, iPtor)) {
                for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
                    if (params.GetPreprocessLevelVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(
                                wxString::Format("PreprocessLevel (step %d, predictor %d), ", iStep, iPtor));
                    }
                    if (params.GetPreprocessTimeHoursVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessTimeHoursV (step %d, predictor %d), ", iStep,
                                                           iPtor));
                    }
                }
            }
        }
    }

    if (!checkSizes) {
        errorField = errorField.Remove(errorField.Length() - 3, 2); // Removes the last coma
        wxString errorMessage =
                _("The following parameters are not compatible with the single assessment: ") + errorField;
        wxLogError(errorMessage);
        return false;
    }

    // Extract the stations IDs
    vvi stationsId = params.GetPredictandStationIdsVector();

    // Create result object to save the final parameters sets
    asResultsParametersArray results_all;
    results_all.Init(_("all_station_parameters"));

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsAnalogsDates anaDatesPrevious;

    for (unsigned int iStat = 0; iStat < stationsId.size(); iStat++) {
        ClearAll();

        vi stationId = stationsId[iStat];

        // Reset the score of the climatology
        m_scoreClimatology.clear();

        // Create results objects
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScores;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

        // Create result objects to save the parameters sets
        asResultsParametersArray results_tested;
        results_tested.Init(
                wxString::Format(_("station_%s_tested_parameters"), GetPredictandStationIdsList(stationId)));

        // Set the next station ID
        params.SetPredictandStationIds(stationId);

        // Process every step one after the other
        int stepsNb = params.GetStepsNb();
        for (int iStep = 0; iStep < stepsNb; iStep++) {
            bool containsNaNs = false;
            if (iStep == 0) {
                if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                    return false;
            } else {
                if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                    return false;
            }
            if (containsNaNs) {
                wxLogError(_("The dates selection contains NaNs"));
                return false;
            }
            if (!GetAnalogsValues(anaValues, params, anaDates, iStep))
                return false;
            if (!GetAnalogsForecastScores(anaScores, params, anaValues, iStep))
                return false;
            if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, iStep))
                return false;

            // Store the result
            results_tested.Add(params, anaScoreFinal.GetForecastScore(), m_scoreValid);

            // Keep the analogs dates of the best parameters set
            anaDatesPrevious = anaDates;
        }

        // Validate
        SaveDetails(params);
        Validate(params);

        // Keep the best parameters set
        results_all.Add(params, anaScoreFinal.GetForecastScore(), m_scoreValid);
        if (!results_all.Print())
            return false;
        if (!results_tested.Print())
            return false;
    }

    return true;
}

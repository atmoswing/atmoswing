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

#include "asMethodCalibratorSingleOnlyValues.h"

asMethodCalibratorSingleOnlyValues::asMethodCalibratorSingleOnlyValues()
    : asMethodCalibrator() {}

asMethodCalibratorSingleOnlyValues::~asMethodCalibratorSingleOnlyValues() {}

bool asMethodCalibratorSingleOnlyValues::Calibrate(asParametersCalibration& params) {
    // Check that we really handle a single case
    bool checkSizes = true;
    wxString errorField = wxEmptyString;
    if (params.GetTimeArrayAnalogsIntervalDaysVector().size() > 1) {
        checkSizes = false;
        errorField.Append("IntervalDays, ");
    }

    for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
        if (params.GetAnalogsNumberVector(iStep).size() > 1) {
            checkSizes = false;
            errorField.Append(asStrF("analogsNumber (step %d), ", iStep));
        }
        for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
            if (params.NeedsPreprocessing(iStep, iPtor)) {
                for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
                    if (params.GetPreprocessDataIdVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(
                            asStrF("preprocessDataId (step %d, predictor %d, preprocess %d), ", iStep, iPtor, iPre));
                    }
                    if (params.GetPreprocessLevelVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(
                            asStrF("PreprocessLevel (step %d, predictor %d, preprocess %d), ", iStep, iPtor, iPre));
                    }
                    if (params.GetPreprocessHourVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(
                            asStrF("preprocessHours (step %d, predictor %d, preprocess %d), ", iStep, iPtor, iPre));
                    }
                }
            }

            // Do the other ones anyway
            if (params.GetPredictorDataIdVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorDataId (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorLevelVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorLevel (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorHourVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorHours (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorXminVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorXmin (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorXptsnbVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorXptsnb (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorYminVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorYmin (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorYptsnbVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorYptsnb (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorCriteriaVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorCriteria (step %d, predictor %d), ", iStep, iPtor));
            }
            if (params.GetPredictorWeightVector(iStep, iPtor).size() > 1) {
                checkSizes = false;
                errorField.Append(asStrF("PredictorWeight (step %d, predictor %d), ", iStep, iPtor));
            }

            if (params.NeedsPreprocessing(iStep, iPtor)) {
                for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
                    if (params.GetPreprocessLevelVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(asStrF("PreprocessLevel (step %d, predictor %d), ", iStep, iPtor));
                    }
                    if (params.GetPreprocessHourVector(iStep, iPtor, iPre).size() > 1) {
                        checkSizes = false;
                        errorField.Append(asStrF("PreprocessHoursV (step %d, predictor %d), ", iStep, iPtor));
                    }
                }
            }
        }
    }

    if (!checkSizes) {
        errorField = errorField.Remove(errorField.Length() - 3, 2);  // Removes the last coma
        wxString errorMessage = _("The following parameters are not compatible with the single assessment: ") +
                                errorField;
        wxLogError(errorMessage);
        return false;
    }

    // Extract the stations IDs
    vvi stationsId = params.GetPredictandStationIdsVector();

    // Create result object to save the final parameters sets
    asResultsParametersArray results_all;
    results_all.Init(_("all_station_parameters"));

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsDates anaDatesPrevious;

    for (const auto& stationId : stationsId) {
        ClearAll();

        // Create results objects
        asResultsDates anaDates;
        asResultsValues anaValues;

        // Set the next station ID
        params.SetPredictandStationIds(stationId);

        // Process every step one after the other
        int stepsNb = params.GetStepsNb();
        for (int iStep = 0; iStep < stepsNb; iStep++) {
            bool containsNaNs = false;
            if (iStep == 0) {
                if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs)) return false;
            } else {
                if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs)) return false;
            }
            if (containsNaNs) {
                wxLogError(_("The dates selection contains NaNs"));
                return false;
            }
            if (!GetAnalogsValues(anaValues, &params, anaDates, iStep)) return false;

            // Keep the analogs dates of the best parameters set
            anaDatesPrevious = anaDates;
        }

        anaDates.Save();
        anaValues.Save();
    }

    return true;
}

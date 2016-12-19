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

#include "asProcessorForecastScore.h"

#include <asParametersCalibration.h>
#include <asPostprocessor.h>
#include <asForecastScore.h>
#include <asForecastScoreFinal.h>
#include <asResultsAnalogsValues.h>
#include <asResultsAnalogsForecastScores.h>
#include <asResultsAnalogsForecastScoreFinal.h>
#include <asResultsAnalogsScoresMap.h>
//#include <asDialogProgressBar.h>
#include <asFileAscii.h>

#ifndef UNIT_TESTING

#include <AtmoswingAppOptimizer.h>

#endif

bool asProcessorForecastScore::GetAnalogsForecastScores(asResultsAnalogsValues &anaValues,
                                                        asForecastScore *forecastScore, asParametersScoring &params,
                                                        asResultsAnalogsForecastScores &results,
                                                        VectorFloat &scoresClimatology)
{
    // Extract Data
    Array1DFloat timeTargetSelection = anaValues.GetTargetDates();
    VArray1DFloat targetValues = anaValues.GetTargetValues();
    Array2DFloat analogsCriteria = anaValues.GetAnalogsCriteria();
    VArray2DFloat analogsValues = anaValues.GetAnalogsValues();
    wxASSERT(timeTargetSelection.size()>0);
    wxASSERT(analogsValues.size()>0);
    int timeTargetSelectionLength = anaValues.GetTargetDatesLength();
    int analogsNbDates = analogsValues[0].cols();
    int stationsNb = targetValues.size();

    // Put values in final containers
    results.SetTargetDates(timeTargetSelection);

    // Check analogs number coherence
    if (params.GetForecastScoreAnalogsNumber() > analogsNbDates)
        asThrowException(wxString::Format(
                _("The given analogs number for the forecast score (%d) processing is superior to the analogs dates number (%d)."),
                params.GetForecastScoreAnalogsNumber(), analogsNbDates));

    if (forecastScore->SingleValue()) {
        // Containers for final results
        Array1DFloat finalForecastScores = Array1DFloat::Zero(timeTargetSelectionLength);
        VArray1DFloat vectForecastScores(stationsNb, Array1DFloat(timeTargetSelectionLength));

        for (int i_st = 0; i_st < stationsNb; i_st++) {
            if (forecastScore->UsesClimatology()) {
                forecastScore->SetScoreClimatology(scoresClimatology[i_st]);
            }

            for (int i_targtime = 0; i_targtime < timeTargetSelectionLength; i_targtime++) {
                if (!asTools::IsNaN(targetValues[i_st](i_targtime))) {
                    if (params.ForecastScoreNeedsPostprocessing()) {
                        //Array2DFloat analogsValuesNew(asPostprocessor::Postprocess(analogsValues.row(i_targtime), analogsCriteria.row(i_targtime), params));
                        //finalForecastScores(i_targtime) = forecastScore->Assess(targetValues(i_targtime), analogsValuesNew.row(i_targtime), params.GetForecastScoreAnalogsNumber());
                    } else {
                        vectForecastScores[i_st](i_targtime) = forecastScore->Assess(targetValues[i_st](i_targtime),
                                                                                     analogsValues[i_st].row(i_targtime),
                                                                                     params.GetForecastScoreAnalogsNumber());
                    }
                } else {
                    vectForecastScores[i_st](i_targtime) = NaNFloat;
                }
            }
        }

        // Merge of the different scores
        if (stationsNb == 1) {
            finalForecastScores = vectForecastScores[0];
        } else {
            // Process the average
            for (int i_st = 0; i_st < stationsNb; i_st++) {
                finalForecastScores += vectForecastScores[i_st];
            }
            finalForecastScores /= stationsNb;
        }

        // Put values in final containers
        results.SetForecastScores(finalForecastScores);
    } else {
        if (stationsNb > 1) {
            wxLogError(_("The processing of multivariate complex scores is not implemented yet."));
            return false;
        }

        if (forecastScore->UsesClimatology()) {
            forecastScore->SetScoreClimatology(scoresClimatology[0]);
        }

        // Containers for final results
        Array2DFloat forecastScores(timeTargetSelectionLength, 3 * (params.GetForecastScoreAnalogsNumber() + 1));

        for (int i_targtime = 0; i_targtime < timeTargetSelectionLength; i_targtime++) {
            if (!asTools::IsNaN(targetValues[0](i_targtime))) {
                if (params.ForecastScoreNeedsPostprocessing()) {
                    //Array2DFloat analogsValuesNew(asPostprocessor::Postprocess(analogsValues.row(i_targtime), analogsCriteria.row(i_targtime), params));
                    //finalForecastScores(i_targtime) = forecastScore->Assess(targetValues(i_targtime), analogsValuesNew.row(i_targtime), params.GetForecastScoreAnalogsNumber());
                } else {
                    forecastScores.row(i_targtime) = forecastScore->AssessOnArray(targetValues[0](i_targtime),
                                                                                  analogsValues[0].row(i_targtime),
                                                                                  params.GetForecastScoreAnalogsNumber());
                }
            } else {
                forecastScores.row(i_targtime) =
                        Array1DFloat::Ones(3 * (params.GetForecastScoreAnalogsNumber() + 1)) * NaNFloat;
            }
        }

        // Put values in final containers
        results.SetForecastScores2DArray(forecastScores);
    }

    return true;
}

bool asProcessorForecastScore::GetAnalogsForecastScoreFinal(asResultsAnalogsForecastScores &anaScores,
                                                            asTimeArray &timeArray, asParametersScoring &params,
                                                            asResultsAnalogsForecastScoreFinal &results)
{
    // TODO (phorton#1#): Specify the period in the parameter
    asForecastScoreFinal *finalScore = asForecastScoreFinal::GetInstance(params.GetForecastScoreName(), "Total");

    // Ranks number set for all, but only used for the rank histogram
    finalScore->SetRanksNb(params.GetForecastScoreAnalogsNumber() + 1);

    if (finalScore->Has2DArrayArgument()) {
        float result = finalScore->Assess(anaScores.GetTargetDates(), anaScores.GetForecastScores2DArray(), timeArray);
        results.SetForecastScore(result);
    } else {
        if (finalScore->SingleValue()) {
            float result = finalScore->Assess(anaScores.GetTargetDates(), anaScores.GetForecastScores(), timeArray);
            results.SetForecastScore(result);
        } else {
            Array1DFloat result = finalScore->AssessOnArray(anaScores.GetTargetDates(), anaScores.GetForecastScores(),
                                                            timeArray);
            results.SetForecastScore(result);
        }
    }

    wxDELETE(finalScore);

    return true;
}


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

#include "asProcessorScore.h"

#include "asParametersCalibration.h"
#include "asPostprocessor.h"
#include "asResultsScores.h"
#include "asResultsScoresMap.h"
#include "asResultsTotalScore.h"
#include "asResultsValues.h"
#include "asScore.h"
#include "asTotalScore.h"
// #include "asDialogProgressBar.h"

#ifndef UNIT_TESTING

#endif

bool asProcessorScore::GetAnalogsScores(asResultsValues& anaValues, asScore* score, asParametersScoring* params,
                                        asResultsScores& results, vf& scoresClimatology) {
    // Extract Data
    a1f timeTargetSelection = anaValues.GetTargetDates();
    va1f targetValues = anaValues.GetTargetValues();
    a2f analogsCriteria = anaValues.GetAnalogsCriteria();
    va2f analogsValues = anaValues.GetAnalogsValues();
    wxASSERT(timeTargetSelection.size() > 0);
    wxASSERT(!analogsValues.empty());
    int timeTargetSelectionLength = anaValues.GetTargetDatesLength();
    int analogsNbDates = analogsValues[0].cols();
    int stationsNb = targetValues.size();

    // Put values in final containers
    results.SetTargetDates(timeTargetSelection);

    // Check analogs number coherence
    if (params->GetScoreAnalogsNumber() > analogsNbDates)
        asThrowException(wxString::Format(
            _("The given analogs number for the score (%d) processing is superior to the analogs dates number (%d)."),
            params->GetScoreAnalogsNumber(), analogsNbDates));

    if (score->SingleValue()) {
        // Containers for final results
        a1f totalScores = a1f::Zero(timeTargetSelectionLength);
        va1f vectScores(stationsNb, a1f(timeTargetSelectionLength));

        for (int iStat = 0; iStat < stationsNb; iStat++) {
            if (score->UsesClimatology()) {
                score->SetScoreClimatology(scoresClimatology[iStat]);
            }

            for (int iTargetTime = 0; iTargetTime < timeTargetSelectionLength; iTargetTime++) {
                if (!asIsNaN(targetValues[iStat](iTargetTime))) {
                    if (params->ScoreNeedsPostprocessing()) {
                        // a2f analogsValuesNew(asPostprocessor::Postprocess(analogsValues.row(iTargetTime),
                        // analogsCriteria.row(iTargetTime), params)); totalScores(iTargetTime) =
                        // score->Assess(targetValues(iTargetTime), analogsValuesNew.row(iTargetTime),
                        // params->GetScoreAnalogsNumber());
                    } else {
                        vectScores[iStat](iTargetTime) = score->Assess(targetValues[iStat](iTargetTime),
                                                                       analogsValues[iStat].row(iTargetTime),
                                                                       params->GetScoreAnalogsNumber());
                    }
                } else {
                    vectScores[iStat](iTargetTime) = NaNf;
                }
            }
        }

        // Merge of the different scores
        if (stationsNb == 1) {
            totalScores = vectScores[0];
        } else {
            // Process the average
            for (int iStat = 0; iStat < stationsNb; iStat++) {
                totalScores += vectScores[iStat];
            }
            totalScores /= stationsNb;
        }

        // Put values in final containers
        results.SetScores(totalScores);
    } else {
        if (stationsNb > 1) {
            wxLogError(_("The processing of multivariate complex scores is not implemented yet."));
            return false;
        }

        if (score->UsesClimatology()) {
            score->SetScoreClimatology(scoresClimatology[0]);
        }

        // Containers for final results
        a2f scores(timeTargetSelectionLength, 3 * (params->GetScoreAnalogsNumber() + 1));

        for (int iTargetTime = 0; iTargetTime < timeTargetSelectionLength; iTargetTime++) {
            if (!asIsNaN(targetValues[0](iTargetTime))) {
                if (params->ScoreNeedsPostprocessing()) {
                    // a2f analogsValuesNew(asPostprocessor::Postprocess(analogsValues.row(iTargetTime),
                    // analogsCriteria.row(iTargetTime), params)); finalScores(iTargetTime) =
                    // score->Assess(targetValues(iTargetTime), analogsValuesNew.row(iTargetTime),
                    // params->GetScoreAnalogsNumber());
                } else {
                    scores.row(iTargetTime) = score->AssessOnArray(targetValues[0](iTargetTime),
                                                                   analogsValues[0].row(iTargetTime),
                                                                   params->GetScoreAnalogsNumber());
                }
            } else {
                scores.row(iTargetTime) = a1f::Ones(3 * (params->GetScoreAnalogsNumber() + 1)) * NaNf;
            }
        }

        // Put values in final containers
        results.SetScores2DArray(scores);
    }

    return true;
}

bool asProcessorScore::GetAnalogsTotalScore(asResultsScores& anaScores, asTimeArray& timeArray,
                                            asParametersScoring* params, asResultsTotalScore& results) {
    // TODO: Specify the period in the parameter
    asTotalScore* finalScore = asTotalScore::GetInstance(params->GetScoreName(), "Total");

    // Ranks number set for all, but only used for the rank histogram
    finalScore->SetRanksNb(params->GetScoreAnalogsNumber() + 1);

    if (finalScore->Has2DArrayArgument()) {
        float result = finalScore->Assess(anaScores.GetTargetDates(), anaScores.GetScores2DArray(), timeArray);
        results.SetScore(result);
    } else {
        if (finalScore->SingleValue()) {
            float result = finalScore->Assess(anaScores.GetTargetDates(), anaScores.GetScores(), timeArray);
            results.SetScore(result);
        } else {
            a1f result = finalScore->AssessOnArray(anaScores.GetTargetDates(), anaScores.GetScores(), timeArray);
            results.SetScore(result);
        }
    }

    wxDELETE(finalScore);

    return true;
}

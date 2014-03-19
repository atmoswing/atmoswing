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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#include "asProcessorForecastScore.h"

#include <asParametersCalibration.h>
#include <asPostprocessor.h>
#include <asForecastScore.h>
#include <asForecastScoreCRPSSloadF0.h>
#include <asForecastScoreFinal.h>
#include <asResultsAnalogsValues.h>
#include <asResultsAnalogsForecastScores.h>
#include <asResultsAnalogsForecastScoreFinal.h>
#include <asResultsAnalogsScoresMap.h>
//#include <asDialogProgressBar.h>
#include <asFileAscii.h>
#ifndef UNIT_TESTING
    #include <AtmoswingAppCalibrator.h>
#endif

bool asProcessorForecastScore::GetAnalogsForecastScores(asResultsAnalogsValues &anaValues,
                                           asForecastScore *forecastScore,
                                           asParametersScoring &params,
                                           asResultsAnalogsForecastScores &results)
{
    // Extract Data
    Array1DFloat timeTargetSelection = anaValues.GetTargetDates();
    Array1DFloat targetValues = anaValues.GetTargetValues();
    Array2DFloat analogsCriteria = anaValues.GetAnalogsCriteria();
    Array2DFloat analogsValues = anaValues.GetAnalogsValues();
    int timeTargetSelectionLength = anaValues.GetTargetDatesLength();
    int analogsNbDates = analogsValues.cols();

    // Put values in final containers
    results.SetTargetDates(timeTargetSelection);

    // Check analogs number coherence
    if(params.GetForecastScoreAnalogsNumber()>analogsNbDates) asThrowException(wxString::Format(_("The given analogs number for the forecast score (%d) processing is superior to the analogs dates number (%d)."), params.GetForecastScoreAnalogsNumber(), analogsNbDates));

    if (forecastScore->SingleValue())
    {
        // Containers for final results
        Array1DFloat finalForecastScores(timeTargetSelectionLength);

        for (int i_targtime=0; i_targtime<timeTargetSelectionLength; i_targtime++)
        {
            if (!asTools::IsNaN(targetValues(i_targtime)))
            {
                if (params.ForecastScoreNeedsPostprocessing())
                {
                    //Array2DFloat analogsValuesNew(asPostprocessor::Postprocess(analogsValues.row(i_targtime), analogsCriteria.row(i_targtime), params));
                    //finalForecastScores(i_targtime) = forecastScore->Assess(targetValues(i_targtime), analogsValuesNew.row(i_targtime), params.GetForecastScoreAnalogsNumber());
                }
                else
                {
                    finalForecastScores(i_targtime) = forecastScore->Assess(targetValues(i_targtime), analogsValues.row(i_targtime), params.GetForecastScoreAnalogsNumber());
                }
            }
            else
            {
                finalForecastScores(i_targtime) = NaNFloat;
            }
        }

        // Put values in final containers
        results.SetForecastScores(finalForecastScores);
    }
    else
    {
        // Containers for final results
        Array2DFloat finalForecastScores(timeTargetSelectionLength, 3*(params.GetForecastScoreAnalogsNumber()+1));

        for (int i_targtime=0; i_targtime<timeTargetSelectionLength; i_targtime++)
        {
            if (!asTools::IsNaN(targetValues(i_targtime)))
            {
                if (params.ForecastScoreNeedsPostprocessing())
                {
                    //Array2DFloat analogsValuesNew(asPostprocessor::Postprocess(analogsValues.row(i_targtime), analogsCriteria.row(i_targtime), params));
                    //finalForecastScores(i_targtime) = forecastScore->Assess(targetValues(i_targtime), analogsValuesNew.row(i_targtime), params.GetForecastScoreAnalogsNumber());
                }
                else
                {
                    finalForecastScores.row(i_targtime) = forecastScore->AssessOnArray(targetValues(i_targtime), analogsValues.row(i_targtime), params.GetForecastScoreAnalogsNumber());
                }
            }
            else
            {
                finalForecastScores.row(i_targtime) = Array1DFloat::Ones(3*(params.GetForecastScoreAnalogsNumber()+1))*NaNFloat;
            }
        }

        // Put values in final containers
        results.SetForecastScores2DArray(finalForecastScores);
    }

    return true;
}

bool asProcessorForecastScore::GetAnalogsForecastScoresLoadF0(asResultsAnalogsValues &anaValues,
                                           asForecastScore *forecastScore,
                                           asParametersScoring &params,
                                           asResultsAnalogsForecastScores &results)
{
    asLogWarning(_("This forecast score is for research purposes only !"));

    // Extract Data
    Array1DFloat timeTargetSelection = anaValues.GetTargetDates();
    Array1DFloat targetValues = anaValues.GetTargetValues();
    Array2DFloat analogsCriteria = anaValues.GetAnalogsCriteria();
    Array2DFloat analogsValues = anaValues.GetAnalogsValues();
    int timeTargetSelectionLength = anaValues.GetTargetDatesLength();
    int analogsNbDates = analogsValues.cols();

    // Check analogs number coherence
    if(params.GetForecastScoreAnalogsNumber()>analogsNbDates) asThrowException(wxString::Format(_("The given analogs number for the forecast score (%d) processing is superior to the analogs dates number (%d)."), params.GetForecastScoreAnalogsNumber(), analogsNbDates));

    // Load data from file
    wxString filePath = asConfig::GetTempDir() + "F0.txt";
    asLogWarning(wxString::Format(_("The F(0) will be read from the following file: %s"), filePath.c_str()));

    if(!wxFileName::FileExists(filePath))
    {
        asLogError(wxString::Format(_("The file %s could not be found."), filePath.c_str()));
        return false;
    }

    asFileAscii file(filePath, asFile::ReadOnly);
    VectorFloat freq;
    if(!file.Open())
    {
        asLogError(wxString::Format(_("The file %s could not be opened."), filePath.c_str()));
        return false;
    }
    while (!file.EndOfFile())
    {
        wxString line = file.GetLineContent();
        if(!line.IsEmpty())
        {
            double val;
            line.ToDouble(&val);

            if (!asTools::IsNaN(val))
            {
                freq.push_back((float)val);
            }
        }

    }
    file.Close();

    if ((unsigned)timeTargetSelection.size()!=freq.size())
    {
        asLogError(wxString::Format(_("Time series length don't match: timeTargetSelection.size()=%d, file size=%d"), (int)timeTargetSelection.size(), (int)freq.size()));
        asLogError(wxString::Format(_("timeTargetSelection: %s-%s"), asTime::GetStringTime(timeTargetSelection[0]).c_str(), asTime::GetStringTime(timeTargetSelection[timeTargetSelection.size()-1]).c_str()));
        return false;
    }
    asLogMessage(_("File correctly read"));

    // Replace the forecast score
    asForecastScoreCRPSSloadF0* forecastScoreF0 = new asForecastScoreCRPSSloadF0();
    forecastScoreF0->SetScoreClimatology(forecastScore->GetScoreClimatology());

    // Containers for final results
    Array1DFloat finalForecastScores(timeTargetSelectionLength);

    for (int i_targtime=0; i_targtime<timeTargetSelectionLength; i_targtime++)
    {
        if (!asTools::IsNaN(targetValues(i_targtime)))
        {
            if (params.ForecastScoreNeedsPostprocessing())
            {
                //Array2DFloat analogsValuesNew(asPostprocessor::Postprocess(analogsValues.row(i_targtime), analogsCriteria.row(i_targtime), params));
                //finalForecastScores(i_targtime) = forecastScoreF0->Assess(targetValues(i_targtime), freq[i_targtime], analogsValuesNew.row(i_targtime), params.GetForecastScoreAnalogsNumber());
            }
            else
            {
                finalForecastScores(i_targtime) = forecastScoreF0->Assess(targetValues(i_targtime), freq[i_targtime], analogsValues.row(i_targtime), params.GetForecastScoreAnalogsNumber());
            }
        }
        else
        {
            finalForecastScores(i_targtime) = NaNFloat;
        }
    }

    // Put values in final containers
    results.SetTargetDates(timeTargetSelection);
    results.SetForecastScores(finalForecastScores);

    wxDELETE(forecastScoreF0);

    return true;
}

bool asProcessorForecastScore::GetAnalogsForecastScoreFinal(asResultsAnalogsForecastScores &anaScores,
                                               asTimeArray &timeArray,
                                               asParametersScoring &params,
                                               asResultsAnalogsForecastScoreFinal &results)
{
// TODO (phorton#1#): Specify the period in the parameter
    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance(params.GetForecastScoreName(), "Total");

    // Ranks number set for all, but only used for the rank histogram
    finalScore->SetRanksNb(params.GetForecastScoreAnalogsNumber()+1);

    if (finalScore->Has2DArrayArgument())
    {
        float result = finalScore->Assess(anaScores.GetTargetDates(), anaScores.GetForecastScores2DArray(), timeArray);
        results.SetForecastScore(result);
    }
    else
    {
        if (finalScore->SingleValue())
        {
            float result = finalScore->Assess(anaScores.GetTargetDates(), anaScores.GetForecastScores(), timeArray);
            results.SetForecastScore(result);
        }
        else
        {
            Array1DFloat result = finalScore->AssessOnArray(anaScores.GetTargetDates(), anaScores.GetForecastScores(), timeArray);
            results.SetForecastScore(result);
        }
    }

    wxDELETE(finalScore);

    return true;
}


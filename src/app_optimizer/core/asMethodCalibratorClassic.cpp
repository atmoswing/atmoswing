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

#include "asMethodCalibratorClassic.h"

asMethodCalibratorClassic::asMethodCalibratorClassic()
        : asMethodCalibrator(),
          m_plus(false),
          m_stepsLatPertinenceMap(1),
          m_stepsLonPertinenceMap(1),
          m_resizingIterations(1),
          m_proceedSequentially(true)
{

}

asMethodCalibratorClassic::~asMethodCalibratorClassic() = default;

bool asMethodCalibratorClassic::Calibrate(asParametersCalibration &params)
{
    // Get preferences
    GetPlusOptions();

    // Extract the stations IDs
    vvi stationsId = params.GetPredictandStationIdsVector();

    // Preload data
    if (!DoPreloadData(params))
        return false;

    // Preloading data is necessary
    for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
            if (!params.NeedsPreloading(iStep, iPtor)) {
                wxLogError(_("You need to preload the data for the classic calibration."));
                return false;
            }
        }
    }

    // Copy of the original parameters set.
    m_originalParams = params;

    // Create result object to save the final parameters sets
    asResultsParametersArray resultsAll;
    resultsAll.Init(_("all_station_best_parameters"));

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsDates anaDatesPrevious;

    for (auto stationId : stationsId) {
        wxLogVerbose(_("Calibrating station %s."), GetPredictandStationIdsList(stationId));

        // Reset the score of the climatology
        m_scoreClimatology.clear();

        // Clear previous results
        ClearAll();

        // Create result objects to save the parameters sets
        asResultsParametersArray resultsTested;
        resultsTested.Init(wxString::Format(_("station_%s_tested_parameters"), GetPredictandStationIdsList(stationId)));
        asResultsParametersArray resultsBest;
        resultsBest.Init(wxString::Format(_("station_%s_best_parameters"), GetPredictandStationIdsList(stationId)));
        wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/Paths/ResultsDir",
                                                                asConfig::GetDefaultUserWorkingDir());
        wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
        resultsXmlFilePath.Append(wxString::Format("/%s_station_%s_best_parameters.xml", time,
                                                   GetPredictandStationIdsList(stationId)));

        // Create a complete relevance map
        wxLogVerbose(_("Creating the complete relevance map for a given predictor."));

        // Get a copy of the original parameters
        params = m_originalParams;

        // Set the next station ID
        params.SetPredictandStationIds(stationId);

        // Process every step one after the other
        for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
            // Restore previous best parameters
            if (iStep > 0) {
                params = m_parameters[0];
            }

            // Clear previous results
            ClearAll();

            // Set the same weight to every predictors
            BalanceWeights(params, iStep);

            // Get spatial boundaries
            ParamExploration explo = GetSpatialBoundaries(params, iStep);

            // Set the initial analogs numbers.
            GetInitialAnalogNumber(params, iStep);

            if (explo.xPtsNbIter != 0 || explo.yPtsNbIter != 0) {
                // Set the minimal size
                SetMinimalArea(params, iStep, explo);

                // Build map to explore
                GenerateRelevanceMapParameters(params, iStep, explo);

                // Process the relevance map
                if (!EvaluateRelevanceMap(params, anaDatesPrevious, resultsTested, iStep))
                    return false;

                // Keep the best parameter set
                wxASSERT(!m_parametersTemp.empty());
                RemoveNaNsInTemp();
                if (!PushBackBestTemp())
                    return false;

                wxASSERT(m_parameters.size() == 1);
                ClearTemp();

                wxLogMessage(_("Best point on relevance map: %.2f lat, %.2f lon"),
                             m_parameters[m_parameters.size() - 1].GetPredictorYmin(iStep, 0),
                             m_parameters[m_parameters.size() - 1].GetPredictorXmin(iStep, 0));

                // Resize domain
                if (!AssessDomainResizing(params, anaDatesPrevious, resultsTested, iStep, explo))
                    return false;

                // Resize domain (plus)
                if (m_plus) {
                    if (!AssessDomainResizingPlus(params, anaDatesPrevious, resultsTested, iStep, explo))
                        return false;
                }

                // Consider the best point in previous iteration
                params = m_parameters[0];
            } else {
                // Fixes and checks
                params.FixWeights();
                params.FixCoordinates();
                m_parameters.push_back(params);
            }

            // Keep the analogs dates of the best parameters set
            if (!GetDatesOfBestParameters(params, anaDatesPrevious, iStep))
                return false;
        }

        // Finally calibrate the number of analogs for every step
        wxLogVerbose(_("Find the analogs number for every step."));
        ClearTemp();
        asResultsDates tempDates;
        if (!SubProcessAnalogsNumber(params, tempDates))
            return false;

        // Extract intermediate results from temporary vectors
        for (unsigned int iRes = 0; iRes < m_parametersTemp.size(); iRes++) {
            resultsTested.Add(m_parametersTemp[iRes], m_scoresCalibTemp[iRes]);
        }
        resultsTested.Print();

        // Keep the best parameter set
        wxASSERT(!m_parameters.empty());
        wxASSERT(!m_parametersTemp.empty());
        wxASSERT(!m_scoresCalibTemp.empty());
        KeepBestTemp();
        ClearTemp();

        // Validate
        SaveDetails(&m_parameters[0]);
        Validate(&m_parameters[0]);

        // Keep the best parameters set
        SetBestParameters(resultsBest);
        if (!resultsBest.Print())
            return false;
        resultsAll.Add(m_parameters[0], m_scoresCalib[0], m_scoreValid);
        if (!resultsAll.Print())
            return false;
        if (!m_parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath))
            return false;
    }

    return true;
}

void asMethodCalibratorClassic::GetPlusOptions()
{
    if (m_plus) {
        ThreadsManager().CritSectionConfig().Enter();
        wxConfigBase *pConfig = wxConfigBase::Get();
        pConfig->Read("/Optimizer/ClassicPlus/StepsLatPertinenceMap", &m_stepsLatPertinenceMap, 2);
        if (m_stepsLatPertinenceMap < 1)
            m_stepsLatPertinenceMap = 1;
        pConfig->Read("/Optimizer/ClassicPlus/StepsLonPertinenceMap", &m_stepsLonPertinenceMap, 2);
        if (m_stepsLonPertinenceMap < 1)
            m_stepsLonPertinenceMap = 1;
        pConfig->Read("/Optimizer/ClassicPlus/ResizingIterations", &m_resizingIterations, 1);
        if (m_resizingIterations < 1)
            m_resizingIterations = 1;
        pConfig->Read("/Optimizer/ClassicPlus/ProceedSequentially", &m_proceedSequentially, true);
        ThreadsManager().CritSectionConfig().Leave();
    }
}

bool asMethodCalibratorClassic::DoPreloadData(asParametersCalibration &params)
{
    try {
        wxLogMessage("Preloading data (if required).");
        if (!PreloadArchiveData(&params)) {
            wxLogError(_("Could not preload the data."));
            return false;
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught in the data preloading: %s"), msg);
        DeletePreloadedArchiveData();
        return false;
    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception in the data preloading: %s"), msg);
        DeletePreloadedArchiveData();
        return false;
    }
    wxLogMessage("Data preloading is over.");
    return true;
}

asMethodCalibrator::ParamExploration asMethodCalibratorClassic::GetSpatialBoundaries(const asParametersCalibration &params,
                                                                                     int iStep) const
{
    ParamExploration explo;

    explo.xMinStart = params.GetPredictorXminLowerLimit(iStep, 0);
    explo.xMinEnd = params.GetPredictorXminUpperLimit(iStep, 0);
    explo.xPtsNbIter = params.GetPredictorXptsnbIteration(iStep, 0);
    explo.xPtsNbStart = params.GetPredictorXptsnbLowerLimit(iStep, 0);
    explo.xPtsNbEnd = params.GetPredictorXptsnbUpperLimit(iStep, 0);
    explo.yMinStart = params.GetPredictorYminLowerLimit(iStep, 0);
    explo.yMinEnd = params.GetPredictorYminUpperLimit(iStep, 0);
    explo.yPtsNbIter = params.GetPredictorYptsnbIteration(iStep, 0);
    explo.yPtsNbStart = params.GetPredictorYptsnbLowerLimit(iStep, 0);
    explo.yPtsNbEnd = params.GetPredictorYptsnbUpperLimit(iStep, 0);

    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        explo.xMinStart = wxMax(explo.xMinStart, params.GetPredictorXminLowerLimit(iStep, iPtor));
        explo.xMinEnd = wxMin(explo.xMinEnd, params.GetPredictorXminUpperLimit(iStep, iPtor));
        explo.xPtsNbIter = wxMin(explo.xPtsNbIter, params.GetPredictorXptsnbIteration(iStep, iPtor));
        explo.xPtsNbStart = wxMax(explo.xPtsNbStart, params.GetPredictorXptsnbLowerLimit(iStep, iPtor));
        explo.xPtsNbEnd = wxMin(explo.xPtsNbEnd, params.GetPredictorXptsnbUpperLimit(iStep, iPtor));
        explo.yMinStart = wxMax(explo.yMinStart, params.GetPredictorYminLowerLimit(iStep, iPtor));
        explo.yMinEnd = wxMin(explo.yMinEnd, params.GetPredictorYminUpperLimit(iStep, iPtor));
        explo.yPtsNbIter = wxMin(explo.yPtsNbIter, params.GetPredictorYptsnbIteration(iStep, iPtor));
        explo.yPtsNbStart = wxMax(explo.yPtsNbStart, params.GetPredictorYptsnbLowerLimit(iStep, iPtor));
        explo.yPtsNbEnd = wxMax(explo.yPtsNbEnd, params.GetPredictorYptsnbUpperLimit(iStep, iPtor));
    }

    if ((explo.xMinStart != explo.xMinEnd) && explo.xPtsNbIter == 0)
        explo.xPtsNbIter = 1;
    if ((explo.yMinStart != explo.yMinEnd) && explo.yPtsNbIter == 0)
        explo.yPtsNbIter = 1;

    return explo;
}

void asMethodCalibratorClassic::GetInitialAnalogNumber(asParametersCalibration &params, int iStep) const
{
    int initalAnalogsNb = 0;
    vi initalAnalogsNbVect = params.GetAnalogsNumberVector(iStep);
    if (initalAnalogsNbVect.size() > 1) {
        int indexAnb = floor(initalAnalogsNbVect.size() / 2.0);
        initalAnalogsNb = initalAnalogsNbVect[indexAnb]; // Take the median
    } else {
        initalAnalogsNb = initalAnalogsNbVect[0];
    }

    // For the current step
    params.SetAnalogsNumber(iStep, initalAnalogsNb);
    // And the next ones
    if (m_proceedSequentially) {
        for (int i = iStep; i < params.GetStepsNb(); i++) {
            params.SetAnalogsNumber(i, initalAnalogsNb);
        }
    }
    params.FixAnalogsNb();
}

void asMethodCalibratorClassic::SetMinimalArea(asParametersCalibration &params, int iStep,
                                               const asMethodCalibrator::ParamExploration &explo) const
{
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        if (params.GetPredictorFlatAllowed(iStep, iPtor)) {
            params.SetPredictorXptsnb(iStep, iPtor, 1);
            params.SetPredictorYptsnb(iStep, iPtor, 1);
        } else {
            params.SetPredictorXptsnb(iStep, iPtor, explo.xPtsNbStart);
            params.SetPredictorYptsnb(iStep, iPtor, explo.yPtsNbStart);
        }
    }
}

void asMethodCalibratorClassic::GetSpatialAxes(const asParametersCalibration &params, int iStep,
                                               const asMethodCalibrator::ParamExploration &explo, a1d &xAxis,
                                               a1d &yAxis) const
{
    wxASSERT(m_preloadedArchive[iStep][0][0][0][0]);

    xAxis = m_preloadedArchive[iStep][0][0][0][0]->GetLonAxis();

    // Check longitude range
    if (xAxis[0] >= params.GetPredictorXminUpperLimit(iStep, 0) &&
        xAxis[0] - 360 <= params.GetPredictorXminUpperLimit(iStep, 0) &&
        xAxis[xAxis.size() - 1] - 360 >= params.GetPredictorXminLowerLimit(iStep, 0)) {
        for (int i = 0; i < xAxis.size(); i++) {
            xAxis[i] -= 360;
        }
    }

    yAxis = m_preloadedArchive[iStep][0][0][0][0]->GetLatAxis();
    asSortArray(&yAxis[0], &yAxis[yAxis.size()-1], Asc);
}

void asMethodCalibratorClassic::GenerateRelevanceMapParameters(asParametersCalibration &params, int iStep,
                                                               const asMethodCalibrator::ParamExploration &explo)
{
    ClearTemp();

    a1d xAxis;
    a1d yAxis;
    GetSpatialAxes(params, iStep, explo, xAxis, yAxis);

    for (int iX = 0; iX < xAxis.size() - m_stepsLonPertinenceMap; iX += m_stepsLonPertinenceMap) {
        for (int iY = 0; iY < yAxis.size() - m_stepsLatPertinenceMap; iY += m_stepsLatPertinenceMap) {

            if (xAxis[iX] >= params.GetPredictorXminLowerLimit(iStep, 0) &&
                xAxis[iX] <= params.GetPredictorXminUpperLimit(iStep, 0) &&
                yAxis[iY] >= params.GetPredictorYminLowerLimit(iStep, 0) &&
                yAxis[iY] <= params.GetPredictorYminUpperLimit(iStep, 0)) {

                for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
                    params.SetPredictorXmin(iStep, iPtor, xAxis[iX]);
                    params.SetPredictorYmin(iStep, iPtor, yAxis[iY]);

                    // Fixes and checks
                    params.FixWeights();
                    params.FixCoordinates();
                }

                m_parametersTemp.push_back(params);
            }
        }
    }
}

void asMethodCalibratorClassic::BalanceWeights(asParametersCalibration &params, int iStep) const
{
    int ptorsNb = params.GetPredictorsNb(iStep);
    float weight = (float) 1 / (float) (ptorsNb);
    for (int iPtor = 0; iPtor < ptorsNb; iPtor++) {
        params.SetPredictorWeight(iStep, iPtor, weight);
    }
}

bool asMethodCalibratorClassic::EvaluateRelevanceMap(const asParametersCalibration &params,
                                                     asResultsDates &anaDatesPrevious,
                                                     asResultsParametersArray &resultsTested, int iStep)
{
    asResultsDates anaDates;
    asResultsDates anaDatesPreviousSubRuns;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    wxStopWatch swMap;

    for (auto &param : m_parametersTemp) {
        if (m_proceedSequentially) {
            bool containsNaNs = false;
            if (iStep == 0) {
                if (!GetAnalogsDates(anaDates, &param, iStep, containsNaNs))
                    return false;
            } else {
                if (!GetAnalogsSubDates(anaDates, &param, anaDatesPrevious, iStep, containsNaNs))
                    return false;
            }
            if (containsNaNs) {
                m_scoresCalibTemp.push_back(NaNf);
                continue;
            }
            if (!GetAnalogsValues(anaValues, &param, anaDates, iStep))
                return false;
            if (!GetAnalogsScores(anaScores, &param, anaValues, iStep))
                return false;
            if (!GetAnalogsTotalScore(anaScoreFinal, &param, anaScores, iStep))
                return false;
        } else {
            bool continueLoop = true;
            anaDatesPreviousSubRuns = anaDatesPrevious;
            for (int sub_step = iStep; sub_step < params.GetStepsNb(); sub_step++) {
                wxLogVerbose(_("Process sub-level %d"), sub_step);
                bool containsNaNs = false;
                if (sub_step == 0) {
                    if (!GetAnalogsDates(anaDates, &param, sub_step, containsNaNs))
                        return false;
                } else {
                    if (!GetAnalogsSubDates(anaDates, &param, anaDatesPreviousSubRuns, sub_step,
                                            containsNaNs))
                        return false;
                }
                if (containsNaNs) {
                    continueLoop = false;
                    m_scoresCalibTemp.push_back(NaNf);
                    continue;
                }
                anaDatesPreviousSubRuns = anaDates;
            }
            if (continueLoop) {
                if (!GetAnalogsValues(anaValues, &param, anaDates, params.GetStepsNb() - 1))
                    return false;
                if (!GetAnalogsScores(anaScores, &param, anaValues, params.GetStepsNb() - 1))
                    return false;
                if (!GetAnalogsTotalScore(anaScoreFinal, &param, anaScores, params.GetStepsNb() - 1))
                    return false;
            }
        }

        // Store the result
        m_scoresCalibTemp.push_back(anaScoreFinal.GetScore());
        resultsTested.Add(param, anaScoreFinal.GetScore());
    }

    wxLogMessage(_("Time to process the relevance map: %.3f min."), float(swMap.Time()) / 60000.0f);

    return true;
}

bool asMethodCalibratorClassic::AssessDomainResizing(asParametersCalibration &params, asResultsDates &anaDatesPrevious,
                                                     asResultsParametersArray &resultsTested, int iStep,
                                                     const asMethodCalibrator::ParamExploration &explo)
{
    wxLogVerbose(_("Resize the spatial domain for every predictor."));

    wxStopWatch swEnlarge;

    // Get axes
    a1d xAxis;
    a1d yAxis;
    GetSpatialAxes(params, iStep, explo, xAxis, yAxis);

    bool isover = false;
    while (!isover) {
        isover = true;

        ClearTemp();

        for (int iResizing = 0; iResizing < 4; iResizing++) {
            // Consider the best point in previous iteration
            params = m_parameters[0];

            for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
                switch (iResizing) {
                    case 0: {
                        // Enlarge top
                        WidenNorth(params, explo, iStep, iPtor);
                        break;
                    }
                    case 1: {
                        // Enlarge right
                        WidenEast(params, explo, iStep, iPtor);
                        break;
                    }
                    case 2: {
                        // Enlarge bottom
                        MoveSouth(params, explo, yAxis, iStep, iPtor);
                        WidenNorth(params, explo, iStep, iPtor);
                        break;
                    }
                    case 3: {
                        // Enlarge left
                        MoveWest(params, explo, xAxis, iStep, iPtor);
                        WidenEast(params, explo, iStep, iPtor);
                        break;
                    }
                    default:
                        wxLogError(_("Resizing not correctly defined."));
                }
            }

            // Fixes and checks
            params.FixWeights();
            params.FixCoordinates();

            // Assess parameters
            asResultsDates anaDates;
            asResultsDates anaDatesPreviousSubRuns;
            asResultsValues anaValues;
            asResultsScores anaScores;
            asResultsTotalScore anaScoreFinal;

            if (m_proceedSequentially) {
                bool containsNaNs = false;
                if (iStep == 0) {
                    if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs))
                        return false;
                } else {
                    if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs))
                        return false;
                }
                if (containsNaNs) {
                    isover = false;
                    continue;
                }
                if (!GetAnalogsValues(anaValues, &params, anaDates, iStep))
                    return false;
                if (!GetAnalogsScores(anaScores, &params, anaValues, iStep))
                    return false;
                if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, iStep))
                    return false;
            } else {
                bool continueLoop = true;
                anaDatesPreviousSubRuns = anaDatesPrevious;
                for (int sub_step = iStep; sub_step < params.GetStepsNb(); sub_step++) {
                    wxLogVerbose(_("Process sub-level %d"), sub_step);
                    bool containsNaNs = false;
                    if (sub_step == 0) {
                        if (!GetAnalogsDates(anaDates, &params, sub_step, containsNaNs))
                            return false;
                    } else {
                        if (!GetAnalogsSubDates(anaDates, &params, anaDatesPreviousSubRuns, sub_step, containsNaNs))
                            return false;
                    }
                    if (containsNaNs) {
                        continueLoop = false;
                        isover = false;
                        continue;
                    }
                    anaDatesPreviousSubRuns = anaDates;
                }
                if (continueLoop) {
                    if (!GetAnalogsValues(anaValues, &params, anaDates, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsScores(anaScores, &params, anaValues, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, params.GetStepsNb() - 1))
                        return false;
                }
            }

            resultsTested.Add(params, anaScoreFinal.GetScore());

            // If better, store it and try again to resize.
            if (PushBackInTempIfBetter(params, anaScoreFinal)) {
                isover = false;
            }
        }

        // Apply the resizing that provides the best improvement
        if (!m_parametersTemp.empty()) {
            KeepBestTemp();
        }
    }

    wxLogMessage(_("Time to process the first resizing procedure: %.3f min."),
                 float(swEnlarge.Time()) / 60000.0f);

    return true;
}

bool asMethodCalibratorClassic::AssessDomainResizingPlus(asParametersCalibration &params,
                                                         asResultsDates &anaDatesPrevious,
                                                         asResultsParametersArray &resultsTested, int iStep,
                                                         const asMethodCalibrator::ParamExploration &explo)
{
    wxLogVerbose(_("Reshape again (calibration plus) the spatial domain for every predictor."));

    ClearTemp();

    wxStopWatch swResize;

    // Get axes
    a1d xAxis;
    a1d yAxis;
    GetSpatialAxes(params, iStep, explo, xAxis, yAxis);

    // Try other moves. No while loop but reinitialize the for loops
    for (int multipleFactor = 1; multipleFactor <= m_resizingIterations; multipleFactor++) {
        for (int iResizing = 0; iResizing < 22; iResizing++) {
            // Consider the best point in previous iteration
            params = m_parameters[0];

            for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
                switch (iResizing) {
                    case 0: {
                        // Enlarge all
                        MoveSouth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        MoveWest(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        WidenNorth(params, explo, iStep, iPtor, 2 * multipleFactor);
                        WidenEast(params, explo, iStep, iPtor, 2 * multipleFactor);
                        break;
                    }
                    case 1: {
                        // Reduce all
                        MoveNorth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        MoveEast(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        ReduceNorth(params, explo, iStep, iPtor, 2 * multipleFactor);
                        ReduceEast(params, explo, iStep, iPtor, 2 * multipleFactor);
                        break;
                    }
                    case 2: {
                        // Reduce top
                        ReduceNorth(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 3: {
                        // Reduce right
                        ReduceEast(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 4: {
                        // Reduce bottom
                        MoveNorth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        ReduceNorth(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 5: {
                        // Reduce left
                        MoveEast(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        ReduceEast(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 6: {
                        // Reduce top & bottom
                        MoveNorth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        ReduceNorth(params, explo, iStep, iPtor, 2 * multipleFactor);
                        break;
                    }
                    case 7: {
                        // Reduce right & left
                        MoveEast(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        ReduceEast(params, explo, iStep, iPtor, 2 * multipleFactor);
                        break;
                    }
                    case 8: {
                        // Enlarge top
                        WidenNorth(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 9: {
                        // Enlarge right
                        WidenEast(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 10: {
                        // Enlarge bottom
                        MoveSouth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        WidenNorth(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 11: {
                        // Enlarge left
                        MoveWest(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        WidenEast(params, explo, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 12: {
                        // Enlarge top & bottom
                        MoveSouth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        WidenNorth(params, explo, iStep, iPtor, 2 * multipleFactor);
                        break;
                    }
                    case 13: {
                        // Enlarge right & left
                        MoveWest(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        WidenEast(params, explo, iStep, iPtor, 2 * multipleFactor);
                        break;
                    }
                    case 14: {
                        // Move top
                        MoveNorth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 15: {
                        // Move right
                        MoveEast(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 16: {
                        // Move bottom
                        MoveSouth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 17: {
                        // Move left
                        MoveWest(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 18: {
                        // Move top-left
                        MoveNorth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        MoveWest(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 19: {
                        // Move top-right
                        MoveNorth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        MoveEast(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 20: {
                        // Move bottom-left
                        MoveSouth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        MoveWest(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    case 21: {
                        // Move bottom-right
                        MoveSouth(params, explo, yAxis, iStep, iPtor, multipleFactor);
                        MoveEast(params, explo, xAxis, iStep, iPtor, multipleFactor);
                        break;
                    }
                    default:
                        wxLogError(_("Resizing not correctly defined."));
                }
            }

            // Fixes and checks
            params.FixWeights();
            params.FixCoordinates();

            // Assess parameters

            // Assess parameters
            asResultsDates anaDates;
            asResultsDates anaDatesPreviousSubRuns;
            asResultsValues anaValues;
            asResultsScores anaScores;
            asResultsTotalScore anaScoreFinal;

            if (m_proceedSequentially) {
                bool containsNaNs = false;
                if (iStep == 0) {
                    if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs))
                        return false;
                } else {
                    if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs))
                        return false;
                }
                if (containsNaNs) {
                    continue;
                }
                if (!GetAnalogsValues(anaValues, &params, anaDates, iStep))
                    return false;
                if (!GetAnalogsScores(anaScores, &params, anaValues, iStep))
                    return false;
                if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, iStep))
                    return false;
            } else {
                bool continueLoop = true;
                anaDatesPreviousSubRuns = anaDatesPrevious;
                for (int sub_step = iStep; sub_step < params.GetStepsNb(); sub_step++) {
                    wxLogVerbose(_("Process sub-level %d"), sub_step);
                    bool containsNaNs = false;
                    if (sub_step == 0) {
                        if (!GetAnalogsDates(anaDates, &params, sub_step, containsNaNs))
                            return false;
                    } else {
                        if (!GetAnalogsSubDates(anaDates, &params, anaDatesPreviousSubRuns, sub_step, containsNaNs))
                            return false;
                    }
                    if (containsNaNs) {
                        continueLoop = false;
                        continue;
                    }
                    anaDatesPreviousSubRuns = anaDates;
                }
                if (continueLoop) {
                    if (!GetAnalogsValues(anaValues, &params, anaDates, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsScores(anaScores, &params, anaValues, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, params.GetStepsNb() - 1))
                        return false;
                }
            }

            resultsTested.Add(params, anaScoreFinal.GetScore());

            // If better, keep it and start again
            if (KeepIfBetter(params, anaScoreFinal)) {
                wxLogMessage("Improved spatial window size and position (move %d, factor %d)", iResizing,
                             multipleFactor);
                iResizing = 0;
                multipleFactor = 1;
            }
        }
    }

    wxLogMessage(_("Time to process the second resizing procedure: %.3f min"),
                 float(swResize.Time()) / 60000.0f);

    return true;
}

void asMethodCalibratorClassic::MoveEast(asParametersCalibration &params,
                                         const asMethodCalibrator::ParamExploration &explo, const a1d &xAxis,
                                         int iStep, int iPtor, int multipleFactor) const
{
    double xtmp = params.GetPredictorXmin(iStep, iPtor);
    int ix = asFind(&xAxis[0], &xAxis[xAxis.size() - 1], xtmp);
    ix = wxMin(ix + multipleFactor * explo.xPtsNbIter, (int) xAxis.size() - 1);
    xtmp = wxMax(wxMin(xAxis[ix], explo.xMinEnd), explo.xMinStart);
    params.SetPredictorXmin(iStep, iPtor, xtmp);
}

void asMethodCalibratorClassic::MoveSouth(asParametersCalibration &params,
                                          const asMethodCalibrator::ParamExploration &explo, const a1d &yAxis,
                                          int iStep, int iPtor, int multipleFactor) const
{
    double ytmp = params.GetPredictorYmin(iStep, iPtor);
    int iy = asFind(&yAxis[0], &yAxis[yAxis.size() - 1], ytmp);
    iy = wxMax(iy - multipleFactor * explo.yPtsNbIter, 0);
    ytmp = wxMax(wxMin(yAxis[iy], explo.yMinEnd), explo.yMinStart);
    params.SetPredictorYmin(iStep, iPtor, ytmp);
}

void asMethodCalibratorClassic::MoveWest(asParametersCalibration &params,
                                         const asMethodCalibrator::ParamExploration &explo, const a1d &xAxis,
                                         int iStep, int iPtor, int multipleFactor) const
{
    double xtmp = params.GetPredictorXmin(iStep, iPtor);
    int ix = asFind(&xAxis[0], &xAxis[xAxis.size() - 1], xtmp);
    ix = wxMax(ix - multipleFactor * explo.xPtsNbIter, 0);
    xtmp = wxMax(wxMin(xAxis[ix], explo.xMinEnd), explo.xMinStart);
    params.SetPredictorXmin(iStep, iPtor, xtmp);
}

void asMethodCalibratorClassic::MoveNorth(asParametersCalibration &params,
                                          const asMethodCalibrator::ParamExploration &explo, const a1d &yAxis,
                                          int iStep, int iPtor, int multipleFactor) const
{
    double ytmp = params.GetPredictorYmin(iStep, iPtor);
    int iy = asFind(&yAxis[0], &yAxis[yAxis.size() - 1], ytmp);
    iy = wxMin(iy + multipleFactor * explo.yPtsNbIter, (int) yAxis.size() - 2);
    ytmp = wxMax(wxMin(yAxis[iy], explo.yMinEnd), explo.yMinStart);
    params.SetPredictorYmin(iStep, iPtor, ytmp);
}

void asMethodCalibratorClassic::WidenEast(asParametersCalibration &params,
                                          const asMethodCalibrator::ParamExploration &explo, int iStep, int iPtor,
                                          int multipleFactor) const
{
    int xptsnbtmp = params.GetPredictorXptsnb(iStep, iPtor) + multipleFactor * explo.xPtsNbIter;
    xptsnbtmp = wxMax(wxMin(xptsnbtmp, explo.xPtsNbEnd), explo.xPtsNbStart);
    params.SetPredictorXptsnb(iStep, iPtor, xptsnbtmp);
}

void asMethodCalibratorClassic::WidenNorth(asParametersCalibration &params,
                                           const asMethodCalibrator::ParamExploration &explo, int iStep, int iPtor,
                                           int multipleFactor) const
{
    int yptsnbtmp = params.GetPredictorYptsnb(iStep, iPtor) + multipleFactor * explo.yPtsNbIter;
    yptsnbtmp = wxMax(wxMin(yptsnbtmp, explo.yPtsNbEnd), explo.yPtsNbStart);
    params.SetPredictorYptsnb(iStep, iPtor, yptsnbtmp);
}

void asMethodCalibratorClassic::ReduceEast(asParametersCalibration &params,
                                           const asMethodCalibrator::ParamExploration &explo, int iStep, int iPtor,
                                           int multipleFactor) const
{
    int xptsnbtmp = params.GetPredictorXptsnb(iStep, iPtor) - multipleFactor * explo.xPtsNbIter;
    xptsnbtmp = wxMax(wxMin(xptsnbtmp, explo.xPtsNbEnd), explo.xPtsNbStart);
    params.SetPredictorXptsnb(iStep, iPtor, xptsnbtmp);
}

void asMethodCalibratorClassic::ReduceNorth(asParametersCalibration &params,
                                            const asMethodCalibrator::ParamExploration &explo, int iStep, int iPtor,
                                            int multipleFactor) const
{
    int yptsnbtmp = params.GetPredictorYptsnb(iStep, iPtor) - multipleFactor * explo.yPtsNbIter;
    yptsnbtmp = wxMax(wxMin(yptsnbtmp, explo.yPtsNbEnd), explo.yPtsNbStart);
    params.SetPredictorYptsnb(iStep, iPtor, yptsnbtmp);
}

bool asMethodCalibratorClassic::GetDatesOfBestParameters(asParametersCalibration &params,
                                                         asResultsDates &anaDatesPrevious, int iStep)
{
    bool containsNaNs = false;
    if (iStep == 0) {
        if (!GetAnalogsDates(anaDatesPrevious, &params, iStep, containsNaNs))
            return false;
    } else if (iStep < params.GetStepsNb()) {
        asResultsDates anaDatesPreviousNew;
        if (!GetAnalogsSubDates(anaDatesPreviousNew, &params, anaDatesPrevious, iStep, containsNaNs))
            return false;
        anaDatesPrevious = anaDatesPreviousNew;
    }
    if (containsNaNs) {
        wxLogError(_("The final dates selection contains NaNs"));

        double tmpYmin = m_parameters[m_parameters.size() - 1].GetPredictorYmin(iStep, 0);
        double tmpXmin = m_parameters[m_parameters.size() - 1].GetPredictorXmin(iStep, 0);
        int tmpYptsnb = m_parameters[m_parameters.size() - 1].GetPredictorYptsnb(iStep, 0);
        int tmpXptsnb = m_parameters[m_parameters.size() - 1].GetPredictorXptsnb(iStep, 0);
        wxLogMessage(_("Area: yMin = %.2f, yPtsNb = %d, xMin = %.2f, xPtsNb = %d"), tmpYmin, tmpYptsnb, tmpXmin,
                     tmpXptsnb);

        return false;
    }
    return true;
}

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

asMethodCalibratorClassic::~asMethodCalibratorClassic()
{

}

bool asMethodCalibratorClassic::Calibrate(asParametersCalibration &params)
{
    // Copy of the original parameters set.
    m_originalParams = params;

    // Get preferences
    GetPlusOptions();

    // Extract the stations IDs
    VVectorInt stationsId = params.GetPredictandStationIdsVector();

    // Preload data
    if(!DoPreloadData(params))
        return false;

    // Create result object to save the final parameters sets
    asResultsParametersArray resultsAll;
    resultsAll.Init(_("all_station_best_parameters"));

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsAnalogsDates anaDatesPrevious;

    for (unsigned int i_stat = 0; i_stat < stationsId.size(); i_stat++) {
        VectorInt stationId = stationsId[i_stat];

        asLogMessage(wxString::Format(_("Calibrating station %s."), GetPredictandStationIdsList(stationId)));

        // Reset the score of the climatology
        m_scoreClimatology.clear();

        // Clear previous results
        ClearAll();

        // Create result objects to save the parameters sets
        asResultsParametersArray resultsTested;
        resultsTested.Init(wxString::Format(_("station_%s_tested_parameters"), GetPredictandStationIdsList(stationId)));
        asResultsParametersArray resultsBest;
        resultsBest.Init(wxString::Format(_("station_%s_best_parameters"), GetPredictandStationIdsList(stationId)));
        wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/Paths/OptimizerResultsDir",
                                                                asConfig::GetDefaultUserWorkingDir());
        wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
        resultsXmlFilePath.Append(wxString::Format("/Optimizer/%s_station_%s_best_parameters.xml", time,
                                                   GetPredictandStationIdsList(stationId)));

        // Create a complete relevance map
        asLogMessage(_("Creating the complete relevance map for a given predictor."));

        // Get a copy of the original parameters
        params = m_originalParams;

        // Set the next station ID
        params.SetPredictandStationIds(stationId);

        // Process every step one after the other
        for (int i_step = 0; i_step < params.GetStepsNb(); i_step++) {
            // Restore previous best parameters
            if (i_step > 0) {
                params = m_parameters[0];
            }

            // Clear previous results
            ClearAll();

            // Set the same weight to every predictors
            BalanceWeights(params, i_step);

            // Get spatial boundaries
            ParamExploration explo = GetSpatialBoundaries(params, i_step);

            // Set the initial analogs numbers.
            GetInitialAnalogNumber(params, i_step);

            if (explo.xPtsNbIter != 0 || explo.yPtsNbIter != 0) {
                // Set the minimal size
                SetMinimalArea(params, i_step, explo);

                // Build map to explore
                GenerateRelevanceMapParameters(params, i_step, explo);

                // Process the relevance map
                if(!EvaluateRelevanceMap(params, anaDatesPrevious, resultsTested, i_step))
                    return false;

                // Keep the best parameter set
                wxASSERT(m_parametersTemp.size() > 0);
                RemoveNaNsInTemp();
                PushBackBestTemp();
                wxASSERT(m_parameters.size() == 1);
                ClearTemp();

                asLogMessageImportant(wxString::Format(_("Best point on relevance map: %.2f lat, %.2f lon"),
                                                       m_parameters[m_parameters.size() - 1].GetPredictorYmin(i_step, 0),
                                                       m_parameters[m_parameters.size() - 1].GetPredictorXmin(i_step, 0)));

                // Resize domain
                if(!AssessDomainResizing(params, anaDatesPrevious, resultsTested, i_step, explo))
                    return false;

                // Resize domain (plus)
                if(m_plus) {
                    if(!AssessDomainResizingPlus(params, anaDatesPrevious, resultsTested, i_step, explo))
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
            if(!GetDatesOfBestParameters(params, anaDatesPrevious, i_step))
                return false;
        }

        // Finally calibrate the number of analogs for every step
        asLogMessage(_("Find the analogs number for every step."));
        ClearTemp();
        asResultsAnalogsDates tempDates;
        if (!SubProcessAnalogsNumber(params, tempDates))
            return false;

        // Extract intermediate results from temporary vectors
        for (unsigned int i_res = 0; i_res < m_parametersTemp.size(); i_res++) {
            resultsTested.Add(m_parametersTemp[i_res], m_scoresCalibTemp[i_res]);
        }
        resultsTested.Print();

        // Keep the best parameter set
        wxASSERT(m_parameters.size() > 0);
        wxASSERT(m_parametersTemp.size() > 0);
        wxASSERT(m_scoresCalibTemp.size() > 0);
        KeepBestTemp();
        ClearTemp();

        // Validate
        Validate();

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
        if (!PreloadData(params)) {
            asLogError(_("Could not preload the data."));
            return false;
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Bad allocation caught in the data preloading: %s"), msg));
        DeletePreloadedData();
        return false;
    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Exception in the data preloading: %s"), msg));
        DeletePreloadedData();
        return false;
    }
    return true;
}

asMethodCalibrator::ParamExploration asMethodCalibratorClassic::GetSpatialBoundaries(const asParametersCalibration &params,
                                                                                     int i_step) const
{
    ParamExploration explo;

    explo.xMinStart = params.GetPredictorXminLowerLimit(i_step, 0);
    explo.xMinEnd = params.GetPredictorXminUpperLimit(i_step, 0);
    explo.xPtsNbIter = params.GetPredictorXptsnbIteration(i_step, 0);
    explo.xPtsnbStart = params.GetPredictorXptsnbLowerLimit(i_step, 0);
    explo.xPtsnbEnd = params.GetPredictorXptsnbUpperLimit(i_step, 0);
    explo.yMinStart = params.GetPredictorYminLowerLimit(i_step, 0);
    explo.yMinEnd = params.GetPredictorYminUpperLimit(i_step, 0);
    explo.yPtsNbIter = params.GetPredictorYptsnbIteration(i_step, 0);
    explo.yPtsnbStart = params.GetPredictorYptsnbLowerLimit(i_step, 0);
    explo.yPtsnbEnd = params.GetPredictorYptsnbUpperLimit(i_step, 0);

    for (int i_ptor = 0; i_ptor < params.GetPredictorsNb(i_step); i_ptor++) {
        explo.xMinStart = wxMax(explo.xMinStart, params.GetPredictorXminLowerLimit(i_step, i_ptor));
        explo.xMinEnd = wxMin(explo.xMinEnd, params.GetPredictorXminUpperLimit(i_step, i_ptor));
        explo.xPtsNbIter = wxMin(explo.xPtsNbIter, params.GetPredictorXptsnbIteration(i_step, i_ptor));
        explo.xPtsnbStart = wxMax(explo.xPtsnbStart, params.GetPredictorXptsnbLowerLimit(i_step, i_ptor));
        explo.xPtsnbEnd = wxMin(explo.xPtsnbEnd, params.GetPredictorXptsnbUpperLimit(i_step, i_ptor));
        explo.yMinStart = wxMax(explo.yMinStart, params.GetPredictorYminLowerLimit(i_step, i_ptor));
        explo.yMinEnd = wxMin(explo.yMinEnd, params.GetPredictorYminUpperLimit(i_step, i_ptor));
        explo.yPtsNbIter = wxMin(explo.yPtsNbIter, params.GetPredictorYptsnbIteration(i_step, i_ptor));
        explo.yPtsnbStart = wxMax(explo.yPtsnbStart, params.GetPredictorYptsnbLowerLimit(i_step, i_ptor));
        explo.yPtsnbEnd = wxMax(explo.yPtsnbEnd, params.GetPredictorYptsnbUpperLimit(i_step, i_ptor));
    }

    if ((explo.xMinStart != explo.xMinEnd) && explo.xPtsNbIter==0)
        explo.xPtsNbIter = 1;
    if ((explo.yMinStart != explo.yMinEnd) && explo.yPtsNbIter==0)
        explo.yPtsNbIter = 1;

    return explo;
}

void asMethodCalibratorClassic::GetInitialAnalogNumber(asParametersCalibration &params, int i_step) const
{
    int initalAnalogsNb = 0;
    VectorInt initalAnalogsNbVect = params.GetAnalogsNumberVector(i_step);
    if (initalAnalogsNbVect.size() > 1) {
        int indexAnb = floor(initalAnalogsNbVect.size() / 2.0);
        initalAnalogsNb = initalAnalogsNbVect[indexAnb]; // Take the median
    } else {
        initalAnalogsNb = initalAnalogsNbVect[0];
    }

    // For the current step
    params.SetAnalogsNumber(i_step, initalAnalogsNb);
    // And the next ones
    if (m_proceedSequentially) {
        for (int i = i_step; i < params.GetPredictorsNb(i_step); i++) {
            params.SetAnalogsNumber(i, initalAnalogsNb);
        }
    }
    params.FixAnalogsNb();
}

void asMethodCalibratorClassic::SetMinimalArea(asParametersCalibration &params, int i_step,
                                               const asMethodCalibrator::ParamExploration &explo) const
{
    for (int i_ptor = 0; i_ptor < params.GetPredictorsNb(i_step); i_ptor++) {
        if (params.GetPredictorFlatAllowed(i_step, i_ptor)) {
            params.SetPredictorXptsnb(i_step, i_ptor, 1);
            params.SetPredictorYptsnb(i_step, i_ptor, 1);
        } else {
            params.SetPredictorXptsnb(i_step, i_ptor, explo.xPtsnbStart);
            params.SetPredictorYptsnb(i_step, i_ptor, explo.yPtsnbStart);
        }
    }
}

void asMethodCalibratorClassic::GetSpatialAxes(const asParametersCalibration &params, int i_step,
                                               const asMethodCalibrator::ParamExploration &explo, Array1DDouble &xAxis,
                                               Array1DDouble &yAxis) const
{
    int areaXptnNb = explo.xPtsnbEnd;
    int areaYptnNb = explo.yPtsnbEnd;

    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, 0),
                                                                          explo.xMinStart, explo.xPtsnbEnd, 0,
                                                                          explo.yMinStart, explo.yPtsnbEnd, 0);

    while (geoArea->GetXmax() < explo.xMinEnd) {
        wxDELETE(geoArea);
        areaXptnNb++;
        geoArea = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, 0), explo.xMinStart,
                                                      areaXptnNb, 0, explo.yMinStart, areaYptnNb, 0);
    }
    while (geoArea->GetYmax() < explo.yMinEnd) {
        wxDELETE(geoArea);
        areaYptnNb++;
        geoArea = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, 0), explo.xMinStart,
                                                      areaXptnNb, 0, explo.yMinStart, areaYptnNb, 0);
    }

    xAxis= geoArea->GetXaxis();
    yAxis= geoArea->GetYaxis();

    wxDELETE(geoArea);
}

void asMethodCalibratorClassic::GenerateRelevanceMapParameters(asParametersCalibration &params, int i_step,
                                                               const asMethodCalibrator::ParamExploration &explo)
{
    ClearTemp();

    Array1DDouble xAxis;
    Array1DDouble yAxis;
    GetSpatialAxes(params, i_step, explo, xAxis, yAxis);

    for (int i_x = 0; i_x < xAxis.size(); i_x += m_stepsLonPertinenceMap) {
        for (int i_y = 0; i_y < yAxis.size(); i_y += m_stepsLatPertinenceMap) {
            for (int i_ptor = 0; i_ptor < params.GetPredictorsNb(i_step); i_ptor++) {
                params.SetPredictorXmin(i_step, i_ptor, xAxis[i_x]);
                params.SetPredictorYmin(i_step, i_ptor, yAxis[i_y]);

                // Fixes and checks
                params.FixWeights();
                params.FixCoordinates();
            }

            m_parametersTemp.push_back(params);
        }
    }
}

void asMethodCalibratorClassic::BalanceWeights(asParametersCalibration &params, int i_step) const
{
    int ptorsNb = params.GetPredictorsNb(i_step);
    float weight = (float) 1 / (float) (ptorsNb);
    for (int i_ptor = 0; i_ptor < ptorsNb; i_ptor++) {
        params.SetPredictorWeight(i_step, i_ptor, weight);
    }
}

bool asMethodCalibratorClassic::EvaluateRelevanceMap(const asParametersCalibration &params,
                                                     asResultsAnalogsDates &anaDatesPrevious,
                                                     asResultsParametersArray &resultsTested, int i_step)
{
    asResultsAnalogsDates anaDates;
    asResultsAnalogsDates anaDatesPreviousSubRuns;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    wxStopWatch swMap;

    for (unsigned int i_param = 0; i_param < m_parametersTemp.size(); i_param++) {
        if (m_proceedSequentially) {
            bool containsNaNs = false;
            if (i_step == 0) {
                if (!GetAnalogsDates(anaDates, m_parametersTemp[i_param], i_step, containsNaNs))
                    return false;
            } else {
                if (!GetAnalogsSubDates(anaDates, m_parametersTemp[i_param], anaDatesPrevious, i_step, containsNaNs))
                    return false;
            }
            if (containsNaNs) {
                m_scoresCalibTemp.push_back(NaNFloat);
                continue;
            }
            if (!GetAnalogsValues(anaValues, m_parametersTemp[i_param], anaDates, i_step))
                return false;
            if (!GetAnalogsForecastScores(anaScores, m_parametersTemp[i_param], anaValues, i_step))
                return false;
            if (!GetAnalogsForecastScoreFinal(anaScoreFinal, m_parametersTemp[i_param], anaScores, i_step))
                return false;
        } else {
            bool continueLoop = true;
            anaDatesPreviousSubRuns = anaDatesPrevious;
            for (int sub_step = i_step; sub_step < params.GetStepsNb(); sub_step++) {
                asLogMessage(wxString::Format(_("Process sub-level %d"), sub_step));
                bool containsNaNs = false;
                if (sub_step == 0) {
                    if (!GetAnalogsDates(anaDates, m_parametersTemp[i_param], sub_step, containsNaNs))
                        return false;
                } else {
                    if (!GetAnalogsSubDates(anaDates, m_parametersTemp[i_param], anaDatesPreviousSubRuns, sub_step,
                                            containsNaNs))
                        return false;
                }
                if (containsNaNs) {
                    continueLoop = false;
                    m_scoresCalibTemp.push_back(NaNFloat);
                    continue;
                }
                anaDatesPreviousSubRuns = anaDates;
            }
            if (continueLoop) {
                if (!GetAnalogsValues(anaValues, m_parametersTemp[i_param], anaDates, params.GetStepsNb() - 1))
                    return false;
                if (!GetAnalogsForecastScores(anaScores, m_parametersTemp[i_param], anaValues, params.GetStepsNb() - 1))
                    return false;
                if (!GetAnalogsForecastScoreFinal(anaScoreFinal, m_parametersTemp[i_param], anaScores,
                                                  params.GetStepsNb() - 1))
                    return false;
            }
        }

        // Store the result
        m_scoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
        resultsTested.Add(m_parametersTemp[i_param], anaScoreFinal.GetForecastScore());
    }

    asLogMessageImportant(wxString::Format(_("Time to process the relevance map: %ldms"), swMap.Time()));

    return true;
}

bool asMethodCalibratorClassic::AssessDomainResizing(asParametersCalibration &params,
                                                     asResultsAnalogsDates &anaDatesPrevious,
                                                     asResultsParametersArray &resultsTested, int i_step,
                                                     const asMethodCalibrator::ParamExploration &explo)
{
    asLogMessage(wxString::Format(_("Resize the spatial domain for every predictor.")));

    wxStopWatch swEnlarge;

    // Get axes
    Array1DDouble xAxis;
    Array1DDouble yAxis;
    GetSpatialAxes(params, i_step, explo, xAxis, yAxis);

    bool isover = false;
    while (!isover) {
        isover = true;

        ClearTemp();

        for (int i_resizing = 0; i_resizing < 4; i_resizing++) {
            // Consider the best point in previous iteration
            params = m_parameters[0];

            for (int i_ptor = 0; i_ptor < params.GetPredictorsNb(i_step); i_ptor++) {
                switch (i_resizing) {
                    case 0: {
                        // Enlarge top
                        WidenNorth(params, explo, i_step, i_ptor);
                        break;
                    }
                    case 1: {
                        // Enlarge right
                        WidenEast(params, explo, i_step, i_ptor);
                        break;
                    }
                    case 2: {
                        // Enlarge bottom
                        MoveSouth(params, explo, yAxis, i_step, i_ptor);
                        WidenNorth(params, explo, i_step, i_ptor);
                        break;
                    }
                    case 3: {
                        // Enlarge left
                        MoveWest(params, explo, xAxis, i_step, i_ptor);
                        WidenEast(params, explo, i_step, i_ptor);
                        break;
                    }
                    default:
                        asLogError(_("Resizing not correctly defined."));
                }
            }

            // Fixes and checks
            params.FixWeights();
            params.FixCoordinates();

            // Assess parameters
            asResultsAnalogsDates anaDates;
            asResultsAnalogsDates anaDatesPreviousSubRuns;
            asResultsAnalogsValues anaValues;
            asResultsAnalogsForecastScores anaScores;
            asResultsAnalogsForecastScoreFinal anaScoreFinal;

            if (m_proceedSequentially) {
                bool containsNaNs = false;
                if (i_step == 0) {
                    if (!GetAnalogsDates(anaDates, params, i_step, containsNaNs))
                        return false;
                } else {
                    if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs))
                        return false;
                }
                if (containsNaNs) {
                    isover = false;
                    continue;
                }
                if (!GetAnalogsValues(anaValues, params, anaDates, i_step))
                    return false;
                if (!GetAnalogsForecastScores(anaScores, params, anaValues, i_step))
                    return false;
                if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step))
                    return false;
            } else {
                bool continueLoop = true;
                anaDatesPreviousSubRuns = anaDatesPrevious;
                for (int sub_step = i_step; sub_step < params.GetStepsNb(); sub_step++) {
                    asLogMessage(wxString::Format(_("Process sub-level %d"), sub_step));
                    bool containsNaNs = false;
                    if (sub_step == 0) {
                        if (!GetAnalogsDates(anaDates, params, sub_step, containsNaNs))
                            return false;
                    } else {
                        if (!GetAnalogsSubDates(anaDates, params, anaDatesPreviousSubRuns, sub_step, containsNaNs))
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
                    if (!GetAnalogsValues(anaValues, params, anaDates, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsForecastScores(anaScores, params, anaValues, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, params.GetStepsNb() - 1))
                        return false;
                }
            }

            resultsTested.Add(params, anaScoreFinal.GetForecastScore());

            // If better, store it and try again to resize.
            if (PushBackInTempIfBetter(params, anaScoreFinal)) {
                isover = false;
            }
        }

        // Apply the resizing that provides the best improvement
        if (m_parametersTemp.size() > 0) {
            KeepBestTemp();
        }
    }

    asLogMessageImportant(wxString::Format(_("Time to process the first resizing procedure: %ldms"), swEnlarge.Time()));

    return true;
}

bool asMethodCalibratorClassic::AssessDomainResizingPlus(asParametersCalibration &params,
                                                         asResultsAnalogsDates &anaDatesPrevious,
                                                         asResultsParametersArray &resultsTested, int i_step,
                                                         const asMethodCalibrator::ParamExploration &explo)
{
    asLogMessage(wxString::Format(_("Reshape again (calibration plus) the spatial domain for every predictor.")));

    ClearTemp();

    wxStopWatch swResize;

    // Get axes
    Array1DDouble xAxis;
    Array1DDouble yAxis;
    GetSpatialAxes(params, i_step, explo, xAxis, yAxis);

    // Try other moves. No while loop but reinitialize the for loops
    for (int multipleFactor = 1; multipleFactor <= m_resizingIterations; multipleFactor++) {
        for (int i_resizing = 0; i_resizing < 22; i_resizing++) {
            // Consider the best point in previous iteration
            params = m_parameters[0];

            for (int i_ptor = 0; i_ptor < params.GetPredictorsNb(i_step); i_ptor++) {
                switch (i_resizing) {
                    case 0: {
                        // Enlarge all
                        MoveSouth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        MoveWest(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        WidenNorth(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        WidenEast(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        break;
                    }
                    case 1: {
                        // Reduce all
                        MoveNorth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        MoveEast(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        ReduceNorth(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        ReduceEast(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        break;
                    }
                    case 2: {
                        // Reduce top
                        ReduceNorth(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 3: {
                        // Reduce right
                        ReduceEast(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 4: {
                        // Reduce bottom
                        MoveNorth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        ReduceNorth(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 5: {
                        // Reduce left
                        MoveEast(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        ReduceEast(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 6: {
                        // Reduce top & bottom
                        MoveNorth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        ReduceNorth(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        break;
                    }
                    case 7: {
                        // Reduce right & left
                        MoveEast(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        ReduceEast(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        break;
                    }
                    case 8: {
                        // Enlarge top
                        WidenNorth(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 9: {
                        // Enlarge right
                        WidenEast(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 10: {
                        // Enlarge bottom
                        MoveSouth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        WidenNorth(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 11: {
                        // Enlarge left
                        MoveWest(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        WidenEast(params, explo, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 12: {
                        // Enlarge top & bottom
                        MoveSouth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        WidenNorth(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        break;
                    }
                    case 13: {
                        // Enlarge right & left
                        MoveWest(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        WidenEast(params, explo, i_step, i_ptor, 2 * multipleFactor);
                        break;
                    }
                    case 14: {
                        // Move top
                        MoveNorth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 15: {
                        // Move right
                        MoveEast(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 16: {
                        // Move bottom
                        MoveSouth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 17: {
                        // Move left
                        MoveWest(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 18: {
                        // Move top-left
                        MoveNorth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        MoveWest(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 19: {
                        // Move top-right
                        MoveNorth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        MoveEast(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 20: {
                        // Move bottom-left
                        MoveSouth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        MoveWest(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    case 21: {
                        // Move bottom-right
                        MoveSouth(params, explo, yAxis, i_step, i_ptor, multipleFactor);
                        MoveEast(params, explo, xAxis, i_step, i_ptor, multipleFactor);
                        break;
                    }
                    default:
                        asLogError(_("Resizing not correctly defined."));
                }
            }

            // Fixes and checks
            params.FixWeights();
            params.FixCoordinates();

            // Assess parameters

            // Assess parameters
            asResultsAnalogsDates anaDates;
            asResultsAnalogsDates anaDatesPreviousSubRuns;
            asResultsAnalogsValues anaValues;
            asResultsAnalogsForecastScores anaScores;
            asResultsAnalogsForecastScoreFinal anaScoreFinal;

            if (m_proceedSequentially) {
                bool containsNaNs = false;
                if (i_step == 0) {
                    if (!GetAnalogsDates(anaDates, params, i_step, containsNaNs))
                        return false;
                } else {
                    if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs))
                        return false;
                }
                if (containsNaNs) {
                    continue;
                }
                if (!GetAnalogsValues(anaValues, params, anaDates, i_step))
                    return false;
                if (!GetAnalogsForecastScores(anaScores, params, anaValues, i_step))
                    return false;
                if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step))
                    return false;
            } else {
                bool continueLoop = true;
                anaDatesPreviousSubRuns = anaDatesPrevious;
                for (int sub_step = i_step; sub_step < params.GetStepsNb(); sub_step++) {
                    asLogMessage(wxString::Format(_("Process sub-level %d"), sub_step));
                    bool containsNaNs = false;
                    if (sub_step == 0) {
                        if (!GetAnalogsDates(anaDates, params, sub_step, containsNaNs))
                            return false;
                    } else {
                        if (!GetAnalogsSubDates(anaDates, params, anaDatesPreviousSubRuns, sub_step, containsNaNs))
                            return false;
                    }
                    if (containsNaNs) {
                        continueLoop = false;
                        continue;
                    }
                    anaDatesPreviousSubRuns = anaDates;
                }
                if (continueLoop) {
                    if (!GetAnalogsValues(anaValues, params, anaDates, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsForecastScores(anaScores, params, anaValues, params.GetStepsNb() - 1))
                        return false;
                    if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, params.GetStepsNb() - 1))
                        return false;
                }
            }

            resultsTested.Add(params, anaScoreFinal.GetForecastScore());

            // If better, keep it and start again
            if (KeepIfBetter(params, anaScoreFinal)) {
                asLogMessageImportant(
                        wxString::Format("Improved spatial window size and position (move %d, factor %d)", i_resizing,
                                         multipleFactor));
                i_resizing = 0;
                multipleFactor = 1;
            }
        }
    }

    asLogMessageImportant(wxString::Format(_("Time to process the second resizing procedure: %ldms"), swResize.Time()));

    return true;
}

void asMethodCalibratorClassic::MoveEast(asParametersCalibration &params,
                                         const asMethodCalibrator::ParamExploration &explo, const Array1DDouble &xAxis,
                                         int i_step, int i_ptor, int multipleFactor) const
{
    double xtmp = params.GetPredictorXmin(i_step, i_ptor);
    int ix = asTools::SortedArraySearch(&xAxis[0], &xAxis[xAxis.size() - 1], xtmp);
    ix = wxMin(ix + multipleFactor * explo.xPtsNbIter, (int)xAxis.size() - 1);
    xtmp = wxMax(wxMin(xAxis[ix], explo.xMinEnd), explo.xMinStart);
    params.SetPredictorXmin(i_step, i_ptor, xtmp);
}

void asMethodCalibratorClassic::MoveSouth(asParametersCalibration &params,
                                          const asMethodCalibrator::ParamExploration &explo, const Array1DDouble &yAxis,
                                          int i_step, int i_ptor, int multipleFactor) const
{
    double ytmp = params.GetPredictorYmin(i_step, i_ptor);
    int iy = asTools::SortedArraySearch(&yAxis[0], &yAxis[yAxis.size() - 1], ytmp);
    iy = wxMax(iy - multipleFactor * explo.yPtsNbIter, 0);
    ytmp = wxMax(wxMin(yAxis[iy], explo.yMinEnd), explo.yMinStart);
    params.SetPredictorYmin(i_step, i_ptor, ytmp);
}

void asMethodCalibratorClassic::MoveWest(asParametersCalibration &params,
                                         const asMethodCalibrator::ParamExploration &explo, const Array1DDouble &xAxis,
                                         int i_step, int i_ptor, int multipleFactor) const
{
    double xtmp = params.GetPredictorXmin(i_step, i_ptor);
    int ix = asTools::SortedArraySearch(&xAxis[0], &xAxis[xAxis.size() - 1], xtmp);
    ix = wxMax(ix - multipleFactor * explo.xPtsNbIter, 0);
    xtmp = wxMax(wxMin(xAxis[ix], explo.xMinEnd), explo.xMinStart);
    params.SetPredictorXmin(i_step, i_ptor, xtmp);
}

void asMethodCalibratorClassic::MoveNorth(asParametersCalibration &params,
                                          const asMethodCalibrator::ParamExploration &explo, const Array1DDouble &yAxis,
                                          int i_step, int i_ptor, int multipleFactor) const
{
    double ytmp = params.GetPredictorYmin(i_step, i_ptor);
    int iy = asTools::SortedArraySearch(&yAxis[0], &yAxis[yAxis.size() - 1], ytmp);
    iy = wxMin(iy + multipleFactor * explo.yPtsNbIter, (int)yAxis.size() - 2);
    ytmp = wxMax(wxMin(yAxis[iy], explo.yMinEnd), explo.yMinStart);
    params.SetPredictorYmin(i_step, i_ptor, ytmp);
}

void asMethodCalibratorClassic::WidenEast(asParametersCalibration &params,
                                          const asMethodCalibrator::ParamExploration &explo, int i_step, int i_ptor,
                                          int multipleFactor) const
{
    int xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor) + multipleFactor * explo.xPtsNbIter;
    xptsnbtmp = wxMax(wxMin(xptsnbtmp, explo.xPtsnbEnd), explo.xPtsnbStart);
    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
}

void asMethodCalibratorClassic::WidenNorth(asParametersCalibration &params,
                                           const asMethodCalibrator::ParamExploration &explo, int i_step, int i_ptor,
                                           int multipleFactor) const
{
    int yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor) + multipleFactor * explo.yPtsNbIter;
    yptsnbtmp = wxMax(wxMin(yptsnbtmp, explo.yPtsnbEnd), explo.yPtsnbStart);
    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
}

void asMethodCalibratorClassic::ReduceEast(asParametersCalibration &params,
                                           const asMethodCalibrator::ParamExploration &explo, int i_step, int i_ptor,
                                           int multipleFactor) const
{
    int xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor) - multipleFactor * explo.xPtsNbIter;
    xptsnbtmp = wxMax(wxMin(xptsnbtmp, explo.xPtsnbEnd), explo.xPtsnbStart);
    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
}

void asMethodCalibratorClassic::ReduceNorth(asParametersCalibration &params,
                                            const asMethodCalibrator::ParamExploration &explo, int i_step, int i_ptor,
                                            int multipleFactor) const
{
    int yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor) - multipleFactor * explo.yPtsNbIter;
    yptsnbtmp = wxMax(wxMin(yptsnbtmp, explo.yPtsnbEnd), explo.yPtsnbStart);
    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
}

bool asMethodCalibratorClassic::GetDatesOfBestParameters(asParametersCalibration &params,
                                                         asResultsAnalogsDates &anaDatesPrevious, int i_step)
{
    bool containsNaNs = false;
    if (i_step == 0) {
        if (!GetAnalogsDates(anaDatesPrevious, params, i_step, containsNaNs))
            return false;
    } else if (i_step < params.GetStepsNb()) {
        asResultsAnalogsDates anaDatesPreviousNew;
        if (!GetAnalogsSubDates(anaDatesPreviousNew, params, anaDatesPrevious, i_step, containsNaNs))
            return false;
        anaDatesPrevious = anaDatesPreviousNew;
    }
    if (containsNaNs) {
        asLogError(_("The final dates selection contains NaNs"));

        double tmpYmin = m_parameters[m_parameters.size() - 1].GetPredictorYmin(i_step, 0);
        double tmpXmin = m_parameters[m_parameters.size() - 1].GetPredictorXmin(i_step, 0);
        int tmpYptsnb = m_parameters[m_parameters.size() - 1].GetPredictorYptsnb(i_step, 0);
        int tmpXptsnb = m_parameters[m_parameters.size() - 1].GetPredictorXptsnb(i_step, 0);
        asLogMessageImportant(
                wxString::Format(_("Area: Ymin = %.2f, Yptsnb = %d, Xmin = %.2f, Xptsnb = %d"), tmpYmin, tmpYptsnb,
                                 tmpXmin, tmpXptsnb));

        return false;
    }
    return true;
}

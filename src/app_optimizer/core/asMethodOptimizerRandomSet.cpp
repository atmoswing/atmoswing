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

#include "asMethodOptimizerRandomSet.h"

#include <asThreadRandomSet.h>

#ifndef UNIT_TESTING

#include <AtmoswingAppOptimizer.h>

#endif

asMethodOptimizerRandomSet::asMethodOptimizerRandomSet()
        : asMethodOptimizer()
{

}

asMethodOptimizerRandomSet::~asMethodOptimizerRandomSet()
{
    //dtor
}

bool asMethodOptimizerRandomSet::Manager()
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool parallelEvaluations;
    pConfig->Read("/Processing/ParallelEvaluations", &parallelEvaluations, false);
    ThreadsManager().CritSectionConfig().Leave();

    // Seeds the random generator
    asInitRandom();

    // Load parameters
    asParametersOptimization params;
    if (!params.LoadFromFile(m_paramsFilePath))
        return false;
    if (!m_predictandStationIds.empty()) {
        params.SetPredictandStationIds(m_predictandStationIds);
    }

    // Reset the score of the climatology
    m_scoreClimatology.clear();

    // Create a result object to save the parameters sets
    vi stationId = params.GetPredictandStationIds();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
    asResultsParametersArray results_all;
    results_all.Init(wxString::Format(_("station_%s_tested_parameters"),
                                      GetPredictandStationIdsList(stationId).c_str()));
    asResultsParametersArray results_best;
    results_best.Init(wxString::Format(_("station_%s_best_parameters"),
                                       GetPredictandStationIdsList(stationId).c_str()));
    wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    resultsXmlFilePath.Append(wxString::Format("/Optimizer/%s_station_%s_best_parameters.xml", time.c_str(),
                                               GetPredictandStationIdsList(stationId).c_str()));

    // Preload data
    if (!PreloadArchiveData(&params)) {
        wxLogError(_("Could not preload the data."));
        return false;
    }

    // Store parameter after preloading !
    InitParameters(params);
    m_originalParams = params;

    // Get a score object to extract the score order
    asScore *score = asScore::GetInstance(params.GetScoreName());
    Order scoreOrder = score->GetOrder();
    wxDELETE(score);
    SetScoreOrder(scoreOrder);

    // Load the Predictand DB
    wxLogVerbose(_("Loading the Predictand DB."));
    if (!LoadPredictandDB(m_predictandDBFilePath))
        return false;
    wxLogVerbose(_("Predictand DB loaded."));

    // Watch
    wxStopWatch sw;

    // Optimizer
    while (!IsOver()) {
        // Get a parameters set
        asParametersOptimization *nextParams = GetNextParameters();

        if (!SkipNext() && !IsOver()) {
            if (parallelEvaluations) {
#ifndef UNIT_TESTING
                if (g_responsive)
                    wxGetApp().Yield();
#endif
                if (m_cancel)
                    return false;

                vf scoreClim = m_scoreClimatology;

                // Push the first parameters set
                asThreadRandomSet *firstThread = new asThreadRandomSet(this, nextParams, &m_scoresCalib[m_iterator],
                                                                       &m_scoreClimatology);
                int threadType = firstThread->GetType();
                ThreadsManager().AddThread(firstThread);

                // Wait until done to get the score of the climatology
                if (scoreClim.empty()) {
                    ThreadsManager().Wait(threadType);

#ifndef UNIT_TESTING
                    if (g_responsive)
                        wxGetApp().Yield();
#endif
                    if (m_cancel)
                        return false;
                }

                // Increment iterator
                IncrementIterator();

                // Get available threads nb
                int threadsNb = ThreadsManager().GetAvailableThreadsNb();
                threadsNb = wxMin(threadsNb, m_paramsNb - m_iterator);

                // Fill up the thread array
                for (int iThread = 0; iThread < threadsNb; iThread++) {
                    // Get a parameters set
                    nextParams = GetNextParameters();

                    // Add it to the threads
                    asThreadRandomSet *thread = new asThreadRandomSet(this, nextParams, &m_scoresCalib[m_iterator],
                                                                      &m_scoreClimatology);
                    ThreadsManager().AddThread(thread);

                    wxASSERT(m_scoresCalib.size() <= (unsigned) m_paramsNb);

                    // Increment iterator
                    IncrementIterator();
                }

                // Continue adding when threads become available
                while (m_iterator < m_paramsNb) {
#ifndef UNIT_TESTING
                    if (g_responsive)
                        wxGetApp().Yield();
#endif
                    if (m_cancel)
                        return false;

                    ThreadsManager().WaitForFreeThread(threadType);

                    // Get a parameters set
                    nextParams = GetNextParameters();

                    // Add it to the threads
                    asThreadRandomSet *thread = new asThreadRandomSet(this, nextParams, &m_scoresCalib[m_iterator],
                                                                      &m_scoreClimatology);
                    ThreadsManager().AddThread(thread);

                    wxASSERT(m_scoresCalib.size() <= (unsigned) m_paramsNb);

                    // Increment iterator
                    IncrementIterator();
                }

                // Wait until all done
                ThreadsManager().Wait(threadType);

                // Check results
                bool checkOK = true;
                for (unsigned int iCheck = 0; iCheck < m_scoresCalib.size(); iCheck++) {
                    if (asIsNaN(m_scoresCalib[iCheck])) {
                        wxLogError(_("NaN found in the scores (element %d on %d in m_scoresCalib)."), (int) iCheck + 1,
                                   (int) m_scoresCalib.size());
                        checkOK = false;
                    }
                }

                if (!checkOK)
                    return false;

            } else {
#ifndef UNIT_TESTING
                if (g_responsive)
                    wxGetApp().Yield();
#endif
                if (m_cancel)
                    return false;

                // Create results objects
                asResultsDates anaDates;
                asResultsDates anaDatesPrevious;
                asResultsValues anaValues;
                asResultsScores anaScores;
                asResultsTotalScore anaScoreFinal;

                // Process every step one after the other
                int stepsNb = params.GetStepsNb();
                for (int iStep = 0; iStep < stepsNb; iStep++) {
                    bool containsNaNs = false;
                    if (iStep == 0) {
                        if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs))
                            return false;
                        anaDatesPrevious = anaDates;
                    } else {
                        if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs))
                            return false;
                        anaDatesPrevious = anaDates;
                    }
                    if (containsNaNs) {
                        wxLogError(_("The dates selection contains NaNs"));
                        return false;
                    }
                }
                if (!GetAnalogsValues(anaValues, &params, anaDates, stepsNb - 1))
                    return false;
                if (!GetAnalogsScores(anaScores, &params, anaValues, stepsNb - 1))
                    return false;
                if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, stepsNb - 1))
                    return false;

                // Store the result
                if (((m_optimizerStage == asINITIALIZATION) | (m_optimizerStage == asREASSESSMENT)) &&
                    m_iterator < m_paramsNb) {
                    m_scoresCalib[m_iterator] = anaScoreFinal.GetScore();
                } else {
                    m_scoresCalibTemp.push_back(anaScoreFinal.GetScore());
                }
                wxASSERT(m_scoresCalib.size() <= (unsigned) m_paramsNb);

                // Save all tested parameters in a text file
                results_all.Add(params, anaScoreFinal.GetScore());

                // Increment iterator
                IncrementIterator();
            }
        }
    }

    // Display processing time
    wxLogMessage(_("The whole processing took %.3f min to execute"), static_cast<float>(sw.Time()) / 60000.0f);
#if wxUSE_GUI
    wxLogStatus(_("Optimization over."));
#endif

    // Print parameters in a text file
    if (!results_all.Print())
        return false;
    SetBestParameters(results_best);
    if (!results_best.Print())
        return false;

    // Generate xml file with the best parameters set
    if (!m_parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath))
        return false;

    // Delete preloaded data
    DeletePreloadedArchiveData();

    return true;
}

void asMethodOptimizerRandomSet::InitParameters(asParametersOptimization &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/Optimizer/MonteCarlo/RandomNb", &m_paramsNb, 1000);
    ThreadsManager().CritSectionConfig().Leave();

    // Get the number of runs
    params.InitRandomValues();

    // Create the corresponding number of parameters
    m_scoresCalib.resize((unsigned long) m_paramsNb);
    for (int iVar = 0; iVar < m_paramsNb; iVar++) {
        asParametersOptimization paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        m_parameters.push_back(paramsCopy);
    }
}

asParametersOptimization *asMethodOptimizerRandomSet::GetNextParameters()
{
    asParametersOptimization *params = NULL;
    m_skipNext = false;

    if (((m_optimizerStage == asINITIALIZATION) | (m_optimizerStage == asREASSESSMENT)) && m_iterator < m_paramsNb) {
        params = &m_parameters[m_iterator];
    } else {
        if (!Optimize(*params))
            wxLogError(_("The parameters could not be optimized"));
    }

    return params;
}

bool asMethodOptimizerRandomSet::SetBestParameters(asResultsParametersArray &results)
{
    wxASSERT(!m_parameters.empty());
    wxASSERT(!m_scoresCalib.empty());

    // Extract selected parameters & best parameters
    float bestscore = m_scoresCalib[0];
    int bestscorerow = 0;

    for (unsigned int i = 0; i < m_parameters.size(); i++) {
        if (m_scoreOrder == Asc) {
            if (m_scoresCalib[i] < bestscore) {
                bestscore = m_scoresCalib[i];
                bestscorerow = i;
            }
        } else {
            if (m_scoresCalib[i] > bestscore) {
                bestscore = m_scoresCalib[i];
                bestscorerow = i;
            }
        }
    }

    if (bestscorerow != 0) {
        // Re-validate
        SaveDetails(m_parameters[bestscorerow]);
        Validate(m_parameters[bestscorerow]);
    }

    // Sort according to the level and the observation time
    asParametersScoring sortedParams = m_parameters[bestscorerow];
    sortedParams.SortLevelsAndTime();

    results.Add(sortedParams, m_scoresCalib[bestscorerow], m_scoreValid);

    return true;
}

bool asMethodOptimizerRandomSet::Optimize(asParametersOptimization &params)
{
    m_isOver = true;
    wxLogVerbose(_("Random method over."));
    return true;
}

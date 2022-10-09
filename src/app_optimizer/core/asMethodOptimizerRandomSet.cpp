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

#include "asThreadRandomSet.h"

#ifndef UNIT_TESTING

#include "AtmoswingAppOptimizer.h"

#endif

asMethodOptimizerRandomSet::asMethodOptimizerRandomSet()
    : asMethodOptimizer() {}

asMethodOptimizerRandomSet::~asMethodOptimizerRandomSet() {}

bool asMethodOptimizerRandomSet::Manager() {
    // Seeds the random generator
    asInitRandom();

    // Load parameters
    asParametersOptimization params;
    if (!params.LoadFromFile(m_paramsFilePath)) return false;
    if (!m_predictandStationIds.empty()) {
        params.SetPredictandStationIds(m_predictandStationIds);
    }

    // Reset the score of the climatology
    m_scoreClimatology.clear();

    // Create a result object to save the parameters sets
    vi stationId = params.GetPredictandStationIds();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), YYYYMMDD_hhmm);
    asResultsParametersArray results_all;
    results_all.Init(
        wxString::Format(_("station_%s_tested_parameters"), GetPredictandStationIdsList(stationId).c_str()));
    asResultsParametersArray results_best;
    results_best.Init(
        wxString::Format(_("station_%s_best_parameters"), GetPredictandStationIdsList(stationId).c_str()));
    wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    resultsXmlFilePath.Append(wxString::Format("/%s_station_%s_best_parameters.xml", time.c_str(),
                                               GetPredictandStationIdsList(stationId).c_str()));

    // Preload data
    if (!PreloadArchiveData(&params)) {
        wxLogError(_("Could not preload the data."));
        return false;
    }

    // Store parameter after preloading !
    InitParameters(params);

    // Get a score object to extract the score order
    asScore* score = asScore::GetInstance(params.GetScoreName());
    Order scoreOrder = score->GetOrder();
    wxDELETE(score);
    SetScoreOrder(scoreOrder);

    // Load the Predictand DB
    wxLogVerbose(_("Loading the Predictand DB."));
    if (!LoadPredictandDB(m_predictandDBFilePath)) return false;
    wxLogVerbose(_("Predictand DB loaded."));

    // Watch
    wxStopWatch sw;

    int threadType = asThread::MethodOptimizerRandomSet;
    bool firstRun = true;

    // Add threads when they become available
    while (m_iterator < m_paramsNb) {
#ifndef UNIT_TESTING
        if (g_responsive) wxTheApp->Yield();
#endif
        if (m_cancel) {
            return false;
        }

        wxLog::FlushActive();

        ThreadsManager().WaitForFreeThread(threadType);

        // Get a parameters set
        asParametersOptimization* nextParams = GetNextParameters();

        if (nextParams) {
            // Add it to the threads
            auto* thread = new asThreadRandomSet(this, nextParams, &m_scoresCalib[m_iterator], &m_scoreClimatology);
            ThreadsManager().AddThread(thread);

            // Wait until done to get the score of the climatology
            if (firstRun) {
                ThreadsManager().Wait(threadType);
                firstRun = false;

#ifndef UNIT_TESTING
                if (g_responsive) wxTheApp->Yield();
#endif

                if (m_cancel) return false;
            }
        }

        wxASSERT(m_scoresCalib.size() <= m_paramsNb);

        // Increment iterator
        IncrementIterator();
    }

    // Wait until all done
    ThreadsManager().Wait(threadType);

    wxLog::FlushActive();

    // Check results
    for (int iCheck = 0; iCheck < m_scoresCalib.size(); iCheck++) {
        if (asIsNaN(m_scoresCalib[iCheck])) {
            wxLogError(_("NaN found in the scores (element %d on %d in m_scoresCalib)."), (int)iCheck + 1,
                       (int)m_scoresCalib.size());
            return false;
        }
    }

    wxASSERT(m_parameters.size() == m_scoresCalib.size());
    for (int iRes = 0; iRes < m_scoresCalib.size(); ++iRes) {
        results_all.Add(m_parameters[iRes], m_scoresCalib[iRes]);
    }

    wxASSERT(m_iterator == m_paramsNb);

    wxLogVerbose(_("Random method over."));

    // Display processing time
    wxLogMessage(_("The whole processing took %.3f min to execute"), float(sw.Time()) / 60000.0f);
#if USE_GUI
    wxLogStatus(_("Optimization over."));
#endif

    // Print parameters in a text file
    if (!results_all.Print()) return false;
    SetBestParameters(results_best);
    if (!results_best.Print()) return false;

    // Generate xml file with the best parameters set
    if (!m_parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;

    // Delete preloaded data
    DeletePreloadedArchiveData();

    return true;
}

void asMethodOptimizerRandomSet::InitParameters(asParametersOptimization& params) {
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Read("/MonteCarlo/RandomNb", &m_paramsNb, 1000);
    ThreadsManager().CritSectionConfig().Leave();

    // Get the number of runs
    params.InitRandomValues();

    // Create the corresponding number of parameters
    m_scoresCalib.resize((long)m_paramsNb);
    for (int iVar = 0; iVar < m_paramsNb; iVar++) {
        asParametersOptimization paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        m_parameters.push_back(paramsCopy);
    }
}

asParametersOptimization* asMethodOptimizerRandomSet::GetNextParameters() {
    return &m_parameters[m_iterator];
}

bool asMethodOptimizerRandomSet::SetBestParameters(asResultsParametersArray& results) {
    wxASSERT(!m_parameters.empty());
    wxASSERT(!m_scoresCalib.empty());

    // Extract selected parameters & best parameters
    float bestscore = m_scoresCalib[0];
    int bestscorerow = 0;

    for (int i = 0; i < m_parameters.size(); i++) {
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

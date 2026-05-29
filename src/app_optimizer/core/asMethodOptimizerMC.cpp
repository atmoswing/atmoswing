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

#include "asMethodOptimizerMC.h"

#include <wx/fileconf.h>

#include "asIncludes.h"
#include "asThreadMC.h"

#ifndef UNIT_TESTING

#include "AtmoSwingAppOptimizer.h"

#endif

asMethodOptimizerMC::asMethodOptimizerMC()
    : asMethodOptimizer() {}

asMethodOptimizerMC::~asMethodOptimizerMC() {}

bool asMethodOptimizerMC::Manager() {
    // Seeds the random generator
    asInitRandom();

    // Load parameters
    asParametersOptimization params;
    if (!params.LoadFromFile(_paramsFilePath)) return false;
    if (!_predictandStationIds.empty()) {
        params.SetPredictandStationIds(_predictandStationIds);
    }

    // Reset the score of the climatology
    _scoreClimatology.clear();

    // Create a result object to save the parameters sets
    vi stationId = params.GetPredictandStationIds();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), YYYYMMDD_hhmm);
    asResultsParametersArray results_all;
    results_all.Init(asStrF(_("station_%s_tested_parameters"), GetStationIdsList(stationId)));
    asResultsParametersArray results_best;
    results_best.Init(asStrF(_("station_%s_best_parameters"), GetStationIdsList(stationId)));
    wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    resultsXmlFilePath.Append(asStrF("/%s_station_%s_best_parameters.xml", time, GetStationIdsList(stationId)));

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
    if (!LoadPredictandDB(_predictandDBFilePath)) return false;

    // Watch
    wxStopWatch sw;

    int threadType = asThread::MethodOptimizerMC;
    bool firstRun = true;

    // Add threads when they become available
    while (_iterator < _paramsNb) {
#ifndef UNIT_TESTING
        if (g_responsive) wxTheApp->Yield();
#endif
        if (_cancel) {
            return false;
        }

        wxLog::FlushActive();

        ThreadsManager().WaitForFreeThread(threadType);

        // Get a parameters set
        asParametersOptimization* nextParams = GetNextParameters();

        if (nextParams) {
            // Add it to the threads
            auto thread = new asThreadMC(this, nextParams, &_scoresCalib[_iterator], &_scoreClimatology);
            ThreadsManager().AddThread(thread);

            // Wait until done to get the score of the climatology
            if (firstRun) {
                ThreadsManager().Wait(threadType);
                firstRun = false;

#ifndef UNIT_TESTING
                if (g_responsive) wxTheApp->Yield();
#endif

                if (_cancel) return false;
            }
        }

        wxASSERT(_scoresCalib.size() <= _paramsNb);

        // Increment iterator
        IncrementIterator();
    }

    // Wait until all done
    ThreadsManager().Wait(threadType);

    wxLog::FlushActive();

    // Check results
    for (int iCheck = 0; iCheck < _scoresCalib.size(); iCheck++) {
        if (isnan(_scoresCalib[iCheck])) {
            wxLogError(_("NaN found in the scores (element %d on %d in _scoresCalib)."), (int)iCheck + 1,
                       (int)_scoresCalib.size());
            return false;
        }
    }

    wxASSERT(_parameters.size() == _scoresCalib.size());
    for (int iRes = 0; iRes < _scoresCalib.size(); ++iRes) {
        results_all.Add(_parameters[iRes], _scoresCalib[iRes]);
    }

    wxASSERT(_iterator == _paramsNb);

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
    if (!_parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;

    // Delete preloaded data
    DeletePreloadedArchiveData();

    return true;
}

void asMethodOptimizerMC::InitParameters(asParametersOptimization& params) {
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Read("/MonteCarlo/RandomNb", &_paramsNb, 1000);
    ThreadsManager().CritSectionConfig().Leave();

    // Get the number of runs
    params.InitRandomValues();

    // Create the corresponding number of parameters
    _scoresCalib.resize((long)_paramsNb);
    for (int iVar = 0; iVar < _paramsNb; iVar++) {
        asParametersOptimization paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        _parameters.push_back(paramsCopy);
    }
}

asParametersOptimization* asMethodOptimizerMC::GetNextParameters() {
    return &_parameters[_iterator];
}

bool asMethodOptimizerMC::SetBestParameters(asResultsParametersArray& results) {
    wxASSERT(!_parameters.empty());
    wxASSERT(!_scoresCalib.empty());

    // Extract selected parameters & best parameters
    float bestscore = _scoresCalib[0];
    int bestscorerow = 0;

    for (int i = 0; i < _parameters.size(); i++) {
        if (_scoreOrder == Asc) {
            if (_scoresCalib[i] < bestscore) {
                bestscore = _scoresCalib[i];
                bestscorerow = i;
            }
        } else {
            if (_scoresCalib[i] > bestscore) {
                bestscore = _scoresCalib[i];
                bestscorerow = i;
            }
        }
    }

    if (bestscorerow != 0) {
        // Re-validate
        SaveDetails(_parameters[bestscorerow]);
        Validate(_parameters[bestscorerow]);
    }

    // Sort according to the level and the observation time
    asParametersScoring sortedParams = _parameters[bestscorerow];
    sortedParams.SortLevelsAndTime();

    results.Add(sortedParams, _scoresCalib[bestscorerow], _scoreValid);

    return true;
}

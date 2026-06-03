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

#include "asMethodOptimizerGAs.h"

#include <wx/dir.h>
#include <wx/fileconf.h>

#include "asIncludes.h"
#include "asFileText.h"
#include "asThreadGAs.h"
#ifndef UNIT_TESTING
#include "AtmoSwingAppOptimizer.h"
#endif

#ifdef USE_CUDA
#include "asProcessorCuda.cuh"
#endif

asMethodOptimizerGAs::asMethodOptimizerGAs()
    : asMethodOptimizer(),
      _scoreCalibBest(NAN),
      _generationNb(0),
      _assessmentCounter(0),
      _popSize(0),
      _naturalSelectionType(0),
      _couplesSelectionType(0),
      _crossoverType(0),
      _mutationsModeType(0),
      _allowElitismForTheBest(true),
      _reassessBatchBests(true),
      _batchSize(912),
      _batchSizeMax(0),
      _epoch(1),
      _epochMax(10) {
    _warnFailedLoadingData = false;
}

asMethodOptimizerGAs::~asMethodOptimizerGAs() = default;

void asMethodOptimizerGAs::ClearAll() {
    _parameters.clear();
    _parametersBatchBests.clear();
    _scoresCalib.clear();
    _scoreCalibBest = NAN;
    _scoreValid = NAN;
    _bestScores.clear();
    _meanScores.clear();
}

void asMethodOptimizerGAs::SortScoresAndParameters() {
    wxASSERT(_scoresCalib.size() == _parameters.size());
    wxASSERT(_scoresCalib.size() >= 1);
    wxASSERT(_parameters.size() >= 1);

    if (_parameters.size() == 1) return;

    int paramsNb = _scoresCalib.size();

    // Sort according to the score
    a1f vIndices = a1f::LinSpaced(paramsNb, 0, paramsNb - 1);
    asSortArrays(&_scoresCalib[0], &_scoresCalib[paramsNb - 1], &vIndices[0], &vIndices[paramsNb - 1], _scoreOrder);

    // Sort the parameters sets as the scores
    vector<asParametersOptimizationGAs> copyParameters;
    for (int i = 0; i < paramsNb; i++) {
        copyParameters.push_back(_parameters[i]);
    }
    for (int i = 0; i < paramsNb; i++) {
        int index = vIndices(i);
        _parameters[i] = copyParameters[index];
    }
}

bool asMethodOptimizerGAs::Manager() {
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    _popSize = pConfig->ReadLong("/GAs/PopulationSize", 500);
    _paramsNb = _popSize;
    _allowElitismForTheBest = pConfig->ReadBool("/GAs/AllowElitismForTheBest", true);
    _naturalSelectionType = (int)pConfig->ReadLong("/GAs/NaturalSelectionOperator", 0l);
    _couplesSelectionType = (int)pConfig->ReadLong("/GAs/CouplesSelectionOperator", 0l);
    _crossoverType = (int)pConfig->ReadLong("/GAs/CrossoverOperator", 0l);
    _mutationsModeType = (int)pConfig->ReadLong("/GAs/MutationOperator", 0l);
    _useBatches = pConfig->ReadBool("/GAs/UseBatches", false);
    _batchSize = (int)pConfig->ReadLong("/GAs/BatchSize", 912l);
    _epochMax = (int)pConfig->ReadLong("/GAs/NumberOfEpochs", 10l);
    ThreadsManager().CritSectionConfig().Leave();

    // Reset the score of the climatology
    _scoreClimatology.clear();

    try {
        ClearAll();
        if (!ManageOneRun()) {
            DeletePreloadedArchiveData();
            return false;
        }
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught in GAs: %s"), msg);
        DeletePreloadedArchiveData();
        return false;
    } catch (std::runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught in the GAs: %s"), msg);
        DeletePreloadedArchiveData();
        return false;
    }

    // Delete preloaded data
    DeletePreloadedArchiveData();

    return true;
}

int asMethodOptimizerGAs::GetGpusNb() {
    int gpusNb = 0;
#ifdef USE_CUDA
    // Number of GPUs
    int method = (int)wxFileConfig::Get()->Read("/Processing/Method", (long)asMULTITHREADS);
    if (method == asCUDA) {
        gpusNb = (int)wxFileConfig::Get()->ReadLong("/Processing/GpusNb", 1);
        int devicesFound = asProcessorCuda::GetDeviceCount();
        if (gpusNb > devicesFound) {
            wxLogWarning(_("%d GPUs provided, but only %d found."), gpusNb, devicesFound);
            gpusNb = devicesFound;
        }
    }
#endif
    return gpusNb;
}

bool asMethodOptimizerGAs::ManageOneRun() {
    // Reset some data members
    _iterator = 0;
    _assessmentCounter = 0;
    _generationNb = 1;
    _epoch = 1;

    // Seeds the random generator
    asInitRandom();

    // Load parameters
    asParametersOptimizationGAs params;
    if (!params.LoadFromFile(_paramsFilePath)) return false;
    if (!_predictandStationIds.empty()) {
        params.SetPredictandStationIds(_predictandStationIds);
    }

    // Create a result object to save the parameters sets
    vi stationId = params.GetPredictandStationIds();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), YYYYMMDD_hhmm);
    asResultsParametersArray resFinalPopulation;
    resFinalPopulation.Init(asStrF(_("station_%s_final_population"), GetStationIdsList(stationId)));
    asResultsParametersArray resBestIndividual;
    resBestIndividual.Init(asStrF(_("station_%s_best_individual"), GetStationIdsList(stationId)));
    _resGenerations.Init(asStrF(_("station_%s_generations"), GetStationIdsList(stationId)));
    wxString resXmlFilePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    resXmlFilePath.Append(asStrF("/%s_station_%s_best_parameters.xml", time, GetStationIdsList(stationId)));
    wxString operatorsFilePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    operatorsFilePath.Append(asStrF("/%s_station_%s_operators.txt", time, GetStationIdsList(stationId)));

    // Initialize parameters before loading data.
    InitParameters(params);

    // Check if previously finished
    if (HasPreviousRunConverged(params)) {
        wxLogError(_("Optimization has already converged."));
        return true;
    }

    // Preload data
    try {
        if (!PreloadArchiveData(&params)) {
            wxLogError(_("Could not preload the data."));
            return false;
        }
        wxLogMessage(_("Predictor data preloaded."));
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught during data preloading (in GAs): %s"), msg);
        DeletePreloadedArchiveData();
        return false;
    } catch (std::runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught during data preloading (in GAs): %s"), msg);
        DeletePreloadedArchiveData();
        return false;
    }

    // Reload previous results
    if (!ResumePreviousRun(params, operatorsFilePath)) {
        wxLogError(_("Failed to resume previous runs"));
        return false;
    }

    // Create operators file if needed
    if (!wxFileExists(operatorsFilePath)) {
        asFileText fileOperators(operatorsFilePath, asFileText::Replace);
        if (!fileOperators.Open()) {
            wxLogError(_("Could not create the operators file."));
            return false;
        }
        if (!fileOperators.Close()) {
            wxLogError(_("Could not close the operators file."));
            return false;
        }
    }

    // Get a score object to extract the score order
    asScore* score = asScore::GetInstance(params.GetScoreName());
    Order scoreOrder = score->GetOrder();
    wxDELETE(score);
    SetScoreOrder(scoreOrder);

    // Load the Predictand DB
    if (!LoadPredictandDB(_predictandDBFilePath)) return false;

    // Define time range if using batches
    if (_useBatches) {
        // Create the target time array as reference for batches
        asTimeArray timeArrayTarget(GetTimeStartCalibration(&params), GetTimeEndCalibration(&params),
                                    params.GetTargetTimeStepHours(), params.GetTimeArrayTargetMode());
        if (!_validationMode && params.HasValidationPeriod()) {
            timeArrayTarget.SetForbiddenYears(params.GetValidationYearsVector());
        }

        if (!timeArrayTarget.Init()) {
            wxLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }

        _batchSizeMax = timeArrayTarget.GetSize();
        _batchStart = 0;
        _batchEnd = std::min(_batchStart + _batchSize, _batchSizeMax) - 1;
    }

    // Watch
    wxStopWatch sw;

    int threadType = asThread::MethodOptimizerGAs;
    bool firstRun = true;

#ifdef USE_CUDA
    // Number of GPUs
    int method = (int)wxFileConfig::Get()->Read("/Processing/Method", (long)asMULTITHREADS);
    int gpusNb = GetGpusNb();
    int device = 0;
#endif

    // Optimizer
    while (true) {
        // Reassess the best parameter if batch as the period has changed
        if (_useBatches && !firstRun) {
            auto thread = new asThreadGAs(this, &_parameterBest, &_scoreCalibBest, &_scoreClimatology);
#ifdef USE_CUDA
            if (method == asCUDA) {
                device = ThreadsManager().GetFreeDevice(gpusNb);
                thread->SetDevice(device);
            }
#endif
            ThreadsManager().AddThread(thread);
        }

        // Add threads when they become available
        while (_iterator < _paramsNb) {
#ifndef UNIT_TESTING
            if (g_responsive) wxTheApp->Yield();
#endif
            if (_cancel) {
                return false;
            }

            ThreadsManager().WaitForFreeThread(threadType);

#ifdef USE_CUDA
            if (method == asCUDA) {
                device = ThreadsManager().GetFreeDevice(gpusNb);
            }
#endif

            // Get a parameters set
            asParametersOptimizationGAs* nextParams = GetNextParameters();

            if (nextParams) {
                // Add it to the threads
                auto thread = new asThreadGAs(this, nextParams, &_scoresCalib[_iterator], &_scoreClimatology);
#ifdef USE_CUDA
                if (method == asCUDA) {
                    thread->SetDevice(device);
                }
#endif
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
                wxString paramsContent = _parameters[iCheck].Print();
                wxLogError(_("Parameters #%d: %s"), (int)iCheck + 1, paramsContent);
                return false;
            }
        }

        wxLog::FlushActive();

        wxASSERT(_iterator == _paramsNb);

        // Different operators consider that the scores are sorted !
        SortScoresAndParameters();

        // Elitism after mutation must occur after evaluation
        ElitismAfterMutation();

        if (_assessmentCounter > 0) {  // Skip if is resuming

            _resGenerations.Clear();

            // Save the full generation
            for (int i = 0; i < _parameters.size(); i++) {
                _resGenerations.Add(_parameters[i], _scoresCalib[i]);
            }

            // Save operators status
            SaveOperators(operatorsFilePath);

            // Print results
            _resGenerations.Print(_resGenerations.GetCount() - _parameters.size());
        }

        // Display stats
        float meanScore = asMean(&_scoresCalib[0], &_scoresCalib[_scoresCalib.size() - 1]);
        float bestScore = 0;
        switch (_scoreOrder) {
            case (Asc): {
                bestScore = asMinArray(&_scoresCalib[0], &_scoresCalib[_scoresCalib.size() - 1]);
                break;
            }
            case (Desc): {
                bestScore = asMaxArray(&_scoresCalib[0], &_scoresCalib[_scoresCalib.size() - 1]);
                break;
            }
        }
        _bestScores.push_back(bestScore);
        _meanScores.push_back(meanScore);

        wxLogMessage(_("Mean %g, best %g"), meanScore, bestScore);

        // Update best
        if (_useBatches) {
            if (isnan(_scoreCalibBest)) {
                _parameterBest = _parameters[0];
                _scoreCalibBest = _scoresCalib[0];
                if (_reassessBatchBests) {
                    _parametersBatchBests.push_back(_parameterBest);
                }
            } else {
                if (_scoreOrder == Asc && _scoresCalib[0] < _scoreCalibBest) {
                    _scoreCalibBest = _scoresCalib[0];
                    _parameterBest = _parameters[0];
                    if (_reassessBatchBests) {
                        _parametersBatchBests.push_back(_parameterBest);
                    }
                } else if (_scoreOrder == Desc && _scoresCalib[0] > _scoreCalibBest) {
                    _scoreCalibBest = _scoresCalib[0];
                    _parameterBest = _parameters[0];
                    if (_reassessBatchBests) {
                        _parametersBatchBests.push_back(_parameterBest);
                    }
                }
            }
        } else {
            _parameterBest = _parameters[0];
            _scoreCalibBest = _scoresCalib[0];
        }

        // Update batches
        if (_useBatches) {
            _batchStart += _batchSize;
            int minNbDays = 32;
            if (_batchStart + minNbDays >= _batchSizeMax) {
                _batchStart = 0;
                _epoch++;
                wxLogMessage(_("Epoch number %d"), _epoch);
            }
            _batchEnd = std::min(_batchStart + _batchSize, _batchSizeMax) - 1;
        }

        // Check if we should end
        if (HasConverged()) {
            // If finished, reassess all parameters on the full period
            if (_useBatches) {
                // Disable the batch mode.
                _useBatches = false;

                // Clear previous results.
                for (int i = 0; i < _parameters.size(); i++) {
                    _scoresCalib[i] = NAN;
                }

                // Reassess on the whole period.
                if (!ComputeAllScoresOnFullPeriod()) {
                    return false;
                }
                SortScoresAndParameters();

                // The current best parameter might not be in the population !
                if (_scoreOrder == Asc && _scoresCalib[0] < _scoreCalibBest) {
                    _scoreCalibBest = _scoresCalib[0];
                    _parameterBest = _parameters[0];
                } else if (_scoreOrder == Desc && _scoresCalib[0] > _scoreCalibBest) {
                    _scoreCalibBest = _scoresCalib[0];
                    _parameterBest = _parameters[0];
                }
            }
            wxLogVerbose(_("Optimization process over."));
            break;
        } else {
            // Always reset the score values for the batch approach as the sample changes.
            if (_useBatches) {
                for (int i = 0; i < _parameters.size(); i++) {
                    _scoresCalib[i] = NAN;
                }
            }
            if (!Optimize()) {
                wxLogError(_("The parameters could not be optimized"));
                return false;
            }
        }
    }

    // Display processing time
    wxLogMessage(_("The whole processing took %.3f min to execute"), float(sw.Time()) / 60000.0f);
#if USE_GUI
    wxLogStatus(_("Optimization over."));
#endif

    // Clear device
#ifdef USE_CUDA
    cudaDeviceReset();
#endif

    // Validate
    SaveDetails(_parameterBest);
    Validate(_parameterBest);

    // Sort according to level and time
    _parameterBest.SortLevelsAndTime();

    // Print parameters in a text file
    for (int i = 0; i < _parameters.size(); i++) {
        resFinalPopulation.Add(_parameters[i], _scoresCalib[i]);
    }
    if (!resFinalPopulation.Print()) {
        wxLogError(_("The file containing the final population could not be generated."));
        return false;
    }
    resBestIndividual.Add(_parameterBest, _scoreCalibBest, _scoreValid);

    if (!resBestIndividual.Print()) {
        wxLogError(_("The file containing the best individual could not be generated."));
        return false;
    }
    if (!_resGenerations.Print(_resGenerations.GetCount() - _parameters.size())) return false;

    // Generate xml file with the best parameters set
    if (!_parameterBest.GenerateSimpleParametersFile(resXmlFilePath)) {
        wxLogError(_("The output xml parameters file could not be generated."));
    }

    // Print stats
    ThreadsManager().CritSectionConfig().Enter();
    wxString statsFilePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    statsFilePath.Append(asStrF("%s_stats.txt", time));
    asFileText stats(statsFilePath, asFile::New);

    return true;
}

float asMethodOptimizerGAs::ComputeScoreFullPeriod(asParametersOptimizationGAs& param) {
    int batchStart = _batchStart;
    int batchEnd = _batchEnd;

    _batchStart = 0;
    _batchEnd = _batchSizeMax - 1;

    float scoreFullPeriod = NAN;
    auto thread = new asThreadGAs(this, &param, &scoreFullPeriod, &_scoreClimatology);
#ifdef USE_CUDA
    int method = (int)wxFileConfig::Get()->Read("/Processing/Method", (long)asMULTITHREADS);
    if (method == asCUDA) {
        int gpusNb = GetGpusNb();
        int device = ThreadsManager().GetFreeDevice(gpusNb);
        thread->SetDevice(device);
    }
#endif
    ThreadsManager().AddThread(thread);
    ThreadsManager().Wait(asThread::MethodOptimizerGAs);

    _batchStart = batchStart;
    _batchEnd = batchEnd;

    return scoreFullPeriod;
}

bool asMethodOptimizerGAs::ComputeAllScoresOnFullPeriod() {
    // Reassess the best parameter if batch as the period has changed
    auto thread = new asThreadGAs(this, &_parameterBest, &_scoreCalibBest, &_scoreClimatology);
#ifdef USE_CUDA
    int method = (int)wxFileConfig::Get()->Read("/Processing/Method", (long)asMULTITHREADS);
    int device = 0;
    int gpusNb = GetGpusNb();
    if (method == asCUDA) {
        device = ThreadsManager().GetFreeDevice(gpusNb);
        thread->SetDevice(device);
    }
#endif
    ThreadsManager().AddThread(thread);

    // Restore all previously-selected best ones
    if (_reassessBatchBests) {
        for (const auto& param : _parametersBatchBests) {
            _parameters.push_back(param);
        }
        _paramsNb = _parameters.size();
        _scoresCalib.resize(_paramsNb);
    }

    // Add threads when they become available
    _iterator = 0;
    while (_iterator < _paramsNb) {
#ifndef UNIT_TESTING
        if (g_responsive) wxTheApp->Yield();
#endif
        if (_cancel) {
            return false;
        }

        ThreadsManager().WaitForFreeThread(asThread::MethodOptimizerGAs);

#ifdef USE_CUDA
        if (method == asCUDA) {
            device = ThreadsManager().GetFreeDevice(gpusNb);
        }
#endif

        // Add it to the threads
        thread = new asThreadGAs(this, &_parameters[_iterator], &_scoresCalib[_iterator], &_scoreClimatology);
#ifdef USE_CUDA
        if (method == asCUDA) {
            thread->SetDevice(device);
        }
#endif
        ThreadsManager().AddThread(thread);

        wxASSERT(_scoresCalib.size() <= _paramsNb);

        // Increment iterator
        IncrementIterator();
    }

    // Wait until all done
    ThreadsManager().Wait(asThread::MethodOptimizerGAs);

    return true;
}

bool asMethodOptimizerGAs::ResumePreviousRun(asParametersOptimizationGAs& params, const wxString& operatorsFilePath) {
    if (!g_resumePreviousRun) {
        return true;
    }

    wxString resultsDir = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());

    wxDir dir(resultsDir);
    if (!dir.IsOpened()) {
        wxLogVerbose(_("The directory %s could not be opened."), resultsDir);
        return false;
    }

    // Check if the resulting file is already present
    vi stationId = params.GetPredictandStationIds();
    wxString finalFilePattern = asStrF("*_station_%s_best_individual.txt", GetStationIdsList(stationId));
    if (dir.HasFiles(finalFilePattern)) {
        wxLogMessage(_("The directory %s already contains the resulting file."), resultsDir);
        return false;
    }

    // Look for intermediate results to load
    wxString generationsFilePattern = asStrF("*_station_%s_generations.txt", GetStationIdsList(stationId));
    if (!dir.HasFiles(generationsFilePattern)) {
        return true;
    }

    wxArrayString filesGen;
    wxDir::GetAllFiles(resultsDir, &filesGen, generationsFilePattern, wxDIR_FILES);
    filesGen.Sort();
    wxString filePath = filesGen.Last();
    filesGen.Clear();

    // Check that the length is consistent with the population size
    int nLines = asFileText::CountLines(filePath) - 1;
    if (nLines % _popSize != 0) {
        wxLogError(_("The number of former results is not consistent with the population size (%d)."), _popSize);
        return false;
    }

    wxString msg = asStrF(_("Previous intermediate results were found and will be loaded (%d lines)."), nLines);
    wxLogWarning(msg);
    asLog::PrintToConsole(msg);
    asFileText prevResults(filePath, asFile::ReadOnly);
    if (!prevResults.Open()) {
        wxLogError(_("Couldn't open the file %s."), filePath);
        return false;
    }
    if (!prevResults.SkipLines(1)) {
        wxLogError(_("Couldn't read the file %s."), filePath);
        return false;
    }

    // Check that the content match the current parameters
    wxString fileLine = prevResults.GetNextLine();
    wxString firstLineCopy = fileLine;
    wxString currentParamsPrint = params.Print();
    int indexInFile, indexInParams;

    // Compare number of steps
    while (true) {
        indexInFile = firstLineCopy.Find("Step");
        indexInParams = currentParamsPrint.Find("Step");
        if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND) {
            break;
        } else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) ||
                   (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND)) {
            wxLogError(
                _("The number of steps do not correspond between the "
                  "current and the previous parameters."));
            return false;
        }

        firstLineCopy.Replace("Step", wxEmptyString, false);
        currentParamsPrint.Replace("Step", wxEmptyString, false);
    }

    // Compare number of predictors
    while (true) {
        indexInFile = firstLineCopy.Find("Ptor");
        indexInParams = currentParamsPrint.Find("Ptor");
        if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND) {
            break;
        } else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) ||
                   (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND)) {
            wxLogError(
                _("The number of predictors do not correspond between the "
                  "current and the previous parameters."));
            return false;
        }

        firstLineCopy.Replace("Ptor", wxEmptyString, false);
        currentParamsPrint.Replace("Ptor", wxEmptyString, false);
    }

    // Compare number of levels
    while (true) {
        indexInFile = firstLineCopy.Find("Level");
        indexInParams = currentParamsPrint.Find("Level");
        if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND) {
            break;
        } else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) ||
                   (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND)) {
            wxLogError(
                _("The number of atmospheric levels do not correspond between "
                  "the current and the previous parameters."));
            return false;
        }

        firstLineCopy.Replace("Level", wxEmptyString, false);
        currentParamsPrint.Replace("Level", wxEmptyString, false);
    }

    int genNb = nLines / _popSize;
    int iLastGen = (genNb - 1) * _popSize;

    asParametersOptimizationGAs prevParams;

    // Parse the parameters data
    vector<float> vectScores;
    vectScores.reserve(nLines);

    int iLine = 0, iVar = 0;
    do {
        if (fileLine.IsEmpty()) break;

        // Get the score
        int indexScoreCalib = fileLine.Find("Calib");
        int indexScoreValid = fileLine.Find("Valid");
        wxString strScore = fileLine.SubString(indexScoreCalib + 6, indexScoreValid - 2);
        double scoreVal;
        strScore.ToDouble(&scoreVal);
        auto prevScoresCalib = float(scoreVal);
        vectScores.push_back(prevScoresCalib);

        // Get the parameters
        if (iLine >= iLastGen) {
            prevParams = _parameters[0];
            if (!prevParams.GetValuesFromString(fileLine)) {
                return false;
            }
            _resGenerations.AddWithoutProcessingMedian(prevParams, prevScoresCalib);

            // Restore the last generation
            _parameters[iVar] = prevParams;
            _scoresCalib[iVar] = prevScoresCalib;
            iVar++;
        } else if (_useBatches && _reassessBatchBests && iLine % _popSize == 0) {
            // Keep the best ones from previous generations
            prevParams = _parameters[0];
            if (!prevParams.GetValuesFromString(fileLine)) {
                return false;
            }

            _parametersBatchBests.push_back(prevParams);
        }

        // Get next line
        fileLine = prevResults.GetNextLine();
        iLine++;
    } while (!prevResults.EndOfFile());
    if (!prevResults.Close()) {
        wxLogError(_("Failed to close the former results file."));
        return false;
    }

    wxLogMessage(_("%d former results have been reloaded."), _resGenerations.GetCount());
    asLog::PrintToConsole(asStrF(_("%d former results have been reloaded.\n"), _resGenerations.GetCount()));

    if (_useBatches && _reassessBatchBests) {
        wxLogMessage(_("%d generation bests have been reloaded."), int(_parametersBatchBests.size()));
        asLog::PrintToConsole(
            asStrF(_("%d generation bests have been reloaded.\n"), int(_parametersBatchBests.size())));
    }

    // Restore best and mean scores
    _bestScores.resize(genNb);
    _meanScores.resize(genNb);
    for (int iGen = 0; iGen < genNb; iGen++) {
        int iBest = iGen * _popSize;
        _bestScores[iGen] = vectScores[iBest];

        float mean = 0;
        for (int iNext = 0; iNext < _popSize; iNext++) {
            mean += vectScores[iNext];
        }

        _meanScores[iGen] = mean / float(_popSize);
    }

    _iterator = _paramsNb;
    _generationNb = genNb;

    // Update the epoch
    if (_useBatches) {
        wxString parentDirStr(dir.GetName());
        parentDirStr = parentDirStr.Mid(0, parentDirStr.Len() - 7);
        wxDir parentDir(parentDirStr);
        wxString logFilePattern = asStrF("*.log");
        if (!parentDir.HasFiles(logFilePattern)) {
            wxLogError(_("No log file found to restore the number of epochs (directory: %s)"), parentDir.GetName());
            return false;
        }

        wxArrayString logFiles;
        wxDir::GetAllFiles(parentDir.GetName(), &logFiles, logFilePattern, wxDIR_FILES);

        _epoch = 1;
        for (const auto& logFilePath : logFiles) {
            asFileText logContent(logFilePath, asFile::ReadOnly);
            if (!logContent.Open()) {
                wxLogWarning(_("Couldn't open the file %s."), logFilePath);
                continue;
            }
            fileLine = logContent.GetNextLine();

            do {
                if (fileLine.IsEmpty()) break;

                // Get the epoch nb
                int locEpochNb = fileLine.Find("Epoch number");
                if (locEpochNb != wxNOT_FOUND) {
                    wxString epochNbStr = fileLine.Mid(22);
                    long epochNb;
                    epochNbStr.ToLong(&epochNb);

                    // Overwrite to the last value
                    _epoch = std::max(_epoch, int(epochNb));
                }

                // Get next line
                fileLine = logContent.GetNextLine();
            } while (!logContent.EndOfFile());
            if (!logContent.Close()) {
                wxLogError(_("Failed to close the log file."));
                return false;
            }
        }

        wxLogMessage(_("Starting again from epoch %d."), _epoch);
    }

    // Copy file to the new target
    wxCopyFile(filePath, _resGenerations.GetFilePath());

    // Restore operators
    wxString operatorsFilePattern = asStrF("*_station_%s_operators.txt", GetStationIdsList(stationId));
    if (dir.HasFiles(operatorsFilePattern)) {
        return true;
    }

    wxArrayString filesOper;
    wxDir::GetAllFiles(resultsDir, &filesOper, operatorsFilePattern, wxDIR_FILES);
    filesOper.Sort();
    wxString operFilePath = filesOper.Last();
    filesOper.Clear();

    wxLogWarning(_("Previous operators were found and will be loaded."));
    asLog::PrintToConsole(_("Previous operators were found and will be loaded.\n"));

    // Copy file to the new target
    wxCopyFile(operFilePath, operatorsFilePath);

    // Check length
    int nLinesOper = asFileText::CountLines(operFilePath);
    if (nLines != nLinesOper) {
        wxLogError(_("Mismatch between number of parameters (%d) and operators (%d)."), nLines, nLinesOper);
    }

    // Open file
    asFileText prevOperators(operFilePath, asFile::ReadOnly);
    if (!prevOperators.Open()) {
        wxLogError(_("Couldn't open the file %s."), operFilePath);
        return false;
    }

    // Extract file content for the last generation
    if (!prevOperators.SkipLines((genNb - 1) * _popSize)) {
        wxLogError(_("Couldn't read the file %s."), operFilePath);
        return false;
    }
    wxString fileLineOper = prevOperators.GetNextLine();
    iVar = 0;
    do {
        if (fileLineOper.IsEmpty()) break;

        if (iVar >= _parameters.size()) {
            wxLogError(_("Mismatch between number of parameters (%d) and operators (%d)."), (int)_parameters.size(),
                       iVar + 1);
            return false;
        }

        switch (_mutationsModeType) {
            case (RandomUniformConstant):
            case (RandomUniformVariable):
            case (RandomNormalConstant):
            case (RandomNormalVariable):
            case (MultiScale):
            case (NoMutation):
            case (NonUniform): {
                // Nothing to do
                break;
            }

            case (SelfAdaptationRate): {
                int indexMutationRate = fileLineOper.Find("MutationRate");
                wxString strMutationRate = fileLineOper.Mid(indexMutationRate + 13);
                double mutationRate;
                strMutationRate.ToDouble(&mutationRate);
                _parameters[iVar].SetAdaptMutationRate((float)mutationRate);
                break;
            }

            case (SelfAdaptationRadius): {
                int indexMutationRate = fileLineOper.Find("MutationRate");
                int indexMutationRadius = fileLineOper.Find("MutationRadius");
                wxString strMutationRate = fileLineOper.SubString(indexMutationRate + 13, indexMutationRadius - 2);
                double mutationRate;
                strMutationRate.ToDouble(&mutationRate);
                _parameters[iVar].SetAdaptMutationRate((float)mutationRate);
                wxString strMutationRadius = fileLineOper.Mid(indexMutationRadius + 15);
                double mutationRadius;
                strMutationRadius.ToDouble(&mutationRadius);
                _parameters[iVar].SetAdaptMutationRadius((float)mutationRadius);
                break;
            }

            case (SelfAdaptationRateChromosome): {
                int indexMutationRate = fileLineOper.Find("ChromosomeMutationRate");
                wxString strMutationRate = fileLineOper.Mid(indexMutationRate + 23);
                vf mutationRate = asExtractVectorFrom(strMutationRate);
                _parameters[iVar].SetChromosomeMutationRate(mutationRate);
                break;
            }

            case (SelfAdaptationRadiusChromosome): {
                int indexMutationRate = fileLineOper.Find("ChromosomeMutationRate");
                int indexMutationRadius = fileLineOper.Find("ChromosomeMutationRadius");
                wxString strMutationRate = fileLineOper.SubString(indexMutationRate + 23, indexMutationRadius - 2);
                vf mutationRate = asExtractVectorFrom(strMutationRate);
                _parameters[iVar].SetChromosomeMutationRate(mutationRate);
                wxString strMutationRadius = fileLineOper.Mid(indexMutationRadius + 25);
                vf mutationRadius = asExtractVectorFrom(strMutationRadius);
                _parameters[iVar].SetChromosomeMutationRadius(mutationRadius);
                break;
            }

            default: {
                wxLogError(_("The mutation method was not found when saving operators."));
            }
        }

        fileLineOper = prevOperators.GetNextLine();
        iVar++;
    } while (!prevOperators.EndOfFile());
    if (!prevOperators.Close()) {
        wxLogError(_("Failed to close the operators file."));
        return false;
    }

    return true;
}

bool asMethodOptimizerGAs::HasPreviousRunConverged(asParametersOptimizationGAs& params) {
    wxString resultsDir = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    wxDir dir(resultsDir);

    if (!dir.IsOpened()) {
        wxLogVerbose(_("The directory %s could not be opened."), resultsDir);
    } else {
        // Check if the resulting file is already present
        vi stationId = params.GetPredictandStationIds();
        wxString finalFilePattern = asStrF("*_station_%s_best_individual.txt", GetStationIdsList(stationId));
        if (dir.HasFiles(finalFilePattern)) {
            wxLogMessage(_("The directory %s already contains the resulting file."), resultsDir);
            return true;
        }
    }

    return false;
}

bool asMethodOptimizerGAs::SaveOperators(const wxString& filePath) {
    // Open the file
    asFileText fileRes(filePath, asFileText::Append);
    if (!fileRes.Open()) return false;

    wxString content = wxEmptyString;

    // Write every parameter one after the other
    for (auto& parameter : _parameters) {
        switch (_mutationsModeType) {
            case (RandomUniformConstant):
            case (RandomUniformVariable):
            case (RandomNormalConstant):
            case (RandomNormalVariable):
            case (MultiScale):
            case (NoMutation):
            case (NonUniform): {
                // Nothing to do
                content.Append("-");
                break;
            }

            case (SelfAdaptationRate): {
                content.Append("MutationRate\t");
                content << parameter.GetAdaptMutationRate();
                break;
            }

            case (SelfAdaptationRadius): {
                content.Append("MutationRate\t");
                content << parameter.GetAdaptMutationRate();
                content.Append("\t");
                content.Append("MutationRadius\t");
                content << parameter.GetAdaptMutationRadius();
                break;
            }

            case (SelfAdaptationRateChromosome): {
                content.Append("ChromosomeMutationRate\t");
                vf chromRates = parameter.GetChromosomeMutationRate();
                content << asVectorToString(chromRates);
                break;
            }

            case (SelfAdaptationRadiusChromosome): {
                content.Append("ChromosomeMutationRate\t");
                vf chromRates = parameter.GetChromosomeMutationRate();
                content << asVectorToString(chromRates);
                content.Append("\t");
                content.Append("ChromosomeMutationRadius\t");
                vf chromRadius = parameter.GetChromosomeMutationRadius();
                content << asVectorToString(chromRadius);
                break;
            }

            default: {
                wxLogError(_("The mutation method was not found when saving operators."));
            }
        }

        content.Append("\n");
    }

    fileRes.AddContent(content);

    if (!fileRes.Close()) {
        wxLogError(_("Failed to close the operators file."));
        return false;
    }

    return true;
}

void asMethodOptimizerGAs::InitParameters(asParametersOptimizationGAs& params) {
    // Get a first parameters set to get the number of unknown variables
    params.InitRandomValues();
    wxLogVerbose(_("The population is made of %d individuals."), _popSize);

    // Create the corresponding number of parameters
    _scoresCalib.resize(_popSize);
    _parameters.resize(_popSize);
    for (int iVar = 0; iVar < _popSize; iVar++) {
        asParametersOptimizationGAs paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        paramsCopy.BuildChromosomes();

        // Create arrays for the self-adaptation methods
        switch (_mutationsModeType) {
            case (SelfAdaptationRate): {
                paramsCopy.InitIndividualSelfAdaptationMutationRate();
                break;
            }

            case (SelfAdaptationRadius): {
                paramsCopy.InitIndividualSelfAdaptationMutationRate();
                paramsCopy.InitIndividualSelfAdaptationMutationRadius();
                break;
            }

            case (SelfAdaptationRateChromosome): {
                paramsCopy.InitChromosomeSelfAdaptationMutationRate();
                break;
            }

            case (SelfAdaptationRadiusChromosome): {
                paramsCopy.InitChromosomeSelfAdaptationMutationRate();
                paramsCopy.InitChromosomeSelfAdaptationMutationRadius();
                break;
            }

            default: {
                // No self-adaptation required.
            }
        }

        _parameters[iVar] = paramsCopy;
        _scoresCalib[iVar] = NAN;
    }
    _scoreValid = NAN;
}

asParametersOptimizationGAs* asMethodOptimizerGAs::GetNextParameters() {
    wxASSERT(_iterator <= _paramsNb);

    while (_iterator < _paramsNb) {
        // Parameters did not change
        if (!isnan(_scoresCalib[_iterator])) {
            _iterator++;
            continue;
        }

        _assessmentCounter++;

        wxLogVerbose(_("_parameters[%d] = %s"), _iterator, _parameters[_iterator].Print());

        return &_parameters[_iterator];
    }

    return nullptr;
}

bool asMethodOptimizerGAs::Optimize() {
    // Proceed to a new generation
    if (!NaturalSelection()) {
        return false;
    }
    if (!Mating()) {
        return false;
    }
    if (!Mutation()) {
        return false;
    }

    _iterator = 0;
    _generationNb++;

    wxLogMessage(_("Generation number %d"), _generationNb);

    return true;
}

bool asMethodOptimizerGAs::HasConverged() {
    // NB: The parameters and scores are already sorted !

    if (_useBatches) {
        if (_epoch > _epochMax) {
            return true;
        }
        return false;
    }

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    int convergenceStepsNb;
    pConfig->Read("/GAs/ConvergenceStepsNb", &convergenceStepsNb, 20);
    ThreadsManager().CritSectionConfig().Leave();

    // Check if enough generations
    wxASSERT(convergenceStepsNb > 0);
    if (_bestScores.size() < static_cast<size_t>(convergenceStepsNb)) {
        return false;
    }

    // Check the best convergenceStepsNb scores. The guard above ensures
    // _bestScores.size() >= convergenceStepsNb, so the unsigned subtraction below cannot underflow.
    const size_t n = _bestScores.size();
    const size_t k = static_cast<size_t>(convergenceStepsNb);
    for (size_t i = n - 1; i > n - k; i--) {
        float lastScore = _bestScores[n - 1];

        if (lastScore == 0) {
            if (_bestScores[i] != _bestScores[i - 1]) {
                return false;
            }
        } else {
            float relDiff = std::abs((lastScore - _bestScores[i - 1]) / lastScore);
            if (relDiff > 0.001) {
                return false;
            }
        }
    }

    return true;
}

bool asMethodOptimizerGAs::ElitismAfterMutation() {
    // Apply elitism: If the best has been degraded during previous mutations, replace a random individual by the
    // previous best.
    if (_allowElitismForTheBest && !isnan(_scoreCalibBest)) {
        float actualBest = _scoresCalib[0];
        switch (_scoreOrder) {
            case (Asc): {
                if (_scoreCalibBest < actualBest) {
                    wxLogMessage(_("Application of elitism after mutation."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, _scoresCalib.size() - 1, 1);
                    _parameters[randomRow] = _parameterBest;
                    _scoresCalib[randomRow] = _scoreCalibBest;
                    SortScoresAndParameters();
                }
                break;
            }
            case (Desc): {
                if (_scoreCalibBest > actualBest) {
                    wxLogMessage(_("Application of elitism after mutation."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, _scoresCalib.size() - 1, 1);
                    _parameters[randomRow] = _parameterBest;
                    _scoresCalib[randomRow] = _scoreCalibBest;
                    SortScoresAndParameters();
                }
                break;
            }
            default: {
                wxLogError(_("Score order not correctly defined."));
                return false;
            }
        }
    }

    return true;
}

bool asMethodOptimizerGAs::ElitismAfterSelection() {
    // Apply elitism: If the best has not been selected, replace a random individual by the best.
    if (_allowElitismForTheBest && !isnan(_scoreCalibBest)) {
        SortScoresAndParameters();
        float actualBest = _scoresCalib[0];
        switch (_scoreOrder) {
            case (Asc): {
                if (_scoreCalibBest < actualBest) {
                    wxLogMessage(_("Application of elitism in the natural selection."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, _scoresCalib.size() - 1, 1);
                    _parameters[randomRow] = _parameterBest;
                    _scoresCalib[randomRow] = _scoreCalibBest;
                }
                break;
            }
            case (Desc): {
                if (_scoreCalibBest > actualBest) {
                    wxLogMessage(_("Application of elitism in the natural selection."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, _scoresCalib.size() - 1, 1);
                    _parameters[randomRow] = _parameterBest;
                    _scoresCalib[randomRow] = _scoreCalibBest;
                }
                break;
            }
            default: {
                wxLogError(_("Score order not correctly defined."));
                return false;
            }
        }
    }

    return true;
}

bool asMethodOptimizerGAs::NaturalSelection() {
    // NB: The parameters and scores are already sorted !

    wxLogVerbose(_("Applying natural selection."));

    vector<asParametersOptimizationGAs> parameters = _parameters;
    vf scores = _scoresCalib;
    _parameters.clear();
    _scoresCalib.clear();

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    double ratioIntermediateGeneration;
    pConfig->Read("/GAs/RatioIntermediateGeneration", &ratioIntermediateGeneration, 0.5);
    ThreadsManager().CritSectionConfig().Leave();

    // Get intermediate generation size
    int intermediateGenerationSize = ratioIntermediateGeneration * _popSize;

    switch (_naturalSelectionType) {
        case (RatioElitism): {
            wxLogVerbose(_("Natural selection: ratio elitism"));

            for (int i = 0; i < intermediateGenerationSize; i++) {
                _parameters.push_back(parameters[i]);
                _scoresCalib.push_back(scores[i]);
            }
            break;
        }

        case (Tournament): {
            wxLogVerbose(_("Natural selection: tournament"));

            double tournamentSelectionProbability;
            pConfig->Read("/GAs/NaturalSelectionTournamentProbability", &tournamentSelectionProbability, 0.9);

            for (int i = 0; i < intermediateGenerationSize; i++) {
                // Choose candidates
                int candidateFinal = 0;
                int candidate1 = asRandom(0, parameters.size() - 1, 1);
                int candidate2 = asRandom(0, parameters.size() - 1, 1);

                // Check they are not the same
                while (candidate1 == candidate2) {
                    candidate2 = asRandom(0, parameters.size() - 1, 1);
                }

                // Check probability of selection of the best
                bool keepBest = (asRandom(0.0, 1.0) <= tournamentSelectionProbability);

                // Use indexes as scores are already sorted (smaller is better)
                if (keepBest) {
                    if (candidate1 < candidate2) {
                        candidateFinal = candidate1;
                    } else {
                        candidateFinal = candidate2;
                    }
                } else {
                    if (candidate1 < candidate2) {
                        candidateFinal = candidate2;
                    } else {
                        candidateFinal = candidate1;
                    }
                }

                // If both scores are equal, select randomly and overwrite previous selection
                if (_scoresCalib[candidate1] == scores[candidate2]) {
                    double randomIndex = asRandom(0.0, 1.0);
                    if (randomIndex <= 0.5) {
                        candidateFinal = candidate1;
                    } else {
                        candidateFinal = candidate2;
                    }
                }

                _parameters.push_back(parameters[candidateFinal]);
                _scoresCalib.push_back(scores[candidateFinal]);
            }
            break;
        }

        default: {
            wxLogError(_("The given natural selection method couldn't be found."));
            return false;
        }
    }

    ElitismAfterSelection();

    return true;
}

bool asMethodOptimizerGAs::Mating() {
    // Different operators consider that the scores are sorted !
    SortScoresAndParameters();

    wxASSERT(_parameters.size() == _scoresCalib.size());

    wxLogVerbose(_("Applying mating."));

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    ThreadsManager().CritSectionConfig().Leave();

    int sizeParents = int(_parameters.size());
    int counter = 0;
    int counterSame = 0;
    bool initialized = false;
    vd probabilities;

    while (_parameters.size() < _popSize) {
        // Couples selection only in the parents pool
        wxLogVerbose(_("Selecting couples."));
        int partner1 = 0, partner2 = 0;
        switch (_couplesSelectionType) {
            case (RankPairing): {
                wxLogVerbose(_("Couples selection: rank pairing"));

                partner1 = counter * 2;      // pairs
                partner2 = counter * 2 + 1;  // impairs

                // Check that we don't overflow from the array
                if (partner2 >= sizeParents) {
                    partner1 = asRandom(0, sizeParents - 1, 1);
                    partner2 = asRandom(0, sizeParents - 1, 1);
                }
                break;
            }

            case (Random): {
                wxLogVerbose(_("Couples selection: random"));

                partner1 = asRandom(0, sizeParents - 1, 1);
                partner2 = asRandom(0, sizeParents - 1, 1);
                break;
            }

            case (RouletteWheelRank): {
                wxLogVerbose(_("Couples selection: roulette wheel rank"));

                // If the first round, initialize the probabilities.
                if (!initialized) {
                    double sum = 0;
                    probabilities.push_back(0.0);
                    for (int i = 0; i < sizeParents; i++) {
                        sum += i + 1;
                    }
                    for (int i = 0; i < sizeParents; i++) {
                        double currentRank = i + 1.0;
                        double prob = (sizeParents - currentRank + 1.0) / sum;
                        double probCumul = prob + probabilities[probabilities.size() - 1];
                        probabilities.push_back(probCumul);
                    }
                    if (fabs(probabilities[probabilities.size() - 1] - 1.0) > 0.00001) {
                        double diff = probabilities[probabilities.size() - 1] - 1.0;
                        wxLogWarning(_("probabilities[last]-1.0=%f"), diff);
                    }
                    initialized = true;
                }

                // Select mates
                double partner1prob = asRandom(0.0, 1.0);
                partner1 = asFindFloor(&probabilities[0], &probabilities[probabilities.size() - 1], partner1prob);
                double partner2prob = asRandom(0.0, 1.0);
                partner2 = asFindFloor(&probabilities[0], &probabilities[probabilities.size() - 1], partner2prob);

                break;
            }

            case (RouletteWheelScore): {
                wxLogVerbose(_("Couples selection: roulette wheel score"));

                // If the first round, initialize the probabilities.
                if (!initialized) {
                    double sum = 0;
                    probabilities.push_back(0.0);
                    for (int i = 0; i < sizeParents; i++) {
                        sum += _scoresCalib[i] - _scoresCalib[sizeParents - 1] + 0.001;  // 0.001 to avoid null probs
                    }
                    for (int i = 0; i < sizeParents; i++) {
                        if (sum > 0) {
                            double currentScore = _scoresCalib[i] - _scoresCalib[sizeParents - 1] + 0.001;
                            double prob = currentScore / sum;
                            double probCumul = prob + probabilities[probabilities.size() - 1];
                            probabilities.push_back(probCumul);
                        } else {
                            wxLogError(_("The sum of the probabilities is null."));
                            return false;
                        }
                    }
                    if (fabs(probabilities[probabilities.size() - 1] - 1.0) > 0.00001) {
                        double diff = probabilities[probabilities.size() - 1] - 1.0;
                        wxLogWarning(_("probabilities[last]-1.0=%f"), diff);
                    }
                    initialized = true;
                }

                wxASSERT(probabilities.size() > 2);

                // Select mates
                double partner1prob = asRandom(0.0, 1.0);
                partner1 = asFindFloor(&probabilities[0], &probabilities[probabilities.size() - 1], partner1prob);
                double partner2prob = asRandom(0.0, 1.0);
                partner2 = asFindFloor(&probabilities[0], &probabilities[probabilities.size() - 1], partner2prob);

                if (partner1 < 0) {
                    wxLogError(_("Could not find a value in the probability distribution."));
                    wxLogError("probabilities[0] = %g, &probabilities[%d] = %g, partner1prob = %g", probabilities[0],
                               (int)probabilities.size() - 1, probabilities[probabilities.size() - 1], partner1prob);
                    return false;
                }
                if (partner2 < 0) {
                    wxLogError(_("Could not find a value in the probability distribution."));
                    wxLogError("probabilities[0] = %g, &probabilities[%d] = %g, partner2prob = %g", probabilities[0],
                               (int)probabilities.size() - 1, probabilities[probabilities.size() - 1], partner2prob);
                    return false;
                }

                break;
            }

            case (TournamentCompetition): {
                wxLogVerbose(_("Couples selection: tournament"));

                // Get nb of points
                int couplesSelectionTournamentNb;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/GAs/CouplesSelectionTournamentNb", &couplesSelectionTournamentNb, 3);
                ThreadsManager().CritSectionConfig().Leave();
                if (couplesSelectionTournamentNb < 2) {
                    wxLogWarning(_("The number of individuals for tournament selection is inferior to 2."));
                    wxLogWarning(_("The number of individuals for tournament selection has been changed."));
                    couplesSelectionTournamentNb = 2;
                }
                if (couplesSelectionTournamentNb > sizeParents / 2) {
                    wxLogWarning(
                        _("The number of individuals for tournament selection superior to the half of the intermediate "
                          "population."));
                    wxLogWarning(_("The number of individuals for tournament selection has been changed."));
                    couplesSelectionTournamentNb = sizeParents / 2;
                }

                // Select partner 1
                partner1 = sizeParents;
                for (int i = 0; i < couplesSelectionTournamentNb; i++) {
                    int candidate = asRandom(0, sizeParents - 1);
                    if (candidate < partner1)  // Smaller rank reflects better score
                    {
                        partner1 = candidate;
                    }
                }

                // Select partner 2
                partner2 = sizeParents;
                for (int i = 0; i < couplesSelectionTournamentNb; i++) {
                    int candidate = asRandom(0, sizeParents - 1);
                    if (candidate < partner2)  // Smaller rank reflects better score
                    {
                        partner2 = candidate;
                    }
                }

                break;
            }

            default: {
                wxLogError(_("The desired couples selection method is not yet implemented."));
            }
        }

        // Check that we don't have the same individual
        if (partner1 == partner2) {
            counterSame++;
            if (counterSame >= 100) {
                for (int i = 0; i < sizeParents; i++) {
                    wxLogWarning(_("_scoresCalib[%d] = %f"), i, _scoresCalib[i]);
                }

                for (int i = 0; i < probabilities.size(); i++) {
                    wxLogWarning(_("probabilities[%d] = %f"), i, probabilities[i]);
                }
                wxLogError(_("The same two partners were chosen more than 100 times. Lost in a loop."));
                return false;
            }
            continue;
        } else {
            counterSame = 0;
        }

        // Chromosomes crossings
        wxLogVerbose(_("Crossing chromosomes."));
        switch (_crossoverType) {
            case (SinglePointCrossover): {
                wxLogVerbose(_("Crossing: single point crossover"));

                // Get nb of points
                int crossoverNbPoints = 1;

                // Get points
                wxASSERT(partner1 >= 0);
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }

                break;
            }

            case (DoublePointsCrossover): {
                wxLogVerbose(_("Crossing: double points crossover"));

                // Get nb of points
                int crossoverNbPoints = 2;

                // Get points
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    if (!crossingPoints.empty()) {
                        // Check that is not already stored
                        if (chromosomeLength > crossoverNbPoints) {
                            for (int iPts = 0; iPts < crossingPoints.size(); iPts++) {
                                if (crossingPoints[iPts] == crossingPoint) {
                                    crossingPoints.erase(crossingPoints.begin() + iPts);
                                    wxLogVerbose(_("Crossing point already selected. Selection of a new one."));
                                    iCross--;
                                    break;
                                }
                            }
                        } else {
                            wxLogVerbose(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            case (MultiplePointsCrossover): {
                wxLogVerbose(_("Crossing: multiple points crossover"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/GAs/CrossoverMultiplePointsNb", &crossoverNbPoints, 3);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                if (crossoverNbPoints >= chromosomeLength) {
                    wxLogError(_("The desired crossings number is superior or equal to the genes number."));
                    return false;
                }

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    if (!crossingPoints.empty()) {
                        for (int iPts = 0; iPts < crossingPoints.size(); iPts++) {
                            if (crossingPoints[iPts] == crossingPoint) {
                                crossingPoints.erase(crossingPoints.begin() + iPts);
                                wxLogVerbose(_("Crossing point already selected. Selection of a new one."));
                                iCross--;
                                break;
                            }
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            case (UniformCrossover): {
                wxLogVerbose(_("Crossing: uniform crossover"));

                // Get points
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                vi crossingPoints;
                bool previouslyCrossed = false;  // flag

                for (int iGene = 0; iGene < chromosomeLength; iGene++) {
                    double doCross = asRandom(0.0, 1.0);

                    if (doCross >= 0.5)  // Yes
                    {
                        if (!previouslyCrossed)  // If situation changes
                        {
                            crossingPoints.push_back(iGene);
                        }
                        previouslyCrossed = true;
                    } else  // No
                    {
                        if (previouslyCrossed)  // If situation changes
                        {
                            crossingPoints.push_back(iGene);
                        }
                        previouslyCrossed = false;
                    }
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                if (!crossingPoints.empty()) {
                    param1.SimpleCrossover(param2, crossingPoints);
                }

                param1.CheckRange();

                // Add the new parameters if there is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            case (LimitedBlending): {
                wxLogVerbose(_("Crossing: limited blending"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/GAs/CrossoverBlendingPointsNb", &crossoverNbPoints, 2);

                // Get option to share beta or to generate a new one at every step
                bool shareBeta;
                pConfig->Read("/GAs/CrossoverBlendingShareBeta", &shareBeta, true);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    if (!crossingPoints.empty()) {
                        // Check that is not already stored
                        if (chromosomeLength > crossoverNbPoints) {
                            for (int iPts = 0; iPts < crossingPoints.size(); iPts++) {
                                if (crossingPoints[iPts] == crossingPoint) {
                                    crossingPoints.erase(crossingPoints.begin() + iPts);
                                    wxLogVerbose(_("Crossing point already selected. Selection of a new one."));
                                    iCross--;
                                    break;
                                }
                            }
                        } else {
                            wxLogVerbose(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.BlendingCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            case (LinearCrossover): {
                wxLogVerbose(_("Crossing: linear crossover"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/GAs/CrossoverLinearPointsNb", &crossoverNbPoints, 2);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    if (!crossingPoints.empty()) {
                        // Check that is not already stored
                        if (chromosomeLength > crossoverNbPoints) {
                            for (int iPts = 0; iPts < crossingPoints.size(); iPts++) {
                                if (crossingPoints[iPts] == crossingPoint) {
                                    crossingPoints.erase(crossingPoints.begin() + iPts);
                                    wxLogVerbose(_("Crossing point already selected. Selection of a new one."));
                                    iCross--;
                                    break;
                                }
                            }
                        } else {
                            wxLogVerbose(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                asParametersOptimizationGAs param3;
                param3 = _parameters[partner2];
                param1.LinearCrossover(param2, param3, crossingPoints);

                if (param1.IsInRange()) {
                    param1.CheckRange();

                    _parameters.push_back(param1);
                    _scoresCalib.push_back(NAN);
                }

                // Add the other parameters if ther is enough room
                if (_popSize - _parameters.size() > 0) {
                    if (param2.IsInRange()) {
                        param2.CheckRange();

                        _parameters.push_back(param2);
                        _scoresCalib.push_back(NAN);
                    }
                }
                if (_popSize - _parameters.size() > 0) {
                    if (param3.IsInRange()) {
                        param3.CheckRange();

                        _parameters.push_back(param3);
                        _scoresCalib.push_back(NAN);
                    }
                }

                break;
            }

            case (HeuristicCrossover): {
                wxLogVerbose(_("Crossing: heuristic crossover"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/GAs/CrossoverHeuristicPointsNb", &crossoverNbPoints, 2);

                // Get option to share beta or to generate a new one at every step
                bool shareBeta;
                pConfig->Read("/GAs/CrossoverHeuristicShareBeta", &shareBeta, true);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    if (!crossingPoints.empty()) {
                        // Check that is not already stored
                        if (chromosomeLength > crossoverNbPoints) {
                            for (int iPts = 0; iPts < crossingPoints.size(); iPts++) {
                                if (crossingPoints[iPts] == crossingPoint) {
                                    crossingPoints.erase(crossingPoints.begin() + iPts);
                                    wxLogVerbose(_("Crossing point already selected. Selection of a new one."));
                                    iCross--;
                                    break;
                                }
                            }
                        } else {
                            wxLogVerbose(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.HeuristicCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            case (BinaryLikeCrossover): {
                wxLogVerbose(_("Crossing: binary-like crossover"));

                // Get nb of points
                ThreadsManager().CritSectionConfig().Enter();
                int crossoverNbPoints;
                pConfig->Read("/GAs/CrossoverBinaryLikePointsNb", &crossoverNbPoints, 2);

                // Get option to share beta or to generate a new one at every step
                bool shareBeta;
                pConfig->Read("/GAs/CrossoverBinaryLikeShareBeta", &shareBeta, true);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = _parameters[partner1].GetChromosomeLength();
                wxASSERT(chromosomeLength > 0);

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    if (!crossingPoints.empty()) {
                        // Check that is not already stored
                        if (chromosomeLength > crossoverNbPoints) {
                            for (int iPts = 0; iPts < crossingPoints.size(); iPts++) {
                                if (crossingPoints[iPts] == crossingPoint) {
                                    crossingPoints.erase(crossingPoints.begin() + iPts);
                                    wxLogVerbose(_("Crossing point already selected. Selection of a new one."));
                                    iCross--;
                                    break;
                                }
                            }
                        } else {
                            wxLogVerbose(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.BinaryLikeCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            case (LinearInterpolation): {
                wxLogVerbose(_("Crossing: linear interpolation"));

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.LinearInterpolation(param2, true);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            case (FreeInterpolation): {
                wxLogVerbose(_("Crossing: free interpolation"));

                // Proceed to crossover
                wxASSERT(_parameters.size() > partner1);
                wxASSERT(_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = _parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = _parameters[partner2];
                param1.LinearInterpolation(param2, false);

                param1.CheckRange();

                // Add the new parameters if there is enough room
                _parameters.push_back(param1);
                _scoresCalib.push_back(NAN);
                if (_popSize - _parameters.size() > 0) {
                    param2.CheckRange();

                    _parameters.push_back(param2);
                    _scoresCalib.push_back(NAN);
                }
                break;
            }

            default: {
                wxLogError(_("The desired chromosomes crossing is not yet implemented."));
            }
        }

        counter++;
    }

    wxASSERT_MSG(_parameters.size() == _popSize,
                 asStrF("_parameters.size() = %d, _popSize = %d", (int)_parameters.size(), _popSize));
    wxASSERT(_parameters.size() == _scoresCalib.size());

    return true;
}

bool asMethodOptimizerGAs::Mutation() {
    // NB: The parameters and scores are already sorted !

    wxLogVerbose(_("Applying mutations."));

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    ThreadsManager().CritSectionConfig().Leave();

    switch (_mutationsModeType) {
        case (RandomUniformConstant): {
            double mutationsProbability;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsUniformConstantProbability", &mutationsProbability, 0.2);
            ThreadsManager().CritSectionConfig().Leave();

            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateUniformDistribution(mutationsProbability, hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (RandomUniformVariable): {
            int nbGenMax;
            double probStart, probEnd;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsUniformVariableMaxGensNbVar", &nbGenMax, 50);
            pConfig->Read("/GAs/MutationsUniformVariableProbabilityStart", &probStart, 0.5);
            pConfig->Read("/GAs/MutationsUniformVariableProbabilityEnd", &probEnd, 0.01);
            ThreadsManager().CritSectionConfig().Leave();

            double probIncrease = (probStart - probEnd) / (double)nbGenMax;
            double mutationsProbability = probStart + probIncrease * std::min(_generationNb - 1, nbGenMax);

            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateUniformDistribution(mutationsProbability, hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (RandomNormalConstant): {
            double mutationsProbability;
            double stdDevRatioRange;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsNormalConstantProbability", &mutationsProbability, 0.2);
            pConfig->Read("/GAs/MutationsNormalConstantStdDevRatioRange", &stdDevRatioRange, 0.10);
            ThreadsManager().CritSectionConfig().Leave();

            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateNormalDistribution(mutationsProbability, stdDevRatioRange, hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (RandomNormalVariable): {
            int nbGenMaxProb, nbGenMaxStdDev;
            double probStart, probEnd;
            double stdDevStart, stdDevEnd;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsNormalVariableMaxGensNbVarProb", &nbGenMaxProb, 50);
            pConfig->Read("/GAs/MutationsNormalVariableMaxGensNbVarStdDev", &nbGenMaxStdDev, 50);
            pConfig->Read("/GAs/MutationsNormalVariableProbabilityStart", &probStart, 0.5);
            pConfig->Read("/GAs/MutationsNormalVariableProbabilityEnd", &probEnd, 0.05);
            pConfig->Read("/GAs/MutationsNormalVariableStdDevStart", &stdDevStart, 0.5);
            pConfig->Read("/GAs/MutationsNormalVariableStdDevEnd", &stdDevEnd, 0.01);
            ThreadsManager().CritSectionConfig().Leave();

            double probIncrease = (probStart - probEnd) / (double)nbGenMaxProb;
            double mutationsProbability = probStart + probIncrease * std::min(_generationNb - 1, nbGenMaxProb);

            double stdDevIncrease = (stdDevStart - stdDevEnd) / (double)nbGenMaxStdDev;
            double stdDevRatioRange = stdDevStart + stdDevIncrease * std::min(_generationNb - 1, nbGenMaxStdDev);

            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateNormalDistribution(mutationsProbability, stdDevRatioRange, hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (NonUniform): {
            int nbGenMax;
            double mutationsProbability, minRate;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsNonUniformProbability", &mutationsProbability, 0.2);
            pConfig->Read("/GAs/MutationsNonUniformMaxGensNbVar", &nbGenMax, 50);
            pConfig->Read("/GAs/MutationsNonUniformMinRate", &minRate, 0.10);
            ThreadsManager().CritSectionConfig().Leave();

            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateNonUniform(mutationsProbability, _generationNb, nbGenMax, minRate, hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRate): {
            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateSelfAdaptationRate(hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRadius): {
            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateSelfAdaptationRadius(hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRateChromosome): {
            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateSelfAdaptationRateChromosome(hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRadiusChromosome): {
            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateSelfAdaptationRadiusChromosome(hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (MultiScale): {
            double mutationsProbability;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsMultiScaleProbability", &mutationsProbability, 0.1);
            ThreadsManager().CritSectionConfig().Leave();

            for (int iInd = 0; iInd < _parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                _parameters[iInd].MutateMultiScale(mutationsProbability, hasMutated);
                if (hasMutated) _scoresCalib[iInd] = NAN;

                _parameters[iInd].FixWeights();
                _parameters[iInd].FixCoordinates();
                _parameters[iInd].CheckRange();
                _parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (NoMutation): {
            // Nothing to do
            break;
        }

        default: {
            wxLogError(_("The desired mutation method is not yet implemented."));
        }
    }

    wxASSERT(_parameters.size() == _scoresCalib.size());

    return true;
}

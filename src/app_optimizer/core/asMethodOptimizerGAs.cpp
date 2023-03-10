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

#include "asFileText.h"
#include "asThreadGAs.h"

#ifndef UNIT_TESTING
#include "AtmoswingAppOptimizer.h"
#endif

#ifdef USE_CUDA
#include "asProcessorCuda.cuh"
#endif

asMethodOptimizerGAs::asMethodOptimizerGAs()
    : asMethodOptimizer(),
      m_scoreCalibBest(NaNf),
      m_generationNb(0),
      m_assessmentCounter(0),
      m_popSize(0),
      m_naturalSelectionType(0),
      m_couplesSelectionType(0),
      m_crossoverType(0),
      m_mutationsModeType(0),
      m_allowElitismForTheBest(true),
      m_reassessMiniBatchBests(true),
      m_miniBatchSize(365),
      m_miniBatchSizeMax(0),
      m_epoch(1),
      m_epochMax(30) {
    m_warnFailedLoadingData = false;
}

asMethodOptimizerGAs::~asMethodOptimizerGAs() = default;

void asMethodOptimizerGAs::ClearAll() {
    m_parameters.clear();
    m_parametersBatchBests.clear();
    m_scoresCalib.clear();
    m_scoreCalibBest = NaNf;
    m_scoreValid = NaNf;
    m_bestScores.clear();
    m_meanScores.clear();
}

void asMethodOptimizerGAs::SortScoresAndParameters() {
    wxASSERT(m_scoresCalib.size() == m_parameters.size());
    wxASSERT(m_scoresCalib.size() >= 1);
    wxASSERT(m_parameters.size() >= 1);

    if (m_parameters.size() == 1) return;

    int paramsNb = m_scoresCalib.size();

    // Sort according to the score
    a1f vIndices = a1f::LinSpaced(paramsNb, 0, paramsNb - 1);
    asSortArrays(&m_scoresCalib[0], &m_scoresCalib[paramsNb - 1], &vIndices[0], &vIndices[paramsNb - 1], m_scoreOrder);

    // Sort the parameters sets as the scores
    std::vector<asParametersOptimizationGAs> copyParameters;
    for (int i = 0; i < paramsNb; i++) {
        copyParameters.push_back(m_parameters[i]);
    }
    for (int i = 0; i < paramsNb; i++) {
        int index = vIndices(i);
        m_parameters[i] = copyParameters[index];
    }
}

bool asMethodOptimizerGAs::Manager() {
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    m_popSize = pConfig->ReadLong("/GAs/PopulationSize", 500);
    m_paramsNb = m_popSize;
    m_allowElitismForTheBest = pConfig->ReadBool("/GAs/AllowElitismForTheBest", true);
    m_naturalSelectionType = (int)pConfig->ReadLong("/GAs/NaturalSelectionOperator", 0l);
    m_couplesSelectionType = (int)pConfig->ReadLong("/GAs/CouplesSelectionOperator", 0l);
    m_crossoverType = (int)pConfig->ReadLong("/GAs/CrossoverOperator", 0l);
    m_mutationsModeType = (int)pConfig->ReadLong("/GAs/MutationOperator", 0l);
    m_useMiniBatches = pConfig->ReadBool("/GAs/UseMiniBatches", false);
    m_miniBatchSize = (int)pConfig->ReadLong("/GAs/MiniBatchSize", 1825l);
    m_epochMax = (int)pConfig->ReadLong("/GAs/NumberOfEpochs", 10l);
    ThreadsManager().CritSectionConfig().Leave();

    // Reset the score of the climatology
    m_scoreClimatology.clear();

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
    } catch (std::exception& e) {
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
    m_iterator = 0;
    m_assessmentCounter = 0;
    m_generationNb = 1;
    m_epoch = 1;

    // Seeds the random generator
    asInitRandom();

    // Load parameters
    asParametersOptimizationGAs params;
    if (!params.LoadFromFile(m_paramsFilePath)) return false;
    if (!m_predictandStationIds.empty()) {
        params.SetPredictandStationIds(m_predictandStationIds);
    }

    // Create a result object to save the parameters sets
    vi stationId = params.GetPredictandStationIds();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), YYYYMMDD_hhmm);
    asResultsParametersArray resFinalPopulation;
    resFinalPopulation.Init(asStrF(_("station_%s_final_population"), GetStationIdsList(stationId)));
    asResultsParametersArray resBestIndividual;
    resBestIndividual.Init(asStrF(_("station_%s_best_individual"), GetStationIdsList(stationId)));
    m_resGenerations.Init(asStrF(_("station_%s_generations"), GetStationIdsList(stationId)));
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
    } catch (std::exception& e) {
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
        fileOperators.Close();
    }

    // Get a score object to extract the score order
    asScore* score = asScore::GetInstance(params.GetScoreName());
    Order scoreOrder = score->GetOrder();
    wxDELETE(score);
    SetScoreOrder(scoreOrder);

    // Load the Predictand DB
    wxLogVerbose(_("Loading the Predictand DB."));
    if (!LoadPredictandDB(m_predictandDBFilePath)) return false;
    wxLogVerbose(_("Predictand DB loaded."));

    // Define time range if using mini-batches
    if (m_useMiniBatches) {
        // Create the target time array as reference for mini-batches
        asTimeArray timeArrayTarget(GetTimeStartCalibration(&params), GetTimeEndCalibration(&params),
                                    params.GetTargetTimeStepHours(), params.GetTimeArrayTargetMode());
        if (!m_validationMode && params.HasValidationPeriod()) {
            timeArrayTarget.SetForbiddenYears(params.GetValidationYearsVector());
        }

        if (!timeArrayTarget.Init()) {
            wxLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }

        m_miniBatchSizeMax = timeArrayTarget.GetSize();
        m_miniBatchStart = 0;
        m_miniBatchEnd = wxMin(m_miniBatchStart + m_miniBatchSize, m_miniBatchSizeMax) - 1;
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
        // Reassess the best parameter if mini-batch as the period has changed
        if (m_useMiniBatches && !firstRun) {
            auto* thread = new asThreadGAs(this, &m_parameterBest, &m_scoreCalibBest, &m_scoreClimatology);
#ifdef USE_CUDA
            if (method == asCUDA) {
                device = ThreadsManager().GetFreeDevice(gpusNb);
                thread->SetDevice(device);
            }
#endif
            ThreadsManager().AddThread(thread);
        }

        // Add threads when they become available
        while (m_iterator < m_paramsNb) {
#ifndef UNIT_TESTING
            if (g_responsive) wxTheApp->Yield();
#endif
            if (m_cancel) {
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
                auto* thread = new asThreadGAs(this, nextParams, &m_scoresCalib[m_iterator], &m_scoreClimatology);
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
                wxString paramsContent = m_parameters[iCheck].Print();
                wxLogError(_("Parameters #%d: %s"), (int)iCheck + 1, paramsContent);
                return false;
            }
        }

        wxLog::FlushActive();

        wxASSERT(m_iterator == m_paramsNb);

        // Different operators consider that the scores are sorted !
        SortScoresAndParameters();

        // Elitism after mutation must occur after evaluation
        ElitismAfterMutation();

        if (m_assessmentCounter > 0) {  // Skip if is resuming

            m_resGenerations.Clear();

            // Save the full generation
            for (int i = 0; i < m_parameters.size(); i++) {
                m_resGenerations.Add(m_parameters[i], m_scoresCalib[i]);
            }

            // Save operators status
            SaveOperators(operatorsFilePath);

            // Print results
            m_resGenerations.Print(m_resGenerations.GetCount() - m_parameters.size());
        }

        // Display stats
        float meanScore = asMean(&m_scoresCalib[0], &m_scoresCalib[m_scoresCalib.size() - 1]);
        float bestScore = 0;
        switch (m_scoreOrder) {
            case (Asc): {
                bestScore = asMinArray(&m_scoresCalib[0], &m_scoresCalib[m_scoresCalib.size() - 1]);
                break;
            }
            case (Desc): {
                bestScore = asMaxArray(&m_scoresCalib[0], &m_scoresCalib[m_scoresCalib.size() - 1]);
                break;
            }
        }
        m_bestScores.push_back(bestScore);
        m_meanScores.push_back(meanScore);

        wxLogMessage(_("Mean %g, best %g"), meanScore, bestScore);

        // Update best
        if (m_useMiniBatches) {
            if (asIsNaN(m_scoreCalibBest)) {
                m_parameterBest = m_parameters[0];
                m_scoreCalibBest = m_scoresCalib[0];
                if (m_reassessMiniBatchBests) {
                    m_parametersBatchBests.push_back(m_parameterBest);
                }
            } else {
                if (m_scoreOrder == Asc && m_scoresCalib[0] < m_scoreCalibBest) {
                    m_scoreCalibBest = m_scoresCalib[0];
                    m_parameterBest = m_parameters[0];
                    if (m_reassessMiniBatchBests) {
                        m_parametersBatchBests.push_back(m_parameterBest);
                    }
                } else if (m_scoreOrder == Desc && m_scoresCalib[0] > m_scoreCalibBest) {
                    m_scoreCalibBest = m_scoresCalib[0];
                    m_parameterBest = m_parameters[0];
                    if (m_reassessMiniBatchBests) {
                        m_parametersBatchBests.push_back(m_parameterBest);
                    }
                }
            }
        } else {
            m_parameterBest = m_parameters[0];
            m_scoreCalibBest = m_scoresCalib[0];
        }

        // Update mini-batches
        if (m_useMiniBatches) {
            m_miniBatchStart += m_miniBatchSize;
            int minNbDays = 32;
            if (m_miniBatchStart + minNbDays >= m_miniBatchSizeMax) {
                m_miniBatchStart = 0;
                m_epoch++;
                wxLogMessage(_("Epoch number %d"), m_epoch);
            }
            m_miniBatchEnd = wxMin(m_miniBatchStart + m_miniBatchSize, m_miniBatchSizeMax) - 1;
        }

        // Check if we should end
        if (HasConverged()) {
            // If finished, reassess all parameters on the full period
            if (m_useMiniBatches) {
                if (!ComputeAllScoresOnFullPeriod()) {
                    return false;
                }
                SortScoresAndParameters();
                // The current best parameter might not be in the population !
                if (m_scoreOrder == Asc && m_scoresCalib[0] < m_scoreCalibBest) {
                    m_scoreCalibBest = m_scoresCalib[0];
                    m_parameterBest = m_parameters[0];
                } else if (m_scoreOrder == Desc && m_scoresCalib[0] > m_scoreCalibBest) {
                    m_scoreCalibBest = m_scoresCalib[0];
                    m_parameterBest = m_parameters[0];
                }
            }
            wxLogVerbose(_("Optimization process over."));
            break;
        } else {
            // Always reset the score values for the mini-batch approach as the sample changes.
            if (m_useMiniBatches) {
                for (int i = 0; i < m_parameters.size(); i++) {
                    m_scoresCalib[i] = NaNf;
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

    // Sort according to level and time
    m_parameterBest.SortLevelsAndTime();

    // Validate
    SaveDetails(m_parameterBest);
    Validate(m_parameterBest);

    // Print parameters in a text file
    for (int i = 0; i < m_parameters.size(); i++) {
        resFinalPopulation.Add(m_parameters[i], m_scoresCalib[i]);
    }
    if (!resFinalPopulation.Print()) {
        wxLogError(_("The file containing the final population could not be generated."));
        return false;
    }
    resBestIndividual.Add(m_parameterBest, m_scoreCalibBest, m_scoreValid);

    if (!resBestIndividual.Print()) {
        wxLogError(_("The file containing the best individual could not be generated."));
        return false;
    }
    if (!m_resGenerations.Print(m_resGenerations.GetCount() - m_parameters.size())) return false;

    // Generate xml file with the best parameters set
    if (!m_parameterBest.GenerateSimpleParametersFile(resXmlFilePath)) {
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
    int miniBatchStart = m_miniBatchStart;
    int miniBatchEnd = m_miniBatchEnd;

    m_miniBatchStart = 0;
    m_miniBatchEnd = m_miniBatchSizeMax - 1;

    float scoreFullPeriod = NaNf;
    auto* thread = new asThreadGAs(this, &param, &scoreFullPeriod, &m_scoreClimatology);
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

    m_miniBatchStart = miniBatchStart;
    m_miniBatchEnd = miniBatchEnd;

    return scoreFullPeriod;
}

bool asMethodOptimizerGAs::ComputeAllScoresOnFullPeriod() {
    m_miniBatchStart = 0;
    m_miniBatchEnd = m_miniBatchSizeMax - 1;

    // Reassess the best parameter if mini-batch as the period has changed
    auto* thread = new asThreadGAs(this, &m_parameterBest, &m_scoreCalibBest, &m_scoreClimatology);
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
    if (m_reassessMiniBatchBests) {
        for (const auto& param : m_parametersBatchBests) {
            m_parameters.push_back(param);
        }
        m_paramsNb = m_parameters.size();
        m_scoresCalib.resize(m_paramsNb);
    }

    // Add threads when they become available
    m_iterator = 0;
    while (m_iterator < m_paramsNb) {
#ifndef UNIT_TESTING
        if (g_responsive) wxTheApp->Yield();
#endif
        if (m_cancel) {
            return false;
        }

        ThreadsManager().WaitForFreeThread(asThread::MethodOptimizerGAs);

#ifdef USE_CUDA
        if (method == asCUDA) {
            device = ThreadsManager().GetFreeDevice(gpusNb);
        }
#endif

        // Add it to the threads
        thread = new asThreadGAs(this, &m_parameters[m_iterator], &m_scoresCalib[m_iterator], &m_scoreClimatology);
#ifdef USE_CUDA
        if (method == asCUDA) {
            thread->SetDevice(device);
        }
#endif
        ThreadsManager().AddThread(thread);

        wxASSERT(m_scoresCalib.size() <= m_paramsNb);

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
    if (nLines % m_popSize != 0) {
        wxLogError(_("The number of former results is not consistent with the population size (%d)."), m_popSize);
        return false;
    }

    wxString msg = asStrF(_("Previous intermediate results were found "
                            "and will be loaded (%d lines)."),
                          nLines);
    wxLogWarning(msg);
    asLog::PrintToConsole(msg);
    asFileText prevResults(filePath, asFile::ReadOnly);
    if (!prevResults.Open()) {
        wxLogError(_("Couldn't open the file %s."), filePath);
        return false;
    }
    prevResults.SkipLines(1);

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

    int genNb = nLines / m_popSize;
    int iLastGen = (genNb - 1) * m_popSize;

    asParametersOptimizationGAs prevParams;

    // Parse the parameters data
    std::vector<float> vectScores;
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
            prevParams = m_parameters[0];
            if (!prevParams.GetValuesFromString(fileLine)) {
                return false;
            }
            m_resGenerations.AddWithoutProcessingMedian(prevParams, prevScoresCalib);
        }

        // Add to the new array
        if (iLine >= iLastGen) {
            // Restore the last generation
            m_parameters[iVar] = prevParams;
            m_scoresCalib[iVar] = prevScoresCalib;
            iVar++;
        }

        // Get next line
        fileLine = prevResults.GetNextLine();
        iLine++;
    } while (!prevResults.EndOfFile());
    prevResults.Close();

    wxLogMessage(_("%d former results have been reloaded."), m_resGenerations.GetCount());
    asLog::PrintToConsole(asStrF(_("%d former results have been reloaded.\n"), m_resGenerations.GetCount()));

    // Restore best and mean scores
    m_bestScores.resize(genNb);
    m_meanScores.resize(genNb);
    for (int iGen = 0; iGen < genNb; iGen++) {
        int iBest = iGen * m_popSize;
        m_bestScores[iGen] = vectScores[iBest];

        float mean = 0;
        for (int iNext = 0; iNext < m_popSize; iNext++) {
            mean += vectScores[iNext];
        }

        m_meanScores[iGen] = mean / float(m_popSize);
    }

    m_iterator = m_paramsNb;
    m_generationNb = genNb;

    // Update the epoch
    if (m_useMiniBatches) {
        wxString parentDirStr(dir.GetName());
        parentDirStr = parentDirStr.Mid(0, parentDirStr.Len() - 7);
        wxDir parentDir(parentDirStr);
        wxString logFilePattern = asStrF("*.log");
        if (!parentDir.HasFiles(logFilePattern)) {
            wxLogError("No log file found to restore the number of epochs (directory: %s)", parentDir.GetName());
            return false;
        }

        wxArrayString logFiles;
        wxDir::GetAllFiles(parentDir.GetName(), &logFiles, logFilePattern, wxDIR_FILES);

        m_epoch = 1;
        for (auto logFilePath : logFiles) {
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
                    m_epoch = wxMax(m_epoch, int(epochNb));
                }

                // Get next line
                fileLine = logContent.GetNextLine();
            } while (!logContent.EndOfFile());
            logContent.Close();
        }

        wxLogMessage(_("Starting again from epoch %d."), m_epoch);
    }

    // Copy file to the new target
    wxCopyFile(filePath, m_resGenerations.GetFilePath());

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
    prevOperators.SkipLines((genNb - 1) * m_popSize);
    wxString fileLineOper = prevOperators.GetNextLine();
    iVar = 0;
    do {
        if (fileLineOper.IsEmpty()) break;

        if (iVar >= m_parameters.size()) {
            wxLogError(_("Mismatch between number of parameters (%d) and operators (%d)."), (int)m_parameters.size(),
                       iVar + 1);
            return false;
        }

        switch (m_mutationsModeType) {
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
                m_parameters[iVar].SetAdaptMutationRate((float)mutationRate);
                break;
            }

            case (SelfAdaptationRadius): {
                int indexMutationRate = fileLineOper.Find("MutationRate");
                int indexMutationRadius = fileLineOper.Find("MutationRadius");
                wxString strMutationRate = fileLineOper.SubString(indexMutationRate + 13, indexMutationRadius - 2);
                double mutationRate;
                strMutationRate.ToDouble(&mutationRate);
                m_parameters[iVar].SetAdaptMutationRate((float)mutationRate);
                wxString strMutationRadius = fileLineOper.Mid(indexMutationRadius + 15);
                double mutationRadius;
                strMutationRadius.ToDouble(&mutationRadius);
                m_parameters[iVar].SetAdaptMutationRadius((float)mutationRadius);
                break;
            }

            case (SelfAdaptationRateChromosome): {
                int indexMutationRate = fileLineOper.Find("ChromosomeMutationRate");
                wxString strMutationRate = fileLineOper.Mid(indexMutationRate + 23);
                vf mutationRate = asExtractVectorFrom(strMutationRate);
                m_parameters[iVar].SetChromosomeMutationRate(mutationRate);
                break;
            }

            case (SelfAdaptationRadiusChromosome): {
                int indexMutationRate = fileLineOper.Find("ChromosomeMutationRate");
                int indexMutationRadius = fileLineOper.Find("ChromosomeMutationRadius");
                wxString strMutationRate = fileLineOper.SubString(indexMutationRate + 23, indexMutationRadius - 2);
                vf mutationRate = asExtractVectorFrom(strMutationRate);
                m_parameters[iVar].SetChromosomeMutationRate(mutationRate);
                wxString strMutationRadius = fileLineOper.Mid(indexMutationRadius + 25);
                vf mutationRadius = asExtractVectorFrom(strMutationRadius);
                m_parameters[iVar].SetChromosomeMutationRadius(mutationRadius);
                break;
            }

            default: {
                wxLogError(_("The mutation method was not found when saving operators."));
            }
        }

        fileLineOper = prevOperators.GetNextLine();
        iVar++;
    } while (!prevOperators.EndOfFile());
    prevOperators.Close();

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
    for (auto& parameter : m_parameters) {
        switch (m_mutationsModeType) {
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

    fileRes.Close();

    return true;
}

void asMethodOptimizerGAs::InitParameters(asParametersOptimizationGAs& params) {
    // Get a first parameters set to get the number of unknown variables
    params.InitRandomValues();
    wxLogVerbose(_("The population is made of %d individuals."), m_popSize);

    // Create the corresponding number of parameters
    m_scoresCalib.resize(m_popSize);
    m_parameters.resize(m_popSize);
    for (int iVar = 0; iVar < m_popSize; iVar++) {
        asParametersOptimizationGAs paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        paramsCopy.BuildChromosomes();

        // Create arrays for the self-adaptation methods
        switch (m_mutationsModeType) {
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

        m_parameters[iVar] = paramsCopy;
        m_scoresCalib[iVar] = NaNf;
    }
    m_scoreValid = NaNf;
}

asParametersOptimizationGAs* asMethodOptimizerGAs::GetNextParameters() {
    wxASSERT(m_iterator <= m_paramsNb);

    while (m_iterator < m_paramsNb) {
        // Parameters did not change
        if (!asIsNaN(m_scoresCalib[m_iterator])) {
            m_iterator++;
            continue;
        }

        m_assessmentCounter++;

        wxLogVerbose(_("m_parameters[%d] = %s"), m_iterator, m_parameters[m_iterator].Print());

        return &m_parameters[m_iterator];
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

    m_iterator = 0;
    m_generationNb++;

    wxLogMessage(_("Generation number %d"), m_generationNb);

    return true;
}

bool asMethodOptimizerGAs::HasConverged() {
    // NB: The parameters and scores are already sorted !

    if (m_useMiniBatches) {
        if (m_epoch > m_epochMax) {
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
    if (m_bestScores.size() < convergenceStepsNb) {
        return false;
    }

    // Check the best convergenceStepsNb scores
    for (int i = m_bestScores.size() - 1; i > m_bestScores.size() - convergenceStepsNb; i--)  // Checked
    {
        float lastScore = m_bestScores[m_bestScores.size() - 1];

        if (lastScore == 0) {
            if (m_bestScores[i] != m_bestScores[i - 1]) {
                return false;
            }
        } else {
            float relDiff = std::abs((lastScore - m_bestScores[i - 1]) / lastScore);
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
    if (m_allowElitismForTheBest && !asIsNaN(m_scoreCalibBest)) {
        float actualBest = m_scoresCalib[0];
        switch (m_scoreOrder) {
            case (Asc): {
                if (m_scoreCalibBest < actualBest) {
                    wxLogMessage(_("Application of elitism after mutation."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, m_scoresCalib.size() - 1, 1);
                    m_parameters[randomRow] = m_parameterBest;
                    m_scoresCalib[randomRow] = m_scoreCalibBest;
                    SortScoresAndParameters();
                }
                break;
            }
            case (Desc): {
                if (m_scoreCalibBest > actualBest) {
                    wxLogMessage(_("Application of elitism after mutation."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, m_scoresCalib.size() - 1, 1);
                    m_parameters[randomRow] = m_parameterBest;
                    m_scoresCalib[randomRow] = m_scoreCalibBest;
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
    if (m_allowElitismForTheBest && !asIsNaN(m_scoreCalibBest)) {
        SortScoresAndParameters();
        float actualBest = m_scoresCalib[0];
        switch (m_scoreOrder) {
            case (Asc): {
                if (m_scoreCalibBest < actualBest) {
                    wxLogMessage(_("Application of elitism in the natural selection."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, m_scoresCalib.size() - 1, 1);
                    m_parameters[randomRow] = m_parameterBest;
                    m_scoresCalib[randomRow] = m_scoreCalibBest;
                }
                break;
            }
            case (Desc): {
                if (m_scoreCalibBest > actualBest) {
                    wxLogMessage(_("Application of elitism in the natural selection."));
                    // Randomly select a row to replace
                    int randomRow = asRandom(0, m_scoresCalib.size() - 1, 1);
                    m_parameters[randomRow] = m_parameterBest;
                    m_scoresCalib[randomRow] = m_scoreCalibBest;
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

    std::vector<asParametersOptimizationGAs> parameters = m_parameters;
    vf scores = m_scoresCalib;
    m_parameters.clear();
    m_scoresCalib.clear();

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    double ratioIntermediateGeneration;
    pConfig->Read("/GAs/RatioIntermediateGeneration", &ratioIntermediateGeneration, 0.5);
    ThreadsManager().CritSectionConfig().Leave();

    // Get intermediate generation size
    int intermediateGenerationSize = ratioIntermediateGeneration * m_popSize;

    switch (m_naturalSelectionType) {
        case (RatioElitism): {
            wxLogVerbose(_("Natural selection: ratio elitism"));

            for (int i = 0; i < intermediateGenerationSize; i++) {
                m_parameters.push_back(parameters[i]);
                m_scoresCalib.push_back(scores[i]);
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
                if (m_scoresCalib[candidate1] == scores[candidate2]) {
                    double randomIndex = asRandom(0.0, 1.0);
                    if (randomIndex <= 0.5) {
                        candidateFinal = candidate1;
                    } else {
                        candidateFinal = candidate2;
                    }
                }

                m_parameters.push_back(parameters[candidateFinal]);
                m_scoresCalib.push_back(scores[candidateFinal]);
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

    wxASSERT(m_parameters.size() == m_scoresCalib.size());

    wxLogVerbose(_("Applying mating."));

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    ThreadsManager().CritSectionConfig().Leave();

    // Build chromosomes
    for (int i = 0; i < m_parameters.size(); i++) {
        wxASSERT(m_parameters[i].GetChromosomeLength() > 0);
    }

    int sizeParents = int(m_parameters.size());
    int counter = 0;
    int counterSame = 0;
    bool initialized = false;
    vd probabilities;

    while (m_parameters.size() < m_popSize) {
        // Couples selection only in the parents pool
        wxLogVerbose(_("Selecting couples."));
        int partner1 = 0, partner2 = 0;
        switch (m_couplesSelectionType) {
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
                        sum += m_scoresCalib[i] - m_scoresCalib[sizeParents - 1] + 0.001;  // 0.001 to avoid null probs
                    }
                    for (int i = 0; i < sizeParents; i++) {
                        if (sum > 0) {
                            double currentScore = m_scoresCalib[i] - m_scoresCalib[sizeParents - 1] + 0.001;
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

        wxLogVerbose("partner1 = %d, partner2 = %d", partner1, partner2);

        // Check that we don't have the same individual
        if (partner1 == partner2) {
            counterSame++;
            if (counterSame >= 100) {
                for (int i = 0; i < sizeParents; i++) {
                    wxLogWarning(_("m_scoresCalib[%d] = %f"), i, m_scoresCalib[i]);
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
        switch (m_crossoverType) {
            case (SinglePointCrossover): {
                wxLogVerbose(_("Crossing: single point crossover"));

                // Get nb of points
                int crossoverNbPoints = 1;

                // Get points
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

                vi crossingPoints;
                for (int iCross = 0; iCross < crossoverNbPoints; iCross++) {
                    int crossingPoint = asRandom(0, chromosomeLength - 1, 1);
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
                }

                break;
            }

            case (DoublePointsCrossover): {
                wxLogVerbose(_("Crossing: double points crossover"));

                // Get nb of points
                int crossoverNbPoints = 2;

                // Get points
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

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
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
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
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

                if (crossoverNbPoints >= chromosomeLength) {
                    wxLogError(_("The desired crossings number is superior or equal to the genes number."));
                    return false;
                }

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
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
                }
                break;
            }

            case (UniformCrossover): {
                wxLogVerbose(_("Crossing: uniform crossover"));

                // Get points
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

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
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                if (!crossingPoints.empty()) {
                    param1.SimpleCrossover(param2, crossingPoints);
                }

                param1.CheckRange();

                // Add the new parameters if there is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
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
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

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
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.BlendingCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
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
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

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
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                asParametersOptimizationGAs param3;
                param3 = m_parameters[partner2];
                param1.LinearCrossover(param2, param3, crossingPoints);

                if (param1.IsInRange()) {
                    param1.CheckRange();

                    m_parameters.push_back(param1);
                    m_scoresCalib.push_back(NaNf);
                }

                // Add the other parameters if ther is enough room
                if (m_popSize - m_parameters.size() > 0) {
                    if (param2.IsInRange()) {
                        param2.CheckRange();

                        m_parameters.push_back(param2);
                        m_scoresCalib.push_back(NaNf);
                    }
                }
                if (m_popSize - m_parameters.size() > 0) {
                    if (param3.IsInRange()) {
                        param3.CheckRange();

                        m_parameters.push_back(param3);
                        m_scoresCalib.push_back(NaNf);
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
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

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
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.HeuristicCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
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
                int chromosomeLength = m_parameters[partner1].GetChromosomeLength();

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
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.BinaryLikeCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
                }
                break;
            }

            case (LinearInterpolation): {
                wxLogVerbose(_("Crossing: linear interpolation"));

                // Proceed to crossover
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.LinearInterpolation(param2, true);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
                }
                break;
            }

            case (FreeInterpolation): {
                wxLogVerbose(_("Crossing: free interpolation"));

                // Proceed to crossover
                wxASSERT(m_parameters.size() > partner1);
                wxASSERT(m_parameters.size() > partner2);
                asParametersOptimizationGAs param1;
                param1 = m_parameters[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_parameters[partner2];
                param1.LinearInterpolation(param2, false);

                param1.CheckRange();

                // Add the new parameters if there is enough room
                m_parameters.push_back(param1);
                m_scoresCalib.push_back(NaNf);
                if (m_popSize - m_parameters.size() > 0) {
                    param2.CheckRange();

                    m_parameters.push_back(param2);
                    m_scoresCalib.push_back(NaNf);
                }
                break;
            }

            default: {
                wxLogError(_("The desired chromosomes crossing is not yet implemented."));
            }
        }

        counter++;
    }

    wxASSERT_MSG(m_parameters.size() == m_popSize,
                 asStrF("m_parameters.size() = %d, m_popSize = %d", (int)m_parameters.size(), m_popSize));
    wxASSERT(m_parameters.size() == m_scoresCalib.size());

    return true;
}

bool asMethodOptimizerGAs::Mutation() {
    // NB: The parameters and scores are already sorted !

    wxLogVerbose(_("Applying mutations."));

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    ThreadsManager().CritSectionConfig().Leave();

    switch (m_mutationsModeType) {
        case (RandomUniformConstant): {
            double mutationsProbability;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsUniformConstantProbability", &mutationsProbability, 0.2);
            ThreadsManager().CritSectionConfig().Leave();

            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateUniformDistribution(mutationsProbability, hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
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
            double mutationsProbability = probStart + probIncrease * wxMin(m_generationNb - 1, nbGenMax);

            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateUniformDistribution(mutationsProbability, hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
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

            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateNormalDistribution(mutationsProbability, stdDevRatioRange, hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
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
            double mutationsProbability = probStart + probIncrease * wxMin(m_generationNb - 1, nbGenMaxProb);

            double stdDevIncrease = (stdDevStart - stdDevEnd) / (double)nbGenMaxStdDev;
            double stdDevRatioRange = stdDevStart + stdDevIncrease * wxMin(m_generationNb - 1, nbGenMaxStdDev);

            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateNormalDistribution(mutationsProbability, stdDevRatioRange, hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (NonUniform): {
            int nbGenMax;
            double mutationsProbability, minRate;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsNonUniformProbability", &mutationsProbability, 0.2);
            pConfig->Read("/GAs/MutationsNonUniformMaxGensNbVar", &nbGenMax, 50);
            pConfig->Read("/GAs/MutationsNonUniformMinRate", &minRate, 0.20);
            ThreadsManager().CritSectionConfig().Leave();

            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateNonUniform(mutationsProbability, m_generationNb, nbGenMax, minRate,
                                                    hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRate): {
            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateSelfAdaptationRate(hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRadius): {
            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateSelfAdaptationRadius(hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRateChromosome): {
            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateSelfAdaptationRateChromosome(hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRadiusChromosome): {
            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateSelfAdaptationRadiusChromosome(hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
            }
            break;
        }

        case (MultiScale): {
            double mutationsProbability;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/GAs/MutationsMultiScaleProbability", &mutationsProbability, 0.2);
            ThreadsManager().CritSectionConfig().Leave();

            for (int iInd = 0; iInd < m_parameters.size(); iInd++) {
                // Mutate
                bool hasMutated = false;
                m_parameters[iInd].MutateMultiScale(mutationsProbability, hasMutated);
                if (hasMutated) m_scoresCalib[iInd] = NaNf;

                m_parameters[iInd].FixWeights();
                m_parameters[iInd].FixCoordinates();
                m_parameters[iInd].CheckRange();
                m_parameters[iInd].FixAnalogsNb();
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

    wxASSERT(m_parameters.size() == m_scoresCalib.size());

    return true;
}

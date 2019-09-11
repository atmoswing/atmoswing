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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asGlobVars.h"
#include "asMethodCalibratorSingle.h"
#include "gtest/gtest.h"
#include "benchmark/benchmark.h"

#define _CHECK_CUDA_RESULTS false

asMethodCalibratorSingle *g_calibrator;
asParametersCalibration *g_params;

const int maxPointsNb = 2048;

int main(int argc, char **argv)
{
    try {
        ::benchmark::Initialize(&argc, argv);

        // Override some globals
        g_unitTesting = true;
        g_silentMode = true;
        g_guiMode = false;

        // Initialize the library because wxApp is not called
        wxInitialize();

        // Set the log
        Log()->CreateFile("AtmoSwingBench.log");
        Log()->SetLevel(2);

        // Set the local config object
        auto *pConfig = new wxFileConfig("AtmoSwing", wxEmptyString, asConfig::GetTempDir() + "AtmoSwingBench.ini",
                                         asConfig::GetTempDir() + "AtmoSwingBench.ini", wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
        pConfig->Write("/Processing/AllowMultithreading", true);
        pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
        pConfig->Write("/Processing/MaxThreadNb", 8);

        // Check path
        wxString filePath = wxFileName::GetCwd();
        wxString filePath1 = filePath;
        filePath1.Append("/benchmark/files");
        if (wxFileName::DirExists(filePath1)) {
            filePath.Append("/benchmark");
            wxSetWorkingDirectory(filePath);
        } else {
            wxString filePath2 = filePath;
            filePath2.Append("/../benchmark/files");
            if (wxFileName::DirExists(filePath2)) {
                filePath.Append("/../benchmark");
                wxSetWorkingDirectory(filePath);
            } else {
                wxString filePath3 = filePath;
                filePath3.Append("/../../benchmark/files");
                if (wxFileName::DirExists(filePath3)) {
                    filePath.Append("/../../benchmark");
                    wxSetWorkingDirectory(filePath);
                } else {
                    wxPrintf("Cannot find the files directory\n");
                    wxPrintf("Original working directory: %s\n", filePath);
                    return 0;
                }
            }
        }

        // Calibrator
        g_calibrator = new asMethodCalibratorSingle();
        g_calibrator->SetPredictorDataDir(BENCHMARK_DATA_DIR);

        // Get parameters
        wxString paramsFilePath = wxFileName::GetCwd();
        paramsFilePath.Append("/files/");
        paramsFilePath.Append("parameters_bench_1.xml");
        g_params = new asParametersCalibration();
        g_params->LoadFromFile(paramsFilePath);

        // Preload data
        g_calibrator->PreloadDataOnly(g_params);

        ::benchmark::RunSpecifiedBenchmarks();

        // Cleanup
        wxUninitialize();
        DeleteThreadsManager();
        DeleteLog();
        delete wxFileConfig::Set((wxFileConfig *) nullptr);
        wxDELETE(g_calibrator);
        wxDELETE(g_params);

    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxPrintf(_("Exception caught: %s\n"), msg);
    }

    return 0;
}


#ifdef USE_CUDA
static void BM_1Ptor_S1_Cuda(benchmark::State &state)
{
    int step = 0;
    int nbY = wxMin((int) std::sqrt(state.range(0)), 40);
    int nbX = int(state.range(0) / nbY);
    asParametersCalibration params = *g_params;
    params.RemovePredictor(step, 1);
    params.SetPredictorXptsnb(step, 0, nbX);
    params.SetPredictorYptsnb(step, 0, nbY);

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asCUDA);

    bool containsNaNs = false;
    asResultsDates anaDates;

    for (auto _ : state) {
        try {
            ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDates, &params, step, containsNaNs));
            EXPECT_FALSE(containsNaNs);
        } catch (std::exception &e) {
            wxPrintf(e.what());
            return;
        }
    }

#if _CHECK_CUDA_RESULTS
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    asResultsDates anaDatesRef;
    ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDatesRef, &params, step, containsNaNs));

    // Extract data
    a1f resultsTargetDatesCPU(anaDatesRef.GetTargetDates());
    a2f resultsCriteriaCPU(anaDatesRef.GetAnalogsCriteria());
    a2f resultsAnalogDatesCPU(anaDatesRef.GetAnalogsDates());
    a1f resultsTargetDatesGPU(anaDates.GetTargetDates());
    a2f resultsCriteriaGPU(anaDates.GetAnalogsCriteria());
    a2f resultsAnalogDatesGPU(anaDates.GetAnalogsDates());

    // Check results
    for (int i = 0; i < resultsCriteriaCPU.rows(); ++i) {
        EXPECT_FLOAT_EQ(resultsTargetDatesCPU(i), resultsTargetDatesGPU(i));
        for (int j = 0; j < resultsCriteriaCPU.cols(); ++j) {
            EXPECT_NEAR(resultsCriteriaCPU(i, j), resultsCriteriaGPU(i, j), 0.0001);
            if (abs(resultsCriteriaCPU(i, j) - resultsCriteriaGPU(i, j)) > 0.0001) {
                EXPECT_FLOAT_EQ(resultsAnalogDatesCPU(i, j), resultsAnalogDatesGPU(i, j));
            }
        }
    }
#endif
}
BENCHMARK(BM_1Ptor_S1_Cuda)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(2, maxPointsNb);
#endif

static void BM_1Ptor_S1_Standard(benchmark::State &state)
{
    int step = 0;
    int nbY = wxMin((int) std::sqrt(state.range(0)), 40);
    int nbX = int(state.range(0) / nbY);
    asParametersCalibration params = *g_params;
    params.RemovePredictor(step, 1);
    params.SetPredictorXptsnb(step, 0, nbX);
    params.SetPredictorYptsnb(step, 0, nbY);

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    bool containsNaNs = false;
    asResultsDates anaDates;

    for (auto _ : state) {
        try {
            ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDates, &params, step, containsNaNs));
            EXPECT_FALSE(containsNaNs);
        } catch (std::exception &e) {
            wxPrintf(e.what());
            return;
        }
    }
}
BENCHMARK(BM_1Ptor_S1_Standard)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(2, maxPointsNb);

static void BM_1Ptor_S1_Multithreaded(benchmark::State &state)
{
    int step = 0;
    int nbY = wxMin((int) std::sqrt(state.range(0)), 40);
    int nbX = int(state.range(0) / nbY);
    asParametersCalibration params = *g_params;
    params.RemovePredictor(step, 1);
    params.SetPredictorXptsnb(step, 0, nbX);
    params.SetPredictorYptsnb(step, 0, nbY);

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    bool containsNaNs = false;
    asResultsDates anaDates;

    for (auto _ : state) {
        try {
            ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDates, &params, step, containsNaNs));
            EXPECT_FALSE(containsNaNs);
        } catch (std::exception &e) {
            wxPrintf(e.what());
            return;
        }
    }
}
BENCHMARK(BM_1Ptor_S1_Multithreaded)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(2, maxPointsNb);

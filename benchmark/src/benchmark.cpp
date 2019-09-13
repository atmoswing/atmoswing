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

#define _CHECK_CUDA_RESULTS true

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
        pConfig->Write("/Processing/MaxThreadNb", 3);

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

void CustomArguments(benchmark::internal::Benchmark *b);

void CustomArgumentsWarmup(benchmark::internal::Benchmark *b);

void CompareResults(asResultsDates &anaDates, asResultsDates &anaDatesRef);

asParametersCalibration GetParameters(int nSteps, int nPtors, int nPts, const wxString &criteria);

#ifdef USE_CUDA
template <class ...ExtraArgs>
void BM_Cuda(benchmark::State &state, ExtraArgs&&... extra_args)
{
    std::tuple<ExtraArgs...> argsTuple{extra_args...};
    wxString criteria(std::get<0>(argsTuple));

    int nSteps = state.range(0);
    int nPtors = state.range(1);
    int nPts = state.range(2);

    asParametersCalibration params = GetParameters(nSteps, nPtors, nPts, criteria);

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asCUDA);

    bool containsNaNs = false;
    asResultsDates anaDates1;
    asResultsDates anaDates2;

    for (auto _ : state) {
        try {
            ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDates1, &params, 0, containsNaNs));
            EXPECT_FALSE(containsNaNs);
            if (nSteps == 2) {
                ASSERT_TRUE(g_calibrator->GetAnalogsSubDates(anaDates2, &params, anaDates1, 1, containsNaNs));
                EXPECT_FALSE(containsNaNs);
            }
        } catch (std::exception &e) {
            wxPrintf(e.what());
            return;
        }
    }

#if _CHECK_CUDA_RESULTS
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    asResultsDates anaDatesRef1;
    ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDatesRef1, &params, 0, containsNaNs));
    CompareResults(anaDates1, anaDatesRef1);

    if (nSteps == 2) {
        asResultsDates anaDatesRef2;
        ASSERT_TRUE(g_calibrator->GetAnalogsSubDates(anaDatesRef2, &params, anaDatesRef1, 1, containsNaNs));
        CompareResults(anaDates2, anaDatesRef2);
    }
#endif
}

BENCHMARK_CAPTURE(BM_Cuda, warmum, "S1")->Unit(benchmark::kMillisecond)->Apply(CustomArgumentsWarmup);
BENCHMARK_CAPTURE(BM_Cuda, S1, "S1")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, RMSE, "RMSE")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, S0, "S0")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, S2, "S2")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, MD, "MD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, RSE, "RSE")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, SAD, "SAD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, DMV, "DMV")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Cuda, DSD, "DSD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);

#endif

template <class ...ExtraArgs>
void BM_Standard(benchmark::State &state, ExtraArgs&&... extra_args)
{
    std::tuple<ExtraArgs...> argsTuple{extra_args...};
    wxString criteria(std::get<0>(argsTuple));

    int nSteps = state.range(0);
    int nPtors = state.range(1);
    int nPts = state.range(2);

    asParametersCalibration params = GetParameters(nSteps, nPtors, nPts, criteria);

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    bool containsNaNs = false;
    asResultsDates anaDates1;
    asResultsDates anaDates2;

    for (auto _ : state) {
        try {
            ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDates1, &params, 0, containsNaNs));
            EXPECT_FALSE(containsNaNs);
            if (nSteps == 2) {
                ASSERT_TRUE(g_calibrator->GetAnalogsSubDates(anaDates2, &params, anaDates1, 1, containsNaNs));
                EXPECT_FALSE(containsNaNs);
            }
        } catch (std::exception &e) {
            wxPrintf(e.what());
            return;
        }
    }
}

BENCHMARK_CAPTURE(BM_Standard, S1, "S1")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, RMSE, "RMSE")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, S0, "S0")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, S2, "S2")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, MD, "MD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, RSE, "RSE")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, SAD, "SAD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, DMV, "DMV")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Standard, DSD, "DSD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);


template <class ...ExtraArgs>
void BM_Multithreaded(benchmark::State &state, ExtraArgs&&... extra_args)
{
    std::tuple<ExtraArgs...> argsTuple{extra_args...};
    wxString criteria(std::get<0>(argsTuple));

    int nSteps = state.range(0);
    int nPtors = state.range(1);
    int nPts = state.range(2);

    asParametersCalibration params = GetParameters(nSteps, nPtors, nPts, criteria);

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    bool containsNaNs = false;
    asResultsDates anaDates1;
    asResultsDates anaDates2;

    for (auto _ : state) {
        try {
            ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDates1, &params, 0, containsNaNs));
            EXPECT_FALSE(containsNaNs);
            if (nSteps == 2) {
                ASSERT_TRUE(g_calibrator->GetAnalogsSubDates(anaDates2, &params, anaDates1, 1, containsNaNs));
                EXPECT_FALSE(containsNaNs);
            }
        } catch (std::exception &e) {
            wxPrintf(e.what());
            return;
        }
    }
}

BENCHMARK_CAPTURE(BM_Multithreaded, S1, "S1")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, RMSE, "RMSE")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, S0, "S0")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, S2, "S2")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, MD, "MD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, RSE, "RSE")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, SAD, "SAD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, DMV, "DMV")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
BENCHMARK_CAPTURE(BM_Multithreaded, DSD, "DSD")->Unit(benchmark::kMillisecond)->Apply(CustomArguments);


void CustomArguments(benchmark::internal::Benchmark *b)
{
    for (int level = 1; level <= 2; ++level) {
        for (int ptors = 1; ptors <= 4; ++ptors) {
            for (int pts = 4; pts <= maxPointsNb; pts *= 2) {
                b->Args({level, ptors, pts});
            }
        }
    }
}

void CustomArgumentsWarmup(benchmark::internal::Benchmark *b)
{
    b->Args({1, 1, 4});
}

void CompareResults(asResultsDates &anaDates, asResultsDates &anaDatesRef)
{
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
            float precision = resultsCriteriaCPU(i, j) * 0.000001;
            EXPECT_NEAR(resultsCriteriaCPU(i, j), resultsCriteriaGPU(i, j), precision);
            if (abs(resultsCriteriaCPU(i, j) - resultsCriteriaGPU(i, j)) > precision) {
                EXPECT_FLOAT_EQ(resultsAnalogDatesCPU(i, j), resultsAnalogDatesGPU(i, j));
            }
        }
    }
}

asParametersCalibration GetParameters(int nSteps, int nPtors, int nPts, const wxString &criteria)
{
    asParametersCalibration params = *g_params;

    int nbY = wxMin((int) std::sqrt(nPts), 40);
    int nbX = int(nPts / nbY);

    if (nSteps == 1) {
        params.RemoveStep(1);
    }

    for (int st = 0; st < nSteps; ++st) {
        for (int pt = 3; pt >= nPtors; --pt) {
            params.RemovePredictor(st, pt);
        }
        for (int pt = 0; pt < nPtors; ++pt) {
            params.SetPredictorXptsnb(st, pt, nbX);
            params.SetPredictorYptsnb(st, pt, nbY);
            params.SetPredictorCriteria(st, pt, criteria);
        }
    }
    return params;
}
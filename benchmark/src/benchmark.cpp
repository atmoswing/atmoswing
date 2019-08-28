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
#include "asProcessor.h"
#include "asMethodCalibratorSingle.h"
#include "gtest/gtest.h"
#include "benchmark/benchmark.h"

asMethodCalibratorSingle *g_calibrator;
asParametersCalibration *g_params;

int main(int argc, char **argv)
{
    try {
        wxPrintf(_("Initializing benchmarks...\n"));

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
        wxFileConfig *pConfig = new wxFileConfig("AtmoSwing", wxEmptyString, asConfig::GetTempDir() + "AtmoSwingBench.ini",
                                                 asConfig::GetTempDir() + "AtmoSwingBench.ini", wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);

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

        wxPrintf(_("Preloading data...\n"));

        // Preload data
        asResultsDates anaDates;
        bool containsNaNs = false;
        g_calibrator->GetAnalogsDates(anaDates, g_params, 0, containsNaNs);

        wxPrintf(_("Starting real work...\n"));

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


static void BM_StringCreation(benchmark::State& state) {

    for (auto _ : state) {
        int step = 0;
        bool containsNaNs = false;
        asResultsDates anaDates;

        try {
            ASSERT_TRUE(g_calibrator->GetAnalogsDates(anaDates, g_params, step, containsNaNs));
            EXPECT_FALSE(containsNaNs);
        } catch (std::exception &e) {
            wxPrintf(e.what());
            return;
        }
    }
}
BENCHMARK(BM_StringCreation);

static void BM_StringCopy(benchmark::State& state) {
    std::string x = "hello";
    for (auto _ : state)
        std::string copy(x);
}
BENCHMARK(BM_StringCopy);



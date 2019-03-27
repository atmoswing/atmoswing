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

#include <wx/filename.h>
#include <wx/dir.h>
#include "asPredictandPrecipitation.h"
#include "asProcessor.h"
#include "asMethodCalibratorSingle.h"
#include "asResultsDates.h"
#include "asResultsValues.h"
#include "asResultsScores.h"
#include "asResultsTotalScore.h"
#include "asFileAscii.h"
#include "gtest/gtest.h"


void Ref1(const wxString &paramsFile, bool shortVersion)
{
    // Create predictand database
    auto *predictand = new asPredictandPrecipitation(asPredictand::Precipitation, asPredictand::Daily,
                                                     asPredictand::Station);

    wxString datasetPredictandFilePath = wxFileName::GetCwd();
    datasetPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(datasetPredictandFilePath, dataFileDir, patternFileDir);

    float P10 = 68.42240f;

    // Get parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append(paramsFile);
    asParametersCalibration params;
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    asResultsDates anaDates;
    asResultsValues anaValues;
    asResultsScores anaScoresCRPS;
    asResultsScores anaScoresCRPSsharpness;
    asResultsScores anaScoresCRPSaccuracy;
    asResultsTotalScore anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        wxString predictorFilePath = wxFileName::GetCwd();
        predictorFilePath.Append("/files/data-ncep-r1/others/");
        calibrator.SetPredictorDataDir(predictorFilePath);
        wxASSERT(predictand);
        calibrator.SetPredictandDB(predictand);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, &params, anaDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPS, &params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsTotalScore(anaScoreFinal, &params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetScoreName("CRPSsharpnessAR");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSsharpness, &params, anaValues, step));
        params.SetScoreName("CRPSaccuracyAR");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSaccuracy, &params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    a1f resTargetDates(anaDates.GetTargetDates());
    a1f resTargetValues(anaValues.GetTargetValues()[0]);
    a2f resCriteria(anaDates.GetAnalogsCriteria());
    a2f resDates(anaDates.GetAnalogsDates());
    a2f resValues(anaValues.GetAnalogsValues()[0]);
    a1f resScoreCRPS(anaScoresCRPS.GetScores());
    a1f resScoreCRPSsharpness(anaScoresCRPSsharpness.GetScores());
    a1f resScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetScores());
    float scoreFinal = anaScoreFinal.GetScore();

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    int nbtests = 0;
    if (!shortVersion) {
        resultFilePath.Append("/files/score_04.txt");
        nbtests = 43;
    } else {
        resultFilePath.Append("/files/score_06.txt");
        nbtests = 20;
    }
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int nanalogs = 50;
    a1f fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int iTest = 0; iTest < nbtests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Get target date from file
        int day = file.GetInt();
        int month = file.GetInt();
        int year = file.GetInt();
        float fileTargetDate = (float) asTime::GetMJD(year, month, day);
        float fileTargetValue = (float) sqrt(file.GetFloat() / P10);

        file.SkipLines(2);

        // Get analogs from file
        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[iAnalog] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[iAnalog] = 0;
            }
            fileAnalogsValues[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            fileAnalogsCriteria[iAnalog] = file.GetFloat();

            file.SkipLines(1);
        }

        float fileScoreCRPS = 0, fileScoreCRPSaccuracy = 0, fileScoreCRPSsharpness = 0;

        if (!shortVersion) {
            file.SkipLines(2);
            fileScoreCRPS = file.GetFloat();
            fileScoreCRPSaccuracy = file.GetFloat();
            fileScoreCRPSsharpness = file.GetFloat();
            file.SkipLines(1);
        } else {
            file.SkipLines(3);
        }

        // Find target date in the array
        int rowTargetDate = asFindClosest(&resTargetDates[0], &resTargetDates[resTargetDates.rows() - 1],
                                          fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resTargetValues(rowTargetDate));

        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            if (fileAnalogsDates[iAnalog] > 0) // If we have the data
            {
                EXPECT_FLOAT_EQ(fileAnalogsDates[iAnalog], resDates(rowTargetDate, iAnalog));
                EXPECT_FLOAT_EQ(fileAnalogsValues[iAnalog], resValues(rowTargetDate, iAnalog));
                EXPECT_NEAR(fileAnalogsCriteria[iAnalog], resCriteria(rowTargetDate, iAnalog), 0.1);
            }
        }

        // The CRPS tolerence is huge, as it is not processed with the same P10 !
        if (!shortVersion) {
            EXPECT_NEAR(fileScoreCRPS, resScoreCRPS(rowTargetDate), 0.1);
            EXPECT_NEAR(fileScoreCRPSaccuracy, resScoreCRPSaccuracy(rowTargetDate), 0.1);
            EXPECT_NEAR(fileScoreCRPSsharpness, resScoreCRPSsharpness(rowTargetDate), 0.1);
        }
    }

    if (!shortVersion) {
        EXPECT_FLOAT_EQ(asMean(&resScoreCRPS[0], &resScoreCRPS[resScoreCRPS.size() - 1]), scoreFinal);
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration
}

#ifdef USE_CUDA
TEST(MethodCalibrator, Ref1Cuda)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);

    wxString paramsFile = "parameters_calibration_R1_shorter.xml";

    // Get parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append(paramsFile);
    asParametersCalibration params;
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibratorCPU, calibratorGPU;
    asResultsDates anaDatesCPU, anaDatesGPU;

    try {
        int step = 0;
        bool containsNaNs = false;
        wxString predictorFilePath = wxFileName::GetCwd();
        predictorFilePath.Append("/files/data-ncep-r1/others/");
        calibratorCPU.SetPredictorDataDir(predictorFilePath);
        calibratorGPU.SetPredictorDataDir(predictorFilePath);

        // CPU
        wxStopWatch sw1;
        pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
        ASSERT_TRUE(calibratorCPU.GetAnalogsDates(anaDatesCPU, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        printf(_("        ---> CPU (multithreaded) time: %.3f sec\n"), float(sw1.Time()) / 1000.0f);

        // GPU
        wxStopWatch sw2;
        pConfig->Write("/Processing/Method", (int) asCUDA);
        ASSERT_TRUE(calibratorGPU.GetAnalogsDates(anaDatesGPU, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        printf(_("        ---> GPU time: %.3f sec\n"), float(sw2.Time()) / 1000.0f);

    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    a1f resultsTargetDatesCPU(anaDatesCPU.GetTargetDates());
    a2f resultsCriteriaCPU(anaDatesCPU.GetAnalogsCriteria());
    a2f resultsAnalogDatesCPU(anaDatesCPU.GetAnalogsDates());
    a1f resultsTargetDatesGPU(anaDatesGPU.GetTargetDates());
    a2f resultsCriteriaGPU(anaDatesGPU.GetAnalogsCriteria());
    a2f resultsAnalogDatesGPU(anaDatesGPU.GetAnalogsDates());

    // Check results
    for (int i = 0; i < resultsCriteriaCPU.rows(); ++i) {
        EXPECT_FLOAT_EQ(resultsTargetDatesCPU(i), resultsTargetDatesGPU(i));
        for (int j = 0; j < resultsCriteriaCPU.cols(); ++j) {
            EXPECT_FLOAT_EQ(resultsCriteriaCPU(i, j), resultsCriteriaGPU(i, j));
            if (abs(resultsCriteriaCPU(i, j) - resultsCriteriaGPU(i, j)) > 0.00001) {
                EXPECT_FLOAT_EQ(resultsAnalogDatesCPU(i, j), resultsAnalogDatesGPU(i, j));
            }
        }
    }

}
#endif

TEST(MethodCalibrator, Ref1Multithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1Standard)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1CalibPeriodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref1("parameters_calibration_R1_calib_period.xml", true);
}

TEST(MethodCalibrator, Ref1CalibPeriodStandard)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    Ref1("parameters_calibration_R1_calib_period.xml", true);
}

void Ref2(const wxString &paramsFile, bool shortVersion)
{
    // Create predictand database
    asPredictandPrecipitation *predictand = new asPredictandPrecipitation(asPredictand::Precipitation,
                                                                          asPredictand::Daily, asPredictand::Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append(paramsFile);
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator.SetPredictorDataDir(predictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsDates anaDates;
    asResultsDates anaSubDates;
    asResultsValues anaValues;
    asResultsScores anaScoresCRPS;
    asResultsScores anaScoresCRPSsharpness;
    asResultsScores anaScoresCRPSaccuracy;
    asResultsTotalScore anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;

        EXPECT_TRUE(calibrator.GetAnalogsDates(anaDates, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        step++;
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates, &params, anaDates, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, &params, anaSubDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPS, &params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsTotalScore(anaScoreFinal, &params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetScoreName("CRPSsharpnessEP");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSsharpness, &params, anaValues, step));
        params.SetScoreName("CRPSaccuracyEP");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSaccuracy, &params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    a1f resultsTargetDates(anaSubDates.GetTargetDates());
    a1f resultsTargetValues(anaValues.GetTargetValues()[0]);
    a2f resultsCriteria(anaSubDates.GetAnalogsCriteria());
    a2f resultsDates(anaSubDates.GetAnalogsDates());
    a2f resultsValues(anaValues.GetAnalogsValues()[0]);
    a1f resultsScoreCRPS(anaScoresCRPS.GetScores());
    a1f resultsScoreCRPSsharpness(anaScoresCRPSsharpness.GetScores());
    a1f resultsScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetScores());

    float scoreFinal = 0;
    if (!shortVersion) {
        scoreFinal = anaScoreFinal.GetScore();
    }

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    int nbtests = 0;
    if (!shortVersion) {
        resultFilePath.Append("/files/score_05.txt");
        nbtests = 30;
    } else {
        resultFilePath.Append("/files/score_07.txt");
        nbtests = 4;
    }
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int nanalogs = 30;
    a1f fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int iTest = 0; iTest < nbtests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Get target date from file
        int day = file.GetInt();
        int month = file.GetInt();
        int year = file.GetInt();
        float fileTargetDate = (float) asTime::GetMJD(year, month, day);
        float fileTargetValue = (float) sqrt(file.GetFloat() / P10);

        file.SkipLines(2);

        // Get analogs from file
        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[iAnalog] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[iAnalog] = 0;
            }
            fileAnalogsValues[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[iAnalog] = file.GetFloat();

            file.SkipLines(1);
        }

        float fileScoreCRPS = 0, fileScoreCRPSaccuracy = 0, fileScoreCRPSsharpness = 0;

        if (!shortVersion) {
            file.SkipLines(2);
            fileScoreCRPS = file.GetFloat();
            fileScoreCRPSaccuracy = file.GetFloat();
            fileScoreCRPSsharpness = file.GetFloat();
            file.SkipLines(1);
        } else {
            file.SkipLines(3);
        }

        // Find target date in the array
        int rowTargetDate = asFindClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows() - 1],
                                          fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            if (fileAnalogsDates[iAnalog] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[iAnalog], resultsDates(rowTargetDate, iAnalog));
                EXPECT_FLOAT_EQ(fileAnalogsValues[iAnalog], resultsValues(rowTargetDate, iAnalog));
                EXPECT_NEAR(fileAnalogsCriteria[iAnalog], resultsCriteria(rowTargetDate, iAnalog), 0.1);

                if (std::abs(fileAnalogsDates[iAnalog] - resultsDates(rowTargetDate, iAnalog)) > 0.0001) {
                    wxPrintf(("Date is %s and should be %s.\n"),
                             asTime::GetStringTime(resultsDates(rowTargetDate, iAnalog)),
                             asTime::GetStringTime(fileAnalogsDates[iAnalog]));
                }
            }
        }

        if (!shortVersion) {
            // The CRPS tolerence is huge, as it is not processed with the same P10 !
            EXPECT_NEAR(fileScoreCRPS, resultsScoreCRPS(rowTargetDate), 0.1);
            EXPECT_NEAR(fileScoreCRPSaccuracy, resultsScoreCRPSaccuracy(rowTargetDate), 0.1);
            EXPECT_NEAR(fileScoreCRPSsharpness, resultsScoreCRPSsharpness(rowTargetDate), 0.1);
        }
    }

    if (!shortVersion) {
        EXPECT_FLOAT_EQ(asMean(&resultsScoreCRPS[0], &resultsScoreCRPS[resultsScoreCRPS.size() - 1]), scoreFinal);
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration
}

TEST(MethodCalibrator, Ref2Multithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref2("parameters_calibration_R2_full.xml", false);
}

TEST(MethodCalibrator, Ref2Standard)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    Ref2("parameters_calibration_R2_full.xml", false);
}

TEST(MethodCalibrator, Ref2CalibPeriodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(MethodCalibrator, Ref2CalibPeriodStandard)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    Ref2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(MethodCalibrator, PreloadingSimple)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    // Get parameters
    asParametersCalibration paramsStd;
    asParametersCalibration paramsPreload;
    wxString paramsFilePathStd = wxFileName::GetCwd();
    wxString paramsFilePathPreload = wxFileName::GetCwd();
    paramsFilePathStd.Append("/files/parameters_calibration_compare_no_preload.xml");
    paramsFilePathPreload.Append("/files/parameters_calibration_compare_preload.xml");
    ASSERT_TRUE(paramsStd.LoadFromFile(paramsFilePathStd));
    ASSERT_TRUE(paramsPreload.LoadFromFile(paramsFilePathPreload));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator1;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator1.SetPredictorDataDir(predictorFilePath);
    calibrator1.SetPredictandDB(nullptr);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asResultsDates anaDatesStd;
    asResultsDates anaDatesPreload;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesStd, &paramsStd, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesPreload, &paramsPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    a2f datesStd = anaDatesStd.GetAnalogsDates();
    a2f datesPreload = anaDatesPreload.GetAnalogsDates();
    a2f criteriaStd = anaDatesStd.GetAnalogsCriteria();
    a2f criteriaPreload = anaDatesPreload.GetAnalogsCriteria();

    EXPECT_EQ(datesStd.cols(), datesPreload.cols());
    EXPECT_EQ(datesStd.rows(), datesPreload.rows());
    EXPECT_EQ(criteriaStd.cols(), criteriaPreload.cols());
    EXPECT_EQ(criteriaStd.rows(), criteriaPreload.rows());

    for (int i = 0; i < datesStd.rows(); i++) {
        for (int j = 0; j < datesStd.cols(); j++) {
            EXPECT_EQ(datesStd.coeff(i, j), datesPreload.coeff(i, j));
            EXPECT_EQ(criteriaStd.coeff(i, j), criteriaPreload.coeff(i, j));
        }
    }
}

TEST(MethodCalibrator, PreloadingWithPreprocessing)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    // Get parameters
    asParametersCalibration paramsStd;
    asParametersCalibration paramsPreload;
    wxString paramsFilePathStd = wxFileName::GetCwd();
    wxString paramsFilePathPreload = wxFileName::GetCwd();
    paramsFilePathStd.Append("/files/parameters_calibration_compare_preproc_no_preload.xml");
    paramsFilePathPreload.Append("/files/parameters_calibration_compare_preproc_preload.xml");
    ASSERT_TRUE(paramsStd.LoadFromFile(paramsFilePathStd));
    ASSERT_TRUE(paramsPreload.LoadFromFile(paramsFilePathPreload));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator1;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator1.SetPredictorDataDir(predictorFilePath);
    calibrator1.SetPredictandDB(nullptr);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asResultsDates anaDatesStd;
    asResultsDates anaDatesPreload;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesStd, &paramsStd, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesPreload, &paramsPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    a1f targetDatesStd = anaDatesStd.GetTargetDates();
    a1f targetDatesPreload = anaDatesPreload.GetTargetDates();
    int targetDatesSize = (int) wxMax(targetDatesStd.cols(), targetDatesStd.rows());
    for (int i = 0; i < targetDatesSize; i++) {
        EXPECT_EQ(targetDatesStd[i], targetDatesPreload[i]);
    }

    a2f datesStd = anaDatesStd.GetAnalogsDates();
    a2f datesPreload = anaDatesPreload.GetAnalogsDates();
    a2f criteriaStd = anaDatesStd.GetAnalogsCriteria();
    a2f criteriaPreload = anaDatesPreload.GetAnalogsCriteria();

    EXPECT_EQ(datesStd.cols(), datesPreload.cols());
    EXPECT_EQ(datesStd.rows(), datesPreload.rows());
    EXPECT_EQ(criteriaStd.cols(), criteriaPreload.cols());
    EXPECT_EQ(criteriaStd.rows(), criteriaPreload.rows());

    for (int i = 0; i < datesStd.rows(); i++) {
        for (int j = 0; j < datesStd.cols(); j++) {
            EXPECT_EQ(datesStd.coeff(i, j), datesPreload.coeff(i, j));
            EXPECT_EQ(criteriaStd.coeff(i, j), criteriaPreload.coeff(i, j));
        }
    }
}

TEST(MethodCalibrator, ComplexPredictorHours)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    // Get parameters
    asParametersCalibration paramsStd;
    asParametersCalibration paramsPreload;
    wxString paramsFilePathStd = wxFileName::GetCwd();
    wxString paramsFilePathPreload = wxFileName::GetCwd();
    paramsFilePathStd.Append("/files/parameters_calibration_compare_no_preload.xml");
    paramsFilePathPreload.Append("/files/parameters_calibration_compare_preload.xml");
    ASSERT_TRUE(paramsStd.LoadFromFile(paramsFilePathStd));
    ASSERT_TRUE(paramsPreload.LoadFromFile(paramsFilePathPreload));

    // Set +- 24h
    asParametersCalibration paramsStdPos = paramsStd;
    asParametersCalibration paramsStdNeg = paramsStd;
    asParametersCalibration paramsPreloadPos = paramsPreload;
    asParametersCalibration paramsPreloadNeg = paramsPreload;

    vd hour;

    hour.push_back(36);
    paramsStdPos.SetPredictorHoursVector(0, 0, hour);
    hour[0] = 48;
    paramsStdPos.SetPredictorHoursVector(0, 1, hour);

    hour[0] = -12;
    paramsStdNeg.SetPredictorHoursVector(0, 0, hour);
    hour[0] = 0;
    paramsStdNeg.SetPredictorHoursVector(0, 1, hour);

    hour[0] = 36;
    paramsPreloadPos.SetPredictorHoursVector(0, 0, hour);
    hour[0] = 48;
    paramsPreloadPos.SetPredictorHoursVector(0, 1, hour);

    hour[0] = -12;
    paramsPreloadNeg.SetPredictorHoursVector(0, 0, hour);
    hour[0] = 0;
    paramsPreloadNeg.SetPredictorHoursVector(0, 1, hour);

    paramsStdPos.InitValues();
    paramsStdPos.FixTimeLimits();
    paramsStdNeg.InitValues();
    paramsStdNeg.FixTimeLimits();
    paramsPreloadPos.SetPreloadingProperties();
    paramsPreloadPos.InitValues();
    paramsPreloadPos.FixTimeLimits();
    paramsPreloadNeg.SetPreloadingProperties();
    paramsPreloadNeg.InitValues();
    paramsPreloadNeg.FixTimeLimits();

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator.SetPredictorDataDir(predictorFilePath);
    calibrator.SetPredictandDB(nullptr);
    asMethodCalibratorSingle calibrator1 = calibrator;
    asMethodCalibratorSingle calibrator2 = calibrator;
    asMethodCalibratorSingle calibrator3 = calibrator;
    asResultsDates anaDatesStd, anaDatesStdPos, anaDatesStdNeg;
    asResultsDates anaDatesPreload, anaDatesPreloadPos, anaDatesPreloadNeg;

    try {
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDatesStd, &paramsStd, 0, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDatesStdPos, &paramsStdPos, 0, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDatesStdNeg, &paramsStdNeg, 0, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesPreload, &paramsPreload, 0, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesPreloadPos, &paramsPreloadPos, 0, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator3.GetAnalogsDates(anaDatesPreloadNeg, &paramsPreloadNeg, 0, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }
}

void Ref1Preloading()
{
    // Create predictand database
    asPredictandPrecipitation *predictand = new asPredictandPrecipitation(asPredictand::Precipitation,
                                                                          asPredictand::Daily, asPredictand::Station);

    wxString datasetPredictandFilePath = wxFileName::GetCwd();
    datasetPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(datasetPredictandFilePath, dataFileDir, patternFileDir);

    float P10 = 68.42240f;

    // Get parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R1_preload.xml");
    asParametersCalibration params;
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Force values
    params.SetPredictorXptsnb(0, 0, 9);
    params.SetPredictorYmin(0, 1, 40);

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    asResultsDates anaDates;
    asResultsValues anaValues;
    asResultsScores anaScoresCRPS;
    asResultsScores anaScoresCRPSsharpness;
    asResultsScores anaScoresCRPSaccuracy;
    asResultsTotalScore anaScoreFinal;

    try {
        bool containsNaNs = false;
        int step = 0;
        wxString predictorFilePath = wxFileName::GetCwd();
        predictorFilePath.Append("/files/data-ncep-r1/others/");
        calibrator.SetPredictorDataDir(predictorFilePath);
        wxASSERT(predictand);
        calibrator.SetPredictandDB(predictand);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, &params, anaDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPS, &params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsTotalScore(anaScoreFinal, &params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetScoreName("CRPSsharpnessAR");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSsharpness, &params, anaValues, step));
        params.SetScoreName("CRPSaccuracyAR");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSaccuracy, &params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }


    // Extract data
    a1f resultsTargetDates(anaDates.GetTargetDates());
    a1f resultsTargetValues(anaValues.GetTargetValues()[0]);
    a2f resultsCriteria(anaDates.GetAnalogsCriteria());
    a2f resultsDates(anaDates.GetAnalogsDates());
    a2f resultsValues(anaValues.GetAnalogsValues()[0]);
    a1f resultsScoreCRPS(anaScoresCRPS.GetScores());
    a1f resultsScoreCRPSsharpness(anaScoresCRPSsharpness.GetScores());
    a1f resultsScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/score_06.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 20; //43

    // Resize the containers
    int nanalogs = 50;
    a1f fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int iTest = 0; iTest < nbtests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Get target date from file
        int day = file.GetInt();
        int month = file.GetInt();
        int year = file.GetInt();
        float fileTargetDate = (float) asTime::GetMJD(year, month, day);
        float fileTargetValue = (float) sqrt(file.GetFloat() / P10);

        file.SkipLines(2);

        // Get analogs from file
        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[iAnalog] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[iAnalog] = 0;
            }
            fileAnalogsValues[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            fileAnalogsCriteria[iAnalog] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asFindClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows() - 1],
                                          fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            if (fileAnalogsDates[iAnalog] > 0) // If we have the data
            {
                EXPECT_FLOAT_EQ(fileAnalogsDates[iAnalog], resultsDates(rowTargetDate, iAnalog));
                EXPECT_FLOAT_EQ(fileAnalogsValues[iAnalog], resultsValues(rowTargetDate, iAnalog));
                EXPECT_NEAR(fileAnalogsCriteria[iAnalog], resultsCriteria(rowTargetDate, iAnalog), 0.1);
            }
        }
    }

    file.Close();

    // predictand pointer deleted by asMethodCalibration
}

TEST(MethodCalibrator, Ref1PreloadingMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref1Preloading();
}

void Ref1PreloadingSubset()
{
    // Create predictand database
    asPredictandPrecipitation *predictand = new asPredictandPrecipitation(asPredictand::Precipitation,
                                                                          asPredictand::Daily, asPredictand::Station);

    wxString datasetPredictandFilePath = wxFileName::GetCwd();
    datasetPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(datasetPredictandFilePath, dataFileDir, patternFileDir);

    // Get parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R1_preload.xml");
    asParametersCalibration params;
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Force values
    params.SetPredictorXptsnb(0, 0, 9);
    params.SetPredictorYmin(0, 1, 42.5);
    params.SetPredictorYptsnb(0, 1, 4);

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    asResultsDates anaDates;
    asResultsValues anaValues;
    asResultsScores anaScoresCRPS;
    asResultsScores anaScoresCRPSsharpness;
    asResultsScores anaScoresCRPSaccuracy;
    asResultsTotalScore anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        wxString predictorFilePath = wxFileName::GetCwd();
        predictorFilePath.Append("/files/data-ncep-r1/others/");
        calibrator.SetPredictorDataDir(predictorFilePath);
        wxASSERT(predictand);
        calibrator.SetPredictandDB(predictand);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, &params, anaDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPS, &params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsTotalScore(anaScoreFinal, &params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetScoreName("CRPSsharpnessAR");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSsharpness, &params, anaValues, step));
        params.SetScoreName("CRPSaccuracyAR");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSaccuracy, &params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // No unit test possible, as results will differ from Grenoble's results.

    // predictand pointer deleted by asMethodCalibration
}

TEST(MethodCalibrator, Ref1PreloadingSubsetMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref1PreloadingSubset();
}

TEST(MethodCalibrator, SmallerSpatialArea)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    // Get parameters
    asParametersCalibration paramsNoPreprocNoPreload;
    asParametersCalibration paramsNoPreprocPreload;
    asParametersCalibration paramsPreprocNoPreload;
    asParametersCalibration paramsPreprocPreload;
    wxString paramsFilePathNoPreprocNoPreload = wxFileName::GetCwd();
    wxString paramsFilePathNoPreprocPreload = wxFileName::GetCwd();
    wxString paramsFilePathPreprocNoPreload = wxFileName::GetCwd();
    wxString paramsFilePathPreprocPreload = wxFileName::GetCwd();
    paramsFilePathNoPreprocNoPreload.Append("/files/parameters_calibration_compare_smaller_no_preproc_no_preload.xml");
    paramsFilePathNoPreprocPreload.Append("/files/parameters_calibration_compare_smaller_no_preproc_preload.xml");
    paramsFilePathPreprocNoPreload.Append("/files/parameters_calibration_compare_smaller_preproc_no_preload.xml");
    paramsFilePathPreprocPreload.Append("/files/parameters_calibration_compare_smaller_preproc_preload.xml");
    ASSERT_TRUE(paramsNoPreprocNoPreload.LoadFromFile(paramsFilePathNoPreprocNoPreload));
    ASSERT_TRUE(paramsNoPreprocPreload.LoadFromFile(paramsFilePathNoPreprocPreload));
    ASSERT_TRUE(paramsPreprocNoPreload.LoadFromFile(paramsFilePathPreprocNoPreload));
    ASSERT_TRUE(paramsPreprocPreload.LoadFromFile(paramsFilePathPreprocPreload));

    // Change spatial windows
    paramsNoPreprocNoPreload.SetPredictorXmin(0, 0, 5);
    paramsNoPreprocNoPreload.SetPredictorXmin(0, 1, 5);
    paramsNoPreprocNoPreload.SetPredictorXptsnb(0, 0, 3);
    paramsNoPreprocNoPreload.SetPredictorXptsnb(0, 1, 3);
    paramsNoPreprocNoPreload.SetPredictorYmin(0, 0, 42.5);
    paramsNoPreprocNoPreload.SetPredictorYmin(0, 1, 42.5);
    paramsNoPreprocNoPreload.SetPredictorYptsnb(0, 0, 3);
    paramsNoPreprocNoPreload.SetPredictorYptsnb(0, 1, 3);

    paramsNoPreprocPreload.SetPredictorXmin(0, 0, 5);
    paramsNoPreprocPreload.SetPredictorXmin(0, 1, 5);
    paramsNoPreprocPreload.SetPredictorXptsnb(0, 0, 3);
    paramsNoPreprocPreload.SetPredictorXptsnb(0, 1, 3);
    paramsNoPreprocPreload.SetPredictorYmin(0, 0, 42.5);
    paramsNoPreprocPreload.SetPredictorYmin(0, 1, 42.5);
    paramsNoPreprocPreload.SetPredictorYptsnb(0, 0, 3);
    paramsNoPreprocPreload.SetPredictorYptsnb(0, 1, 3);

    paramsPreprocNoPreload.SetPredictorXmin(0, 0, 5);
    paramsPreprocNoPreload.SetPredictorXmin(0, 1, 5);
    paramsPreprocNoPreload.SetPredictorXptsnb(0, 0, 3);
    paramsPreprocNoPreload.SetPredictorXptsnb(0, 1, 3);
    paramsPreprocNoPreload.SetPredictorYmin(0, 0, 42.5);
    paramsPreprocNoPreload.SetPredictorYmin(0, 1, 42.5);
    paramsPreprocNoPreload.SetPredictorYptsnb(0, 0, 3);
    paramsPreprocNoPreload.SetPredictorYptsnb(0, 1, 3);

    paramsPreprocPreload.SetPredictorXmin(0, 0, 5);
    paramsPreprocPreload.SetPredictorXmin(0, 1, 5);
    paramsPreprocPreload.SetPredictorXptsnb(0, 0, 3);
    paramsPreprocPreload.SetPredictorXptsnb(0, 1, 3);
    paramsPreprocPreload.SetPredictorYmin(0, 0, 42.5);
    paramsPreprocPreload.SetPredictorYmin(0, 1, 42.5);
    paramsPreprocPreload.SetPredictorYptsnb(0, 0, 3);
    paramsPreprocPreload.SetPredictorYptsnb(0, 1, 3);

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator1;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator1.SetPredictorDataDir(predictorFilePath);
    calibrator1.SetPredictandDB(nullptr);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asMethodCalibratorSingle calibrator3 = calibrator1;
    asMethodCalibratorSingle calibrator4 = calibrator1;
    asResultsDates anaDatesNoPreprocNoPreload;
    asResultsDates anaDatesNoPreprocPreload;
    asResultsDates anaDatesPreprocNoPreload;
    asResultsDates anaDatesPreprocPreload;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesNoPreprocNoPreload, &paramsNoPreprocNoPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesNoPreprocPreload, &paramsNoPreprocPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator3.GetAnalogsDates(anaDatesPreprocNoPreload, &paramsPreprocNoPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator4.GetAnalogsDates(anaDatesPreprocPreload, &paramsPreprocPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    a2f datesNoPreprocNoPreload = anaDatesNoPreprocNoPreload.GetAnalogsDates();
    a2f datesNoPreprocPreload = anaDatesNoPreprocPreload.GetAnalogsDates();
    a2f datesPreprocNoPreload = anaDatesPreprocNoPreload.GetAnalogsDates();
    a2f datesPreprocPreload = anaDatesPreprocPreload.GetAnalogsDates();
    a2f criteriaNoPreprocNoPreload = anaDatesNoPreprocNoPreload.GetAnalogsCriteria();
    a2f criteriaNoPreprocPreload = anaDatesNoPreprocPreload.GetAnalogsCriteria();
    a2f criteriaPreprocNoPreload = anaDatesPreprocNoPreload.GetAnalogsCriteria();
    a2f criteriaPreprocPreload = anaDatesPreprocPreload.GetAnalogsCriteria();

    EXPECT_EQ(datesNoPreprocNoPreload.cols(), datesNoPreprocPreload.cols());
    EXPECT_EQ(datesNoPreprocNoPreload.rows(), datesNoPreprocPreload.rows());
    EXPECT_EQ(datesNoPreprocNoPreload.cols(), datesPreprocNoPreload.cols());
    EXPECT_EQ(datesNoPreprocNoPreload.rows(), datesPreprocNoPreload.rows());
    EXPECT_EQ(datesNoPreprocNoPreload.cols(), datesPreprocPreload.cols());
    EXPECT_EQ(datesNoPreprocNoPreload.rows(), datesPreprocPreload.rows());

    EXPECT_EQ(criteriaNoPreprocNoPreload.cols(), criteriaNoPreprocPreload.cols());
    EXPECT_EQ(criteriaNoPreprocNoPreload.rows(), criteriaNoPreprocPreload.rows());
    EXPECT_EQ(criteriaNoPreprocNoPreload.cols(), criteriaPreprocNoPreload.cols());
    EXPECT_EQ(criteriaNoPreprocNoPreload.rows(), criteriaPreprocNoPreload.rows());
    EXPECT_EQ(criteriaNoPreprocNoPreload.cols(), criteriaPreprocPreload.cols());
    EXPECT_EQ(criteriaNoPreprocNoPreload.rows(), criteriaPreprocPreload.rows());

    for (int i = 0; i < datesNoPreprocNoPreload.rows(); i++) {
        for (int j = 0; j < datesNoPreprocNoPreload.cols(); j++) {
            EXPECT_EQ(datesNoPreprocNoPreload.coeff(i, j), datesNoPreprocPreload.coeff(i, j));
            EXPECT_EQ(criteriaNoPreprocNoPreload.coeff(i, j), criteriaNoPreprocPreload.coeff(i, j));
            EXPECT_EQ(datesNoPreprocNoPreload.coeff(i, j), datesPreprocNoPreload.coeff(i, j));
            EXPECT_EQ(criteriaNoPreprocNoPreload.coeff(i, j), criteriaPreprocNoPreload.coeff(i, j));
            EXPECT_EQ(datesNoPreprocNoPreload.coeff(i, j), datesPreprocPreload.coeff(i, j));
            EXPECT_EQ(criteriaNoPreprocNoPreload.coeff(i, j), criteriaPreprocPreload.coeff(i, j));
        }
    }
}

void Ref2Preloading()
{
    // Create predictand database
    asPredictandPrecipitation *predictand = new asPredictandPrecipitation(asPredictand::Precipitation,
                                                                          asPredictand::Daily, asPredictand::Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R2_preload.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator.SetPredictorDataDir(predictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsDates anaDates;
    asResultsDates anaSubDates;
    asResultsValues anaValues;
    asResultsScores anaScoresCRPS;
    asResultsScores anaScoresCRPSsharpness;
    asResultsScores anaScoresCRPSaccuracy;
    asResultsTotalScore anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        step++;
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates, &params, anaDates, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, &params, anaSubDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPS, &params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsTotalScore(anaScoreFinal, &params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetScoreName("CRPSsharpnessEP");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSsharpness, &params, anaValues, step));
        params.SetScoreName("CRPSaccuracyEP");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSaccuracy, &params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    a1f resultsTargetDates(anaSubDates.GetTargetDates());
    a1f resultsTargetValues(anaValues.GetTargetValues()[0]);
    a2f resultsCriteria(anaSubDates.GetAnalogsCriteria());
    a2f resultsDates(anaSubDates.GetAnalogsDates());
    a2f resultsValues(anaValues.GetAnalogsValues()[0]);
    a1f resultsScoreCRPS(anaScoresCRPS.GetScores());
    a1f resultsScoreCRPSsharpness(anaScoresCRPSsharpness.GetScores());
    a1f resultsScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/score_07.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 4;

    // Resize the containers
    int nanalogs = 30;
    a1f fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int iTest = 0; iTest < nbtests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Get target date from file
        int day = file.GetInt();
        int month = file.GetInt();
        int year = file.GetInt();
        float fileTargetDate = asTime::GetMJD(year, month, day);
        float fileTargetValue = sqrt(file.GetFloat() / P10);

        file.SkipLines(2);

        // Get analogs from file
        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[iAnalog] = asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[iAnalog] = 0;
            }
            fileAnalogsValues[iAnalog] = sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[iAnalog] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asFindClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows() - 1],
                                          fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            if (fileAnalogsDates[iAnalog] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[iAnalog], resultsDates(rowTargetDate, iAnalog));
                EXPECT_FLOAT_EQ(fileAnalogsValues[iAnalog], resultsValues(rowTargetDate, iAnalog));
                EXPECT_NEAR(fileAnalogsCriteria[iAnalog], resultsCriteria(rowTargetDate, iAnalog), 0.1);
            }
        }
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration
}

TEST(MethodCalibrator, Ref2PreloadingMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref2Preloading();
}

TEST(MethodCalibrator, Ref2PreloadingStandard)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asSTANDARD);

    Ref2Preloading();
}

void Ref2SavingIntermediateResults()
{
    // Create predictand database
    asPredictandPrecipitation *predictand = new asPredictandPrecipitation(asPredictand::Precipitation,
                                                                          asPredictand::Daily, asPredictand::Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R2_calib_period.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator.SetPredictorDataDir(predictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsDates anaDates1, anaDates2;
    asResultsDates anaSubDates1, anaSubDates2;
    asResultsValues anaValues1, anaValues2;
    asResultsScores anaScoresCRPS1, anaScoresCRPS2;
    asResultsTotalScore anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;

        // Create
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates1, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        anaDates1.Save();
        ASSERT_TRUE(asFile::Exists(anaDates1.GetFilePath()));
        // Reload
        anaDates2.Init(&params);
        ASSERT_TRUE(anaDates2.Load());
        ASSERT_TRUE(anaDates2.GetTargetDatesLength() > 0);
        step++;
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates1, &params, anaDates2, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        anaSubDates1.Save();
        ASSERT_TRUE(asFile::Exists(anaSubDates1.GetFilePath()));
        // Reload
        anaSubDates2.Init(&params);
        anaSubDates2.SetCurrentStep(1);
        ASSERT_TRUE(anaSubDates2.Load());
        ASSERT_TRUE(anaSubDates2.GetTargetDatesLength() > 0);
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues1, &params, anaSubDates2, step));
        anaValues1.Save();
        ASSERT_TRUE(asFile::Exists(anaValues1.GetFilePath()));
        // Reload
        anaValues2.Init(&params);
        anaValues2.SetCurrentStep(1);
        ASSERT_TRUE(anaValues2.Load());
        ASSERT_TRUE(anaValues2.GetTargetDatesLength() > 0);
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPS1, &params, anaValues2, step));
        anaScoresCRPS1.Save();
        ASSERT_TRUE(asFile::Exists(anaScoresCRPS1.GetFilePath()));
        // Reload
        anaScoresCRPS2.Init(&params);
        anaScoresCRPS2.SetCurrentStep(1);
        ASSERT_TRUE(anaScoresCRPS2.Load());
        ASSERT_TRUE(anaScoresCRPS2.GetTargetDates().size() > 0);
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsTotalScore(anaScoreFinal, &params, anaScoresCRPS2, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    a1f resultsTargetDates(anaSubDates2.GetTargetDates());
    a1f resultsTargetValues(anaValues2.GetTargetValues()[0]);
    a2f resultsCriteria(anaSubDates2.GetAnalogsCriteria());
    a2f resultsDates(anaSubDates2.GetAnalogsDates());
    a2f resultsValues(anaValues2.GetAnalogsValues()[0]);
    a1f resultsScoreCRPS(anaScoresCRPS2.GetScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/score_07.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 4;

    // Resize the containers
    int nanalogs = 30;
    a1f fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int iTest = 0; iTest < nbtests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Get target date from file
        int day = file.GetInt();
        int month = file.GetInt();
        int year = file.GetInt();
        float fileTargetDate = asTime::GetMJD(year, month, day);
        float fileTargetValue = sqrt(file.GetFloat() / P10);

        file.SkipLines(2);

        // Get analogs from file
        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[iAnalog] = asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[iAnalog] = 0;
            }
            fileAnalogsValues[iAnalog] = sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[iAnalog] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asFindClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows() - 1],
                                          fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            if (fileAnalogsDates[iAnalog] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[iAnalog], resultsDates(rowTargetDate, iAnalog));
                EXPECT_FLOAT_EQ(fileAnalogsValues[iAnalog], resultsValues(rowTargetDate, iAnalog));
                EXPECT_NEAR(fileAnalogsCriteria[iAnalog], resultsCriteria(rowTargetDate, iAnalog), 0.1);
            }
        }
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration
}

TEST(MethodCalibrator, Ref2SavingIntermediateResults)
{
    wxString tmpDir = asConfig::GetTempDir() + "IntermediateResults";

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    pConfig->Write("/Paths/OptimizerResultsDir", tmpDir);

    Ref2SavingIntermediateResults();

    wxDir::Remove(tmpDir, wxPATH_RMDIR_RECURSIVE);
}

void Ref2MergeByHalfAndMultiply()
{
    // Create predictand database
    asPredictandPrecipitation *predictand = new asPredictandPrecipitation(asPredictand::Precipitation,
                                                                          asPredictand::Daily, asPredictand::Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R2_calib_period_merge_by_half.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator.SetPredictorDataDir(predictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsDates anaDates;
    asResultsDates anaSubDates;
    asResultsValues anaValues;
    asResultsScores anaScoresCRPS;
    asResultsScores anaScoresCRPSsharpness;
    asResultsScores anaScoresCRPSaccuracy;
    asResultsTotalScore anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, &params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        step++;
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates, &params, anaDates, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, &params, anaSubDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPS, &params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsTotalScore(anaScoreFinal, &params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetScoreName("CRPSsharpnessEP");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSsharpness, &params, anaValues, step));
        params.SetScoreName("CRPSaccuracyEP");
        ASSERT_TRUE(calibrator.GetAnalogsScores(anaScoresCRPSaccuracy, &params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data&
    a1f resultsTargetDates(anaSubDates.GetTargetDates());
    a1f resultsTargetValues(anaValues.GetTargetValues()[0]);
    a2f resultsCriteria(anaSubDates.GetAnalogsCriteria());
    a2f resultsDates(anaSubDates.GetAnalogsDates());
    a2f resultsValues(anaValues.GetAnalogsValues()[0]);
    a1f resultsScoreCRPS(anaScoresCRPS.GetScores());
    a1f resultsScoreCRPSsharpness(anaScoresCRPSsharpness.GetScores());
    a1f resultsScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/score_07.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 4;

    // Resize the containers
    int nanalogs = 30;
    a1f fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int iTest = 0; iTest < nbtests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Get target date from file
        int day = file.GetInt();
        int month = file.GetInt();
        int year = file.GetInt();
        float fileTargetDate = (float) asTime::GetMJD(year, month, day);
        float fileTargetValue = (float) sqrt(file.GetFloat() / P10);

        file.SkipLines(2);

        // Get analogs from file
        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[iAnalog] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[iAnalog] = 0;
            }
            fileAnalogsValues[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[iAnalog] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asFindClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows() - 1],
                                          fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int iAnalog = 0; iAnalog < nanalogs; iAnalog++) {
            if (fileAnalogsDates[iAnalog] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[iAnalog], resultsDates(rowTargetDate, iAnalog));
                EXPECT_FLOAT_EQ(fileAnalogsValues[iAnalog], resultsValues(rowTargetDate, iAnalog));
                EXPECT_NEAR(fileAnalogsCriteria[iAnalog], resultsCriteria(rowTargetDate, iAnalog), 0.1);
            }
        }
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration
}

TEST(MethodCalibrator, Ref2MergeByHalfAndMultiply)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);

    Ref2MergeByHalfAndMultiply();
}

TEST(MethodCalibrator, PreloadingWithLevelCorrection)
{
    wxLogNull logNull;

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/General/ParallelDataLoad", false); // In order to avoid warning messages

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_preload_multiple_variables.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Check original parameters
    EXPECT_EQ(params.GetPredictorLevel(0, 0), 0);
    EXPECT_EQ(params.GetPredictorHour(0, 0), 6);
    EXPECT_EQ(params.GetPredictorLevel(0, 1), 0);
    EXPECT_EQ(params.GetPredictorHour(0, 1), 6);

    // Preload data
    asMethodCalibratorSingle calibrator;
    wxString predictorFilePath = wxFileName::GetCwd();
    predictorFilePath.Append("/files/data-ncep-r1/others/");
    calibrator.SetPredictorDataDir(predictorFilePath);
    calibrator.SetPredictandDB(nullptr);
    asResultsDates anaDates;
    asResultsDates anaSubDates;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, &params, step, containsNaNs));
        EXPECT_TRUE(anaDates.GetAnalogsDatesLength() > 0);
        EXPECT_FALSE(containsNaNs);
        step++;
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates, &params, anaDates, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Check corrected parameters
    EXPECT_TRUE(params.GetPredictorLevel(0, 0) > 0);
    EXPECT_EQ(params.GetPredictorHour(0, 0), 24);
    EXPECT_EQ(params.GetPredictorLevel(0, 1), 0);
    EXPECT_TRUE(params.GetPredictorHour(0, 1) > 6);

    // Check pointer sharing
    EXPECT_FALSE(calibrator.IsArchiveDataPointerCopy(0, 0, 1));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(0, 1, 0));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(0, 1, 1));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(0, 2, 0));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(0, 2, 1));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(0, 2, 2));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(0, 2, 3));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 0, 0));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 0, 1));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 0, 2));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 0, 3));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 1, 0));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 1, 1));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 2, 0));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 2, 1));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 2, 2));
    EXPECT_TRUE(calibrator.IsArchiveDataPointerCopy(1, 2, 3));

}

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
#include "asDataPredictandPrecipitation.h"
#include "asProcessor.h"
#include "asMethodCalibratorSingle.h"
#include "asResultsAnalogsDates.h"
#include "asResultsAnalogsValues.h"
#include "asResultsAnalogsForecastScores.h"
#include "asResultsAnalogsForecastScoreFinal.h"
#include "asFileAscii.h"
#include "gtest/gtest.h"


void Ref1(const wxString &paramsFile, bool shortVersion)
{
    // Create predictand database
    asDataPredictandPrecipitation *predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

    wxString datasetPredictandFilePath = wxFileName::GetCwd();
    datasetPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpDir = asConfig::CreateTempFileName("predictandDBtest");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(datasetPredictandFilePath, dataFileDir, patternFileDir, tmpDir);

    float P10 = 68.42240f;

    // Get parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append(paramsFile);
    asParametersCalibration params;
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScoresCRPS;
    asResultsAnalogsForecastScores anaScoresCRPSsharpness;
    asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        wxString dataPredictorFilePath = wxFileName::GetCwd();
        dataPredictorFilePath.Append("/files/");
        calibrator.SetPredictorDataDir(dataPredictorFilePath);
        wxASSERT(predictand);
        calibrator.SetPredictandDB(predictand);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, params, anaDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetForecastScoreName("CRPSsharpnessAR");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step));
        params.SetForecastScoreName("CRPSaccuracyAR");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    Array1DFloat resultsTargetDates(anaDates.GetTargetDates());
    Array1DFloat resultsTargetValues(anaValues.GetTargetValues()[0]);
    Array2DFloat resultsAnalogsCriteria(anaDates.GetAnalogsCriteria());
    Array2DFloat resultsAnalogsDates(anaDates.GetAnalogsDates());
    Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues()[0]);
    Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());
    float scoreFinal = anaScoreFinal.GetForecastScore();

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    int nbtests = 0;
    if (!shortVersion) {
        resultFilePath.Append("/files/forecast_score_04.txt");
        nbtests = 43;
    } else {
        resultFilePath.Append("/files/forecast_score_06.txt");
        nbtests = 20;
    }
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int i_test = 0; i_test < nbtests; i_test++) {
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
        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[i_ana] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[i_ana] = 0;
            }
            fileAnalogsValues[i_ana] = (float) sqrt(file.GetFloat() / P10);
            fileAnalogsCriteria[i_ana] = file.GetFloat();

            file.SkipLines(1);
        }

        float fileForecastScoreCRPS = 0, fileForecastScoreCRPSaccuracy = 0, fileForecastScoreCRPSsharpness = 0;

        if (!shortVersion) {
            file.SkipLines(2);
            fileForecastScoreCRPS = file.GetFloat();
            fileForecastScoreCRPSaccuracy = file.GetFloat();
            fileForecastScoreCRPSsharpness = file.GetFloat();
            file.SkipLines(1);
        } else {
            file.SkipLines(3);
        }

        // Find target date in the array
        int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0],
                                                              &resultsTargetDates[resultsTargetDates.rows() - 1],
                                                              fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            if (fileAnalogsDates[i_ana] > 0) // If we have the data
            {
                EXPECT_FLOAT_EQ(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana));
                EXPECT_FLOAT_EQ(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana));
                EXPECT_NEAR(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
            }
        }

        // The CRPS tolerence is huge, as it is not processed with the same P10 !
        if (!shortVersion) {
            EXPECT_NEAR(fileForecastScoreCRPS, resultsForecastScoreCRPS(rowTargetDate), 0.1);
            EXPECT_NEAR(fileForecastScoreCRPSaccuracy, resultsForecastScoreCRPSaccuracy(rowTargetDate), 0.1);
            EXPECT_NEAR(fileForecastScoreCRPSsharpness, resultsForecastScoreCRPSsharpness(rowTargetDate), 0.1);
        }
    }

    if (!shortVersion) {
        EXPECT_FLOAT_EQ(asTools::Mean(&resultsForecastScoreCRPS[0],
                                      &resultsForecastScoreCRPS[resultsForecastScoreCRPS.size() - 1]), scoreFinal);
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration

    asRemoveDir(tmpDir);
}

#ifdef USE_CUDA
TEST(Ref1ProcessingMethodCuda)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asCUDA);

    // Reset intermediate results option
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep1", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep2", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep3", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep4", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogValues", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveForecastScores", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveFinalForecastScore", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep1", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep2", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep3", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep4", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogValues", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadForecastScores", false);

    Ref1("parameters_calibration_R1_full.xml", false);
}
#endif

TEST(MethodCalibrator, Ref1MultithreadsWithLinAlgebra)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);

    // Reset intermediate results option
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep1", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep2", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep3", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep4", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogValues", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveForecastScores", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveFinalForecastScore", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep1", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep2", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep3", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep4", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogValues", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadForecastScores", false);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1MultithreadsWithLinAlgebraNoVar)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1MultithreadsWithCoeff)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1MultithreadsWithCoeffNoVar)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1Insert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1Splitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1("parameters_calibration_R1_full.xml", false);
}

TEST(MethodCalibrator, Ref1CalibPeriodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1("parameters_calibration_R1_calib_period.xml", true);
}

TEST(MethodCalibrator, Ref1MultithreadsWithLinAlgebraNoVarNoPreprocessing)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1("parameters_calibration_R1_full_no_preproc.xml", false);
}

TEST(MethodCalibrator, Ref1MultithreadsWithCoeffNoVarNoPreprocessing)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);

    Ref1("parameters_calibration_R1_full_no_preproc.xml", false);
}

TEST(MethodCalibrator, Ref1CalibPeriodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1("parameters_calibration_R1_calib_period.xml", true);
}

TEST(MethodCalibrator, Ref1CalibPeriodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1("parameters_calibration_R1_calib_period.xml", true);
}

void Ref2(const wxString &paramsFile, bool shortVersion)
{
    // Create predictand database
    asDataPredictandPrecipitation *predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpDir = asConfig::CreateTempFileName("predictandDBtest");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir, tmpDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append(paramsFile);
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator.SetPredictorDataDir(dataPredictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsAnalogsDates anaDates;
    asResultsAnalogsDates anaSubDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScoresCRPS;
    asResultsAnalogsForecastScores anaScoresCRPSsharpness;
    asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;

        EXPECT_TRUE(calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        step++;
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, params, anaSubDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetForecastScoreName("CRPSsharpnessEP");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step));
        params.SetForecastScoreName("CRPSaccuracyEP");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    Array1DFloat resultsTargetDates(anaSubDates.GetTargetDates());
    Array1DFloat resultsTargetValues(anaValues.GetTargetValues()[0]);
    Array2DFloat resultsAnalogsCriteria(anaSubDates.GetAnalogsCriteria());
    Array2DFloat resultsAnalogsDates(anaSubDates.GetAnalogsDates());
    Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues()[0]);
    Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

    float scoreFinal = 0;
    if (!shortVersion) {
        scoreFinal = anaScoreFinal.GetForecastScore();
    }

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    int nbtests = 0;
    if (!shortVersion) {
        resultFilePath.Append("/files/forecast_score_05.txt");
        nbtests = 30;
    } else {
        resultFilePath.Append("/files/forecast_score_07.txt");
        nbtests = 4;
    }
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int i_test = 0; i_test < nbtests; i_test++) {
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
        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[i_ana] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[i_ana] = 0;
            }
            fileAnalogsValues[i_ana] = (float) sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[i_ana] = file.GetFloat();

            file.SkipLines(1);
        }

        float fileForecastScoreCRPS = 0, fileForecastScoreCRPSaccuracy = 0, fileForecastScoreCRPSsharpness = 0;

        if (!shortVersion) {
            file.SkipLines(2);
            fileForecastScoreCRPS = file.GetFloat();
            fileForecastScoreCRPSaccuracy = file.GetFloat();
            fileForecastScoreCRPSsharpness = file.GetFloat();
            file.SkipLines(1);
        } else {
            file.SkipLines(3);
        }

        // Find target date in the array
        int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0],
                                                              &resultsTargetDates[resultsTargetDates.rows() - 1],
                                                              fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            if (fileAnalogsDates[i_ana] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana));
                EXPECT_FLOAT_EQ(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana));
                EXPECT_NEAR(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);

                if (std::abs(fileAnalogsDates[i_ana] - resultsAnalogsDates(rowTargetDate, i_ana)) > 0.0001) {
                    wxPrintf(("Date is %s and should be %s.\n"),
                             asTime::GetStringTime(resultsAnalogsDates(rowTargetDate, i_ana)),
                             asTime::GetStringTime(fileAnalogsDates[i_ana]));
                }
            }
        }

        if (!shortVersion) {
            // The CRPS tolerence is huge, as it is not processed with the same P10 !
            EXPECT_NEAR(fileForecastScoreCRPS, resultsForecastScoreCRPS(rowTargetDate), 0.1);
            EXPECT_NEAR(fileForecastScoreCRPSaccuracy, resultsForecastScoreCRPSaccuracy(rowTargetDate), 0.1);
            EXPECT_NEAR(fileForecastScoreCRPSsharpness, resultsForecastScoreCRPSsharpness(rowTargetDate), 0.1);
        }
    }

    if (!shortVersion) {
        EXPECT_FLOAT_EQ(asTools::Mean(&resultsForecastScoreCRPS[0],
                                      &resultsForecastScoreCRPS[resultsForecastScoreCRPS.size() - 1]), scoreFinal);
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration

    asRemoveDir(tmpDir);
}

TEST(MethodCalibrator, Ref2Multithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref2("parameters_calibration_R2_full.xml", false);
}

TEST(MethodCalibrator, Ref2Insert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref2("parameters_calibration_R2_full.xml", false);
}

TEST(MethodCalibrator, Ref2Splitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref2("parameters_calibration_R2_full.xml", false);
}

TEST(MethodCalibrator, Ref2CalibPeriodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(MethodCalibrator, Ref2CalibPeriodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(MethodCalibrator, Ref2CalibPeriodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(MethodCalibrator, PreloadingSimple)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

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
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator1.SetPredictorDataDir(dataPredictorFilePath);
    calibrator1.SetPredictandDB(NULL);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asResultsAnalogsDates anaDatesStd;
    asResultsAnalogsDates anaDatesPreload;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesPreload, paramsPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    Array2DFloat datesStd = anaDatesStd.GetAnalogsDates();
    Array2DFloat datesPreload = anaDatesPreload.GetAnalogsDates();
    Array2DFloat criteriaStd = anaDatesStd.GetAnalogsCriteria();
    Array2DFloat criteriaPreload = anaDatesPreload.GetAnalogsCriteria();

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
    pConfig->Write("/Processing/Method", (int) asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

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
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator1.SetPredictorDataDir(dataPredictorFilePath);
    calibrator1.SetPredictandDB(NULL);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asResultsAnalogsDates anaDatesStd;
    asResultsAnalogsDates anaDatesPreload;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesPreload, paramsPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    Array1DFloat targetDatesStd = anaDatesStd.GetTargetDates();
    Array1DFloat targetDatesPreload = anaDatesPreload.GetTargetDates();
    int targetDatesSize = (int) wxMax(targetDatesStd.cols(), targetDatesStd.rows());
    for (int i = 0; i < targetDatesSize; i++) {
        EXPECT_EQ(targetDatesStd[i], targetDatesPreload[i]);
    }

    Array2DFloat datesStd = anaDatesStd.GetAnalogsDates();
    Array2DFloat datesPreload = anaDatesPreload.GetAnalogsDates();
    Array2DFloat criteriaStd = anaDatesStd.GetAnalogsCriteria();
    Array2DFloat criteriaPreload = anaDatesPreload.GetAnalogsCriteria();

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

void Ref1Preloading()
{
    // Create predictand database
    asDataPredictandPrecipitation *predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

    wxString datasetPredictandFilePath = wxFileName::GetCwd();
    datasetPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpDir = asConfig::CreateTempFileName("predictandDBtest");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(datasetPredictandFilePath, dataFileDir, patternFileDir, tmpDir);

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
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScoresCRPS;
    asResultsAnalogsForecastScores anaScoresCRPSsharpness;
    asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    try {
        bool containsNaNs = false;
        int step = 0;
        wxString dataPredictorFilePath = wxFileName::GetCwd();
        dataPredictorFilePath.Append("/files/");
        calibrator.SetPredictorDataDir(dataPredictorFilePath);
        wxASSERT(predictand);
        calibrator.SetPredictandDB(predictand);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, params, anaDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetForecastScoreName("CRPSsharpnessAR");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step));
        params.SetForecastScoreName("CRPSaccuracyAR");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }


    // Extract data
    Array1DFloat resultsTargetDates(anaDates.GetTargetDates());
    Array1DFloat resultsTargetValues(anaValues.GetTargetValues()[0]);
    Array2DFloat resultsAnalogsCriteria(anaDates.GetAnalogsCriteria());
    Array2DFloat resultsAnalogsDates(anaDates.GetAnalogsDates());
    Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues()[0]);
    Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/forecast_score_06.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 20; //43

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int i_test = 0; i_test < nbtests; i_test++) {
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
        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[i_ana] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[i_ana] = 0;
            }
            fileAnalogsValues[i_ana] = (float) sqrt(file.GetFloat() / P10);
            fileAnalogsCriteria[i_ana] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0],
                                                              &resultsTargetDates[resultsTargetDates.rows() - 1],
                                                              fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            if (fileAnalogsDates[i_ana] > 0) // If we have the data
            {
                EXPECT_FLOAT_EQ(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana));
                EXPECT_FLOAT_EQ(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana));
                EXPECT_NEAR(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
            }
        }
    }

    file.Close();

    // predictand pointer deleted by asMethodCalibration

    asRemoveDir(tmpDir);
}

TEST(MethodCalibrator, Ref1PreloadingMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1Preloading();
}

void Ref1PreloadingSubset()
{
    // Create predictand database
    asDataPredictandPrecipitation *predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

    wxString datasetPredictandFilePath = wxFileName::GetCwd();
    datasetPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpDir = asConfig::CreateTempFileName("predictandDBtest");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(datasetPredictandFilePath, dataFileDir, patternFileDir, tmpDir);

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
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScoresCRPS;
    asResultsAnalogsForecastScores anaScoresCRPSsharpness;
    asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        wxString dataPredictorFilePath = wxFileName::GetCwd();
        dataPredictorFilePath.Append("/files/");
        calibrator.SetPredictorDataDir(dataPredictorFilePath);
        wxASSERT(predictand);
        calibrator.SetPredictandDB(predictand);
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, params, anaDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetForecastScoreName("CRPSsharpnessAR");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step));
        params.SetForecastScoreName("CRPSaccuracyAR");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // No unit test possible, as results will differ from Grenoble's results.

    // predictand pointer deleted by asMethodCalibration

    asRemoveDir(tmpDir);
}

TEST(MethodCalibrator, Ref1PreloadingSubsetMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref1PreloadingSubset();
}

TEST(MethodCalibrator, SmallerSpatialArea)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

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
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator1.SetPredictorDataDir(dataPredictorFilePath);
    calibrator1.SetPredictandDB(NULL);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asMethodCalibratorSingle calibrator3 = calibrator1;
    asMethodCalibratorSingle calibrator4 = calibrator1;
    asResultsAnalogsDates anaDatesNoPreprocNoPreload;
    asResultsAnalogsDates anaDatesNoPreprocPreload;
    asResultsAnalogsDates anaDatesPreprocNoPreload;
    asResultsAnalogsDates anaDatesPreprocPreload;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(
                calibrator1.GetAnalogsDates(anaDatesNoPreprocNoPreload, paramsNoPreprocNoPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesNoPreprocPreload, paramsNoPreprocPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator3.GetAnalogsDates(anaDatesPreprocNoPreload, paramsPreprocNoPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator4.GetAnalogsDates(anaDatesPreprocPreload, paramsPreprocPreload, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    Array2DFloat datesNoPreprocNoPreload = anaDatesNoPreprocNoPreload.GetAnalogsDates();
    Array2DFloat datesNoPreprocPreload = anaDatesNoPreprocPreload.GetAnalogsDates();
    Array2DFloat datesPreprocNoPreload = anaDatesPreprocNoPreload.GetAnalogsDates();
    Array2DFloat datesPreprocPreload = anaDatesPreprocPreload.GetAnalogsDates();
    Array2DFloat criteriaNoPreprocNoPreload = anaDatesNoPreprocNoPreload.GetAnalogsCriteria();
    Array2DFloat criteriaNoPreprocPreload = anaDatesNoPreprocPreload.GetAnalogsCriteria();
    Array2DFloat criteriaPreprocNoPreload = anaDatesPreprocNoPreload.GetAnalogsCriteria();
    Array2DFloat criteriaPreprocPreload = anaDatesPreprocPreload.GetAnalogsCriteria();

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
    asDataPredictandPrecipitation *predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpDir = asConfig::CreateTempFileName("predictandDBtest");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir, tmpDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R2_preload.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator.SetPredictorDataDir(dataPredictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsAnalogsDates anaDates;
    asResultsAnalogsDates anaSubDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScoresCRPS;
    asResultsAnalogsForecastScores anaScoresCRPSsharpness;
    asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        step++;
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, params, anaSubDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetForecastScoreName("CRPSsharpnessEP");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step));
        params.SetForecastScoreName("CRPSaccuracyEP");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    Array1DFloat resultsTargetDates(anaSubDates.GetTargetDates());
    Array1DFloat resultsTargetValues(anaValues.GetTargetValues()[0]);
    Array2DFloat resultsAnalogsCriteria(anaSubDates.GetAnalogsCriteria());
    Array2DFloat resultsAnalogsDates(anaSubDates.GetAnalogsDates());
    Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues()[0]);
    Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/forecast_score_07.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 4;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int i_test = 0; i_test < nbtests; i_test++) {
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
        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[i_ana] = 0;
            }
            fileAnalogsValues[i_ana] = sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[i_ana] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0],
                                                              &resultsTargetDates[resultsTargetDates.rows() - 1],
                                                              fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            if (fileAnalogsDates[i_ana] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana));
                EXPECT_FLOAT_EQ(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana));
                EXPECT_NEAR(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
            }
        }
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration

    asRemoveDir(tmpDir);
}

TEST(MethodCalibrator, Ref2PreloadingMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);

    Ref2Preloading();
}

TEST(MethodCalibrator, Ref2PreloadingInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);

    Ref2Preloading();
}

void Ref2SavingIntermediateResults()
{
    // Create predictand database
    asDataPredictandPrecipitation *predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpDir = asConfig::CreateTempFileName("predictandDBtest");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir, tmpDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R2_calib_period.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator.SetPredictorDataDir(dataPredictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsAnalogsDates anaDates1, anaDates2;
    asResultsAnalogsDates anaSubDates1, anaSubDates2;
    asResultsAnalogsValues anaValues1, anaValues2;
    asResultsAnalogsForecastScores anaScoresCRPS1, anaScoresCRPS2;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;

        // Create
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates1, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        // Reload
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates2, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        step++;
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates1, params, anaDates2, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        // Reload
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates2, params, anaDates2, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues1, params, anaSubDates2, step));
        // Reload
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues2, params, anaSubDates2, step));
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS1, params, anaValues2, step));
        // Reload
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS2, params, anaValues2, step));
        // Create
        ASSERT_TRUE(calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS2, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    Array1DFloat resultsTargetDates(anaSubDates2.GetTargetDates());
    Array1DFloat resultsTargetValues(anaValues2.GetTargetValues()[0]);
    Array2DFloat resultsAnalogsCriteria(anaSubDates2.GetAnalogsCriteria());
    Array2DFloat resultsAnalogsDates(anaSubDates2.GetAnalogsDates());
    Array2DFloat resultsAnalogsValues(anaValues2.GetAnalogsValues()[0]);
    Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS2.GetForecastScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/forecast_score_07.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 4;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int i_test = 0; i_test < nbtests; i_test++) {
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
        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[i_ana] = 0;
            }
            fileAnalogsValues[i_ana] = sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[i_ana] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0],
                                                              &resultsTargetDates[resultsTargetDates.rows() - 1],
                                                              fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            if (fileAnalogsDates[i_ana] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana));
                EXPECT_FLOAT_EQ(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana));
                EXPECT_NEAR(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
            }
        }
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration

    asRemoveDir(tmpDir);
}

TEST(MethodCalibrator, Ref2SavingIntermediateResults)
{
    wxString tmpDir = asConfig::GetTempDir() + "IntermediateResults";

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);

    pConfig->Write("/Paths/IntermediateResultsDir", tmpDir);

    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep1", true);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep2", true);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep3", true);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep4", true);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogValues", true);
    pConfig->Write("/Optimizer/IntermediateResults/SaveForecastScores", true);
    pConfig->Write("/Optimizer/IntermediateResults/SaveFinalForecastScore", true);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep1", true);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep2", true);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep3", true);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep4", true);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogValues", true);
    pConfig->Write("/Optimizer/IntermediateResults/LoadForecastScores", true);

    Ref2SavingIntermediateResults();

    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep1", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep2", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep3", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogDatesStep4", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveAnalogValues", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveForecastScores", false);
    pConfig->Write("/Optimizer/IntermediateResults/SaveFinalForecastScore", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep1", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep2", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep3", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogDatesStep4", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadAnalogValues", false);
    pConfig->Write("/Optimizer/IntermediateResults/LoadForecastScores", false);

    EXPECT_TRUE(wxDir::Remove(tmpDir, wxPATH_RMDIR_RECURSIVE));
}

void Ref2MergeByHalfAndMultiply()
{
    // Create predictand database
    asDataPredictandPrecipitation *predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

    wxString catalogPredictandFilePath = wxFileName::GetCwd();
    catalogPredictandFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpDir = asConfig::CreateTempFileName("predictandDBtest");

    predictand->SetIsSqrt(true);
    predictand->SetReturnPeriodNormalization(10);
    predictand->BuildPredictandDB(catalogPredictandFilePath, dataFileDir, patternFileDir, tmpDir);

    float P10 = 68.42240f;

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_R2_calib_period_merge_by_half.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator;
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator.SetPredictorDataDir(dataPredictorFilePath);
    wxASSERT(predictand);
    calibrator.SetPredictandDB(predictand);
    asResultsAnalogsDates anaDates;
    asResultsAnalogsDates anaSubDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScoresCRPS;
    asResultsAnalogsForecastScores anaScoresCRPSsharpness;
    asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        step++;
        ASSERT_TRUE(calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator.GetAnalogsValues(anaValues, params, anaSubDates, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step));
        ASSERT_TRUE(calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step));

        // Sharpness and Accuracy
        params.SetForecastScoreName("CRPSsharpnessEP");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step));
        params.SetForecastScoreName("CRPSaccuracyEP");
        ASSERT_TRUE(calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step));
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Extract data
    Array1DFloat resultsTargetDates(anaSubDates.GetTargetDates());
    Array1DFloat resultsTargetValues(anaValues.GetTargetValues()[0]);
    Array2DFloat resultsAnalogsCriteria(anaSubDates.GetAnalogsCriteria());
    Array2DFloat resultsAnalogsDates(anaSubDates.GetAnalogsDates());
    Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues()[0]);
    Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
    Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

    // Open a result file from Grenoble
    wxString resultFilePath = wxFileName::GetCwd();
    resultFilePath.Append("/files/forecast_score_07.txt");
    asFileAscii file(resultFilePath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 4;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
    fileAnalogsDates.resize(nanalogs);
    fileAnalogsCriteria.resize(nanalogs);
    fileAnalogsValues.resize(nanalogs);

    for (int i_test = 0; i_test < nbtests; i_test++) {
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
        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            file.SkipElements(1);

            day = file.GetInt();
            month = file.GetInt();
            year = file.GetInt();
            if (year > 0) {
                fileAnalogsDates[i_ana] = (float) asTime::GetMJD(year, month, day);
            } else {
                fileAnalogsDates[i_ana] = 0;
            }
            fileAnalogsValues[i_ana] = (float) sqrt(file.GetFloat() / P10);
            file.SkipElements(1); // Skip S1
            fileAnalogsCriteria[i_ana] = file.GetFloat();

            file.SkipLines(1);
        }

        file.SkipLines(3);

        // Find target date in the array
        int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0],
                                                              &resultsTargetDates[resultsTargetDates.rows() - 1],
                                                              fileTargetDate);

        // Compare the file and the processing
        EXPECT_FLOAT_EQ(fileTargetValue, resultsTargetValues(rowTargetDate));

        for (int i_ana = 0; i_ana < nanalogs; i_ana++) {
            if (fileAnalogsDates[i_ana] > 0) {
                EXPECT_FLOAT_EQ(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana));
                EXPECT_FLOAT_EQ(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana));
                EXPECT_NEAR(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
            }
        }
    }

    file.Close();
    // predictand pointer deleted by asMethodCalibration

    asRemoveDir(tmpDir);
}

TEST(MethodCalibrator, Ref2MergeByHalfAndMultiply)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    Ref2MergeByHalfAndMultiply();
}

TEST(MethodCalibrator, PrelodingWithLevelCorrection)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
    pConfig->Write("/General/ParallelDataLoad", false); // In order to avoid warning messages

    // Get parameters
    asParametersCalibration params;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_preload_multiple_variables.xml");
    ASSERT_TRUE(params.LoadFromFile(paramsFilePath));

    // Check original parameters
    EXPECT_EQ(params.GetPredictorLevel(0, 0), 0);
    EXPECT_EQ(params.GetPredictorTimeHours(0, 0), 6);
    EXPECT_EQ(params.GetPredictorLevel(0, 1), 0);
    EXPECT_EQ(params.GetPredictorTimeHours(0, 1), 6);

    // Preload data
    asMethodCalibratorSingle calibrator;
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator.SetPredictorDataDir(dataPredictorFilePath);
    calibrator.SetPredictandDB(NULL);
    asResultsAnalogsDates anaDates;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs));
        EXPECT_TRUE(anaDates.GetAnalogsDatesLength() > 0);
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    // Check corrected parameters
    EXPECT_TRUE(params.GetPredictorLevel(0, 0) > 0);
    EXPECT_EQ(params.GetPredictorTimeHours(0, 0), 24);
    EXPECT_EQ(params.GetPredictorLevel(0, 1), 0);
    EXPECT_TRUE(params.GetPredictorTimeHours(0, 1) > 6);
}

TEST(MethodCalibrator, NormalizedS1Criteria)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    // Get parameters
    asParametersCalibration paramsStd;
    asParametersCalibration paramsNorm;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_normalized_S1_criteria.xml");
    ASSERT_TRUE(paramsStd.LoadFromFile(paramsFilePath));
    ASSERT_TRUE(paramsNorm.LoadFromFile(paramsFilePath));
    paramsStd.SetPredictorCriteria(0, 0, "S1");
    ASSERT_TRUE(paramsStd.GetPredictorCriteria(0, 0).IsSameAs("S1"));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator1;
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator1.SetPredictorDataDir(dataPredictorFilePath);
    calibrator1.SetPredictandDB(NULL);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asResultsAnalogsDates anaDatesStd;
    asResultsAnalogsDates anaDatesNorm;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesNorm, paramsNorm, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    Array1DFloat targetDatesStd = anaDatesStd.GetTargetDates();
    Array1DFloat targetDatesPreload = anaDatesNorm.GetTargetDates();
    int targetDatesSize = (int) wxMax(targetDatesStd.cols(), targetDatesStd.rows());
    for (int i = 0; i < targetDatesSize; i++) {
        EXPECT_EQ(targetDatesStd[i], targetDatesPreload[i]);
    }

    Array2DFloat datesStd = anaDatesStd.GetAnalogsDates();
    Array2DFloat datesNorm = anaDatesNorm.GetAnalogsDates();
    Array2DFloat criteriaStd = anaDatesStd.GetAnalogsCriteria();
    Array2DFloat criteriaNorm = anaDatesNorm.GetAnalogsCriteria();

    EXPECT_EQ(datesStd.cols(), datesNorm.cols());
    EXPECT_EQ(datesStd.rows(), datesNorm.rows());
    EXPECT_EQ(criteriaStd.cols(), criteriaNorm.cols());
    EXPECT_EQ(criteriaStd.rows(), criteriaNorm.rows());

    for (int i = 0; i < datesStd.rows(); i++) {
        for (int j = 0; j < datesStd.cols(); j++) {
            EXPECT_EQ(datesStd.coeff(i, j), datesNorm.coeff(i, j));
            EXPECT_NE(criteriaStd.coeff(i, j), criteriaNorm.coeff(i, j));
        }
    }
}

TEST(MethodCalibrator, NormalizedRMSECriteria)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int) asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);

    // Get parameters
    asParametersCalibration paramsStd;
    asParametersCalibration paramsNorm;
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/parameters_calibration_normalized_RMSE_criteria.xml");
    ASSERT_TRUE(paramsStd.LoadFromFile(paramsFilePath));
    ASSERT_TRUE(paramsNorm.LoadFromFile(paramsFilePath));
    paramsStd.SetPredictorCriteria(0, 0, "RMSE");
    ASSERT_TRUE(paramsStd.GetPredictorCriteria(0, 0).IsSameAs("RMSE"));

    // Proceed to the calculations
    asMethodCalibratorSingle calibrator1;
    wxString dataPredictorFilePath = wxFileName::GetCwd();
    dataPredictorFilePath.Append("/files/");
    calibrator1.SetPredictorDataDir(dataPredictorFilePath);
    calibrator1.SetPredictandDB(NULL);
    asMethodCalibratorSingle calibrator2 = calibrator1;
    asResultsAnalogsDates anaDatesStd;
    asResultsAnalogsDates anaDatesNorm;

    try {
        int step = 0;
        bool containsNaNs = false;
        ASSERT_TRUE(calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
        ASSERT_TRUE(calibrator2.GetAnalogsDates(anaDatesNorm, paramsNorm, step, containsNaNs));
        EXPECT_FALSE(containsNaNs);
    } catch (asException &e) {
        wxPrintf(e.GetFullMessage());
        return;
    }

    Array1DFloat targetDatesStd = anaDatesStd.GetTargetDates();
    Array1DFloat targetDatesPreload = anaDatesNorm.GetTargetDates();
    int targetDatesSize = (int) wxMax(targetDatesStd.cols(), targetDatesStd.rows());
    for (int i = 0; i < targetDatesSize; i++) {
        EXPECT_EQ(targetDatesStd[i], targetDatesPreload[i]);
    }

    Array2DFloat datesStd = anaDatesStd.GetAnalogsDates();
    Array2DFloat datesNorm = anaDatesNorm.GetAnalogsDates();
    Array2DFloat criteriaStd = anaDatesStd.GetAnalogsCriteria();
    Array2DFloat criteriaNorm = anaDatesNorm.GetAnalogsCriteria();

    EXPECT_EQ(datesStd.cols(), datesNorm.cols());
    EXPECT_EQ(datesStd.rows(), datesNorm.rows());
    EXPECT_EQ(criteriaStd.cols(), criteriaNorm.cols());
    EXPECT_EQ(criteriaStd.rows(), criteriaNorm.rows());

    for (int i = 0; i < datesStd.rows(); i++) {
        for (int j = 0; j < datesStd.cols(); j++) {
            EXPECT_EQ(datesStd.coeff(i, j), datesNorm.coeff(i, j));
            EXPECT_NE(criteriaStd.coeff(i, j), criteriaNorm.coeff(i, j));
        }
    }
}
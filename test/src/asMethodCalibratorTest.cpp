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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#include <wx/filename.h>

#include "include_tests.h"
#include "asDataPredictandPrecipitation.h"
#include "asProcessor.h"
#include "asMethodCalibratorSingle.h"
#include "asResultsAnalogsDates.h"
#include "asResultsAnalogsValues.h"
#include "asResultsAnalogsForecastScores.h"
#include "asResultsAnalogsForecastScoreFinal.h"
#include "asFileAscii.h"

#include "UnitTest++.h"

namespace
{

void GrenobleComparison1()
{
    if (g_UnitTestLongestProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        asParametersCalibration params;
        params.AddStep();

        params.SetArchiveYearStart(1962);
        params.SetArchiveYearEnd(2001);
        params.SetCalibrationYearStart(1962);
        params.SetCalibrationYearEnd(2001);
        params.SetTimeArrayTargetTimeStepHours(24);
        params.SetTimeArrayAnalogsTimeStepHours(24);
        params.SetTimeArrayTargetMode("Simple");
        params.SetTimeArrayAnalogsMode("DaysInterval");
        params.SetTimeArrayAnalogsExcludeDays(60);
        params.SetTimeArrayAnalogsIntervalDays(60);
        params.SetMethodName(0, "Analogs");
        params.SetAnalogsNumber(0, 50);

        params.SetPredictorDatasetId(0, 0, "NCEP_Reanalysis_v1_lthe");
        params.SetPredictorDataId(0, 0, "hgt_1000hPa");
        params.SetPredictorLevel(0, 0, 1000);
        params.SetPredictorUmin(0, 0, -5);
        params.SetPredictorUptsnb(0, 0, 9);
        params.SetPredictorUstep(0, 0, 2.5);
        params.SetPredictorVmin(0, 0, 40);
        params.SetPredictorVptsnb(0, 0, 5);
        params.SetPredictorVstep(0, 0, 2.5);
        params.SetPredictorDTimeHours(0, 0, 12);
        params.SetPredictorCriteria(0, 0, "S1");
        params.SetPredictorWeight(0, 0, 0.5);

        params.AddPredictor();
        params.SetPredictorDatasetId(0, 1, "NCEP_Reanalysis_v1_lthe");
        params.SetPredictorDataId(0, 1, "hgt_500hPa");
        params.SetPredictorLevel(0, 1, 500);
        params.SetPredictorUmin(0, 1, -5);
        params.SetPredictorUptsnb(0, 1, 9);
        params.SetPredictorUstep(0, 1, 2.5);
        params.SetPredictorVmin(0, 1, 40);
        params.SetPredictorVptsnb(0, 1, 5);
        params.SetPredictorVstep(0, 1, 2.5);
        params.SetPredictorDTimeHours(0, 1, 24);
        params.SetPredictorCriteria(0, 1, "S1");
        params.SetPredictorWeight(0, 1, 0.5);

        params.SetForecastScorePostprocess(false);

        params.SetForecastScoreName("CRPSAR");
        params.SetForecastScoreAnalogsNumber(50);
        params.SetPredictandStationId(1);
        params.SetForecastScoreTimeArrayMode("Simple");

        // Fixes
        params.FixTimeShift();
        params.FixWeights();
        params.FixCoordinates();

        // Proceed to the calculations
        int step = 0;
        asMethodCalibratorSingle calibrator;
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScoresCRPS;
        asResultsAnalogsForecastScores anaScoresCRPSsharpness;
        asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;
        bool result;
		bool containsNaNs = false;
        try
        {
            wxString dataPredictorFilePath = wxFileName::GetCwd();
            dataPredictorFilePath.Append("/files/");
            calibrator.SetPredictorDataDir(dataPredictorFilePath);
            wxASSERT(predictand);
            calibrator.SetPredictandDB(predictand);
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
			CHECK_EQUAL(true, containsNaNs);
            result = calibrator.GetAnalogsValues(anaValues, params, anaDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // Extract data
        Array1DFloat resultsTargetDates(anaDates.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaDates.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaDates.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());
        float scoreFinal = anaScoreFinal.GetForecastScore();

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile06.txt");
        asFileAscii file(resultFilePath, asFile::ReadOnly);
        file.Open();

        // Test numbers
        int nbtests = 43; //43

        // Resize the containers
        int nanalogs = 50;
        Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
        fileAnalogsDates.resize(nanalogs);
        fileAnalogsCriteria.resize(nanalogs);
        fileAnalogsValues.resize(nanalogs);

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(2);

            float fileForecastScoreCRPS = file.GetFloat();
            float fileForecastScoreCRPSaccuracy = file.GetFloat();
            float fileForecastScoreCRPSsharpness = file.GetFloat();

            file.SkipLines(1);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
            }

            // The CRPS tolerence is hughe, as it is not processed with the same P10 !
            CHECK_CLOSE(fileForecastScoreCRPS, resultsForecastScoreCRPS(rowTargetDate), 0.01);
            CHECK_CLOSE(fileForecastScoreCRPSaccuracy, resultsForecastScoreCRPSaccuracy(rowTargetDate), 0.01);
            CHECK_CLOSE(fileForecastScoreCRPSsharpness, resultsForecastScoreCRPSsharpness(rowTargetDate), 0.01);
        }

        CHECK_CLOSE(asTools::Mean(&resultsForecastScoreCRPS[0],&resultsForecastScoreCRPS[resultsForecastScoreCRPS.size()-1]), scoreFinal, 0.0001);

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithLinAlgebra)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);

    wxString str("Processing GrenobleComparison1 with the multithreaded option (lin algebra)\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison1();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithLinAlgebraNoVar)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1 with the multithreaded option (lin algebra no var)\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison1();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithCoeff)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);

    wxString str("Processing GrenobleComparison1 with the multithreaded option (coeff)\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison1();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithCoeffNoVar)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);

    wxString str("Processing GrenobleComparison1 with the multithreaded option (coeff no var)\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison1();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

TEST(GrenobleComparison1ProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asINSERT);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1 with the array insertion option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison1();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

TEST(GrenobleComparison1ProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asFULL_ARRAY);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1 with the array splitting option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison1();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

void GrenobleComparison1CalibrationPeriod()
{
    if (g_UnitTestLongerProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        paramsFilePath.Append("/files/parameters_calibration_01.xml");
        asParametersCalibration params;
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        int step = 0;
        asMethodCalibratorSingle calibrator;
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScoresCRPS;
        asResultsAnalogsForecastScores anaScoresCRPSsharpness;
        asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

		bool containsNaNs = false;

        try
        {
            wxString dataPredictorFilePath = wxFileName::GetCwd();
            dataPredictorFilePath.Append("/files/");
            calibrator.SetPredictorDataDir(dataPredictorFilePath);
            wxASSERT(predictand);
            calibrator.SetPredictandDB(predictand);
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsValues(anaValues, params, anaDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // Extract data
        Array1DFloat resultsTargetDates(anaDates.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaDates.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaDates.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile10.txt");
        asFileAscii file(resultFilePath, asFile::ReadOnly);
        file.Open();

        // Test numbers
        int nbtests = 20; //20 //43

        // Resize the containers
        int nanalogs = 50; //50
        Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
        fileAnalogsDates.resize(nanalogs);
        fileAnalogsCriteria.resize(nanalogs);
        fileAnalogsValues.resize(nanalogs);

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                if(year>0)
                {
                    fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                }
                else
                {
                    fileAnalogsDates[i_ana] = 0;
                }
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(3);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                if (fileAnalogsDates[i_ana]>0) // If we have the data
                {
                    CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
                }
            }
        }

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison1CalibrationPeriodProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1 on calibration period with the multithreaded option (lin algebra no var)\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison1CalibrationPeriod();
}

TEST(GrenobleComparison1CalibrationPeriodProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asINSERT);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1 on calibration period with the array insertion option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison1CalibrationPeriod();
}

TEST(GrenobleComparison1CalibrationPeriodProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asFULL_ARRAY);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1 on calibration period with the array splitting option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison1CalibrationPeriod();
}

void GrenobleComparison2()
{
    if (g_UnitTestLongestProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(2);

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        params.AddStep();

        params.SetArchiveYearStart(1962);
        params.SetArchiveYearEnd(2001);
        params.SetCalibrationYearStart(1962);
        params.SetCalibrationYearEnd(2001);
        params.SetTimeArrayTargetTimeStepHours(24);
        params.SetTimeArrayAnalogsTimeStepHours(24);
        params.SetTimeArrayTargetMode("Simple");
        params.SetTimeArrayAnalogsMode("DaysInterval");
        params.SetTimeArrayAnalogsExcludeDays(60);
        params.SetTimeArrayAnalogsIntervalDays(60);

        params.SetMethodName(0, "Analogs");
        params.SetAnalogsNumber(0, 70);

        params.SetPredictorDatasetId(0, 0, "NCEP_Grenoble_start12h_step24h");
        params.SetPredictorDataId(0, 0, "hgt");
        params.SetPredictorLevel(0, 0, 1000);
        params.SetPredictorUmin(0, 0, -5);
        params.SetPredictorUptsnb(0, 0, 9);
        params.SetPredictorUstep(0, 0, 2.5);
        params.SetPredictorVmin(0, 0, 40);
        params.SetPredictorVptsnb(0, 0, 5);
        params.SetPredictorVstep(0, 0, 2.5);
        params.SetPredictorDTimeHours(0, 0, 12);
        params.SetPredictorCriteria(0, 0, "S1");
        params.SetPredictorWeight(0, 0, 0.5);

        params.AddPredictor();
        params.SetPredictorDatasetId(0, 1, "NCEP_Grenoble_start0h_step24h");
        params.SetPredictorDataId(0, 1, "hgt");
        params.SetPredictorLevel(0, 1, 500);
        params.SetPredictorUmin(0, 1, -5);
        params.SetPredictorUptsnb(0, 1, 9);
        params.SetPredictorUstep(0, 1, 2.5);
        params.SetPredictorVmin(0, 1, 40);
        params.SetPredictorVptsnb(0, 1, 5);
        params.SetPredictorVstep(0, 1, 2.5);
        params.SetPredictorDTimeHours(0, 1, 24);
        params.SetPredictorCriteria(0, 1, "S1");
        params.SetPredictorWeight(0, 1, 0.5);

        params.AddStep();
        params.SetMethodName(1, "Analogs");
        params.SetAnalogsNumber(1, 30);

        params.SetPredictorDatasetId(1, 0, "NCEP_Grenoble_start0h_step12h");
        params.SetPredictorDataId(1, 0, "humidity");
        params.SetPreprocess(1, 0, true);
        params.SetPreprocessMethod(1, 0, "MergeCouplesAndMultiply");
        params.SetPreprocessDatasetId(1, 0, 0, "NCEP_Grenoble_start0h_step12h");
        params.SetPreprocessDatasetId(1, 0, 1, "NCEP_Grenoble_start0h_step12h");
        params.SetPreprocessDatasetId(1, 0, 2, "NCEP_Grenoble_start0h_step12h");
        params.SetPreprocessDatasetId(1, 0, 3, "NCEP_Grenoble_start0h_step12h");
        params.SetPreprocessDataId(1, 0, 0, "pwa");
        params.SetPreprocessDataId(1, 0, 1, "pwa");
        params.SetPreprocessDataId(1, 0, 2, "rhum");
        params.SetPreprocessDataId(1, 0, 3, "rhum");
        params.SetPreprocessLevel(1, 0, 0, 0);
        params.SetPreprocessLevel(1, 0, 1, 0);
        params.SetPreprocessLevel(1, 0, 2, 850);
        params.SetPreprocessLevel(1, 0, 3, 850);
        params.SetPreprocessDTimeHours(1, 0, 0, 12);
        params.SetPreprocessDTimeHours(1, 0, 1, 24);
        params.SetPreprocessDTimeHours(1, 0, 2, 12);
        params.SetPreprocessDTimeHours(1, 0, 3, 24);
        params.SetPredictorLevel(1, 0, 0);
        params.SetPredictorUmin(1, 0, 5);
        params.SetPredictorUptsnb(1, 0, 2);
        params.SetPredictorUstep(1, 0, 2.5);
        params.SetPredictorVmin(1, 0, 45);
        params.SetPredictorVptsnb(1, 0, 2);
        params.SetPredictorVstep(1, 0, 2.5);
        params.SetPredictorDTimeHours(1, 0, 12);
        params.SetPredictorCriteria(1, 0, "RSE");
        params.SetPredictorWeight(1, 0, 1);

        params.SetForecastScorePostprocess(false);

        params.SetForecastScoreName("CRPSEP");
        params.SetForecastScoreAnalogsNumber(30);
        params.SetPredictandStationId(1);
        params.SetForecastScoreTimeArrayMode("Simple");

        // Fixes
        params.FixTimeShift();
        params.FixWeights();
        params.FixCoordinates();

        // Proceed to the calculations
        int step = 0;
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
        bool result;
		bool containsNaNs = false;
        try
        {
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            step++;
            result = calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsValues(anaValues, params, anaSubDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // Extract data
        Array1DFloat resultsTargetDates(anaSubDates.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaSubDates.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaSubDates.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());
        float scoreFinal = anaScoreFinal.GetForecastScore();

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile07.txt");
        asFileAscii file(resultFilePath, asFile::ReadOnly);
        file.Open();

        // Test numbers
        int nbtests = 30;

        // Resize the containers
        int nanalogs = 30;
        Array1DFloat fileAnalogsDates, fileAnalogsCriteria, fileAnalogsValues;
        fileAnalogsDates.resize(nanalogs);
        fileAnalogsCriteria.resize(nanalogs);
        fileAnalogsValues.resize(nanalogs);

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                file.SkipElements(1); // Skip S1
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(2);

            float fileForecastScoreCRPS = file.GetFloat();
            float fileForecastScoreCRPSaccuracy = file.GetFloat();
            float fileForecastScoreCRPSsharpness = file.GetFloat();

            file.SkipLines(1);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);

                if (abs(fileAnalogsDates[i_ana]-resultsAnalogsDates(rowTargetDate, i_ana))>0.0001)
                {
                    wxString strdates = wxString::Format(("Date is %s and should be %s.\n"), asTime::GetStringTime(resultsAnalogsDates(rowTargetDate, i_ana)), asTime::GetStringTime(fileAnalogsDates[i_ana]));
                    printf("%s", strdates.mb_str(wxConvUTF8).data());
                }
            }

            // The CRPS tolerence is huge, as it is not processed with the same P10 !
            CHECK_CLOSE(fileForecastScoreCRPS, resultsForecastScoreCRPS(rowTargetDate), 0.01);
            CHECK_CLOSE(fileForecastScoreCRPSaccuracy, resultsForecastScoreCRPSaccuracy(rowTargetDate), 0.01);
            CHECK_CLOSE(fileForecastScoreCRPSsharpness, resultsForecastScoreCRPSsharpness(rowTargetDate), 0.01);
        }

        CHECK_CLOSE(asTools::Mean(&resultsForecastScoreCRPS[0],&resultsForecastScoreCRPS[resultsForecastScoreCRPS.size()-1]), scoreFinal, 0.0001);

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2ProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison2 with the multithreaded option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison2();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

TEST(GrenobleComparison2ProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asINSERT);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison2 with the array insertion option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison2();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

TEST(GrenobleComparison2ProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asFULL_ARRAY);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison2 with the array splitting option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxStopWatch sw;

    GrenobleComparison2();

    wxString msg = wxString::Format(" -> took %ldms to execute\n", sw.Time());
    printf("%s", msg.mb_str(wxConvUTF8).data());
}

void GrenobleComparison2CalibrationPeriod()
{
    if (g_UnitTestLongerProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        paramsFilePath.Append("/files/parameters_calibration_02.xml");
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        int step = 0;
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

		bool containsNaNs = false;

        try
        {
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            step++;
            result = calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsValues(anaValues, params, anaSubDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // Extract data
        Array1DFloat resultsTargetDates(anaSubDates.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaSubDates.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaSubDates.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile11.txt");
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

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                if(year>0)
                {
                    fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                }
                else
                {
                    fileAnalogsDates[i_ana] = 0;
                }
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                file.SkipElements(1); // Skip S1
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(3);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                if(fileAnalogsDates[i_ana]>0)
                {
                    CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
                }
            }
        }

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2CalibrationPeriodProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison2 on calibration period with the multithreaded option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison2CalibrationPeriod();
}

TEST(GrenobleComparison2CalibrationPeriodProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asINSERT);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison2 with the array insertion option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison2CalibrationPeriod();
}

TEST(GrenobleComparison2CalibrationPeriodProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asFULL_ARRAY);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison2 with the array splitting option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison2CalibrationPeriod();
}

TEST(PreloadingSimple)
{
    if (g_UnitTestLongerProcessing)
    {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asINSERT);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);

        wxString str("Processing PreloadingSimple\n");
        printf("%s", str.mb_str(wxConvUTF8).data());

        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        //pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        wxString dataFileDir = wxFileName::GetCwd();
        dataFileDir.Append("/files/");
        wxString patternFileDir = wxFileName::GetCwd();
        patternFileDir.Append("/files/");

        // Get parameters
        asParametersCalibration paramsStd;
        asParametersCalibration paramsPreload;
        wxString paramsFilePathStd = wxFileName::GetCwd();
        wxString paramsFilePathPreload = wxFileName::GetCwd();
        paramsFilePathStd.Append("/files/parameters_calibration_05.xml");
        paramsFilePathPreload.Append("/files/parameters_calibration_06.xml");
        result = paramsStd.LoadFromFile(paramsFilePathStd);
        CHECK_EQUAL(true, result);
        result = paramsPreload.LoadFromFile(paramsFilePathPreload);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        int step = 0;
        asMethodCalibratorSingle calibrator1;
        wxString dataPredictorFilePath = wxFileName::GetCwd();
        dataPredictorFilePath.Append("/files/");
        calibrator1.SetPredictorDataDir(dataPredictorFilePath);
        calibrator1.SetPredictandDB(NULL);
        asMethodCalibratorSingle calibrator2 = calibrator1;
        asResultsAnalogsDates anaDatesStd;
        asResultsAnalogsDates anaDatesPreload;

		bool containsNaNs = false;

        try
        {
            result = calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator2.GetAnalogsDates(anaDatesPreload, paramsPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        Array2DFloat datesStd = anaDatesStd.GetAnalogsDates();
        Array2DFloat datesPreload = anaDatesPreload.GetAnalogsDates();
        Array2DFloat criteriaStd = anaDatesStd.GetAnalogsCriteria();
        Array2DFloat criteriaPreload = anaDatesPreload.GetAnalogsCriteria();

        CHECK_EQUAL(datesStd.cols(),datesPreload.cols());
        CHECK_EQUAL(datesStd.rows(),datesPreload.rows());
        CHECK_EQUAL(criteriaStd.cols(),criteriaPreload.cols());
        CHECK_EQUAL(criteriaStd.rows(),criteriaPreload.rows());

        for (int i=0; i<datesStd.rows(); i++)
        {
            for (int j=0; j<datesStd.cols(); j++)
            {
                CHECK_EQUAL(datesStd.coeff(i,j), datesPreload.coeff(i,j));
                CHECK_EQUAL(criteriaStd.coeff(i,j), criteriaPreload.coeff(i,j));
            }
        }

        wxDELETE(pLog);
    }
}

TEST(PreloadingWithPreprocessing)
{
    if (g_UnitTestLongerProcessing)
    {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asINSERT);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);

        wxString str("Processing PreloadingWithPreprocessing\n");
        printf("%s", str.mb_str(wxConvUTF8).data());

        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        //pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        wxString dataFileDir = wxFileName::GetCwd();
        dataFileDir.Append("/files/");
        wxString patternFileDir = wxFileName::GetCwd();
        patternFileDir.Append("/files/");

        // Get parameters
        asParametersCalibration paramsStd;
        asParametersCalibration paramsPreload;
        wxString paramsFilePathStd = wxFileName::GetCwd();
        wxString paramsFilePathPreload = wxFileName::GetCwd();
        paramsFilePathStd.Append("/files/parameters_calibration_07.xml");
        paramsFilePathPreload.Append("/files/parameters_calibration_08.xml");
        result = paramsStd.LoadFromFile(paramsFilePathStd);
        CHECK_EQUAL(true, result);
        result = paramsPreload.LoadFromFile(paramsFilePathPreload);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        int step = 0;
        asMethodCalibratorSingle calibrator1;
        wxString dataPredictorFilePath = wxFileName::GetCwd();
        dataPredictorFilePath.Append("/files/");
        calibrator1.SetPredictorDataDir(dataPredictorFilePath);
        calibrator1.SetPredictandDB(NULL);
        asMethodCalibratorSingle calibrator2 = calibrator1;
        asResultsAnalogsDates anaDatesStd;
        asResultsAnalogsDates anaDatesPreload;

		bool containsNaNs = false;

        try
        {
            result = calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator2.GetAnalogsDates(anaDatesPreload, paramsPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        Array1DFloat targetDatesStd = anaDatesStd.GetTargetDates();
        Array1DFloat targetDatesPreload = anaDatesPreload.GetTargetDates();
        int targetDatesSize = wxMax(targetDatesStd.cols(),targetDatesStd.rows());
        for (int i=0; i<targetDatesSize; i++)
        {
            CHECK_EQUAL(targetDatesStd[i],targetDatesPreload[i]);
        }

        Array2DFloat datesStd = anaDatesStd.GetAnalogsDates();
        Array2DFloat datesPreload = anaDatesPreload.GetAnalogsDates();
        Array2DFloat criteriaStd = anaDatesStd.GetAnalogsCriteria();
        Array2DFloat criteriaPreload = anaDatesPreload.GetAnalogsCriteria();

        CHECK_EQUAL(datesStd.cols(),datesPreload.cols());
        CHECK_EQUAL(datesStd.rows(),datesPreload.rows());
        CHECK_EQUAL(criteriaStd.cols(),criteriaPreload.cols());
        CHECK_EQUAL(criteriaStd.rows(),criteriaPreload.rows());

        for (int i=0; i<datesStd.rows(); i++)
        {
            for (int j=0; j<datesStd.cols(); j++)
            {
                CHECK_EQUAL(datesStd.coeff(i,j), datesPreload.coeff(i,j));
                CHECK_EQUAL(criteriaStd.coeff(i,j), criteriaPreload.coeff(i,j));

                break;
            }


            break;
        }

        wxDELETE(pLog);
    }
}

void GrenobleComparison1Preloading()
{
    if (g_UnitTestLongerProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFile("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        paramsFilePath.Append("/files/parameters_calibration_03.xml");
        asParametersCalibration params;
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Force values
        params.SetPredictorUptsnb(0, 0, 9);
        params.SetPredictorVmin(0, 1, 40);

        // Proceed to the calculations
        int step = 0;
        asMethodCalibratorSingle calibrator;
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScoresCRPS;
        asResultsAnalogsForecastScores anaScoresCRPSsharpness;
        asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

		bool containsNaNs = false;

        try
        {
            wxString dataPredictorFilePath = wxFileName::GetCwd();
            dataPredictorFilePath.Append("/files/");
            calibrator.SetPredictorDataDir(dataPredictorFilePath);
            wxASSERT(predictand);
            calibrator.SetPredictandDB(predictand);
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsValues(anaValues, params, anaDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }


        // Extract data
        Array1DFloat resultsTargetDates(anaDates.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaDates.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaDates.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile10.txt");
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

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                if(year>0)
                {
                    fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                }
                else
                {
                    fileAnalogsDates[i_ana] = 0;
                }
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(3);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                if (fileAnalogsDates[i_ana]>0) // If we have the data
                {
                    CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
                }
            }
        }

        file.Close();

        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison1PreloadingMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1 with data preloading\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison1Preloading();
}

void GrenobleComparison1PreloadingSubset()
{
    if (g_UnitTestLongerProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFile("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        paramsFilePath.Append("/files/parameters_calibration_03.xml");
        asParametersCalibration params;
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Force values
        params.SetPredictorUptsnb(0, 0, 9);
        params.SetPredictorVmin(0, 1, 42.5);
        params.SetPredictorVptsnb(0, 1, 4);

        // Proceed to the calculations
        int step = 0;
        asMethodCalibratorSingle calibrator;
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScoresCRPS;
        asResultsAnalogsForecastScores anaScoresCRPSsharpness;
        asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

		bool containsNaNs = false;

        try
        {
            wxString dataPredictorFilePath = wxFileName::GetCwd();
            dataPredictorFilePath.Append("/files/");
            calibrator.SetPredictorDataDir(dataPredictorFilePath);
            wxASSERT(predictand);
            calibrator.SetPredictandDB(predictand);
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsValues(anaValues, params, anaDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyAR");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // No unit test possible, as results will differ from Grenoble's results.

        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison1PreloadingSubsetMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

    wxString str("Processing GrenobleComparison1Subset with data preloading\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison1PreloadingSubset();
}

void GrenobleComparison2Preloading()
{
    if (g_UnitTestLongerProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        paramsFilePath.Append("/files/parameters_calibration_04.xml");
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        int step = 0;
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

		bool containsNaNs = false;

        try
        {
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            step++;
            result = calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsValues(anaValues, params, anaSubDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // Extract data
        Array1DFloat resultsTargetDates(anaSubDates.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaSubDates.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaSubDates.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile11.txt");
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

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                if(year>0)
                {
                    fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                }
                else
                {
                    fileAnalogsDates[i_ana] = 0;
                }
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                file.SkipElements(1); // Skip S1
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(3);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                if(fileAnalogsDates[i_ana]>0)
                {
                    CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
                }
            }
        }

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2PreloadingProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);

    wxString str("Processing GrenobleComparison2 preloading with the multithreaded option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison2Preloading();
}

TEST(GrenobleComparison2PreloadingProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asINSERT);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);

    wxString str("Processing GrenobleComparison2 preloading with the array insertion option\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison2Preloading();
}

void GrenobleComparison2SavingIntermediateResults()
{
    if (g_UnitTestLongerProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        paramsFilePath.Append("/files/parameters_calibration_02.xml");
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        int step = 0;
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

		bool containsNaNs = false;

        try
        {
            // Create
            result = calibrator.GetAnalogsDates(anaDates1, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            // Reload
            result = calibrator.GetAnalogsDates(anaDates2, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            step++;
            // Create
            result = calibrator.GetAnalogsSubDates(anaSubDates1, params, anaDates2, step, containsNaNs);
            CHECK_EQUAL(true, result);
            // Reload
            result = calibrator.GetAnalogsSubDates(anaSubDates2, params, anaDates2, step, containsNaNs);
            CHECK_EQUAL(true, result);
            // Create
            result = calibrator.GetAnalogsValues(anaValues1, params, anaSubDates2, step);
            CHECK_EQUAL(true, result);
            // Reload
            result = calibrator.GetAnalogsValues(anaValues2, params, anaSubDates2, step);
            CHECK_EQUAL(true, result);
            // Create
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS1, params, anaValues2, step);
            CHECK_EQUAL(true, result);
            // Reload
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS2, params, anaValues2, step);
            CHECK_EQUAL(true, result);
            // Create
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS2, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // Extract data
        Array1DFloat resultsTargetDates(anaSubDates2.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues2.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaSubDates2.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaSubDates2.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues2.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS2.GetForecastScores());

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile11.txt");
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

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                if(year>0)
                {
                    fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                }
                else
                {
                    fileAnalogsDates[i_ana] = 0;
                }
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                file.SkipElements(1); // Skip S1
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(3);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                if(fileAnalogsDates[i_ana]>0)
                {
                    CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
                }
            }
        }

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2SavingIntermediateResults)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);

    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep1", true);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep2", true);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep3", true);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep4", true);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogValues", true);
    pConfig->Write("/Calibration/IntermediateResults/SaveForecastScores", true);
    pConfig->Write("/Calibration/IntermediateResults/SaveFinalForecastScore", true);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep1", true);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep2", true);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep3", true);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep4", true);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogValues", true);
    pConfig->Write("/Calibration/IntermediateResults/LoadForecastScores", true);

    wxString str("Processing GrenobleComparison2 with saving/loading of intermediate results\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison2SavingIntermediateResults();

    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep1", false);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep2", false);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep3", false);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep4", false);
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogValues", false);
    pConfig->Write("/Calibration/IntermediateResults/SaveForecastScores", false);
    pConfig->Write("/Calibration/IntermediateResults/SaveFinalForecastScore", false);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep1", false);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep2", false);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep3", false);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep4", false);
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogValues", false);
    pConfig->Write("/Calibration/IntermediateResults/LoadForecastScores", false);
}

void GrenobleComparison2MergeByHalfAndMultiply()
{
    if (g_UnitTestLongerProcessing)
    {
        // Set log
        asLog* pLog = new asLog();
        pLog->DisableMessageBoxOnError();
        pLog->CreateFileOnly("AtmoswingUnitTesting.log");
        pLog->SetLevel(1);

        bool result;

        // Create predictand database
        asDataPredictandPrecipitation* predictand = new asDataPredictandPrecipitation(Precipitation, Daily, Station);

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
        paramsFilePath.Append("/files/parameters_calibration_09.xml");
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        int step = 0;
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

		bool containsNaNs = false;

        try
        {
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            step++;
            result = calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsValues(anaValues, params, anaSubDates, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPS, params, anaValues, step);
            CHECK_EQUAL(true, result);
            result = calibrator.GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScoresCRPS, step);
            CHECK_EQUAL(true, result);

            // Sharpness and Accuracy
            params.SetForecastScoreName("CRPSsharpnessEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSsharpness, params, anaValues, step);
            CHECK_EQUAL(true, result);
            params.SetForecastScoreName("CRPSaccuracyEP");
            result = calibrator.GetAnalogsForecastScores(anaScoresCRPSaccuracy, params, anaValues, step);
            CHECK_EQUAL(true, result);
        }
        catch(asException& e)
        {
            wxString eMessage = e.GetFullMessage();
            printf("%s", eMessage.mb_str(wxConvUTF8).data());
            return;
        }

        // Extract data
        Array1DFloat resultsTargetDates(anaSubDates.GetTargetDates());
        Array1DFloat resultsTargetValues(anaValues.GetTargetValues());
        Array2DFloat resultsAnalogsCriteria(anaSubDates.GetAnalogsCriteria());
        Array2DFloat resultsAnalogsDates(anaSubDates.GetAnalogsDates());
        Array2DFloat resultsAnalogsValues(anaValues.GetAnalogsValues());
        Array1DFloat resultsForecastScoreCRPS(anaScoresCRPS.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSsharpness(anaScoresCRPSsharpness.GetForecastScores());
        Array1DFloat resultsForecastScoreCRPSaccuracy(anaScoresCRPSaccuracy.GetForecastScores());

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        resultFilePath.Append("/files/asMethodCalibratorTestFile11.txt");
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

        for (int i_test=0;i_test<nbtests;i_test++)
        {
            // Skip the header
            file.SkipLines(1);

            // Get target date from file
            int day = file.GetInt();
            int month = file.GetInt();
            int year = file.GetInt();
            float fileTargetDate = asTime::GetMJD(year, month, day);
            float fileTargetValue = sqrt(file.GetFloat()/P10);

            file.SkipLines(2);

            // Get analogs from file
            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                file.SkipElements(1);

                day = file.GetInt();
                month = file.GetInt();
                year = file.GetInt();
                if(year>0)
                {
                    fileAnalogsDates[i_ana] = asTime::GetMJD(year, month, day);
                }
                else
                {
                    fileAnalogsDates[i_ana] = 0;
                }
                fileAnalogsValues[i_ana] = sqrt(file.GetFloat()/P10);
                file.SkipElements(1); // Skip S1
                fileAnalogsCriteria[i_ana] = file.GetFloat();

                file.SkipLines(1);
            }

            file.SkipLines(3);

            // Find target date in the array
            int rowTargetDate = asTools::SortedArraySearchClosest(&resultsTargetDates[0], &resultsTargetDates[resultsTargetDates.rows()-1], fileTargetDate);

            // Compare the file and the processing
            CHECK_CLOSE(fileTargetValue, resultsTargetValues(rowTargetDate), 0.0001);

            for (int i_ana=0; i_ana<nanalogs; i_ana++)
            {
                if(fileAnalogsDates[i_ana]>0)
                {
                    CHECK_CLOSE(fileAnalogsDates[i_ana], resultsAnalogsDates(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsValues[i_ana], resultsAnalogsValues(rowTargetDate, i_ana), 0.0001);
                    CHECK_CLOSE(fileAnalogsCriteria[i_ana], resultsAnalogsCriteria(rowTargetDate, i_ana), 0.1);
                }
            }
        }

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        wxDELETE(pLog);

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2MergeByHalfAndMultiply)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/AllowMultithreading", true);
    pConfig->Write("/ProcessingOptions/ProcessingMethod", (int)asMULTITHREADS);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);

    wxString str("Processing GrenobleComparison2 with MergeByHalfAndMultiply preprocessing\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    GrenobleComparison2MergeByHalfAndMultiply();
}

}

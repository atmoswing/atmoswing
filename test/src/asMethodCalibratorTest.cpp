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

void GrenobleComparison1(const wxString &paramsFile, bool shortVersion)
{
    if (g_unitTestLongProcessing)
    {
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
        paramsFilePath.Append("/files/");
        paramsFilePath.Append(paramsFile);
        asParametersCalibration params;
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        asMethodCalibratorSingle calibrator;
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScoresCRPS;
        asResultsAnalogsForecastScores anaScoresCRPSsharpness;
        asResultsAnalogsForecastScores anaScoresCRPSaccuracy;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

        try
        {
            int step = 0;
            bool containsNaNs = false;
            wxString dataPredictorFilePath = wxFileName::GetCwd();
            dataPredictorFilePath.Append("/files/");
            calibrator.SetPredictorDataDir(dataPredictorFilePath);
            wxASSERT(predictand);
            calibrator.SetPredictandDB(predictand);
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
			CHECK_EQUAL(false, containsNaNs);
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
			wxPrintf( e.GetFullMessage());
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
        if (!shortVersion)
        {
            resultFilePath.Append("/files/forecast_score_04.txt");
            nbtests = 43;
        }
        else
        {
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

            float fileForecastScoreCRPS, fileForecastScoreCRPSaccuracy, fileForecastScoreCRPSsharpness;

            if (!shortVersion)
            {
                file.SkipLines(2);
                fileForecastScoreCRPS = file.GetFloat();
                fileForecastScoreCRPSaccuracy = file.GetFloat();
                fileForecastScoreCRPSsharpness = file.GetFloat();
                file.SkipLines(1);
            }
            else
            {
                file.SkipLines(3);
            }

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

            // The CRPS tolerence is huge, as it is not processed with the same P10 !
            if (!shortVersion)
            {
                CHECK_CLOSE(fileForecastScoreCRPS, resultsForecastScoreCRPS(rowTargetDate), 0.01);
                CHECK_CLOSE(fileForecastScoreCRPSaccuracy, resultsForecastScoreCRPSaccuracy(rowTargetDate), 0.01);
                CHECK_CLOSE(fileForecastScoreCRPSsharpness, resultsForecastScoreCRPSsharpness(rowTargetDate), 0.01);
            }
        }

        if (!shortVersion)
        {
            CHECK_CLOSE(asTools::Mean(&resultsForecastScoreCRPS[0],&resultsForecastScoreCRPS[resultsForecastScoreCRPS.size()-1]), scoreFinal, 0.0001);
        }

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        asRemoveDir(tmpDir);
    }
}

#ifdef USE_CUDA
TEST(GrenobleComparison1ProcessingMethodCuda)
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

	wxPrintf("Processing GrenobleComparison1 with CUDA\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}
#endif

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithLinAlgebra)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA);

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

	wxPrintf("Processing GrenobleComparison1 with the multithreaded option (lin algebra)\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full.xml", false);

	wxPrintf( " -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithLinAlgebraNoVar)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 with the multithreaded option (lin algebra no var)\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithCoeff)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asCOEFF);

	wxPrintf("Processing GrenobleComparison1 with the multithreaded option (coeff)\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithCoeffNoVar)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asCOEFF_NOVAR);

	wxPrintf("Processing GrenobleComparison1 with the multithreaded option (coeff no var)\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1ProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 with the array insertion option (lin algebra no var)\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1ProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 with the array splitting option (lin algebra no var)\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1CalibrationPeriodProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 on calibration period with the multithreaded option (lin algebra no var)\n");

    GrenobleComparison1("parameters_calibration_R1_calib_period.xml", true);
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithLinAlgebraNoVarNoPreprocessing)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 with the multithreaded option (lin algebra no var) no preprocessing\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full_no_preproc.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1ProcessingMethodMultithreadsWithCoeffNoVarNoPreprocessing)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asCOEFF_NOVAR);

	wxPrintf("Processing GrenobleComparison1 with the multithreaded option (coeff no var) no preprocessing\n");

    wxStopWatch sw;

    GrenobleComparison1("parameters_calibration_R1_full_no_preproc.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison1CalibrationPeriodProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 on calibration period with the array insertion option\n");

    GrenobleComparison1("parameters_calibration_R1_calib_period.xml", true);
}

TEST(GrenobleComparison1CalibrationPeriodProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 on calibration period with the array splitting option\n");

    GrenobleComparison1("parameters_calibration_R1_calib_period.xml", true);
}

void GrenobleComparison2(const wxString &paramsFile, bool shortVersion)
{
    if (g_unitTestLongProcessing)
    {
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
        paramsFilePath.Append("/files/");
        paramsFilePath.Append(paramsFile);
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

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
		
        try
        {
            int step = 0;
            bool containsNaNs = false;

            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            step++;
            result = calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
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
			wxPrintf( e.GetFullMessage());
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
        if(!shortVersion)
        {
            scoreFinal = anaScoreFinal.GetForecastScore();
        }

        // Open a result file from Grenoble
        wxString resultFilePath = wxFileName::GetCwd();
        int nbtests = 0;
        if(!shortVersion)
        {
            resultFilePath.Append("/files/forecast_score_05.txt");
            nbtests = 30;
        }
        else
        {
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

            float fileForecastScoreCRPS, fileForecastScoreCRPSaccuracy, fileForecastScoreCRPSsharpness;

            if (!shortVersion)
            {
                file.SkipLines(2);
                fileForecastScoreCRPS = file.GetFloat();
                fileForecastScoreCRPSaccuracy = file.GetFloat();
                fileForecastScoreCRPSsharpness = file.GetFloat();
                file.SkipLines(1);
            }
            else
            {
                file.SkipLines(3);
            }

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

                    if (std::abs(fileAnalogsDates[i_ana]-resultsAnalogsDates(rowTargetDate, i_ana))>0.0001)
                    {
						wxPrintf(("Date is %s and should be %s.\n"), asTime::GetStringTime(resultsAnalogsDates(rowTargetDate, i_ana)), asTime::GetStringTime(fileAnalogsDates[i_ana]));
                    }
                }
            }

            if (!shortVersion)
            {
                // The CRPS tolerence is huge, as it is not processed with the same P10 !
                CHECK_CLOSE(fileForecastScoreCRPS, resultsForecastScoreCRPS(rowTargetDate), 0.01);
                CHECK_CLOSE(fileForecastScoreCRPSaccuracy, resultsForecastScoreCRPSaccuracy(rowTargetDate), 0.01);
                CHECK_CLOSE(fileForecastScoreCRPSsharpness, resultsForecastScoreCRPSsharpness(rowTargetDate), 0.01);
            }
        }

        if (!shortVersion)
        {
            CHECK_CLOSE(asTools::Mean(&resultsForecastScoreCRPS[0],&resultsForecastScoreCRPS[resultsForecastScoreCRPS.size()-1]), scoreFinal, 0.0001);
        }

        file.Close();
        // predictand pointer deleted by asMethodCalibration

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2ProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison2 with the multithreaded option\n");

    wxStopWatch sw;

    GrenobleComparison2("parameters_calibration_R2_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison2ProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison2 with the array insertion option\n");

    wxStopWatch sw;

    GrenobleComparison2("parameters_calibration_R2_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison2ProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison2 with the array splitting option\n");

    wxStopWatch sw;

    GrenobleComparison2("parameters_calibration_R2_full.xml", false);

	wxPrintf(" -> took %ld ms to execute\n", sw.Time());
}

TEST(GrenobleComparison2CalibrationPeriodProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison2 on calibration period with the multithreaded option\n");

    GrenobleComparison2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(GrenobleComparison2CalibrationPeriodProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison2 with the array insertion option\n");

    GrenobleComparison2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(GrenobleComparison2CalibrationPeriodProcessingMethodSplitting)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asFULL_ARRAY);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison2 with the array splitting option\n");

    GrenobleComparison2("parameters_calibration_R2_calib_period.xml", true);
}

TEST(PreloadingSimple)
{
    if (g_unitTestLongProcessing)
    {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/Processing/Method", (int)asINSERT);
        pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

		wxPrintf("Processing PreloadingSimple\n");

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
        paramsFilePathStd.Append("/files/parameters_calibration_compare_no_preload.xml");
        paramsFilePathPreload.Append("/files/parameters_calibration_compare_preload.xml");
        result = paramsStd.LoadFromFile(paramsFilePathStd);
        CHECK_EQUAL(true, result);
        result = paramsPreload.LoadFromFile(paramsFilePathPreload);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        asMethodCalibratorSingle calibrator1;
        wxString dataPredictorFilePath = wxFileName::GetCwd();
        dataPredictorFilePath.Append("/files/");
        calibrator1.SetPredictorDataDir(dataPredictorFilePath);
        calibrator1.SetPredictandDB(NULL);
        asMethodCalibratorSingle calibrator2 = calibrator1;
        asResultsAnalogsDates anaDatesStd;
        asResultsAnalogsDates anaDatesPreload;

        try
        {
            int step = 0;
            bool containsNaNs = false;
            result = calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            result = calibrator2.GetAnalogsDates(anaDatesPreload, paramsPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
        }
        catch(asException& e)
        {
			wxPrintf( e.GetFullMessage());
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
    }
}

TEST(PreloadingWithPreprocessing)
{
    if (g_unitTestLongProcessing)
    {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/Processing/Method", (int)asINSERT);
        pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

		wxPrintf("Processing PreloadingWithPreprocessing\n");

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
        paramsFilePathStd.Append("/files/parameters_calibration_compare_preproc_no_preload.xml");
        paramsFilePathPreload.Append("/files/parameters_calibration_compare_preproc_preload.xml");
        result = paramsStd.LoadFromFile(paramsFilePathStd);
        CHECK_EQUAL(true, result);
        result = paramsPreload.LoadFromFile(paramsFilePathPreload);
        CHECK_EQUAL(true, result);

        // Proceed to the calculations
        asMethodCalibratorSingle calibrator1;
        wxString dataPredictorFilePath = wxFileName::GetCwd();
        dataPredictorFilePath.Append("/files/");
        calibrator1.SetPredictorDataDir(dataPredictorFilePath);
        calibrator1.SetPredictandDB(NULL);
        asMethodCalibratorSingle calibrator2 = calibrator1;
        asResultsAnalogsDates anaDatesStd;
        asResultsAnalogsDates anaDatesPreload;

        try
        {
            int step = 0;
            bool containsNaNs = false;
            result = calibrator1.GetAnalogsDates(anaDatesStd, paramsStd, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            result = calibrator2.GetAnalogsDates(anaDatesPreload, paramsPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
        }
        catch(asException& e)
        {
			wxPrintf( e.GetFullMessage());
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
    }
}

void GrenobleComparison1Preloading()
{
    if (g_unitTestLongProcessing)
    {
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
        paramsFilePath.Append("/files/parameters_calibration_R1_preload.xml");
        asParametersCalibration params;
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

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

        try
        {
            bool containsNaNs = false;
            int step = 0;
            wxString dataPredictorFilePath = wxFileName::GetCwd();
            dataPredictorFilePath.Append("/files/");
            calibrator.SetPredictorDataDir(dataPredictorFilePath);
            wxASSERT(predictand);
            calibrator.SetPredictandDB(predictand);
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
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
			wxPrintf( e.GetFullMessage());
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

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison1PreloadingMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1 with data preloading\n");

    GrenobleComparison1Preloading();
}

void GrenobleComparison1PreloadingSubset()
{
    if (g_unitTestLongProcessing)
    {
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
        paramsFilePath.Append("/files/parameters_calibration_R1_preload.xml");
        asParametersCalibration params;
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

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

        try
        {
            int step = 0;
            bool containsNaNs = false;
            wxString dataPredictorFilePath = wxFileName::GetCwd();
            dataPredictorFilePath.Append("/files/");
            calibrator.SetPredictorDataDir(dataPredictorFilePath);
            wxASSERT(predictand);
            calibrator.SetPredictandDB(predictand);
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
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
			wxPrintf( e.GetFullMessage());
            return;
        }

        // No unit test possible, as results will differ from Grenoble's results.

        // predictand pointer deleted by asMethodCalibration

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison1PreloadingSubsetMultithreaded)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison1Subset with data preloading\n");

    GrenobleComparison1PreloadingSubset();
}

TEST(SmallerSpatialArea)
{
    if (g_unitTestLongProcessing)
    {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
        pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

		wxPrintf("Processing SmallerSpatialArea\n");

        bool result;

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
        result = paramsNoPreprocNoPreload.LoadFromFile(paramsFilePathNoPreprocNoPreload);
        CHECK_EQUAL(true, result);
        result = paramsNoPreprocPreload.LoadFromFile(paramsFilePathNoPreprocPreload);
        CHECK_EQUAL(true, result);
        result = paramsPreprocNoPreload.LoadFromFile(paramsFilePathPreprocNoPreload);
        CHECK_EQUAL(true, result);
        result = paramsPreprocPreload.LoadFromFile(paramsFilePathPreprocPreload);
        CHECK_EQUAL(true, result);

        // Change spatial windows
        paramsNoPreprocNoPreload.SetPredictorXmin(0,0,5);
        paramsNoPreprocNoPreload.SetPredictorXmin(0,1,5);
        paramsNoPreprocNoPreload.SetPredictorXptsnb(0,0,3);
        paramsNoPreprocNoPreload.SetPredictorXptsnb(0,1,3);
        paramsNoPreprocNoPreload.SetPredictorYmin(0,0,42.5);
        paramsNoPreprocNoPreload.SetPredictorYmin(0,1,42.5);
        paramsNoPreprocNoPreload.SetPredictorYptsnb(0,0,3);
        paramsNoPreprocNoPreload.SetPredictorYptsnb(0,1,3);

        paramsNoPreprocPreload.SetPredictorXmin(0,0,5);
        paramsNoPreprocPreload.SetPredictorXmin(0,1,5);
        paramsNoPreprocPreload.SetPredictorXptsnb(0,0,3);
        paramsNoPreprocPreload.SetPredictorXptsnb(0,1,3);
        paramsNoPreprocPreload.SetPredictorYmin(0,0,42.5);
        paramsNoPreprocPreload.SetPredictorYmin(0,1,42.5);
        paramsNoPreprocPreload.SetPredictorYptsnb(0,0,3);
        paramsNoPreprocPreload.SetPredictorYptsnb(0,1,3);

        paramsPreprocNoPreload.SetPredictorXmin(0,0,5);
        paramsPreprocNoPreload.SetPredictorXmin(0,1,5);
        paramsPreprocNoPreload.SetPredictorXptsnb(0,0,3);
        paramsPreprocNoPreload.SetPredictorXptsnb(0,1,3);
        paramsPreprocNoPreload.SetPredictorYmin(0,0,42.5);
        paramsPreprocNoPreload.SetPredictorYmin(0,1,42.5);
        paramsPreprocNoPreload.SetPredictorYptsnb(0,0,3);
        paramsPreprocNoPreload.SetPredictorYptsnb(0,1,3);

        paramsPreprocPreload.SetPredictorXmin(0,0,5);
        paramsPreprocPreload.SetPredictorXmin(0,1,5);
        paramsPreprocPreload.SetPredictorXptsnb(0,0,3);
        paramsPreprocPreload.SetPredictorXptsnb(0,1,3);
        paramsPreprocPreload.SetPredictorYmin(0,0,42.5);
        paramsPreprocPreload.SetPredictorYmin(0,1,42.5);
        paramsPreprocPreload.SetPredictorYptsnb(0,0,3);
        paramsPreprocPreload.SetPredictorYptsnb(0,1,3);

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

        try
        {
            int step = 0;
            bool containsNaNs = false;
            result = calibrator1.GetAnalogsDates(anaDatesNoPreprocNoPreload, paramsNoPreprocNoPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            result = calibrator2.GetAnalogsDates(anaDatesNoPreprocPreload, paramsNoPreprocPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            result = calibrator3.GetAnalogsDates(anaDatesPreprocNoPreload, paramsPreprocNoPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            result = calibrator4.GetAnalogsDates(anaDatesPreprocPreload, paramsPreprocPreload, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
        }
        catch(asException& e)
        {
			wxPrintf( e.GetFullMessage());
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

        CHECK_EQUAL(datesNoPreprocNoPreload.cols(),datesNoPreprocPreload.cols());
        CHECK_EQUAL(datesNoPreprocNoPreload.rows(),datesNoPreprocPreload.rows());
        CHECK_EQUAL(datesNoPreprocNoPreload.cols(),datesPreprocNoPreload.cols());
        CHECK_EQUAL(datesNoPreprocNoPreload.rows(),datesPreprocNoPreload.rows());
        CHECK_EQUAL(datesNoPreprocNoPreload.cols(),datesPreprocPreload.cols());
        CHECK_EQUAL(datesNoPreprocNoPreload.rows(),datesPreprocPreload.rows());

        CHECK_EQUAL(criteriaNoPreprocNoPreload.cols(),criteriaNoPreprocPreload.cols());
        CHECK_EQUAL(criteriaNoPreprocNoPreload.rows(),criteriaNoPreprocPreload.rows());
        CHECK_EQUAL(criteriaNoPreprocNoPreload.cols(),criteriaPreprocNoPreload.cols());
        CHECK_EQUAL(criteriaNoPreprocNoPreload.rows(),criteriaPreprocNoPreload.rows());
        CHECK_EQUAL(criteriaNoPreprocNoPreload.cols(),criteriaPreprocPreload.cols());
        CHECK_EQUAL(criteriaNoPreprocNoPreload.rows(),criteriaPreprocPreload.rows());

        for (int i=0; i<datesNoPreprocNoPreload.rows(); i++)
        {
            for (int j=0; j<datesNoPreprocNoPreload.cols(); j++)
            {
                CHECK_EQUAL(datesNoPreprocNoPreload.coeff(i,j), datesNoPreprocPreload.coeff(i,j));
                CHECK_EQUAL(criteriaNoPreprocNoPreload.coeff(i,j), criteriaNoPreprocPreload.coeff(i,j));
                CHECK_EQUAL(datesNoPreprocNoPreload.coeff(i,j), datesPreprocNoPreload.coeff(i,j));
                CHECK_EQUAL(criteriaNoPreprocNoPreload.coeff(i,j), criteriaPreprocNoPreload.coeff(i,j));
                CHECK_EQUAL(datesNoPreprocNoPreload.coeff(i,j), datesPreprocPreload.coeff(i,j));
                CHECK_EQUAL(criteriaNoPreprocNoPreload.coeff(i,j), criteriaPreprocPreload.coeff(i,j));
            }
        }
    }
}

void GrenobleComparison2Preloading()
{
    if (g_unitTestLongProcessing)
    {
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
        paramsFilePath.Append("/files/parameters_calibration_R2_preload.xml");
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

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

        try
        {
            int step = 0;
            bool containsNaNs = false;
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            step++;
            result = calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
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
			wxPrintf( e.GetFullMessage());
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

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2PreloadingProcessingMethodMultithreads)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asCOEFF_NOVAR);

	wxPrintf("Processing GrenobleComparison2 preloading with the multithreaded option\n");

    GrenobleComparison2Preloading();
}

TEST(GrenobleComparison2PreloadingProcessingMethodInsert)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/Method", (int)asINSERT);
    pConfig->Write("/Processing/LinAlgebra", (int)asCOEFF_NOVAR);

	wxPrintf("Processing GrenobleComparison2 preloading with the array insertion option\n");

    GrenobleComparison2Preloading();
}

void GrenobleComparison2SavingIntermediateResults()
{
    if (g_unitTestLongProcessing)
    {
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
        paramsFilePath.Append("/files/parameters_calibration_R2_calib_period.xml");
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

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

        try
        {
            int step = 0;
            bool containsNaNs = false;

            // Create
            result = calibrator.GetAnalogsDates(anaDates1, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            // Reload
            result = calibrator.GetAnalogsDates(anaDates2, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            step++;
            // Create
            result = calibrator.GetAnalogsSubDates(anaSubDates1, params, anaDates2, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            // Reload
            result = calibrator.GetAnalogsSubDates(anaSubDates2, params, anaDates2, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
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
			wxPrintf( e.GetFullMessage());
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

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2SavingIntermediateResults)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asCOEFF);

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

	wxPrintf("Processing GrenobleComparison2 with saving/loading of intermediate results\n");

    GrenobleComparison2SavingIntermediateResults();

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
}

void GrenobleComparison2MergeByHalfAndMultiply()
{
    if (g_unitTestLongProcessing)
    {
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
        paramsFilePath.Append("/files/parameters_calibration_R2_calib_period_merge_by_half.xml");
        result = params.LoadFromFile(paramsFilePath);
        CHECK_EQUAL(true, result);

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

        try
        {
            int step = 0;
            bool containsNaNs = false;
            result = calibrator.GetAnalogsDates(anaDates, params, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
            step++;
            result = calibrator.GetAnalogsSubDates(anaSubDates, params, anaDates, step, containsNaNs);
            CHECK_EQUAL(true, result);
            CHECK_EQUAL(false, containsNaNs);
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
			wxPrintf( e.GetFullMessage());
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

        asRemoveDir(tmpDir);
    }
}

TEST(GrenobleComparison2MergeByHalfAndMultiply)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/Method", (int)asMULTITHREADS);
    pConfig->Write("/Processing/LinAlgebra", (int)asLIN_ALGEBRA_NOVAR);

	wxPrintf("Processing GrenobleComparison2 with MergeByHalfAndMultiply preprocessing\n");

    GrenobleComparison2MergeByHalfAndMultiply();
}

}

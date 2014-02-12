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
#include <asForecastScore.h>
#include <asForecastScoreFinal.h>
#include <asFileAscii.h>

#include "UnitTest++.h"

namespace
{

TEST(ProcessCRPSapproxRectangle)
{
	wxString str("Testing forecast scores...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());
	
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSAR);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.00001);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSexactPrimitive)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("CRPSEP");

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = max(target/500,(float)0.00002);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSapproxRectangle1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("CRPSAR");

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.00002);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSexactPrimitive1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSEP);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = max(target/100,(float)0.00005);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSapproxRectangle30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSAR);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.00002);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSexactPrimitive30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSEP);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = max(target/100,(float)0.0001);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSsharpnessApproxRectangle)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSsharpnessAR);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSsharpnessExactPrimitive)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("CRPSsharpnessEP");

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = max(target/500,(float)0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSsharpnessApproxRectangle1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("CRPSsharpnessAR");

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.0002); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSsharpnessExactPrimitive1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSsharpnessEP);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = max(target/100,(float)0.0005); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSsharpnessApproxRectangle30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSsharpnessAR);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSsharpnessExactPrimitive30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSsharpnessEP);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = max(target/100,(float)0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSaccuracyApproxRectangle)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSaccuracyAR);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.0003);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSaccuracyExactPrimitive)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("CRPSaccuracyEP");

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = max(target/500,(float)0.0003);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSaccuracyApproxRectangle1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("CRPSaccuracyAR");

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.0001);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSaccuracyExactPrimitive1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 50;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSaccuracyEP);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = max(target/100,(float)0.00005);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSaccuracyApproxRectangle30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSaccuracyAR);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = max(target/1000,(float)0.0005);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(ProcessCRPSaccuracyExactPrimitive30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/forecast_score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbtests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nanalogs = 30;
    Array1DFloat ForcastVals;
    ForcastVals.resize(nanalogs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance(asForecastScore::CRPSaccuracyEP);

    for (int i_test=0;i_test<nbtests;i_test++)
    {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float ObservedVal = sqrt(file.GetFloat()/P10);
        file.SkipLines(2);
        for (int i_ana=0; i_ana<nanalogs; i_ana++)
        {
            file.SkipElements(4);
            ForcastVals[i_ana] = sqrt(file.GetFloat()/P10);
            file.SkipLines(1);
        }

        float result = score->Assess(ObservedVal, ForcastVals, nanalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = max(target/100,(float)0.001);
        CHECK_CLOSE(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

void InitConstantDistribution(Array2DFloat &vecForecast, Array1DFloat &vecObs)
{
    // Time
    int timeLength = 410;

    // Resize the containers
    int nanalogs = 50;
    vecForecast = Array2DFloat::Zero(timeLength, nanalogs);
    vecObs = Array1DFloat::Zero(timeLength);
    Array1DFloat singleDay = Array1DFloat::Zero(nanalogs);

    // Not forecasted and no event
    for (int i_time=0;i_time<timeLength;i_time++)
    {
        vecObs[i_time] = asTools::Random(0.0, 0.4999999);
        for (int i_ana=0;i_ana<nanalogs;i_ana++)
        {
            singleDay[i_ana] = asTools::Random(0.0, 0.4999999);
        }
        asTools::SortArray(&singleDay[0],&singleDay[nanalogs-1],Asc);
        vecForecast.row(i_time) = singleDay;
    }

    // Add 28 events forecasted and observed
    vecForecast.row(31).fill(0.54f);
    vecForecast.row(51).fill(0.59f);
    vecForecast.row(71).fill(0.98f);
    vecForecast.row(91).fill(0.95f);
    vecForecast.row(101).fill(0.76f);
    vecForecast.row(181).fill(0.69f);
    vecForecast.row(161).fill(0.58f);
    vecForecast.row(241).fill(0.77f);
    vecForecast.row(371).fill(0.87f);
    vecForecast.row(391).fill(0.68f);
    vecForecast.row(401).fill(0.90f);
    vecObs[31] = 0.54f;
    vecObs[51] = 0.59f;
    vecObs[71] = 0.98f;
    vecObs[91] = 0.95f;
    vecObs[101] = 0.76f;
    vecObs[181] = 0.69f;
    vecObs[161] = 0.58f;
    vecObs[241] = 0.77f;
    vecObs[371] = 0.87f;
    vecObs[391] = 0.68f;
    vecObs[401] = 0.90f;

    // Add 17 events forecasted but not observed
    vecForecast.row(22).fill(0.86f);
    vecForecast.row(42).fill(0.69f);
    vecForecast.row(62).fill(0.60f);
    vecForecast.row(72).fill(0.58f);
    vecForecast.row(82).fill(0.55f);
    vecForecast.row(92).fill(0.589f);
    vecForecast.row(132).fill(0.59f);
    vecForecast.row(152).fill(0.69f);
    vecForecast.row(182).fill(0.97f);
    vecForecast.row(252).fill(0.97f);
    vecForecast.row(262).fill(0.78f);
    vecForecast.row(282).fill(0.68f);
    vecForecast.row(292).fill(0.69f);
    vecForecast.row(352).fill(0.59f);
    vecForecast.row(372).fill(0.95f);
    vecForecast.row(392).fill(0.96f);
    vecForecast.row(402).fill(0.79f);
    vecObs[22] = 0.36f;
    vecObs[42] = 0.49f;
    vecObs[62] = 0.20f;
    vecObs[72] = 0.18f;
    vecObs[82] = 0.05f;
    vecObs[92] = 0.09f;
    vecObs[132] = 0.29f;
    vecObs[152] = 0.49f;
    vecObs[182] = 0.37f;
    vecObs[252] = 0.27f;
    vecObs[262] = 0.28f;
    vecObs[282] = 0.38f;
    vecObs[292] = 0.29f;
    vecObs[352] = 0.49f;
    vecObs[372] = 0.45f;
    vecObs[392] = 0.26f;
    vecObs[402] = 0.19f;

    // Add 9 events not forecasted but observed
    vecForecast.row(33).fill(0.24f);
    vecForecast.row(53).fill(0.32f);
    vecForecast.row(83).fill(0.13f);
    vecForecast.row(163).fill(0.14f);
    vecForecast.row(173).fill(0.46f);
    vecForecast.row(183).fill(0.37f);
    vecForecast.row(273).fill(0.38f);
    vecForecast.row(283).fill(0.48f);
    vecForecast.row(373).fill(0.39f);
    vecObs[33] = 0.64f;
    vecObs[53] = 0.72f;
    vecObs[83] = 0.83f;
    vecObs[163] = 0.74f;
    vecObs[173] = 0.96f;
    vecObs[183] = 0.77f;
    vecObs[273] = 0.58f;
    vecObs[283] = 0.58f;
    vecObs[373] = 0.69f;
}

void InitRealisticDistribution(Array2DFloat &vecForecast, Array1DFloat &vecObs)
{
    // Data from Wilks (2006)

    // Time
    int timeLength = 2803;

    // Resize the containers
    int nanalogs = 50;
    vecForecast = Array2DFloat::Zero(timeLength, nanalogs);
    vecObs = Array1DFloat::Zero(timeLength);
    Array1DFloat singleDay = Array1DFloat::Zero(nanalogs);

    // Not forecasted and no event
    for (int i_time=0;i_time<timeLength;i_time++)
    {
        vecObs[i_time] = asTools::Random(0.0, 0.5999999);
        for (int i_ana=0;i_ana<nanalogs;i_ana++)
        {
            singleDay[i_ana] = asTools::Random(0.0, 0.5999999);
        }
        asTools::SortArray(&singleDay[0],&singleDay[nanalogs-1],Asc);
        vecForecast.row(i_time) = singleDay;
    }

    // Add 28 events forecasted and observed
    Array1DInt indicesA(28);
    indicesA << 11,21,31,41,51,161,171,181,191,301,311,321,1131,1141,1151,1161,1171,1681,1691,1701,1711,1721,2231,2241,2251,2261,2271,2281;
    for (int i=0; i<indicesA.size(); i++)
    {
        int i_time = indicesA[i];
        vecObs[i_time] = asTools::Random(0.6, 1.0);
        for (int i_ana=0;i_ana<20;i_ana++)
        {
            singleDay[i_ana] = asTools::Random(0.0, 0.5999999);
        }
        for (int i_ana=20;i_ana<50;i_ana++)
        {
            singleDay[i_ana] = asTools::Random(0.6, 1.0);
        }
        asTools::SortArray(&singleDay[0],&singleDay[nanalogs-1],Asc);
        vecForecast.row(i_time) = singleDay;
    }

    // Add 17 events forecasted but not observed
    Array1DInt indicesB(72);
    indicesB << 12,22,32,42,52,62,72,82,92,102,112,122,132,142,152,162,372,382,392,402,412,422,432,442,452,462,472,482,492,502,512,522,832,842,852,862,872,882,892,902,912,922,932,942,952,962,972,982,1492,1502,1512,1522,1532,1542,1552,1562,1572,1582,1592,1602,1612,1622,2132,2142,2152,2162,2172,2182,2192,2202,2212,2222;
    for (int i=0; i<indicesB.size(); i++)
    {
        int i_time = indicesB[i];
        vecObs[i_time] = asTools::Random(0.0, 0.5999999);
        for (int i_ana=0;i_ana<20;i_ana++)
        {
            singleDay[i_ana] = asTools::Random(0.0, 0.5999999);
        }
        for (int i_ana=20;i_ana<50;i_ana++)
        {
            singleDay[i_ana] = asTools::Random(0.6, 1.0);
        }
        asTools::SortArray(&singleDay[0],&singleDay[nanalogs-1],Asc);
        vecForecast.row(i_time) = singleDay;
    }

    // Add 9 events not forecasted but observed
    Array1DInt indicesC(23);
    indicesC << 13,23,33,43,53,63,73,83,93,103,113,223,233,243,653,663,673,1183,1193,1203,1213,1223,1233;
    for (int i=0; i<indicesC.size(); i++)
    {
        int i_time = indicesC[i];
        vecObs[i_time] = asTools::Random(0.6, 1.0);
    }

}

TEST(ProcessPCwithConstantDistribution)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitConstantDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("PC");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.5f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("PC", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.936585, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessPC)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("PC");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("PC", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.966, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessTS)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("TS");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("TS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.228, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessBIAS)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("BIAS");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("BIAS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(1.96, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessFARA)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("FARA");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("FARA", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.720, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessH)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("H");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("H", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.549, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessF)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("F");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("F", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.0262, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessHSS)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("HSS");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("HSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.355, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessPSS)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("PSS");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("PSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.523, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessGSS)
{
    // Create data
    Array2DFloat vecForecast;
    Array1DFloat vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("GSS");
    score->SetPercentile(0.5f);
    score->SetThreshold(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 50);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("GSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    CHECK_CLOSE(0.216, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessMAE)
{
    // Sizes
    int timeLength = 10;
    int nanalogs = 20;

    // Resize the containers
    Array2DFloat vecForecast = Array2DFloat::Zero(timeLength, nanalogs);
    Array1DFloat vecObs = Array1DFloat::Zero(timeLength);
    Array1DFloat singleDay = Array1DFloat::Zero(nanalogs);

    // Not forecasted and no event
    vecForecast.row(0) << 0.02f,0.092646306f,0.139052338f,0.197637696f,0.229360704f,0.259909806f,0.298701546f,0.390238317f,0.407640012f,0.452575894f,0.512074354f,0.512345829f,0.58075933f,0.647425783f,0.654962539f,0.686593503f,0.729810476f,0.755282455f,0.799893526f,0.827401513f;
    vecForecast.row(1) << 0.02f,0.058561017f,0.127939716f,0.171685632f,0.265536249f,0.286160135f,0.315265848f,0.373659704f,0.3741501f,0.458286985f,0.506647511f,0.52196153f,0.610837661f,0.648162317f,0.651138364f,0.742684806f,0.80394142f,0.827924274f,0.88050801f,0.883691337f;
    vecForecast.row(2) << 0.02f,0.057409007f,0.124060844f,0.12989179f,0.194506231f,0.238944812f,0.262222184f,0.274957116f,0.276758707f,0.299777457f,0.308798466f,0.335768931f,0.407246414f,0.482673721f,0.530500548f,0.552122915f,0.636896541f,0.703442086f,0.756793177f,0.801346686f;
    vecForecast.row(3) << 0.02f,0.092411597f,0.117131378f,0.15816281f,0.215819448f,0.24559958f,0.250436984f,0.315896104f,0.357809806f,0.41176128f,0.428890994f,0.502444147f,0.510156521f,0.531216004f,0.627005158f,0.679551953f,0.719490245f,0.752477718f,0.758531907f,0.842848077f;
    vecForecast.row(4) << 0.02f,0.025565194f,0.124927271f,0.163237889f,0.182254672f,0.183216729f,0.229018135f,0.309541163f,0.397108137f,0.464487554f,0.545250143f,0.62989469f,0.727740022f,0.739352757f,0.820597597f,0.914068845f,0.956546342f,0.996502564f,1.024902501f,1.038549464f;
    vecForecast.row(5) << 0.02f,0.083376876f,0.140626298f,0.206117695f,0.218892839f,0.234828446f,0.328446981f,0.370601439f,0.417945902f,0.452067833f,0.525719917f,0.612793799f,0.648267108f,0.692725339f,0.694307008f,0.696266998f,0.794462364f,0.861882906f,0.910444299f,0.98822941f;
    vecForecast.row(6) << 0.02f,0.064229562f,0.09309693f,0.126129382f,0.22445095f,0.252971047f,0.348992863f,0.42909501f,0.519460404f,0.550894836f,0.643772657f,0.670622479f,0.688459436f,0.761704166f,0.843085811f,0.942577325f,1.001365175f,1.013441683f,1.041955139f,1.058193308f;
    vecForecast.row(7) << 0.02f,0.026738614f,0.095937412f,0.142691197f,0.215824523f,0.265994552f,0.320279392f,0.416087902f,0.432058177f,0.449941177f,0.466638011f,0.491397644f,0.569040335f,0.614604226f,0.657455658f,0.754066417f,0.826451172f,0.899028592f,0.964815104f,1.012976654f;
    vecForecast.row(8) << 0.02f,0.085296508f,0.183380599f,0.243443873f,0.273040713f,0.273055653f,0.325655881f,0.370962958f,0.376225608f,0.458607287f,0.486447729f,0.580692959f,0.596512866f,0.615277217f,0.702622102f,0.789096489f,0.794578027f,0.824465809f,0.907287888f,0.953155395f;
    vecForecast.row(9) << 0.02f,0.064684235f,0.094707249f,0.131646633f,0.173289652f,0.216579839f,0.241963985f,0.313384425f,0.321065805f,0.361266365f,0.364172913f,0.367698584f,0.438098064f,0.523397878f,0.590133347f,0.661338069f,0.733570663f,0.8022949f,0.821953293f,0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("MAE");
    score->SetPercentile(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 20);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("MAE", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    CHECK_CLOSE(0.311554, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessRMSE)
{
    // Sizes
    int timeLength = 10;
    int nanalogs = 20;

    // Resize the containers
    Array2DFloat vecForecast = Array2DFloat::Zero(timeLength, nanalogs);
    Array1DFloat vecObs = Array1DFloat::Zero(timeLength);
    Array1DFloat singleDay = Array1DFloat::Zero(nanalogs);

    // Not forecasted and no event
    vecForecast.row(0) << 0.02f,0.092646306f,0.139052338f,0.197637696f,0.229360704f,0.259909806f,0.298701546f,0.390238317f,0.407640012f,0.452575894f,0.512074354f,0.512345829f,0.58075933f,0.647425783f,0.654962539f,0.686593503f,0.729810476f,0.755282455f,0.799893526f,0.827401513f;
    vecForecast.row(1) << 0.02f,0.058561017f,0.127939716f,0.171685632f,0.265536249f,0.286160135f,0.315265848f,0.373659704f,0.3741501f,0.458286985f,0.506647511f,0.52196153f,0.610837661f,0.648162317f,0.651138364f,0.742684806f,0.80394142f,0.827924274f,0.88050801f,0.883691337f;
    vecForecast.row(2) << 0.02f,0.057409007f,0.124060844f,0.12989179f,0.194506231f,0.238944812f,0.262222184f,0.274957116f,0.276758707f,0.299777457f,0.308798466f,0.335768931f,0.407246414f,0.482673721f,0.530500548f,0.552122915f,0.636896541f,0.703442086f,0.756793177f,0.801346686f;
    vecForecast.row(3) << 0.02f,0.092411597f,0.117131378f,0.15816281f,0.215819448f,0.24559958f,0.250436984f,0.315896104f,0.357809806f,0.41176128f,0.428890994f,0.502444147f,0.510156521f,0.531216004f,0.627005158f,0.679551953f,0.719490245f,0.752477718f,0.758531907f,0.842848077f;
    vecForecast.row(4) << 0.02f,0.025565194f,0.124927271f,0.163237889f,0.182254672f,0.183216729f,0.229018135f,0.309541163f,0.397108137f,0.464487554f,0.545250143f,0.62989469f,0.727740022f,0.739352757f,0.820597597f,0.914068845f,0.956546342f,0.996502564f,1.024902501f,1.038549464f;
    vecForecast.row(5) << 0.02f,0.083376876f,0.140626298f,0.206117695f,0.218892839f,0.234828446f,0.328446981f,0.370601439f,0.417945902f,0.452067833f,0.525719917f,0.612793799f,0.648267108f,0.692725339f,0.694307008f,0.696266998f,0.794462364f,0.861882906f,0.910444299f,0.98822941f;
    vecForecast.row(6) << 0.02f,0.064229562f,0.09309693f,0.126129382f,0.22445095f,0.252971047f,0.348992863f,0.42909501f,0.519460404f,0.550894836f,0.643772657f,0.670622479f,0.688459436f,0.761704166f,0.843085811f,0.942577325f,1.001365175f,1.013441683f,1.041955139f,1.058193308f;
    vecForecast.row(7) << 0.02f,0.026738614f,0.095937412f,0.142691197f,0.215824523f,0.265994552f,0.320279392f,0.416087902f,0.432058177f,0.449941177f,0.466638011f,0.491397644f,0.569040335f,0.614604226f,0.657455658f,0.754066417f,0.826451172f,0.899028592f,0.964815104f,1.012976654f;
    vecForecast.row(8) << 0.02f,0.085296508f,0.183380599f,0.243443873f,0.273040713f,0.273055653f,0.325655881f,0.370962958f,0.376225608f,0.458607287f,0.486447729f,0.580692959f,0.596512866f,0.615277217f,0.702622102f,0.789096489f,0.794578027f,0.824465809f,0.907287888f,0.953155395f;
    vecForecast.row(9) << 0.02f,0.064684235f,0.094707249f,0.131646633f,0.173289652f,0.216579839f,0.241963985f,0.313384425f,0.321065805f,0.361266365f,0.364172913f,0.367698584f,0.438098064f,0.523397878f,0.590133347f,0.661338069f,0.733570663f,0.8022949f,0.821953293f,0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("RMSE");
    score->SetPercentile(0.6f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 20);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("RMSE", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    CHECK_CLOSE(0.358206612, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessBS)
{
    // Sizes
    int timeLength = 10;
    int nanalogs = 20;

    // Resize the containers
    Array2DFloat vecForecast = Array2DFloat::Zero(timeLength, nanalogs);
    Array1DFloat vecObs = Array1DFloat::Zero(timeLength);
    Array1DFloat singleDay = Array1DFloat::Zero(nanalogs);

    // Not forecasted and no event
    vecForecast.row(0) << 0.02f,0.092646306f,0.139052338f,0.197637696f,0.229360704f,0.259909806f,0.298701546f,0.390238317f,0.407640012f,0.452575894f,0.512074354f,0.512345829f,0.58075933f,0.647425783f,0.654962539f,0.686593503f,0.729810476f,0.755282455f,0.799893526f,0.827401513f;
    vecForecast.row(1) << 0.02f,0.058561017f,0.127939716f,0.171685632f,0.265536249f,0.286160135f,0.315265848f,0.373659704f,0.3741501f,0.458286985f,0.506647511f,0.52196153f,0.610837661f,0.648162317f,0.651138364f,0.742684806f,0.80394142f,0.827924274f,0.88050801f,0.883691337f;
    vecForecast.row(2) << 0.02f,0.057409007f,0.124060844f,0.12989179f,0.194506231f,0.238944812f,0.262222184f,0.274957116f,0.276758707f,0.299777457f,0.308798466f,0.335768931f,0.407246414f,0.482673721f,0.530500548f,0.552122915f,0.636896541f,0.703442086f,0.756793177f,0.801346686f;
    vecForecast.row(3) << 0.02f,0.092411597f,0.117131378f,0.15816281f,0.215819448f,0.24559958f,0.250436984f,0.315896104f,0.357809806f,0.41176128f,0.428890994f,0.502444147f,0.510156521f,0.531216004f,0.627005158f,0.679551953f,0.719490245f,0.752477718f,0.758531907f,0.842848077f;
    vecForecast.row(4) << 0.02f,0.025565194f,0.124927271f,0.163237889f,0.182254672f,0.183216729f,0.229018135f,0.309541163f,0.397108137f,0.464487554f,0.545250143f,0.62989469f,0.727740022f,0.739352757f,0.820597597f,0.914068845f,0.956546342f,0.996502564f,1.024902501f,1.038549464f;
    vecForecast.row(5) << 0.02f,0.083376876f,0.140626298f,0.206117695f,0.218892839f,0.234828446f,0.328446981f,0.370601439f,0.417945902f,0.452067833f,0.525719917f,0.612793799f,0.648267108f,0.692725339f,0.694307008f,0.696266998f,0.794462364f,0.861882906f,0.910444299f,0.98822941f;
    vecForecast.row(6) << 0.02f,0.064229562f,0.09309693f,0.126129382f,0.22445095f,0.252971047f,0.348992863f,0.42909501f,0.519460404f,0.550894836f,0.643772657f,0.670622479f,0.688459436f,0.761704166f,0.843085811f,0.942577325f,1.001365175f,1.013441683f,1.041955139f,1.058193308f;
    vecForecast.row(7) << 0.02f,0.026738614f,0.095937412f,0.142691197f,0.215824523f,0.265994552f,0.320279392f,0.416087902f,0.432058177f,0.449941177f,0.466638011f,0.491397644f,0.569040335f,0.614604226f,0.657455658f,0.754066417f,0.826451172f,0.899028592f,0.964815104f,1.012976654f;
    vecForecast.row(8) << 0.02f,0.085296508f,0.183380599f,0.243443873f,0.273040713f,0.273055653f,0.325655881f,0.370962958f,0.376225608f,0.458607287f,0.486447729f,0.580692959f,0.596512866f,0.615277217f,0.702622102f,0.789096489f,0.794578027f,0.824465809f,0.907287888f,0.953155395f;
    vecForecast.row(9) << 0.02f,0.064684235f,0.094707249f,0.131646633f,0.173289652f,0.216579839f,0.241963985f,0.313384425f,0.321065805f,0.361266365f,0.364172913f,0.367698584f,0.438098064f,0.523397878f,0.590133347f,0.661338069f,0.733570663f,0.8022949f,0.821953293f,0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("BS");
    score->SetPercentile(0.6f);
    score->SetThreshold(0.4f);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 20);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("BS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    CHECK_CLOSE(0.187428558, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(ProcessBSS)
{
    // Sizes
    int timeLength = 10;
    int nanalogs = 20;

    // Resize the containers
    Array2DFloat vecForecast = Array2DFloat::Zero(timeLength, nanalogs);
    Array1DFloat vecObs = Array1DFloat::Zero(timeLength);
    Array1DFloat singleDay = Array1DFloat::Zero(nanalogs);

    // Not forecasted and no event
    vecForecast.row(0) << 0.02f,0.092646306f,0.139052338f,0.197637696f,0.229360704f,0.259909806f,0.298701546f,0.390238317f,0.407640012f,0.452575894f,0.512074354f,0.512345829f,0.58075933f,0.647425783f,0.654962539f,0.686593503f,0.729810476f,0.755282455f,0.799893526f,0.827401513f;
    vecForecast.row(1) << 0.02f,0.058561017f,0.127939716f,0.171685632f,0.265536249f,0.286160135f,0.315265848f,0.373659704f,0.3741501f,0.458286985f,0.506647511f,0.52196153f,0.610837661f,0.648162317f,0.651138364f,0.742684806f,0.80394142f,0.827924274f,0.88050801f,0.883691337f;
    vecForecast.row(2) << 0.02f,0.057409007f,0.124060844f,0.12989179f,0.194506231f,0.238944812f,0.262222184f,0.274957116f,0.276758707f,0.299777457f,0.308798466f,0.335768931f,0.407246414f,0.482673721f,0.530500548f,0.552122915f,0.636896541f,0.703442086f,0.756793177f,0.801346686f;
    vecForecast.row(3) << 0.02f,0.092411597f,0.117131378f,0.15816281f,0.215819448f,0.24559958f,0.250436984f,0.315896104f,0.357809806f,0.41176128f,0.428890994f,0.502444147f,0.510156521f,0.531216004f,0.627005158f,0.679551953f,0.719490245f,0.752477718f,0.758531907f,0.842848077f;
    vecForecast.row(4) << 0.02f,0.025565194f,0.124927271f,0.163237889f,0.182254672f,0.183216729f,0.229018135f,0.309541163f,0.397108137f,0.464487554f,0.545250143f,0.62989469f,0.727740022f,0.739352757f,0.820597597f,0.914068845f,0.956546342f,0.996502564f,1.024902501f,1.038549464f;
    vecForecast.row(5) << 0.02f,0.083376876f,0.140626298f,0.206117695f,0.218892839f,0.234828446f,0.328446981f,0.370601439f,0.417945902f,0.452067833f,0.525719917f,0.612793799f,0.648267108f,0.692725339f,0.694307008f,0.696266998f,0.794462364f,0.861882906f,0.910444299f,0.98822941f;
    vecForecast.row(6) << 0.02f,0.064229562f,0.09309693f,0.126129382f,0.22445095f,0.252971047f,0.348992863f,0.42909501f,0.519460404f,0.550894836f,0.643772657f,0.670622479f,0.688459436f,0.761704166f,0.843085811f,0.942577325f,1.001365175f,1.013441683f,1.041955139f,1.058193308f;
    vecForecast.row(7) << 0.02f,0.026738614f,0.095937412f,0.142691197f,0.215824523f,0.265994552f,0.320279392f,0.416087902f,0.432058177f,0.449941177f,0.466638011f,0.491397644f,0.569040335f,0.614604226f,0.657455658f,0.754066417f,0.826451172f,0.899028592f,0.964815104f,1.012976654f;
    vecForecast.row(8) << 0.02f,0.085296508f,0.183380599f,0.243443873f,0.273040713f,0.273055653f,0.325655881f,0.370962958f,0.376225608f,0.458607287f,0.486447729f,0.580692959f,0.596512866f,0.615277217f,0.702622102f,0.789096489f,0.794578027f,0.824465809f,0.907287888f,0.953155395f;
    vecForecast.row(9) << 0.02f,0.064684235f,0.094707249f,0.131646633f,0.173289652f,0.216579839f,0.241963985f,0.313384425f,0.321065805f,0.361266365f,0.364172913f,0.367698584f,0.438098064f,0.523397878f,0.590133347f,0.661338069f,0.733570663f,0.8022949f,0.821953293f,0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asForecastScore* score = asForecastScore::GetInstance("BSS");
    score->SetPercentile(0.6f);
    score->SetThreshold(0.4f);

    score->ProcessScoreClimatology(vecObs,vecObs);

    Array1DFloat results = Array1DFloat::Zero(vecObs.size());
    Array1DFloat pseudoDates = Array1DFloat::Zero(vecObs.size());

    for (int i_time=0;i_time<vecObs.size();i_time++)
    {
        pseudoDates[i_time] = i_time;
        results[i_time] = score->Assess(vecObs[i_time], vecForecast.row(i_time), 20);
        CHECK(!asTools::IsNaN(results[i_time]));
    }

    asForecastScoreFinal* finalScore = asForecastScoreFinal::GetInstance("BSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0,1,1,asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    CHECK_CLOSE(0.378581034, scoreVal, 0.00001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

}


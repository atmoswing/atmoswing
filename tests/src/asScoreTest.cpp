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
#include <asScore.h>
#include <asTotalScore.h>
#include <asFileAscii.h>
#include "gtest/gtest.h"


TEST(Score, ProcessCRPSapproxRectangle)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSAR);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000, (float) 0.00001);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSexactPrimitive)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSEP");

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 500, (float) 0.00002);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSapproxRectangle1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSAR");

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000, (float) 0.00002);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSexactPrimitive1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSEP);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 100, (float) 0.00005);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSapproxRectangle30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 30;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSAR);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000, (float) 0.00002);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSexactPrimitive30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 30;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSEP);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 100, (float) 0.0001);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSsharpnessApproxRectangle)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSsharpnessAR);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000,
                                (float) 0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSsharpnessExactPrimitive)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSsharpnessEP");

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 500,
                                (float) 0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSsharpnessApproxRectangle1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSsharpnessAR");

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000,
                                (float) 0.0002); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSsharpnessExactPrimitive1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSsharpnessEP);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 100,
                                (float) 0.0005); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSsharpnessApproxRectangle30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 30;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSsharpnessAR);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000,
                                (float) 0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSsharpnessExactPrimitive30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 30;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSsharpnessEP);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(2);
        float target = file.GetFloat();

        float precision = wxMax(target / 100,
                                (float) 0.001); // The tolerance was increased as the median in not interpolated in the Grenoble score processing.
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSaccuracyApproxRectangle)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSaccuracyAR);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000, (float) 0.0003);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSaccuracyExactPrimitive)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_01.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 17;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSaccuracyEP");

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = wxMax(target / 500, (float) 0.0003);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSaccuracyApproxRectangle1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSaccuracyAR");

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000, (float) 0.0001);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSaccuracyExactPrimitive1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_02.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 50;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSaccuracyEP);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = wxMax(target / 100, (float) 0.00005);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSaccuracyApproxRectangle30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 30;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSaccuracyAR);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = wxMax(target / 1000, (float) 0.0005);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

TEST(Score, ProcessCRPSaccuracyExactPrimitive30Analogs1983)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/score_03.txt");
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Test numbers
    int nbTests = 30;

    // P10
    float P10 = 70.82f;

    // Resize the containers
    int nAnalogs = 30;
    a1f values;
    values.resize(nAnalogs);

    // Instantiate the score
    asScore *score = asScore::GetInstance(asScore::CRPSaccuracyEP);

    for (int iTest = 0; iTest < nbTests; iTest++) {
        // Skip the header
        file.SkipLines(1);

        // Load data
        file.SkipElements(3);
        float obs = (float) sqrt(file.GetFloat() / P10);
        file.SkipLines(2);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            file.SkipElements(4);
            values[iAnalog] = (float) sqrt(file.GetFloat() / P10);
            file.SkipLines(1);
        }

        float result = score->Assess(obs, values, nAnalogs);
        file.SkipLines(2);
        file.SkipElements(1);
        float target = file.GetFloat();

        float precision = wxMax(target / 100, (float) 0.001);
        EXPECT_NEAR(target, result, precision);

        // Go to header
        file.SkipLines(1);
    }
    file.Close();

    wxDELETE(score);
}

void InitConstantDistribution(a2f &vecForecast, a1f &vecObs)
{
    // Time
    int timeLength = 410;

    // Resize the containers
    int nAnalogs = 50;
    vecForecast = a2f::Zero(timeLength, nAnalogs);
    vecObs = a1f::Zero(timeLength);
    a1f singleDay = a1f::Zero(nAnalogs);

    // Not forecasted and no event
    for (int iTime = 0; iTime < timeLength; iTime++) {
        vecObs[iTime] = (float) asRandom(0.0, 0.4999999);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            singleDay[iAnalog] = (float) asRandom(0.0, 0.4999999);
        }
        asSortArray(&singleDay[0], &singleDay[nAnalogs - 1], Asc);
        vecForecast.row(iTime) = singleDay;
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

void InitRealisticDistribution(a2f &vecForecast, a1f &vecObs)
{
    // Data from Wilks (2006)

    // Time
    int timeLength = 2803;

    // Resize the containers
    int nAnalogs = 50;
    vecForecast = a2f::Zero(timeLength, nAnalogs);
    vecObs = a1f::Zero(timeLength);
    a1f singleDay = a1f::Zero(nAnalogs);

    // Not forecasted and no event
    for (int iTime = 0; iTime < timeLength; iTime++) {
        vecObs[iTime] = (float) asRandom(0.0, 0.5999999);
        for (int iAnalog = 0; iAnalog < nAnalogs; iAnalog++) {
            singleDay[iAnalog] = (float) asRandom(0.0, 0.5999999);
        }
        asSortArray(&singleDay[0], &singleDay[nAnalogs - 1], Asc);
        vecForecast.row(iTime) = singleDay;
    }

    // Add 28 events forecasted and observed
    a1i indicesA(28);
    indicesA
            << 11, 21, 31, 41, 51, 161, 171, 181, 191, 301, 311, 321, 1131, 1141, 1151, 1161, 1171, 1681, 1691, 1701, 1711, 1721, 2231, 2241, 2251, 2261, 2271, 2281;
    for (int i = 0; i < indicesA.size(); i++) {
        int iTime = indicesA[i];
        vecObs[iTime] = (float) asRandom(0.6, 1.0);
        for (int iAnalog = 0; iAnalog < 20; iAnalog++) {
            singleDay[iAnalog] = (float) asRandom(0.0, 0.5999999);
        }
        for (int iAnalog = 20; iAnalog < 50; iAnalog++) {
            singleDay[iAnalog] = asRandom(0.6, 1.0);
        }
        asSortArray(&singleDay[0], &singleDay[nAnalogs - 1], Asc);
        vecForecast.row(iTime) = singleDay;
    }

    // Add 17 events forecasted but not observed
    a1i indicesB(72);
    indicesB
            << 12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 372, 382, 392, 402, 412, 422, 432, 442, 452, 462, 472, 482, 492, 502, 512, 522, 832, 842, 852, 862, 872, 882, 892, 902, 912, 922, 932, 942, 952, 962, 972, 982, 1492, 1502, 1512, 1522, 1532, 1542, 1552, 1562, 1572, 1582, 1592, 1602, 1612, 1622, 2132, 2142, 2152, 2162, 2172, 2182, 2192, 2202, 2212, 2222;
    for (int i = 0; i < indicesB.size(); i++) {
        int iTime = indicesB[i];
        vecObs[iTime] = (float) asRandom(0.0, 0.5999999);
        for (int iAnalog = 0; iAnalog < 20; iAnalog++) {
            singleDay[iAnalog] = (float) asRandom(0.0, 0.5999999);
        }
        for (int iAnalog = 20; iAnalog < 50; iAnalog++) {
            singleDay[iAnalog] = (float) asRandom(0.6, 1.0);
        }
        asSortArray(&singleDay[0], &singleDay[nAnalogs - 1], Asc);
        vecForecast.row(iTime) = singleDay;
    }

    // Add 9 events not forecasted but observed
    a1i indicesC(23);
    indicesC
            << 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 223, 233, 243, 653, 663, 673, 1183, 1193, 1203, 1213, 1223, 1233;
    for (int i = 0; i < indicesC.size(); i++) {
        int iTime = indicesC[i];
        vecObs[iTime] = (float) asRandom(0.6, 1.0);
    }

}

TEST(Score, ProcessPCwithConstantDistribution)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitConstantDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("PC");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.5f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("PC", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.936585, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessPC)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("PC");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("PC", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.966, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessTS)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("TS");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("TS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.228, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessBIAS)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("BIAS");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("BIAS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(1.96, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessFARA)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("FARA");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("FARA", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.720, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessH)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("H");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("H", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.549, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessF)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("F");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("F", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.0262, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessHSS)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("HSS");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("HSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.355, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessPSS)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("PSS");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("PSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.523, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessGSS)
{
    // Create data
    a2f vecForecast;
    a1f vecObs;
    InitRealisticDistribution(vecForecast, vecObs);

    // Instantiate the score
    asScore *score = asScore::GetInstance("GSS");
    score->SetQuantile(0.5f);
    score->SetThreshold(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 50);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("GSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value according to Wilks (2006), p.268
    EXPECT_NEAR(0.216, scoreVal, 0.001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessMAE)
{
    // Sizes
    int timeLength = 10;
    int nAnalogs = 20;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);
    a1f singleDay = a1f::Zero(nAnalogs);

    // Not forecasted and no event
    vecForecast.row(0)
            << 0.02f, 0.092646306f, 0.139052338f, 0.197637696f, 0.229360704f, 0.259909806f, 0.298701546f, 0.390238317f, 0.407640012f, 0.452575894f, 0.512074354f, 0.512345829f, 0.58075933f, 0.647425783f, 0.654962539f, 0.686593503f, 0.729810476f, 0.755282455f, 0.799893526f, 0.827401513f;
    vecForecast.row(1)
            << 0.02f, 0.058561017f, 0.127939716f, 0.171685632f, 0.265536249f, 0.286160135f, 0.315265848f, 0.373659704f, 0.3741501f, 0.458286985f, 0.506647511f, 0.52196153f, 0.610837661f, 0.648162317f, 0.651138364f, 0.742684806f, 0.80394142f, 0.827924274f, 0.88050801f, 0.883691337f;
    vecForecast.row(2)
            << 0.02f, 0.057409007f, 0.124060844f, 0.12989179f, 0.194506231f, 0.238944812f, 0.262222184f, 0.274957116f, 0.276758707f, 0.299777457f, 0.308798466f, 0.335768931f, 0.407246414f, 0.482673721f, 0.530500548f, 0.552122915f, 0.636896541f, 0.703442086f, 0.756793177f, 0.801346686f;
    vecForecast.row(3)
            << 0.02f, 0.092411597f, 0.117131378f, 0.15816281f, 0.215819448f, 0.24559958f, 0.250436984f, 0.315896104f, 0.357809806f, 0.41176128f, 0.428890994f, 0.502444147f, 0.510156521f, 0.531216004f, 0.627005158f, 0.679551953f, 0.719490245f, 0.752477718f, 0.758531907f, 0.842848077f;
    vecForecast.row(4)
            << 0.02f, 0.025565194f, 0.124927271f, 0.163237889f, 0.182254672f, 0.183216729f, 0.229018135f, 0.309541163f, 0.397108137f, 0.464487554f, 0.545250143f, 0.62989469f, 0.727740022f, 0.739352757f, 0.820597597f, 0.914068845f, 0.956546342f, 0.996502564f, 1.024902501f, 1.038549464f;
    vecForecast.row(5)
            << 0.02f, 0.083376876f, 0.140626298f, 0.206117695f, 0.218892839f, 0.234828446f, 0.328446981f, 0.370601439f, 0.417945902f, 0.452067833f, 0.525719917f, 0.612793799f, 0.648267108f, 0.692725339f, 0.694307008f, 0.696266998f, 0.794462364f, 0.861882906f, 0.910444299f, 0.98822941f;
    vecForecast.row(6)
            << 0.02f, 0.064229562f, 0.09309693f, 0.126129382f, 0.22445095f, 0.252971047f, 0.348992863f, 0.42909501f, 0.519460404f, 0.550894836f, 0.643772657f, 0.670622479f, 0.688459436f, 0.761704166f, 0.843085811f, 0.942577325f, 1.001365175f, 1.013441683f, 1.041955139f, 1.058193308f;
    vecForecast.row(7)
            << 0.02f, 0.026738614f, 0.095937412f, 0.142691197f, 0.215824523f, 0.265994552f, 0.320279392f, 0.416087902f, 0.432058177f, 0.449941177f, 0.466638011f, 0.491397644f, 0.569040335f, 0.614604226f, 0.657455658f, 0.754066417f, 0.826451172f, 0.899028592f, 0.964815104f, 1.012976654f;
    vecForecast.row(8)
            << 0.02f, 0.085296508f, 0.183380599f, 0.243443873f, 0.273040713f, 0.273055653f, 0.325655881f, 0.370962958f, 0.376225608f, 0.458607287f, 0.486447729f, 0.580692959f, 0.596512866f, 0.615277217f, 0.702622102f, 0.789096489f, 0.794578027f, 0.824465809f, 0.907287888f, 0.953155395f;
    vecForecast.row(9)
            << 0.02f, 0.064684235f, 0.094707249f, 0.131646633f, 0.173289652f, 0.216579839f, 0.241963985f, 0.313384425f, 0.321065805f, 0.361266365f, 0.364172913f, 0.367698584f, 0.438098064f, 0.523397878f, 0.590133347f, 0.661338069f, 0.733570663f, 0.8022949f, 0.821953293f, 0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("MAE");
    score->SetQuantile(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 20);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("MAE", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    EXPECT_NEAR(0.311834, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessRMSE)
{
    // Sizes
    int timeLength = 10;
    int nAnalogs = 20;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);
    a1f singleDay = a1f::Zero(nAnalogs);

    // Not forecasted and no event
    vecForecast.row(0)
            << 0.02f, 0.092646306f, 0.139052338f, 0.197637696f, 0.229360704f, 0.259909806f, 0.298701546f, 0.390238317f, 0.407640012f, 0.452575894f, 0.512074354f, 0.512345829f, 0.58075933f, 0.647425783f, 0.654962539f, 0.686593503f, 0.729810476f, 0.755282455f, 0.799893526f, 0.827401513f;
    vecForecast.row(1)
            << 0.02f, 0.058561017f, 0.127939716f, 0.171685632f, 0.265536249f, 0.286160135f, 0.315265848f, 0.373659704f, 0.3741501f, 0.458286985f, 0.506647511f, 0.52196153f, 0.610837661f, 0.648162317f, 0.651138364f, 0.742684806f, 0.80394142f, 0.827924274f, 0.88050801f, 0.883691337f;
    vecForecast.row(2)
            << 0.02f, 0.057409007f, 0.124060844f, 0.12989179f, 0.194506231f, 0.238944812f, 0.262222184f, 0.274957116f, 0.276758707f, 0.299777457f, 0.308798466f, 0.335768931f, 0.407246414f, 0.482673721f, 0.530500548f, 0.552122915f, 0.636896541f, 0.703442086f, 0.756793177f, 0.801346686f;
    vecForecast.row(3)
            << 0.02f, 0.092411597f, 0.117131378f, 0.15816281f, 0.215819448f, 0.24559958f, 0.250436984f, 0.315896104f, 0.357809806f, 0.41176128f, 0.428890994f, 0.502444147f, 0.510156521f, 0.531216004f, 0.627005158f, 0.679551953f, 0.719490245f, 0.752477718f, 0.758531907f, 0.842848077f;
    vecForecast.row(4)
            << 0.02f, 0.025565194f, 0.124927271f, 0.163237889f, 0.182254672f, 0.183216729f, 0.229018135f, 0.309541163f, 0.397108137f, 0.464487554f, 0.545250143f, 0.62989469f, 0.727740022f, 0.739352757f, 0.820597597f, 0.914068845f, 0.956546342f, 0.996502564f, 1.024902501f, 1.038549464f;
    vecForecast.row(5)
            << 0.02f, 0.083376876f, 0.140626298f, 0.206117695f, 0.218892839f, 0.234828446f, 0.328446981f, 0.370601439f, 0.417945902f, 0.452067833f, 0.525719917f, 0.612793799f, 0.648267108f, 0.692725339f, 0.694307008f, 0.696266998f, 0.794462364f, 0.861882906f, 0.910444299f, 0.98822941f;
    vecForecast.row(6)
            << 0.02f, 0.064229562f, 0.09309693f, 0.126129382f, 0.22445095f, 0.252971047f, 0.348992863f, 0.42909501f, 0.519460404f, 0.550894836f, 0.643772657f, 0.670622479f, 0.688459436f, 0.761704166f, 0.843085811f, 0.942577325f, 1.001365175f, 1.013441683f, 1.041955139f, 1.058193308f;
    vecForecast.row(7)
            << 0.02f, 0.026738614f, 0.095937412f, 0.142691197f, 0.215824523f, 0.265994552f, 0.320279392f, 0.416087902f, 0.432058177f, 0.449941177f, 0.466638011f, 0.491397644f, 0.569040335f, 0.614604226f, 0.657455658f, 0.754066417f, 0.826451172f, 0.899028592f, 0.964815104f, 1.012976654f;
    vecForecast.row(8)
            << 0.02f, 0.085296508f, 0.183380599f, 0.243443873f, 0.273040713f, 0.273055653f, 0.325655881f, 0.370962958f, 0.376225608f, 0.458607287f, 0.486447729f, 0.580692959f, 0.596512866f, 0.615277217f, 0.702622102f, 0.789096489f, 0.794578027f, 0.824465809f, 0.907287888f, 0.953155395f;
    vecForecast.row(9)
            << 0.02f, 0.064684235f, 0.094707249f, 0.131646633f, 0.173289652f, 0.216579839f, 0.241963985f, 0.313384425f, 0.321065805f, 0.361266365f, 0.364172913f, 0.367698584f, 0.438098064f, 0.523397878f, 0.590133347f, 0.661338069f, 0.733570663f, 0.8022949f, 0.821953293f, 0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("RMSE");
    score->SetQuantile(0.6f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 20);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("RMSE", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    EXPECT_NEAR(0.358484, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessBS)
{
    // Sizes
    int timeLength = 10;
    int nAnalogs = 20;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);
    a1f singleDay = a1f::Zero(nAnalogs);

    // Not forecasted and no event
    vecForecast.row(0)
            << 0.02f, 0.092646306f, 0.139052338f, 0.197637696f, 0.229360704f, 0.259909806f, 0.298701546f, 0.390238317f, 0.407640012f, 0.452575894f, 0.512074354f, 0.512345829f, 0.58075933f, 0.647425783f, 0.654962539f, 0.686593503f, 0.729810476f, 0.755282455f, 0.799893526f, 0.827401513f;
    vecForecast.row(1)
            << 0.02f, 0.058561017f, 0.127939716f, 0.171685632f, 0.265536249f, 0.286160135f, 0.315265848f, 0.373659704f, 0.3741501f, 0.458286985f, 0.506647511f, 0.52196153f, 0.610837661f, 0.648162317f, 0.651138364f, 0.742684806f, 0.80394142f, 0.827924274f, 0.88050801f, 0.883691337f;
    vecForecast.row(2)
            << 0.02f, 0.057409007f, 0.124060844f, 0.12989179f, 0.194506231f, 0.238944812f, 0.262222184f, 0.274957116f, 0.276758707f, 0.299777457f, 0.308798466f, 0.335768931f, 0.407246414f, 0.482673721f, 0.530500548f, 0.552122915f, 0.636896541f, 0.703442086f, 0.756793177f, 0.801346686f;
    vecForecast.row(3)
            << 0.02f, 0.092411597f, 0.117131378f, 0.15816281f, 0.215819448f, 0.24559958f, 0.250436984f, 0.315896104f, 0.357809806f, 0.41176128f, 0.428890994f, 0.502444147f, 0.510156521f, 0.531216004f, 0.627005158f, 0.679551953f, 0.719490245f, 0.752477718f, 0.758531907f, 0.842848077f;
    vecForecast.row(4)
            << 0.02f, 0.025565194f, 0.124927271f, 0.163237889f, 0.182254672f, 0.183216729f, 0.229018135f, 0.309541163f, 0.397108137f, 0.464487554f, 0.545250143f, 0.62989469f, 0.727740022f, 0.739352757f, 0.820597597f, 0.914068845f, 0.956546342f, 0.996502564f, 1.024902501f, 1.038549464f;
    vecForecast.row(5)
            << 0.02f, 0.083376876f, 0.140626298f, 0.206117695f, 0.218892839f, 0.234828446f, 0.328446981f, 0.370601439f, 0.417945902f, 0.452067833f, 0.525719917f, 0.612793799f, 0.648267108f, 0.692725339f, 0.694307008f, 0.696266998f, 0.794462364f, 0.861882906f, 0.910444299f, 0.98822941f;
    vecForecast.row(6)
            << 0.02f, 0.064229562f, 0.09309693f, 0.126129382f, 0.22445095f, 0.252971047f, 0.348992863f, 0.42909501f, 0.519460404f, 0.550894836f, 0.643772657f, 0.670622479f, 0.688459436f, 0.761704166f, 0.843085811f, 0.942577325f, 1.001365175f, 1.013441683f, 1.041955139f, 1.058193308f;
    vecForecast.row(7)
            << 0.02f, 0.026738614f, 0.095937412f, 0.142691197f, 0.215824523f, 0.265994552f, 0.320279392f, 0.416087902f, 0.432058177f, 0.449941177f, 0.466638011f, 0.491397644f, 0.569040335f, 0.614604226f, 0.657455658f, 0.754066417f, 0.826451172f, 0.899028592f, 0.964815104f, 1.012976654f;
    vecForecast.row(8)
            << 0.02f, 0.085296508f, 0.183380599f, 0.243443873f, 0.273040713f, 0.273055653f, 0.325655881f, 0.370962958f, 0.376225608f, 0.458607287f, 0.486447729f, 0.580692959f, 0.596512866f, 0.615277217f, 0.702622102f, 0.789096489f, 0.794578027f, 0.824465809f, 0.907287888f, 0.953155395f;
    vecForecast.row(9)
            << 0.02f, 0.064684235f, 0.094707249f, 0.131646633f, 0.173289652f, 0.216579839f, 0.241963985f, 0.313384425f, 0.321065805f, 0.361266365f, 0.364172913f, 0.367698584f, 0.438098064f, 0.523397878f, 0.590133347f, 0.661338069f, 0.733570663f, 0.8022949f, 0.821953293f, 0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("BS");
    score->SetQuantile(0.6f);
    score->SetThreshold(0.4f);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 20);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("BS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    EXPECT_NEAR(0.187771, scoreVal, 0.000001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessBSS)
{
    // Sizes
    int timeLength = 10;
    int nAnalogs = 20;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);
    a1f singleDay = a1f::Zero(nAnalogs);

    // Not forecasted and no event
    vecForecast.row(0)
            << 0.02f, 0.092646306f, 0.139052338f, 0.197637696f, 0.229360704f, 0.259909806f, 0.298701546f, 0.390238317f, 0.407640012f, 0.452575894f, 0.512074354f, 0.512345829f, 0.58075933f, 0.647425783f, 0.654962539f, 0.686593503f, 0.729810476f, 0.755282455f, 0.799893526f, 0.827401513f;
    vecForecast.row(1)
            << 0.02f, 0.058561017f, 0.127939716f, 0.171685632f, 0.265536249f, 0.286160135f, 0.315265848f, 0.373659704f, 0.3741501f, 0.458286985f, 0.506647511f, 0.52196153f, 0.610837661f, 0.648162317f, 0.651138364f, 0.742684806f, 0.80394142f, 0.827924274f, 0.88050801f, 0.883691337f;
    vecForecast.row(2)
            << 0.02f, 0.057409007f, 0.124060844f, 0.12989179f, 0.194506231f, 0.238944812f, 0.262222184f, 0.274957116f, 0.276758707f, 0.299777457f, 0.308798466f, 0.335768931f, 0.407246414f, 0.482673721f, 0.530500548f, 0.552122915f, 0.636896541f, 0.703442086f, 0.756793177f, 0.801346686f;
    vecForecast.row(3)
            << 0.02f, 0.092411597f, 0.117131378f, 0.15816281f, 0.215819448f, 0.24559958f, 0.250436984f, 0.315896104f, 0.357809806f, 0.41176128f, 0.428890994f, 0.502444147f, 0.510156521f, 0.531216004f, 0.627005158f, 0.679551953f, 0.719490245f, 0.752477718f, 0.758531907f, 0.842848077f;
    vecForecast.row(4)
            << 0.02f, 0.025565194f, 0.124927271f, 0.163237889f, 0.182254672f, 0.183216729f, 0.229018135f, 0.309541163f, 0.397108137f, 0.464487554f, 0.545250143f, 0.62989469f, 0.727740022f, 0.739352757f, 0.820597597f, 0.914068845f, 0.956546342f, 0.996502564f, 1.024902501f, 1.038549464f;
    vecForecast.row(5)
            << 0.02f, 0.083376876f, 0.140626298f, 0.206117695f, 0.218892839f, 0.234828446f, 0.328446981f, 0.370601439f, 0.417945902f, 0.452067833f, 0.525719917f, 0.612793799f, 0.648267108f, 0.692725339f, 0.694307008f, 0.696266998f, 0.794462364f, 0.861882906f, 0.910444299f, 0.98822941f;
    vecForecast.row(6)
            << 0.02f, 0.064229562f, 0.09309693f, 0.126129382f, 0.22445095f, 0.252971047f, 0.348992863f, 0.42909501f, 0.519460404f, 0.550894836f, 0.643772657f, 0.670622479f, 0.688459436f, 0.761704166f, 0.843085811f, 0.942577325f, 1.001365175f, 1.013441683f, 1.041955139f, 1.058193308f;
    vecForecast.row(7)
            << 0.02f, 0.026738614f, 0.095937412f, 0.142691197f, 0.215824523f, 0.265994552f, 0.320279392f, 0.416087902f, 0.432058177f, 0.449941177f, 0.466638011f, 0.491397644f, 0.569040335f, 0.614604226f, 0.657455658f, 0.754066417f, 0.826451172f, 0.899028592f, 0.964815104f, 1.012976654f;
    vecForecast.row(8)
            << 0.02f, 0.085296508f, 0.183380599f, 0.243443873f, 0.273040713f, 0.273055653f, 0.325655881f, 0.370962958f, 0.376225608f, 0.458607287f, 0.486447729f, 0.580692959f, 0.596512866f, 0.615277217f, 0.702622102f, 0.789096489f, 0.794578027f, 0.824465809f, 0.907287888f, 0.953155395f;
    vecForecast.row(9)
            << 0.02f, 0.064684235f, 0.094707249f, 0.131646633f, 0.173289652f, 0.216579839f, 0.241963985f, 0.313384425f, 0.321065805f, 0.361266365f, 0.364172913f, 0.367698584f, 0.438098064f, 0.523397878f, 0.590133347f, 0.661338069f, 0.733570663f, 0.8022949f, 0.821953293f, 0.886632874f;

    vecObs << 0.3f, 0.6f, 0.7f, 0.9f, 0.3f, 0.2f, 0.1f, 0.3f, 0.1f, 0.4f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("BSS");
    score->SetQuantile(0.6f);
    score->SetThreshold(0.4f);

    score->ProcessScoreClimatology(vecObs, vecObs);

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        results[iTime] = score->Assess(vecObs[iTime], vecForecast.row(iTime), 20);
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("BSS", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Value processed on Excel
    EXPECT_NEAR(0.375521, scoreVal, 0.00001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessRankHistogram)
{
    // Sizes
    int timeLength = 20;
    int nAnalogs = 30;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);

    vecForecast.row(0)
            << 0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.4f, 0.7f, 0.8f, 1.0f, 1.3f, 1.7f, 3.3f, 4.9f, 4.9f, 6.0f, 6.0f, 8.6f, 9.2f, 9.5f;
    vecForecast.row(1)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.8f, 1.2f, 1.3f, 1.6f, 2.6f, 3.4f, 5.1f, 5.3f, 5.6f, 5.6f, 5.7f, 6.3f, 6.3f, 7.4f, 7.7f, 8.5f;
    vecForecast.row(2)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.8f, 1.7f, 2.1f, 2.1f, 2.5f, 3.3f, 4.1f, 4.5f, 4.6f, 5.0f, 6.4f, 7.5f, 8.6f;
    vecForecast.row(3)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.5f, 1.3f, 2.9f, 3.0f, 3.1f, 3.9f, 4.1f, 5.1f, 5.6f, 6.1f, 7.3f, 9.3f;
    vecForecast.row(4)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.9f, 1.8f, 2.1f, 3.5f, 6.2f, 7.0f, 7.1f, 7.4f, 8.1f, 8.6f, 8.6f, 9.1f, 9.2f, 9.8f, 9.9f, 10.0f;
    vecForecast.row(5)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.6f, 0.7f, 0.8f, 1.9f, 3.4f, 4.1f, 4.3f, 5.1f, 5.2f, 5.6f, 5.8f, 6.4f, 6.5f, 6.9f, 7.8f, 9.2f, 9.5f, 9.6f;
    vecForecast.row(6)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.2f, 0.5f, 0.6f, 1.6f, 2.4f, 2.4f, 3.6f, 4.1f, 4.2f, 5.0f, 5.2f, 5.7f, 5.9f, 5.9f, 6.5f, 7.5f, 7.7f, 8.4f, 9.3f;
    vecForecast.row(7)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f, 1.6f, 3.1f, 5.3f, 6.2f, 6.6f, 7.3f, 7.6f, 8.1f, 8.7f, 8.9f, 9.1f, 9.1f, 9.6f, 9.7f;
    vecForecast.row(8)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 2.6f, 3.0f, 3.3f, 3.8f, 5.9f, 6.5f, 6.7f, 6.9f, 7.6f, 9.2f;
    vecForecast.row(9)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.1f, 3.7f, 4.2f, 5.1f, 5.7f, 6.5f, 8.6f, 8.8f, 9.2f, 9.4f;
    vecForecast.row(10)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f, 1.4f, 3.2f, 3.8f, 5.0f, 5.5f, 6.0f, 6.2f, 6.2f, 7.4f, 8.1f, 8.2f, 8.4f, 9.9f;
    vecForecast.row(11)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.7f, 1.8f, 4.4f, 4.9f, 5.1f, 5.9f, 6.2f, 7.8f, 8.3f, 8.6f, 8.8f, 9.2f, 9.3f;
    vecForecast.row(12)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.9f, 1.0f, 2.9f, 3.3f, 3.7f, 3.8f, 3.9f, 5.7f, 5.9f, 6.1f, 6.2f, 8.1f, 9.0f;
    vecForecast.row(13)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.6f, 0.9f, 1.2f, 1.3f, 1.5f, 2.2f, 5.1f, 6.0f, 6.5f, 6.5f, 6.9f, 7.6f, 8.0f, 8.9f, 8.9f, 9.4f, 9.7f;
    vecForecast.row(14)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.6f, 0.7f, 1.3f, 1.8f, 2.9f, 3.2f, 4.0f, 4.5f, 5.8f, 6.0f, 6.1f, 6.5f, 7.1f, 7.8f, 8.5f;
    vecForecast.row(15)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.9f, 1.0f, 2.8f, 3.2f, 3.6f, 4.7f, 5.1f, 6.2f, 6.6f, 6.8f, 7.0f, 7.5f, 8.7f, 8.9f, 8.9f, 10.0f;
    vecForecast.row(16)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.2f, 1.6f, 2.0f, 2.6f, 2.7f, 3.3f, 3.3f, 4.6f, 4.9f, 5.6f, 5.7f, 6.6f, 7.9f, 8.0f, 9.7f;
    vecForecast.row(17)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.2f, 2.4f, 3.8f, 5.4f, 5.8f, 6.3f, 7.5f, 7.6f, 8.7f, 8.9f, 9.2f, 9.5f, 9.5f, 9.8f, 10.0f;
    vecForecast.row(18)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.2f, 0.5f, 0.5f, 1.9f, 2.3f, 4.4f, 4.9f, 5.3f, 5.4f, 6.5f, 6.9f, 7.7f, 7.8f, 7.9f;
    vecForecast.row(19)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.6f, 1.2f, 2.0f, 2.3f, 5.1f, 5.1f, 5.1f, 6.6f, 8.3f, 8.8f, 9.4f;

    vecObs
            << 0.0f, 6.3f, 7.1f, 3.6f, 8.4f, 9.8f, 0.7f, 0.2f, 3.7f, 4.5f, 8.3f, 0.1f, 5.0f, 0.1f, 5.7f, 0.7f, 7.6f, 1.0f, 1.5f, 3.0f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("RankHistogram");

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        float res = score->Assess(vecObs[iTime], vecForecast.row(iTime), nAnalogs);
        results[iTime] = res;
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    // Values processed on Excel
    bool isTrue = (results[0] >= 1 && results[0] < 5); // Contains random value
    EXPECT_EQ(true, isTrue);
    isTrue = (results[1] >= 26 && results[1] <= 28); // Contains random value
    EXPECT_EQ(true, isTrue);
    EXPECT_EQ(29, results[2]);
    EXPECT_EQ(24, results[3]);
    EXPECT_EQ(24, results[4]);
    EXPECT_EQ(31, results[5]);
    EXPECT_EQ(15, results[6]);
    EXPECT_EQ(16, results[7]);
    EXPECT_EQ(24, results[8]);
    EXPECT_EQ(24, results[9]);
    EXPECT_EQ(29, results[10]);
    EXPECT_EQ(18, results[11]);
    EXPECT_EQ(25, results[12]);
    EXPECT_EQ(14, results[13]);
    EXPECT_EQ(24, results[14]);
    EXPECT_EQ(15, results[15]);
    EXPECT_EQ(28, results[16]);
    EXPECT_EQ(16, results[17]);
    EXPECT_EQ(20, results[18]);
    EXPECT_EQ(24, results[19]);

    asTotalScore *finalScore = asTotalScore::GetInstance("RankHistogram", "Total");
    finalScore->SetRanksNb(nAnalogs + 1);
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    a1f scoreVal = finalScore->AssessOnArray(pseudoDates, results, emptyTimeArray);

    float total = scoreVal.sum();
    EXPECT_FLOAT_EQ(100, total);

    // Values processed on Excel
    EXPECT_FLOAT_EQ(5, scoreVal[13]);
    EXPECT_FLOAT_EQ(10, scoreVal[14]);
    EXPECT_FLOAT_EQ(10, scoreVal[15]);
    EXPECT_FLOAT_EQ(5, scoreVal[17]);
    EXPECT_FLOAT_EQ(5, scoreVal[19]);
    EXPECT_FLOAT_EQ(30, scoreVal[23]);
    EXPECT_FLOAT_EQ(5, scoreVal[24]);
    EXPECT_FLOAT_EQ(10, scoreVal[28]);
    EXPECT_FLOAT_EQ(5, scoreVal[30]);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessRankHistogramReliability)
{
    wxLogNull logNo;

    // Sizes
    int timeLength = 20;
    int nAnalogs = 30;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);

    vecForecast.row(0)
            << 0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.4f, 0.7f, 0.8f, 1.0f, 1.3f, 1.7f, 3.3f, 4.9f, 4.9f, 6.0f, 6.0f, 8.6f, 9.2f, 9.5f;
    vecForecast.row(1)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.8f, 1.2f, 1.3f, 1.6f, 2.6f, 3.4f, 5.1f, 5.3f, 5.6f, 5.6f, 5.7f, 6.3f, 6.3f, 7.4f, 7.7f, 8.5f;
    vecForecast.row(2)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.8f, 1.7f, 2.1f, 2.1f, 2.5f, 3.3f, 4.1f, 4.5f, 4.6f, 5.0f, 6.4f, 7.5f, 8.6f;
    vecForecast.row(3)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.5f, 1.3f, 2.9f, 3.0f, 3.1f, 3.9f, 4.1f, 5.1f, 5.6f, 6.1f, 7.3f, 9.3f;
    vecForecast.row(4)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.9f, 1.8f, 2.1f, 3.5f, 6.2f, 7.0f, 7.1f, 7.4f, 8.1f, 8.6f, 8.6f, 9.1f, 9.2f, 9.8f, 9.9f, 10.0f;
    vecForecast.row(5)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.6f, 0.7f, 0.8f, 1.9f, 3.4f, 4.1f, 4.3f, 5.1f, 5.2f, 5.6f, 5.8f, 6.4f, 6.5f, 6.9f, 7.8f, 9.2f, 9.5f, 9.6f;
    vecForecast.row(6)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.2f, 0.5f, 0.6f, 1.6f, 2.4f, 2.4f, 3.6f, 4.1f, 4.2f, 5.0f, 5.2f, 5.7f, 5.9f, 5.9f, 6.5f, 7.5f, 7.7f, 8.4f, 9.3f;
    vecForecast.row(7)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f, 1.6f, 3.1f, 5.3f, 6.2f, 6.6f, 7.3f, 7.6f, 8.1f, 8.7f, 8.9f, 9.1f, 9.1f, 9.6f, 9.7f;
    vecForecast.row(8)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 2.6f, 3.0f, 3.3f, 3.8f, 5.9f, 6.5f, 6.7f, 6.9f, 7.6f, 9.2f;
    vecForecast.row(9)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.1f, 3.7f, 4.2f, 5.1f, 5.7f, 6.5f, 8.6f, 8.8f, 9.2f, 9.4f;
    vecForecast.row(10)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f, 1.4f, 3.2f, 3.8f, 5.0f, 5.5f, 6.0f, 6.2f, 6.2f, 7.4f, 8.1f, 8.2f, 8.4f, 9.9f;
    vecForecast.row(11)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.7f, 1.8f, 4.4f, 4.9f, 5.1f, 5.9f, 6.2f, 7.8f, 8.3f, 8.6f, 8.8f, 9.2f, 9.3f;
    vecForecast.row(12)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.9f, 1.0f, 2.9f, 3.3f, 3.7f, 3.8f, 3.9f, 5.7f, 5.9f, 6.1f, 6.2f, 8.1f, 9.0f;
    vecForecast.row(13)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.6f, 0.9f, 1.2f, 1.3f, 1.5f, 2.2f, 5.1f, 6.0f, 6.5f, 6.5f, 6.9f, 7.6f, 8.0f, 8.9f, 8.9f, 9.4f, 9.7f;
    vecForecast.row(14)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.6f, 0.7f, 1.3f, 1.8f, 2.9f, 3.2f, 4.0f, 4.5f, 5.8f, 6.0f, 6.1f, 6.5f, 7.1f, 7.8f, 8.5f;
    vecForecast.row(15)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.9f, 1.0f, 2.8f, 3.2f, 3.6f, 4.7f, 5.1f, 6.2f, 6.6f, 6.8f, 7.0f, 7.5f, 8.7f, 8.9f, 8.9f, 10.0f;
    vecForecast.row(16)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.2f, 1.6f, 2.0f, 2.6f, 2.7f, 3.3f, 3.3f, 4.6f, 4.9f, 5.6f, 5.7f, 6.6f, 7.9f, 8.0f, 9.7f;
    vecForecast.row(17)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.2f, 2.4f, 3.8f, 5.4f, 5.8f, 6.3f, 7.5f, 7.6f, 8.7f, 8.9f, 9.2f, 9.5f, 9.5f, 9.8f, 10.0f;
    vecForecast.row(18)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.2f, 0.5f, 0.5f, 1.9f, 2.3f, 4.4f, 4.9f, 5.3f, 5.4f, 6.5f, 6.9f, 7.7f, 7.8f, 7.9f;
    vecForecast.row(19)
            << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.6f, 1.2f, 2.0f, 2.3f, 5.1f, 5.1f, 5.1f, 6.6f, 8.3f, 8.8f, 9.4f;

    vecObs
            << 0.3f, 6.4f, 7.1f, 3.6f, 8.4f, 9.8f, 0.7f, 0.2f, 3.7f, 4.5f, 8.3f, 0.1f, 5.0f, 0.1f, 5.7f, 0.7f, 7.6f, 1.0f, 1.5f, 3.0f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("RankHistogramReliability");

    a1f results = a1f::Zero(vecObs.size());
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        float res = score->Assess(vecObs[iTime], vecForecast.row(iTime), nAnalogs);
        results[iTime] = res;
        EXPECT_TRUE(!asIsNaN(results[iTime]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("RankHistogramReliability", "Total");
    finalScore->SetRanksNb(nAnalogs + 1);
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Values processed on Excel
    EXPECT_FLOAT_EQ(2.3300f, scoreVal);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessCRPSreliability)
{
    // Sizes
    int timeLength = 15;
    int nAnalogs = 21;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);

    vecForecast.row(0)
            << 1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 1.0f;
    vecForecast.row(1)
            << -2.0f, -1.0f, -2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -2.0f, -2.0f, -3.0f, 0.0f, 1.0f, 2.0f, 1.0f, 1.0f, -2.0f, -3.0f, 1.0f, -1.0f, 0.0f, 2.0f;
    vecForecast.row(2)
            << 5.0f, 5.0f, 4.0f, 4.0f, 3.0f, 7.0f, 3.0f, 6.0f, 6.0f, 4.0f, 5.0f, 4.0f, 7.0f, 6.0f, 6.0f, 5.0f, 5.0f, 5.0f, 4.0f, 6.0f, 4.0f;
    vecForecast.row(3)
            << 6.0f, 6.0f, 9.0f, 4.0f, 5.0f, 5.0f, 5.0f, 9.0f, 4.0f, 7.0f, 8.0f, 5.0f, 6.0f, 2.0f, 5.0f, 6.0f, 7.0f, 5.0f, 7.0f, 6.0f, 5.0f;
    vecForecast.row(4)
            << 7.0f, 7.0f, 8.0f, 8.0f, 8.0f, 7.0f, 7.0f, 8.0f, 8.0f, 7.0f, 8.0f, 7.0f, 8.0f, 7.0f, 7.0f, 8.0f, 9.0f, 8.0f, 8.0f, 7.0f, 8.0f;
    vecForecast.row(5)
            << 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 0.0f, 2.0f, 2.0f;
    vecForecast.row(6)
            << -2.0f, -2.0f, -1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 0.0f, -2.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f;
    vecForecast.row(7)
            << -5.0f, -2.0f, -4.0f, -2.0f, -2.0f, -1.0f, -2.0f, -3.0f, -4.0f, -4.0f, -4.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -5.0f, -2.0f, -2.0f, -3.0f, -1.0f;
    vecForecast.row(8)
            << -6.0f, -6.0f, -5.0f, -3.0f, -3.0f, -4.0f, -5.0f, -5.0f, -6.0f, -6.0f, -6.0f, -3.0f, -3.0f, -4.0f, -4.0f, -4.0f, -5.0f, -3.0f, -6.0f, -4.0f, -3.0f;
    vecForecast.row(9)
            << -4.0f, -2.0f, -5.0f, -4.0f, -4.0f, -3.0f, -5.0f, -4.0f, -3.0f, -5.0f, -4.0f, -3.0f, -3.0f, -4.0f, -4.0f, -4.0f, -3.0f, -5.0f, -3.0f, -3.0f, -4.0f;
    vecForecast.row(10)
            << -4.0f, -4.0f, -4.0f, -4.0f, -4.0f, -4.0f, -5.0f, -6.0f, -5.0f, -4.0f, -4.0f, -4.0f, -5.0f, -5.0f, -4.0f, -6.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f;
    vecForecast.row(11)
            << -4.0f, -3.0f, -4.0f, -5.0f, -7.0f, -5.0f, -6.0f, -5.0f, -7.0f, -4.0f, -4.0f, -6.0f, -5.0f, -5.0f, -5.0f, -4.0f, -3.0f, -5.0f, -4.0f, -4.0f, -5.0f;
    vecForecast.row(12)
            << -6.0f, -6.0f, -5.0f, -5.0f, -6.0f, -5.0f, -7.0f, -6.0f, -8.0f, -7.0f, -7.0f, -7.0f, -5.0f, -5.0f, -6.0f, -6.0f, -8.0f, -6.0f, -6.0f, -5.0f, -5.0f;
    vecForecast.row(13)
            << -1.0f, -1.0f, -4.0f, -2.0f, -4.0f, -2.0f, -4.0f, -1.0f, -1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -3.0f, -3.0f, -1.0f, -2.0f, -3.0f, -1.0f, -2.0f, -4.0f;
    vecForecast.row(14)
            << 6.0f, 6.0f, 6.0f, 3.0f, 0.0f, 3.0f, 0.0f, 6.0f, 3.0f, 7.0f, 8.0f, 1.0f, 4.0f, 1.0f, 2.0f, 5.0f, 6.0f, 4.0f, 5.0f, 7.0f, 2.0f;

    vecObs << 2.0f, 2.0f, 7.0f, 11.0f, 10.0f, 2.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -7.0f, -5.0f, 0.0f, 2.0f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSreliability");

    a2f results = a2f::Zero(vecObs.size(), 3 * (nAnalogs + 1));
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        a1f res = score->AssessOnArray(vecObs[iTime], vecForecast.row(iTime), nAnalogs);
        results.row(iTime) = res;
        EXPECT_TRUE(!asHasNaN(&res[0], &res[res.size() - 1]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("CRPSreliability", "Total");
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Final values
    EXPECT_NEAR(0.6381, scoreVal, 0.0001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

TEST(Score, ProcessCRPSpotential)
{
    // Sizes
    int timeLength = 15;
    int nAnalogs = 21;

    // Resize the containers
    a2f vecForecast = a2f::Zero(timeLength, nAnalogs);
    a1f vecObs = a1f::Zero(timeLength);

    vecForecast.row(0)
            << 1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 1.0f;
    vecForecast.row(1)
            << -2.0f, -1.0f, -2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -2.0f, -2.0f, -3.0f, 0.0f, 1.0f, 2.0f, 1.0f, 1.0f, -2.0f, -3.0f, 1.0f, -1.0f, 0.0f, 2.0f;
    vecForecast.row(2)
            << 5.0f, 5.0f, 4.0f, 4.0f, 3.0f, 7.0f, 3.0f, 6.0f, 6.0f, 4.0f, 5.0f, 4.0f, 7.0f, 6.0f, 6.0f, 5.0f, 5.0f, 5.0f, 4.0f, 6.0f, 4.0f;
    vecForecast.row(3)
            << 6.0f, 6.0f, 9.0f, 4.0f, 5.0f, 5.0f, 5.0f, 9.0f, 4.0f, 7.0f, 8.0f, 5.0f, 6.0f, 2.0f, 5.0f, 6.0f, 7.0f, 5.0f, 7.0f, 6.0f, 5.0f;
    vecForecast.row(4)
            << 7.0f, 7.0f, 8.0f, 8.0f, 8.0f, 7.0f, 7.0f, 8.0f, 8.0f, 7.0f, 8.0f, 7.0f, 8.0f, 7.0f, 7.0f, 8.0f, 9.0f, 8.0f, 8.0f, 7.0f, 8.0f;
    vecForecast.row(5)
            << 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 0.0f, 2.0f, 2.0f;
    vecForecast.row(6)
            << -2.0f, -2.0f, -1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 0.0f, -2.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f;
    vecForecast.row(7)
            << -5.0f, -2.0f, -4.0f, -2.0f, -2.0f, -1.0f, -2.0f, -3.0f, -4.0f, -4.0f, -4.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -5.0f, -2.0f, -2.0f, -3.0f, -1.0f;
    vecForecast.row(8)
            << -6.0f, -6.0f, -5.0f, -3.0f, -3.0f, -4.0f, -5.0f, -5.0f, -6.0f, -6.0f, -6.0f, -3.0f, -3.0f, -4.0f, -4.0f, -4.0f, -5.0f, -3.0f, -6.0f, -4.0f, -3.0f;
    vecForecast.row(9)
            << -4.0f, -2.0f, -5.0f, -4.0f, -4.0f, -3.0f, -5.0f, -4.0f, -3.0f, -5.0f, -4.0f, -3.0f, -3.0f, -4.0f, -4.0f, -4.0f, -3.0f, -5.0f, -3.0f, -3.0f, -4.0f;
    vecForecast.row(10)
            << -4.0f, -4.0f, -4.0f, -4.0f, -4.0f, -4.0f, -5.0f, -6.0f, -5.0f, -4.0f, -4.0f, -4.0f, -5.0f, -5.0f, -4.0f, -6.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f;
    vecForecast.row(11)
            << -4.0f, -3.0f, -4.0f, -5.0f, -7.0f, -5.0f, -6.0f, -5.0f, -7.0f, -4.0f, -4.0f, -6.0f, -5.0f, -5.0f, -5.0f, -4.0f, -3.0f, -5.0f, -4.0f, -4.0f, -5.0f;
    vecForecast.row(12)
            << -6.0f, -6.0f, -5.0f, -5.0f, -6.0f, -5.0f, -7.0f, -6.0f, -8.0f, -7.0f, -7.0f, -7.0f, -5.0f, -5.0f, -6.0f, -6.0f, -8.0f, -6.0f, -6.0f, -5.0f, -5.0f;
    vecForecast.row(13)
            << -1.0f, -1.0f, -4.0f, -2.0f, -4.0f, -2.0f, -4.0f, -1.0f, -1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -3.0f, -3.0f, -1.0f, -2.0f, -3.0f, -1.0f, -2.0f, -4.0f;
    vecForecast.row(14)
            << 6.0f, 6.0f, 6.0f, 3.0f, 0.0f, 3.0f, 0.0f, 6.0f, 3.0f, 7.0f, 8.0f, 1.0f, 4.0f, 1.0f, 2.0f, 5.0f, 6.0f, 4.0f, 5.0f, 7.0f, 2.0f;

    vecObs << 2.0f, 2.0f, 7.0f, 11.0f, 10.0f, 2.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -7.0f, -5.0f, 0.0f, 2.0f;

    // Instantiate the score
    asScore *score = asScore::GetInstance("CRPSpotential");

    a2f results = a2f::Zero(vecObs.size(), 3 * (nAnalogs + 1));
    a1f pseudoDates = a1f::Zero(vecObs.size());

    for (int iTime = 0; iTime < vecObs.size(); iTime++) {
        pseudoDates[iTime] = iTime;
        a1f res = score->AssessOnArray(vecObs[iTime], vecForecast.row(iTime), nAnalogs);
        results.row(iTime) = res;
        EXPECT_TRUE(!asHasNaN(&res[0], &res[res.size() - 1]));
    }

    asTotalScore *finalScore = asTotalScore::GetInstance("CRPSpotential", "Total");
    finalScore->SetRanksNb(nAnalogs + 1);
    asTimeArray emptyTimeArray = asTimeArray(0, 1, 1, asTimeArray::Simple);
    float scoreVal = finalScore->Assess(pseudoDates, results, emptyTimeArray);

    // Final values
    EXPECT_NEAR(0.5708, scoreVal, 0.0001);

    wxDELETE(score);
    wxDELETE(finalScore);
}

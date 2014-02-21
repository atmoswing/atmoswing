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
 */

#include <wx/filename.h>

#include "include_tests.h"
#include "asDataPredictandPrecipitation.h"
#include "asFile.h"

#include "UnitTest++.h"

namespace
{

TEST(GumbelAdjustment)
{
	wxString str("Testing precipitation predictand...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());
	
    asDataPredictandPrecipitation predictand(Precipitation, Daily, Station);

    wxString datasetFilePath = wxFileName::GetCwd();
    datasetFilePath.Append("/files/catalog_precipitation_somewhere.xml");
    wxString dataFileDir = wxFileName::GetCwd();
    dataFileDir.Append("/files/");
    wxString patternFileDir = wxFileName::GetCwd();
    patternFileDir.Append("/files/");

    wxString tmpPath = wxFileName::CreateTempFileName("predictandDBtest");

    bool success;
    success = predictand.BuildPredictandDB(datasetFilePath, dataFileDir, patternFileDir, tmpPath);
	predictand.SetIsSqrt(false);
	predictand.SetReturnPeriodNormalization(10);
    CHECK_EQUAL(true, success);

    // Checked against Martin Froidevaux
    float P2 = predictand.GetPrecipitationOfReturnPeriod(0, 1, 2);
    CHECK_CLOSE(2.0785084*24, P2, 0.001);
    float P5 = predictand.GetPrecipitationOfReturnPeriod(0, 1, 5);
    CHECK_CLOSE(2.5432442*24, P5, 0.001);
    float P10 = predictand.GetPrecipitationOfReturnPeriod(0, 1, 10);
    CHECK_CLOSE(2.8509397*24, P10, 0.001);
    float P20 = predictand.GetPrecipitationOfReturnPeriod(0, 1, 20);
    CHECK_CLOSE(3.1460887*24, P20, 0.001);
    float P50 = predictand.GetPrecipitationOfReturnPeriod(0, 1, 50);
    CHECK_CLOSE(3.5281287*24, P50, 0.001);
    float P100 = predictand.GetPrecipitationOfReturnPeriod(0, 1, 100);
    CHECK_CLOSE(3.8144139*24, P100, 0.001);

    // Ckecked against myself... To prevent any future bug
    P2 = predictand.GetPrecipitationOfReturnPeriod(1, 1, 2);
    CHECK_CLOSE(49.884, P2, 0.001);
    P5 = predictand.GetPrecipitationOfReturnPeriod(1, 1, 5);
    CHECK_CLOSE(61.03771, P5, 0.00001);
    P10 = predictand.GetPrecipitationOfReturnPeriod(1, 1, 10);
    CHECK_CLOSE(68.42240, P10, 0.00001);
    P20 = predictand.GetPrecipitationOfReturnPeriod(1, 1, 20);
    CHECK_CLOSE(75.50597, P20, 0.00002);
    P50 = predictand.GetPrecipitationOfReturnPeriod(1, 1, 50);
    CHECK_CLOSE(84.67493, P50, 0.00001);
    P100 = predictand.GetPrecipitationOfReturnPeriod(1, 1, 100);
    CHECK_CLOSE(91.54578, P100, 0.00002);

    // Checked against Martin Froidevaux
    P2 = predictand.GetPrecipitationOfReturnPeriod(0, 2, 2);
    CHECK_CLOSE(1.4609545*48, P2, 0.001);
    P5 = predictand.GetPrecipitationOfReturnPeriod(0, 2, 5);
    CHECK_CLOSE(1.8127833*48, P5, 0.001);
    P10 = predictand.GetPrecipitationOfReturnPeriod(0, 2, 10);
    CHECK_CLOSE(2.0457246*48, P10, 0.001);
    P20 = predictand.GetPrecipitationOfReturnPeriod(0, 2, 20);
    CHECK_CLOSE(2.2691675*48, P20, 0.001);
    P50 = predictand.GetPrecipitationOfReturnPeriod(0, 2, 50);
    CHECK_CLOSE(2.5583914*48, P50, 0.001);
    P100 = predictand.GetPrecipitationOfReturnPeriod(0, 2, 100);
    CHECK_CLOSE(2.7751241*48, P100, 0.001);

    // Checked against Martin Froidevaux
    P2 = predictand.GetPrecipitationOfReturnPeriod(0, 3, 2);
    CHECK_CLOSE(1.1455557*72, P2, 0.001);
    P5 = predictand.GetPrecipitationOfReturnPeriod(0, 3, 5);
    CHECK_CLOSE(1.4066929*72, P5, 0.001);
    P10 = predictand.GetPrecipitationOfReturnPeriod(0, 3, 10);
    CHECK_CLOSE(1.5795884*72, P10, 0.001);
    P20 = predictand.GetPrecipitationOfReturnPeriod(0, 3, 20);
    CHECK_CLOSE(1.7454339*72, P20, 0.001);
    P50 = predictand.GetPrecipitationOfReturnPeriod(0, 3, 50);
    CHECK_CLOSE(1.9601040*72, P50, 0.001);
    P100 = predictand.GetPrecipitationOfReturnPeriod(0, 3, 100);
    CHECK_CLOSE(2.1209689*72, P100, 0.001);

    asRemoveDir(tmpPath);

}

}

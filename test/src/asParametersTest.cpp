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

#include "include_tests.h"
#include "asParameters.h"
#include "asParametersCalibration.h"
#include "asParametersForecast.h"

#include "gtest/gtest.h"


TEST(Parameters, ParametersLoadFromFile)
{
	wxPrintf("Testing base parameters...\n");

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_standard_read.xml");

    asParameters params;
    params.LoadFromFile(filepath);

    ASSERT_EQ(asTime::GetMJD(1962,1,1), params.GetArchiveStart());
    ASSERT_EQ(asTime::GetMJD(2008,12,31), params.GetArchiveEnd());
    ASSERT_EQ(24, params.GetTimeArrayAnalogsTimeStepHours());
    ASSERT_EQ(24, params.GetTimeArrayTargetTimeStepHours());
    ASSERT_EQ(60, params.GetTimeArrayAnalogsIntervalDays());
    ASSERT_EQ(60, params.GetTimeArrayAnalogsExcludeDays());
    ASSERT_EQ(true, params.GetTimeArrayAnalogsMode().IsSameAs("days_interval"));
    ASSERT_EQ(true, params.GetTimeArrayTargetMode().IsSameAs("simple"));

    ASSERT_EQ(100, params.GetAnalogsNumber(0));

    ASSERT_EQ(false, params.NeedsPreprocessing(0,0));
    ASSERT_EQ(true, params.GetPredictorDatasetId(0,0).IsSameAs("NCEP_R-1"));
    ASSERT_EQ(true, params.GetPredictorDataId(0,0).IsSameAs("hgt"));
    ASSERT_EQ(500, params.GetPredictorLevel(0,0));
    ASSERT_EQ(24, params.GetPredictorTimeHours(0,0));
    ASSERT_EQ(true, params.GetPredictorGridType(0,0).IsSameAs("regular"));
    ASSERT_EQ(-10, params.GetPredictorXmin(0,0));
    ASSERT_EQ(9, params.GetPredictorXptsnb(0,0));
    ASSERT_EQ(2.5, params.GetPredictorXstep(0,0));
    ASSERT_EQ(30, params.GetPredictorYmin(0,0));
    ASSERT_EQ(5, params.GetPredictorYptsnb(0,0));
    ASSERT_EQ(2.5, params.GetPredictorYstep(0,0));
    ASSERT_EQ(true, params.GetPredictorCriteria(0,0).IsSameAs("S1"));
    CHECK_CLOSE(0.6, params.GetPredictorWeight(0,0), 0.0001);

    ASSERT_EQ(true, params.NeedsPreprocessing(0,1));
    ASSERT_EQ(true, params.GetPreprocessMethod(0,1).IsSameAs("gradients"));
    ASSERT_EQ(true, params.GetPreprocessDatasetId(0,1,0).IsSameAs("NCEP_R-1"));
    ASSERT_EQ(true, params.GetPreprocessDataId(0,1,0).IsSameAs("hgt"));
    ASSERT_EQ(1000, params.GetPreprocessLevel(0,1,0));
    ASSERT_EQ(12, params.GetPreprocessTimeHours(0,1,0));
    ASSERT_EQ(1000, params.GetPredictorLevel(0,1));
    ASSERT_EQ(12, params.GetPredictorTimeHours(0,1));
    ASSERT_EQ(-15, params.GetPredictorXmin(0,1));
    ASSERT_EQ(11, params.GetPredictorXptsnb(0,1));
    ASSERT_EQ(2.5, params.GetPredictorXstep(0,1));
    ASSERT_EQ(35, params.GetPredictorYmin(0,1));
    ASSERT_EQ(7, params.GetPredictorYptsnb(0,1));
    ASSERT_EQ(2.5, params.GetPredictorYstep(0,1));
    ASSERT_EQ(true, params.GetPredictorCriteria(0,1).IsSameAs("S1"));
    CHECK_CLOSE(0.4, params.GetPredictorWeight(0,1), 0.0001);

    ASSERT_EQ(40, params.GetPredictandStationIds()[0]);
}

TEST(Parameters, ParametersLoadFromFileMultipleIds)
{
	wxPrintf("Testing parameters with multiple station ids...\n");

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_standard_multiple_station_ids.xml");

    asParameters params;
    params.LoadFromFile(filepath);

    VectorInt stations = params.GetPredictandStationIds();

    ASSERT_EQ(5, (int)stations.size());
    ASSERT_EQ(40, stations[0]);
    ASSERT_EQ(41, stations[1]);
    ASSERT_EQ(42, stations[2]);
    ASSERT_EQ(43, stations[3]);
    ASSERT_EQ(44, stations[4]);
}

TEST(Parameters, GenerateSimpleParametersFileCalibration)
{
    // Get original parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append("parameters_calibration_R1_calib_period.xml");
    asParametersCalibration params;
    bool result = params.LoadFromFile(paramsFilePath);
    ASSERT_EQ(true, result);

    // Generate simple file
    wxString tmpPath = wxFileName::CreateTempFileName("GenerateSimpleParametersFileCalibrationTest");
    result = params.GenerateSimpleParametersFile(tmpPath);
    ASSERT_EQ(true, result);
}

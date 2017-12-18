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

#include "asParameters.h"
#include "asParametersCalibration.h"
#include "gtest/gtest.h"


TEST(Parameters, ParametersLoadFromFile)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_standard_read.xml");

    asParameters params;
    params.LoadFromFile(filepath);

    EXPECT_EQ(asTime::GetMJD(1962, 1, 1), params.GetArchiveStart());
    EXPECT_EQ(asTime::GetMJD(2008, 12, 31), params.GetArchiveEnd());
    EXPECT_EQ(24, params.GetTimeArrayAnalogsTimeStepHours());
    EXPECT_EQ(24, params.GetTimeArrayTargetTimeStepHours());
    EXPECT_EQ(60, params.GetTimeArrayAnalogsIntervalDays());
    EXPECT_EQ(60, params.GetTimeArrayAnalogsExcludeDays());
    EXPECT_TRUE(params.GetTimeArrayAnalogsMode().IsSameAs("days_interval"));
    EXPECT_TRUE(params.GetTimeArrayTargetMode().IsSameAs("simple"));

    EXPECT_EQ(100, params.GetAnalogsNumber(0));

    EXPECT_FALSE(params.NeedsPreprocessing(0, 0));
    EXPECT_TRUE(params.GetPredictorDatasetId(0, 0).IsSameAs("NCEP_R-1"));
    EXPECT_TRUE(params.GetPredictorDataId(0, 0).IsSameAs("hgt"));
    EXPECT_EQ(500, params.GetPredictorLevel(0, 0));
    EXPECT_EQ(24, params.GetPredictorTimeHours(0, 0));
    EXPECT_TRUE(params.GetPredictorGridType(0, 0).IsSameAs("regular"));
    EXPECT_EQ(-10, params.GetPredictorXmin(0, 0));
    EXPECT_EQ(9, params.GetPredictorXptsnb(0, 0));
    EXPECT_EQ(2.5, params.GetPredictorXstep(0, 0));
    EXPECT_EQ(30, params.GetPredictorYmin(0, 0));
    EXPECT_EQ(5, params.GetPredictorYptsnb(0, 0));
    EXPECT_EQ(2.5, params.GetPredictorYstep(0, 0));
    EXPECT_TRUE(params.GetPredictorCriteria(0, 0).IsSameAs("S1"));
    EXPECT_FLOAT_EQ(0.6f, params.GetPredictorWeight(0, 0));

    EXPECT_TRUE(params.NeedsPreprocessing(0, 1));
    EXPECT_TRUE(params.GetPreprocessMethod(0, 1).IsSameAs("gradients"));
    EXPECT_TRUE(params.GetPreprocessDatasetId(0, 1, 0).IsSameAs("NCEP_R-1"));
    EXPECT_TRUE(params.GetPreprocessDataId(0, 1, 0).IsSameAs("hgt"));
    EXPECT_EQ(1000, params.GetPreprocessLevel(0, 1, 0));
    EXPECT_EQ(12, params.GetPreprocessTimeHours(0, 1, 0));
    EXPECT_EQ(1000, params.GetPredictorLevel(0, 1));
    EXPECT_EQ(12, params.GetPredictorTimeHours(0, 1));
    EXPECT_EQ(-15, params.GetPredictorXmin(0, 1));
    EXPECT_EQ(11, params.GetPredictorXptsnb(0, 1));
    EXPECT_EQ(2.5, params.GetPredictorXstep(0, 1));
    EXPECT_EQ(35, params.GetPredictorYmin(0, 1));
    EXPECT_EQ(7, params.GetPredictorYptsnb(0, 1));
    EXPECT_EQ(2.5, params.GetPredictorYstep(0, 1));
    EXPECT_TRUE(params.GetPredictorCriteria(0, 1).IsSameAs("S1"));
    EXPECT_FLOAT_EQ(0.4f, params.GetPredictorWeight(0, 1));

    EXPECT_EQ(40, params.GetPredictandStationIds()[0]);
}

TEST(Parameters, ParametersLoadFromFileMultipleIds)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_standard_multiple_station_ids.xml");

    asParameters params;
    params.LoadFromFile(filepath);

    vi stations = params.GetPredictandStationIds();

    EXPECT_EQ(5, (int) stations.size());
    EXPECT_EQ(40, stations[0]);
    EXPECT_EQ(41, stations[1]);
    EXPECT_EQ(42, stations[2]);
    EXPECT_EQ(43, stations[3]);
    EXPECT_EQ(44, stations[4]);
}

TEST(Parameters, GenerateSimpleParametersFileCalibration)
{
    // Get original parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append("parameters_calibration_R1_calib_period.xml");
    asParametersCalibration params;
    EXPECT_TRUE(params.LoadFromFile(paramsFilePath));

    // Generate simple file
    wxString tmpPath = wxFileName::CreateTempFileName("GenerateSimpleParametersFileCalibrationTest");
    EXPECT_TRUE(params.GenerateSimpleParametersFile(tmpPath));
}

TEST(Parameters, SortLevelsAndTime)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_standard_sort_level_time.xml");

    asParameters params;
    params.LoadFromFile(filepath);
    int s = 0, p = 0;
    
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(18, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(1000, params.GetPredictorLevel(s,p));
    EXPECT_EQ(18, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_FALSE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(500, params.GetPredictorLevel(s,p));
    EXPECT_EQ(24, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(12, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(1000, params.GetPredictorLevel(s,p));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_FALSE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(850, params.GetPredictorLevel(s,p));
    EXPECT_EQ(24, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_FALSE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(850, params.GetPredictorLevel(s,p));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    
    s++;
    p = 0;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(18, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(18, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(12, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(850, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(12, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(700, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(18, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(18, params.GetPredictorTimeHours(s,p));

    // Sort and check
    params.SortLevelsAndTime();

    p = 0;
    s = 0;
    EXPECT_FALSE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(500, params.GetPredictorLevel(s,p));
    EXPECT_EQ(24, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_FALSE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(850, params.GetPredictorLevel(s,p));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_FALSE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(850, params.GetPredictorLevel(s,p));
    EXPECT_EQ(24, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(12, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(1000, params.GetPredictorLevel(s,p));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(18, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(1000, params.GetPredictorLevel(s,p));
    EXPECT_EQ(18, params.GetPredictorTimeHours(s,p));
    
    s++;
    p = 0;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(700, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(18, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(18, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(850, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(12, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(12, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(12, params.GetPredictorTimeHours(s,p));
    p++;
    EXPECT_TRUE(params.NeedsPreprocessing(s,p));
    EXPECT_EQ(1000, params.GetPreprocessLevel(s,p,0));
    EXPECT_EQ(18, params.GetPreprocessTimeHours(s,p,0));
    EXPECT_EQ(18, params.GetPredictorTimeHours(s,p));
}

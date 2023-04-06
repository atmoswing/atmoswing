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
 * Portions Copyright 2019-2020 Pascal Horton, University of Bern.
 */

#include <gtest/gtest.h>

#include "asParameters.h"
#include "asParametersForecast.h"

TEST(ParametersForecasting, ParametersLoadFromFile) {
    wxString filePath = wxFileName::GetCwd();
    filePath.Append("/files/parameters_forecasting_6hrly.xml");

    asParametersForecast params;
    ASSERT_TRUE(params.LoadFromFile(filePath));

    EXPECT_EQ(asTime::GetMJD(2011, 4, 1, 6), params.GetArchiveStart());
    EXPECT_EQ(asTime::GetMJD(2018, 9, 30, 18), params.GetArchiveEnd());
    EXPECT_EQ(15, params.GetLeadTimeNb());
    EXPECT_EQ(6.0 / 24.0, params.GetLeadTimeDaysVector()[0]);
    EXPECT_EQ(90.0 / 24.0, params.GetLeadTimeDaysVector()[14]);
    EXPECT_EQ(6, params.GetAnalogsTimeStepHours());
    EXPECT_EQ(6, params.GetTargetTimeStepHours());
    EXPECT_EQ(200, params.GetAnalogsIntervalDays());
    EXPECT_EQ(0, params.GetAnalogsExcludeDays());
    EXPECT_TRUE(params.GetTimeArrayAnalogsMode().IsSameAs("April_to_September"));
    EXPECT_TRUE(params.GetTimeArrayTargetMode().IsSameAs("simple"));

    EXPECT_EQ(65, params.GetAnalogsNumber(0));

    EXPECT_FALSE(params.NeedsPreprocessing(0, 0));
    EXPECT_TRUE(params.GetRealtimeStandardize(0, 0));
    EXPECT_EQ(333.5554787289083, params.GetRealtimeStandardizeMean(0, 0));
    EXPECT_EQ(19.838536332114153, params.GetRealtimeStandardizeSd(0, 0));
    EXPECT_FALSE(params.GetArchiveStandardize(0, 0));
    EXPECT_TRUE(asIsNaN(params.GetArchiveStandardizeMean(0, 0)));
    EXPECT_TRUE(asIsNaN(params.GetArchiveStandardizeSd(0, 0)));
    EXPECT_TRUE(params.GetPredictorRealtimeDatasetId(0, 0).IsSameAs("Custom_MeteoFVG_Forecast"));
    EXPECT_TRUE(params.GetPredictorRealtimeDataId(0, 0).IsSameAs("thetaES_925"));
    EXPECT_TRUE(params.GetPredictorArchiveDatasetId(0, 0).IsSameAs("Generic"));
    EXPECT_TRUE(params.GetPredictorArchiveDataId(0, 0).IsSameAs("thetaES_925"));
    EXPECT_EQ(925, params.GetPredictorLevel(0, 0));
    EXPECT_EQ(0, params.GetPredictorHour(0, 0));
    EXPECT_TRUE(params.GetPredictorGridType(0, 0).IsSameAs("regular"));
    EXPECT_EQ(6.125, params.GetPredictorXmin(0, 0));
    EXPECT_EQ(78, params.GetPredictorXptsnb(0, 0));
    EXPECT_EQ(0.125, params.GetPredictorXstep(0, 0));
    EXPECT_EQ(45.375, params.GetPredictorYmin(0, 0));
    EXPECT_EQ(7, params.GetPredictorYptsnb(0, 0));
    EXPECT_EQ(0.125, params.GetPredictorYstep(0, 0));
    EXPECT_TRUE(params.GetPredictorCriteria(0, 0).IsSameAs("S1"));
    EXPECT_FLOAT_EQ(0.3f, params.GetPredictorWeight(0, 0));

    EXPECT_FALSE(params.NeedsPreprocessing(0, 1));
    EXPECT_TRUE(params.GetRealtimeStandardize(0, 1));
    EXPECT_EQ(0.00043102220253530437, params.GetRealtimeStandardizeMean(0, 1));
    EXPECT_EQ(0.0014176902243256072, params.GetRealtimeStandardizeSd(0, 1));
    EXPECT_FALSE(params.GetArchiveStandardize(0, 1));
    EXPECT_TRUE(asIsNaN(params.GetArchiveStandardizeMean(0, 1)));
    EXPECT_TRUE(asIsNaN(params.GetArchiveStandardizeSd(0, 1)));
    EXPECT_TRUE(params.GetPredictorRealtimeDatasetId(0, 1).IsSameAs("Custom_MeteoFVG_Forecast"));
    EXPECT_TRUE(params.GetPredictorRealtimeDataId(0, 1).IsSameAs("cp_sfc"));
    EXPECT_TRUE(params.GetPredictorArchiveDatasetId(0, 1).IsSameAs("Generic"));
    EXPECT_TRUE(params.GetPredictorArchiveDataId(0, 1).IsSameAs("cp_sfc"));
    EXPECT_EQ(0, params.GetPredictorLevel(0, 1));
    EXPECT_EQ(0, params.GetPredictorHour(0, 1));
    EXPECT_EQ(11.75, params.GetPredictorXmin(0, 1));
    EXPECT_EQ(25, params.GetPredictorXptsnb(0, 1));
    EXPECT_EQ(0.125, params.GetPredictorXstep(0, 1));
    EXPECT_EQ(45.25, params.GetPredictorYmin(0, 1));
    EXPECT_EQ(13, params.GetPredictorYptsnb(0, 1));
    EXPECT_EQ(0.125, params.GetPredictorYstep(0, 1));
    EXPECT_TRUE(params.GetPredictorCriteria(0, 1).IsSameAs("S1"));
    EXPECT_FLOAT_EQ(0.7f, params.GetPredictorWeight(0, 1));

    EXPECT_FALSE(params.NeedsPreprocessing(1, 0));
    EXPECT_TRUE(params.GetRealtimeStandardize(1, 0));
    EXPECT_EQ(4.858158690266188, params.GetRealtimeStandardizeMean(1, 0));
    EXPECT_EQ(7.742704912270005, params.GetRealtimeStandardizeSd(1, 0));
    EXPECT_FALSE(params.GetArchiveStandardize(1, 0));
    EXPECT_TRUE(asIsNaN(params.GetArchiveStandardizeMean(1, 0)));
    EXPECT_TRUE(asIsNaN(params.GetArchiveStandardizeSd(1, 0)));
    EXPECT_TRUE(params.GetPredictorRealtimeDatasetId(1, 0).IsSameAs("Custom_MeteoFVG_Forecast"));
    EXPECT_TRUE(params.GetPredictorRealtimeDataId(1, 0).IsSameAs("MB500925"));
    EXPECT_TRUE(params.GetPredictorArchiveDatasetId(1, 0).IsSameAs("Generic"));
    EXPECT_TRUE(params.GetPredictorArchiveDataId(1, 0).IsSameAs("MB500925"));
    EXPECT_EQ(500, params.GetPredictorLevel(1, 0));
    EXPECT_EQ(0, params.GetPredictorHour(1, 0));
    EXPECT_TRUE(params.GetPredictorCriteria(1, 0).IsSameAs("RMSE"));
    EXPECT_FLOAT_EQ(0.5f, params.GetPredictorWeight(1, 0));

    EXPECT_FALSE(params.NeedsPreprocessing(1, 1));
    EXPECT_TRUE(params.GetRealtimeStandardize(1, 1));
    EXPECT_EQ(0.36561869859522483, params.GetRealtimeStandardizeMean(1, 1));
    EXPECT_EQ(2.4421665133229036, params.GetRealtimeStandardizeSd(1, 1));
    EXPECT_TRUE(params.GetPredictorRealtimeDatasetId(1, 1).IsSameAs("Custom_MeteoFVG_Forecast"));
    EXPECT_TRUE(params.GetPredictorRealtimeDataId(1, 1).IsSameAs("10u_sfc"));
    EXPECT_TRUE(params.GetPredictorArchiveDatasetId(1, 1).IsSameAs("Generic"));
    EXPECT_TRUE(params.GetPredictorArchiveDataId(1, 1).IsSameAs("10u_sfc"));
    EXPECT_EQ(0, params.GetPredictorLevel(1, 1));
    EXPECT_EQ(0, params.GetPredictorHour(1, 1));
    EXPECT_TRUE(params.GetPredictorCriteria(1, 1).IsSameAs("RMSE"));
    EXPECT_FLOAT_EQ(0.5f, params.GetPredictorWeight(1, 1));

    EXPECT_EQ(1, params.GetPredictandStationIds()[0]);
}

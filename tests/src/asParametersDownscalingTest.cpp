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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include <gtest/gtest.h>
#include "asParameters.h"
#include "asParametersDownscaling.h"

TEST(ParametersDownscaling, ParametersLoadFromFile) {
  wxString filepath = wxFileName::GetCwd();
  filepath.Append("/files/parameters_downscaling.xml");

  asParametersDownscaling params;
  ASSERT_TRUE(params.LoadFromFile(filepath));

  EXPECT_EQ(asTime::GetMJD(1962, 1, 1), params.GetArchiveStart());
  EXPECT_EQ(asTime::GetMJD(2008, 12, 31), params.GetArchiveEnd());
  EXPECT_EQ(asTime::GetMJD(2050, 1, 1), params.GetDownscalingStart());
  EXPECT_EQ(asTime::GetMJD(2099, 12, 31), params.GetDownscalingEnd());
  EXPECT_EQ(24, params.GetAnalogsTimeStepHours());
  EXPECT_EQ(24, params.GetTargetTimeStepHours());
  EXPECT_EQ(60, params.GetAnalogsIntervalDays());
  EXPECT_EQ(60, params.GetAnalogsExcludeDays());
  EXPECT_TRUE(params.GetTimeArrayAnalogsMode().IsSameAs("days_interval"));
  EXPECT_TRUE(params.GetTimeArrayTargetMode().IsSameAs("simple"));

  EXPECT_EQ(100, params.GetAnalogsNumber(0));

  EXPECT_FALSE(params.NeedsPreprocessing(0, 0));
  EXPECT_TRUE(params.GetPredictorProjDatasetId(0, 0).IsSameAs("CORDEX"));
  EXPECT_TRUE(params.GetPredictorProjDataId(0, 0).IsSameAs("zg500"));
  EXPECT_TRUE(params.GetPredictorDatasetId(0, 0).IsSameAs("NCEP_R1"));
  EXPECT_TRUE(params.GetPredictorDataId(0, 0).IsSameAs("hgt"));
  EXPECT_EQ(500, params.GetPredictorLevel(0, 0));
  EXPECT_EQ(24, params.GetPredictorHour(0, 0));
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
  EXPECT_TRUE(params.GetPreprocessMethod(0, 1).IsSameAs("SimpleGradients", false));
  EXPECT_TRUE(params.GetPreprocessProjDatasetId(0, 1, 0).IsSameAs("CORDEX"));
  EXPECT_TRUE(params.GetPreprocessProjDataId(0, 1, 0).IsSameAs("zg850"));
  EXPECT_TRUE(params.GetPreprocessDatasetId(0, 1, 0).IsSameAs("NCEP_R1"));
  EXPECT_TRUE(params.GetPreprocessDataId(0, 1, 0).IsSameAs("hgt"));
  EXPECT_EQ(850, params.GetPreprocessLevel(0, 1, 0));
  EXPECT_EQ(12, params.GetPreprocessHour(0, 1, 0));
  EXPECT_EQ(850, params.GetPredictorLevel(0, 1));
  EXPECT_EQ(12, params.GetPredictorHour(0, 1));
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

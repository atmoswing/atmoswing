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
 * Portions Copyright 2016 Pascal Horton, University of Bern.
 */

#include "asParameters.h"
#include "asParametersOptimization.h"
#include "gtest/gtest.h"


TEST(ParametersOptimization, LoadFromFile)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_optimization.xml");

    asParametersOptimization params;
    ASSERT_TRUE(params.LoadFromFile(filepath));

    params.InitRandomValues();

    EXPECT_EQ(asTime::GetMJD(1962, 1, 1), params.GetArchiveStart());
    EXPECT_EQ(asTime::GetMJD(2008, 12, 31), params.GetArchiveEnd());
    EXPECT_EQ(asTime::GetMJD(1970, 1, 1), params.GetCalibrationStart());
    EXPECT_EQ(asTime::GetMJD(2000, 12, 31), params.GetCalibrationEnd());
    EXPECT_EQ(24, params.GetTimeArrayAnalogsTimeStepHours());
    EXPECT_EQ(24, params.GetTimeArrayTargetTimeStepHours());
    EXPECT_EQ(1, params.GetTimeArrayAnalogsIntervalDaysIteration());
    EXPECT_EQ(10, params.GetTimeArrayAnalogsIntervalDaysLowerLimit());
    EXPECT_EQ(182, params.GetTimeArrayAnalogsIntervalDaysUpperLimit());
    EXPECT_FALSE(params.IsTimeArrayAnalogsIntervalDaysLocked());
    EXPECT_EQ(60, params.GetTimeArrayAnalogsExcludeDays());
    EXPECT_TRUE(params.GetTimeArrayAnalogsMode().IsSameAs("days_interval"));
    EXPECT_TRUE(params.GetTimeArrayTargetMode().IsSameAs("simple"));

    EXPECT_EQ(1, params.GetAnalogsNumberIteration(0));
    EXPECT_EQ(5, params.GetAnalogsNumberLowerLimit(0));
    EXPECT_EQ(200, params.GetAnalogsNumberUpperLimit(0));
    EXPECT_FALSE(params.IsAnalogsNumberLocked(0));

    EXPECT_FALSE(params.NeedsPreprocessing(0, 0));
    EXPECT_TRUE(params.GetPredictorDatasetId(0, 0).IsSameAs("NCEP_R-1"));
    EXPECT_TRUE(params.GetPredictorDataId(0, 0).IsSameAs("hgt"));
    EXPECT_EQ(500, params.GetPredictorLevel(0, 0));
    EXPECT_EQ(6, params.GetPredictorTimeHoursIteration(0, 0));
    EXPECT_EQ(-48, params.GetPredictorTimeHoursLowerLimit(0, 0));
    EXPECT_EQ(48, params.GetPredictorTimeHoursUpperLimit(0, 0));
    EXPECT_FALSE(params.IsPredictorTimeHoursLocked(0, 0));
    EXPECT_TRUE(params.GetPredictorGridType(0, 0).IsSameAs("regular"));
    EXPECT_EQ(2.5, params.GetPredictorXminIteration(0, 0));
    EXPECT_EQ(300, params.GetPredictorXminLowerLimit(0, 0));
    EXPECT_EQ(450, params.GetPredictorXminUpperLimit(0, 0));
    EXPECT_FALSE(params.IsPredictorXminLocked(0, 0));
    EXPECT_EQ(1, params.GetPredictorXptsnbIteration(0, 0));
    EXPECT_EQ(1, params.GetPredictorXptsnbLowerLimit(0, 0));
    EXPECT_EQ(21, params.GetPredictorXptsnbUpperLimit(0, 0));
    EXPECT_FALSE(params.IsPredictorXptsnbLocked(0, 0));
    EXPECT_EQ(2.5, params.GetPredictorXstep(0, 0));
    EXPECT_EQ(2.5, params.GetPredictorYminIteration(0, 0));
    EXPECT_EQ(0, params.GetPredictorYminLowerLimit(0, 0));
    EXPECT_EQ(70, params.GetPredictorYminUpperLimit(0, 0));
    EXPECT_FALSE(params.IsPredictorYminLocked(0, 0));
    EXPECT_EQ(1, params.GetPredictorYptsnbIteration(0, 0));
    EXPECT_EQ(1, params.GetPredictorYptsnbLowerLimit(0, 0));
    EXPECT_EQ(13, params.GetPredictorYptsnbUpperLimit(0, 0));
    EXPECT_FALSE(params.IsPredictorYptsnbLocked(0, 0));
    EXPECT_EQ(2.5, params.GetPredictorYstep(0, 0));
    EXPECT_TRUE(params.GetPredictorCriteria(0, 0).IsSameAs("S1"));
    EXPECT_NEAR(0.01, params.GetPredictorWeightIteration(0, 0), 0.0001);
    EXPECT_NEAR(0, params.GetPredictorWeightLowerLimit(0, 0), 0.0001);
    EXPECT_NEAR(1, params.GetPredictorWeightUpperLimit(0, 0), 0.0001);
    EXPECT_FALSE(params.IsPredictorWeightLocked(0, 0));

    EXPECT_TRUE(params.NeedsPreprocessing(0, 1));
    EXPECT_TRUE(params.GetPreprocessMethod(0, 1).IsSameAs("Gradients"));
    EXPECT_TRUE(params.GetPreprocessDatasetId(0, 1, 0).IsSameAs("NCEP_R-1"));
    EXPECT_TRUE(params.GetPreprocessDataId(0, 1, 0).IsSameAs("hgt"));
    EXPECT_EQ(2, params.GetPreprocessLevelVector(0, 1, 0).size());
    EXPECT_EQ(850, params.GetPreprocessLevelVector(0, 1, 0)[0]);
    EXPECT_EQ(1000, params.GetPreprocessLevelVector(0, 1, 0)[1]);
    EXPECT_EQ(6, params.GetPreprocessTimeHoursIteration(0, 1, 0));
    EXPECT_EQ(-6, params.GetPreprocessTimeHoursLowerLimit(0, 1, 0));
    EXPECT_EQ(24, params.GetPreprocessTimeHoursUpperLimit(0, 1, 0));
    EXPECT_TRUE(params.GetPredictorGridType(0, 1).IsSameAs("regular"));
    EXPECT_EQ(2.5, params.GetPredictorXminIteration(0, 1));
    EXPECT_EQ(300, params.GetPredictorXminLowerLimit(0, 1));
    EXPECT_EQ(450, params.GetPredictorXminUpperLimit(0, 1));
    EXPECT_FALSE(params.IsPredictorXminLocked(0, 1));
    EXPECT_EQ(1, params.GetPredictorXptsnbIteration(0, 1));
    EXPECT_EQ(3, params.GetPredictorXptsnbLowerLimit(0, 1));
    EXPECT_EQ(19, params.GetPredictorXptsnbUpperLimit(0, 1));
    EXPECT_FALSE(params.IsPredictorXptsnbLocked(0, 1));
    EXPECT_EQ(2.5, params.GetPredictorXstep(0, 1));
    EXPECT_EQ(2.5, params.GetPredictorYminIteration(0, 1));
    EXPECT_EQ(0, params.GetPredictorYminLowerLimit(0, 1));
    EXPECT_EQ(70, params.GetPredictorYminUpperLimit(0, 1));
    EXPECT_FALSE(params.IsPredictorYminLocked(0, 1));
    EXPECT_EQ(1, params.GetPredictorYptsnbIteration(0, 1));
    EXPECT_EQ(1, params.GetPredictorYptsnbLowerLimit(0, 1));
    EXPECT_EQ(9, params.GetPredictorYptsnbUpperLimit(0, 1));
    EXPECT_FALSE(params.IsPredictorYptsnbLocked(0, 1));
    EXPECT_EQ(2.5, params.GetPredictorYstep(0, 1));
    EXPECT_TRUE(params.GetPredictorCriteria(0, 1).IsSameAs("S1grads"));
    EXPECT_NEAR(0.01, params.GetPredictorWeightIteration(0, 1), 0.0001);
    EXPECT_NEAR(0, params.GetPredictorWeightLowerLimit(0, 1), 0.0001);
    EXPECT_NEAR(1, params.GetPredictorWeightUpperLimit(0, 1), 0.0001);
    EXPECT_FALSE(params.IsPredictorWeightLocked(0, 1));

    EXPECT_EQ(40, params.GetPredictandStationIds()[0]);

    EXPECT_TRUE(params.GetScoreName().IsSameAs("CRPSAR"));

    EXPECT_TRUE(params.GetScoreTimeArrayMode().IsSameAs("simple"));
}

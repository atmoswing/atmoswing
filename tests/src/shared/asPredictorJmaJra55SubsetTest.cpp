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
 * Portions Copyright 2016 Pascal Horton, University of Bern.
 */

#include <gtest/gtest.h>
#include <wx/filename.h>

#include "asAreaGridRegular.h"
#include "asPredictor.h"
#include "asTimeArray.h"

TEST(PredictorJmaJra55Subset, GetCorrectPredictors) {
    asPredictor* predictor;

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_p125/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_p125/rh", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_p125/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_p125/w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_surf125/msl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_column125/pwat", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "fcst_phy2m125/tprat3h", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "fcst_phy2m125/tprat6h", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_isentrop125/pv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialVorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_isentrop125/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);
}

TEST(PredictorJmaJra55CSubset, GetCorrectPredictors) {
    asPredictor* predictor;

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_p125/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_p125/rh", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_p125/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_p125/w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_surf125/msl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_column125/pwat", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "fcst_phy2m125/tprat3h", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "fcst_phy2m125/tprat6h", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_isentrop125/pv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialVorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("JMA_JRA_55C_subset", "anl_isentrop125/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);
}

TEST(PredictorJmaJra55Subset, LoadEasy) {
    double xMin = 360;
    double xWidth = 10;
    double xStep = 1.250;
    double yMin = 75;
    double yWidth = 5;
    double yStep = 1.250;
    float level = 1000;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-jma-jra55-ncar-subset/");

    asPredictor* predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_p125/hgt", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    122.4	121.6	121.1	120.4	119.6	118.1	116.9	115.9	115.1
    115.4	113.6	112.1	110.4	108.9	105.9	104.1	102.6	101.4
    102.4	100.1	97.9	95.6	93.6	91.6	89.9	87.9	86.4
    81.9	79.9	78.1	76.4	74.9	73.6	72.4	71.1	69.9
    54.9	54.6	54.9	55.4	56.1	56.9	57.9	58.6	59.6
    */
    EXPECT_NEAR(122.4, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(121.6, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(121.1, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(120.4, hgt[0][0](0, 3), 0.1);
    EXPECT_NEAR(115.1, hgt[0][0](0, 8), 0.1);
    EXPECT_NEAR(115.4, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(102.4, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(54.9, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(59.6, hgt[0][0](4, 8), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    85.8	86.3	86.6	86.3	85.8	84.3	83.3	82.1	81.1
    90.1	89.3	88.1	86.6	85.1	82.1	80.1	78.1	76.3
    88.6	86.6	84.6	82.3	80.1	78.1	75.6	73.3	71.6
    80.8	78.3	75.6	72.8	70.3	67.8	65.3	62.8	61.1
    67.6	64.1	60.3	56.8	53.3	50.1	47.1	44.6	42.8
    */
    EXPECT_NEAR(85.8, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(86.3, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(86.6, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(86.3, hgt[7][0](0, 3), 0.1);
    EXPECT_NEAR(81.1, hgt[7][0](0, 8), 0.1);
    EXPECT_NEAR(90.1, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(88.6, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(67.6, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(42.8, hgt[7][0](4, 8), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorJmaJra55Subset, Around360) {
    double xMin = 355;
    double xWidth = 10;
    double xStep = 1.250;
    double yMin = 75;
    double yWidth = 5;
    double yStep = 1.250;
    float level = 1000;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-jma-jra55-ncar-subset/");

    asPredictor* predictor = asPredictor::GetInstance("JMA_JRA_55_subset", "anl_p125/hgt", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    125.9	124.6	123.6	122.9	122.4	121.6	121.1	120.4	119.6
    118.9	118.4	117.9	116.9	115.4	113.6	112.1	110.4	108.9
    109.9	108.4	106.6	104.9	102.4	100.1	97.9	95.6	93.6
    91.4	89.1	86.6	84.1	81.9	79.9	78.1	76.4	74.9
    61.6	58.9	56.9	55.6	54.9	54.6	54.9	55.4	56.1
    */
    EXPECT_NEAR(125.9, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(124.6, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(123.6, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(122.9, hgt[0][0](0, 3), 0.1);
    EXPECT_NEAR(119.6, hgt[0][0](0, 8), 0.1);
    EXPECT_NEAR(118.9, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(109.9, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(61.6, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(56.1, hgt[0][0](4, 8), 0.1);

    wxDELETE(predictor);
}
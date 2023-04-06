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

TEST(PredictorNasaMerra2, GetCorrectPredictors) {
    asPredictor* predictor;

    predictor = asPredictor::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/slp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);
}

TEST(PredictorNasaMerra2Subset, GetCorrectPredictors) {
    asPredictor* predictor;

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/sh", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/slp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/u", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/v", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/ps", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst6_3d_ana_Np/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst3_3d_asm_Np/pv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialVorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst3_3d_asm_Np/w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst3_3d_asm_Np/r", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst3_3d_asm_Np/slp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst3_3d_asm_Np/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst3_3d_asm_Np/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_int_Nx/tqi", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_int_Nx/tql", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_int_Nx/tqv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_int_Nx/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_asm_Nx/tqi", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_asm_Nx/tql", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_asm_Nx/tqv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_asm_Nx/t10m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "inst1_2d_asm_Nx/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "tavg1_2d_flx_Nx/tp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "tavg1_2d_flx_Nx/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "tavg1_2d_lnd_Nx/tp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NASA_MERRA_2_subset", "tavg1_2d_lnd_Nx/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);
}

TEST(PredictorNasaMerra2, LoadEasy) {
    double xMin = 2.5;
    double xWidth = 5;
    double xStep = 0.625;
    double yMin = 75;
    double yWidth = 2;
    double yStep = 0.5;
    float level = 1000;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-nasa-merra2/");

    asPredictor* predictor = asPredictor::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/h", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    91.8	91.1	90.3	89.5	88.8	88.1	87.5	87.0	86.5
    84.1	83.5	82.9	82.4	81.9	81.5	81.0	80.5	80.0
    75.2	74.8	74.4	74.1	73.9	73.9	73.8	73.7	73.5
    65.8	65.8	65.9	66.1	66.4	66.8	67.0	67.1	67.1
    56.5	57.1	57.8	58.6	59.4	60.1	60.6	61.0	61.3
    */
    EXPECT_NEAR(91.8, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(91.1, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(90.3, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(89.5, hgt[0][0](0, 3), 0.1);
    EXPECT_NEAR(86.5, hgt[0][0](0, 8), 0.1);
    EXPECT_NEAR(84.1, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(75.2, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(56.5, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(61.3, hgt[0][0](4, 8), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    83.4	83.3	82.8	82.1	81.5	81.0	80.3	79.5	78.7
    81.4	80.9	80.2	79.4	78.6	77.8	77.0	76.2	75.4
    77.6	76.9	76.1	75.4	74.7	73.9	73.2	72.4	71.4
    72.5	71.6	70.8	69.9	68.9	67.9	67.0	66.1	65.2
    65.9	64.9	63.9	62.9	61.9	60.9	59.6	58.3	57.1
    */
    EXPECT_NEAR(83.4, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(83.3, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(82.8, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(82.1, hgt[7][0](0, 3), 0.1);
    EXPECT_NEAR(78.7, hgt[7][0](0, 8), 0.1);
    EXPECT_NEAR(81.4, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(77.6, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(65.9, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(57.1, hgt[7][0](4, 8), 0.1);

    // Check lat axis
    a1d lats = predictor->GetLatAxis();
    EXPECT_NEAR(77, lats[0], 0.1);
    EXPECT_NEAR(76.5, lats[1], 0.1);
    EXPECT_NEAR(75, lats[4], 0.1);

    wxDELETE(predictor);
}

TEST(PredictorNasaMerra2, LoadBorderLeft) {
    double xMin = -180;
    double xWidth = 2.5;
    double xStep = 0.625;
    double yMin = 75;
    double yWidth = 2;
    double yStep = 0.5;
    float level = 1000;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-nasa-merra2/");

    asPredictor* predictor = asPredictor::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/h", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |    10.8	8.8	    6.8	    4.8	    2.7
    |    16.9	15.0	13.0	11.0	9.0
    |    23.3	21.5	19.6	17.7	15.8
    |    30.6	28.7	26.7	24.8	22.9
    |    38.1	36.2	34.2	32.2	30.3
    */
    EXPECT_NEAR(10.8, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(8.8, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(6.8, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(4.8, hgt[0][0](0, 3), 0.1);
    EXPECT_NEAR(2.7, hgt[0][0](0, 4), 0.1);
    EXPECT_NEAR(16.9, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(23.3, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(38.1, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(30.3, hgt[0][0](4, 4), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    |   -63.3	-62.1	-61.0	-59.9	-58.8
    |   -59.5	-58.4	-57.5	-56.5	-55.4
    |   -54.2	-53.6	-52.9	-52.1	-51.5
    |   -48.7	-48.2	-47.7	-47.2	-47.3
    |   -42.4	-42.2	-41.9	-42.1	-42.4
    */
    EXPECT_NEAR(-63.3, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(-62.1, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(-61.0, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(-59.9, hgt[7][0](0, 3), 0.1);
    EXPECT_NEAR(-58.8, hgt[7][0](0, 4), 0.1);
    EXPECT_NEAR(-59.5, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(-54.2, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(-42.4, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(-42.4, hgt[7][0](4, 4), 0.1);

    wxDELETE(predictor);
}

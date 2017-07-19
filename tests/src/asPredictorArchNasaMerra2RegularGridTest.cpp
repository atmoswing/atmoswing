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

#include <wx/filename.h>
#include "asPredictorArch.h"
#include "asGeoAreaCompositeRegularGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorArchNasaMerra2Regular, LoadEasy)
{
    double Xmin = 2.5;
    double Xwidth = 5;
    double Xstep = 0.625;
    double Ymin = 75;
    double Ywidth = 2;
    double Ystep = 0.5;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-nasa-merra2/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/h",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

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

    wxDELETE(predictor);
}

TEST(PredictorArchNasaMerra2Regular, LoadComposite)
{
    double Xmin = 177.5;
    double Xwidth = 5;
    double Xstep = 0.625;
    double Ymin = 75;
    double Ywidth = 2;
    double Ystep = 0.5;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-nasa-merra2/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NASA_MERRA_2", "inst6_3d_ana_Np/h",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    18.7	16.8	14.8	12.8   |    10.8	8.8	    6.8	    4.8	    2.7
    24.1	22.2	20.4	18.7   |    16.9	15.0	13.0	11.0	9.0
    30.5	28.6	26.8	25.1   |    23.3	21.5	19.6	17.7	15.8
    38.0	36.2	34.3	32.5   |    30.6	28.7	26.7	24.8	22.9
    45.7	43.8	41.9	40.0   |    38.1	36.2	34.2	32.2	30.3
    */
    EXPECT_NEAR(18.7, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(16.8, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(14.8, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(12.8, hgt[0][0](0, 3), 0.1);
    //EXPECT_NEAR(10.8, hgt[0][0](0, 4), 0.1);
    //EXPECT_NEAR(2.7, hgt[0][0](0, 8), 0.1);
    EXPECT_NEAR(24.1, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(30.5, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(45.7, hgt[0][0](4, 0), 0.1);
    //EXPECT_NEAR(30.3, hgt[0][0](4, 8), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    -68.0	-66.8	-65.7	-64.5   |   -63.3	-62.1	-61.0	-59.9	-58.8
    -63.7	-62.6	-61.6	-60.6   |   -59.5	-58.4	-57.5	-56.5	-55.4
    -57.5	-56.6	-55.9	-55.1   |   -54.2	-53.6	-52.9	-52.1	-51.5
    -51.1	-50.5	-49.9	-49.2   |   -48.7	-48.2	-47.7	-47.2	-47.3
    -43.8	-43.4	-43.0	-42.6   |   -42.4	-42.2	-41.9	-42.1	-42.4
    */
    EXPECT_NEAR(-68.0, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(-66.8, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(-65.7, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(-64.5, hgt[7][0](0, 3), 0.1);
    //EXPECT_NEAR(-63.3, hgt[7][0](0, 4), 0.1);
    //EXPECT_NEAR(-58.8, hgt[7][0](0, 8), 0.1);
    EXPECT_NEAR(-63.7, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(-57.5, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(-43.8, hgt[7][0](4, 0), 0.1);
    //EXPECT_NEAR(-42.4, hgt[7][0](4, 8), 0.1);

    wxDELETE(predictor);
}

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
#include "asAreaCompRegGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorArchNcepCfsrRegular, LoadEasy)
{
    double xMin = 10;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 2;
    double step = 0.5;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-cfsr/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_CFSR", "pgbh/hgt", predictorDataDir);

    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    171.1	170.7	169.1	167.1	164.5	162.1	160.0	158.3	156.8	155.3	153.5
    172.0	171.7	170.7	168.9	166.6	164.6	162.3	159.7	157.5	155.3	153.4
    173.5	172.4	171.5	169.7	167.8	166.5	165.1	163.4	161.1	159.3	157.3
    174.9	173.6	171.9	170.1	168.9	167.9	166.6	165.5	163.3	161.4	159.6
    175.7	173.8	171.9	170.2	169.0	168.7	167.5	166.2	164.6	162.8	161.1
    */
    EXPECT_NEAR(171.1, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(170.7, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(169.1, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(167.1, hgt[0][0](0, 3), 0.1);
    EXPECT_NEAR(153.5, hgt[0][0](0, 10), 0.1);
    EXPECT_NEAR(172.0, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(173.5, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(175.7, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(161.1, hgt[0][0](4, 10), 0.1);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    163.0	161.8	160.8	160.9	160.8	160.9	160.7	159.4	157.9	156.5	154.8
    165.4	163.9	163.4	163.5	163.4	163.3	162.4	161.1	159.5	157.6	156.3
    167.4	165.4	166.2	166.1	165.9	165.7	165.1	164.2	162.7	161.7	160.2
    168.1	168.0	168.7	168.8	168.9	168.7	168.0	167.3	165.5	164.2	162.6
    170.3	170.4	170.6	170.8	170.9	170.9	170.2	169.0	167.7	166.3	164.8
    */
    EXPECT_NEAR(163.0, hgt[1][0](0, 0), 0.1);
    EXPECT_NEAR(161.8, hgt[1][0](0, 1), 0.1);
    EXPECT_NEAR(160.8, hgt[1][0](0, 2), 0.1);
    EXPECT_NEAR(160.9, hgt[1][0](0, 3), 0.1);
    EXPECT_NEAR(154.8, hgt[1][0](0, 10), 0.1);
    EXPECT_NEAR(165.4, hgt[1][0](1, 0), 0.1);
    EXPECT_NEAR(167.4, hgt[1][0](2, 0), 0.1);
    EXPECT_NEAR(170.3, hgt[1][0](4, 0), 0.1);
    EXPECT_NEAR(164.8, hgt[1][0](4, 10), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    147.3	149.4	150.1	150.8	150.5	149.8	149.1	148.1	148.4	150.4	152.7
    143.9	147.2	149.1	150.1	150.8	150.6	149.7	149.0	149.8	151.5	152.8
    143.6	146.6	148.8	150.0	150.9	151.3	151.7	152.3	153.0	154.3	154.5
    142.8	145.7	148.5	150.8	152.1	152.6	152.8	153.9	154.2	154.7	155.1
    141.9	143.9	148.3	150.9	151.9	153.2	153.8	154.5	155.4	155.9	156.3
    */
    EXPECT_NEAR(147.3, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(149.4, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(150.1, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(150.8, hgt[7][0](0, 3), 0.1);
    EXPECT_NEAR(152.7, hgt[7][0](0, 10), 0.1);
    EXPECT_NEAR(143.9, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(143.6, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(141.9, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(156.3, hgt[7][0](4, 10), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchNcepCfsrRegular, LoadComposite)
{
    double xMin = -3;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 2;
    double step = 0.5;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-cfsr/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_CFSR", "pgbh/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    174.8	169.0	161.9	159.3	158.2	156.8   |   153.9	151.8	150.9	150.8	151.1
    159.4	157.8	156.4	156.1	154.4	151.7   |   149.6	148.3	148.0	149.5	150.6
    156.4	156.4	154.9	152.6	150.4	149.2   |   148.0	146.7	149.8	158.7	165.2
    155.9	153.7	150.3	148.7	149.0	150.5   |   151.0	151.9	156.6	166.0	169.5
    156.3	152.3	150.7	155.2	162.2	164.2   |   165.0	165.8	167.7	171.3	170.1
    */
    EXPECT_NEAR(174.8, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(169.0, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(161.9, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(156.8, hgt[0][0](0, 5), 0.1);
    EXPECT_NEAR(153.9, hgt[0][0](0, 6), 0.1);
    EXPECT_NEAR(151.1, hgt[0][0](0, 10), 0.1);
    EXPECT_NEAR(159.4, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(156.4, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(156.3, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(170.1, hgt[0][0](4, 10), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    169.2	163.9	158.3	158.0	158.3	159.0   |   157.9	156.8	156.4	155.6	154.9
    155.7	154.5	154.1	154.9	155.5	155.1   |   154.3	153.6	153.3	154.7	155.9
    152.3	153.1	152.8	152.3	152.1	152.2   |   150.7	147.6	147.9	152.4	155.7
    152.0	151.8	150.0	148.9	148.5	148.7   |   147.4	145.7	146.3	148.8	146.7
    149.7	150.2	150.5	151.8	152.7	151.3   |   150.8	150.8	150.9	150.6	145.9
    */
    EXPECT_NEAR(169.2, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(163.9, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(158.3, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(159.0, hgt[7][0](0, 5), 0.1);
    EXPECT_NEAR(157.9, hgt[7][0](0, 6), 0.1);
    EXPECT_NEAR(154.9, hgt[7][0](0, 10), 0.1);
    EXPECT_NEAR(155.7, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(152.3, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(149.7, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(145.9, hgt[7][0](4, 10), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchNcepCfsrRegular, LoadBorderLeft)
{
    double xMin = 0;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 2;
    double step = 0.5;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-cfsr/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_CFSR", "pgbh/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   153.9	151.8	150.9	150.8	151.1
    |   149.6	148.3	148.0	149.5	150.6
    |   148.0	146.7	149.8	158.7	165.2
    |   151.0	151.9	156.6	166.0	169.5
    |   165.0	165.8	167.7	171.3	170.1
    */
    EXPECT_NEAR(153.9, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(151.8, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(150.9, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(151.1, hgt[0][0](0, 4), 0.1);
    EXPECT_NEAR(149.6, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(148.0, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(165.0, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(170.1, hgt[0][0](4, 4), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    |   157.9	156.8	156.4	155.6	154.9
    |   154.3	153.6	153.3	154.7	155.9
    |   150.7	147.6	147.9	152.4	155.7
    |   147.4	145.7	146.3	148.8	146.7
    |   150.8	150.8	150.9	150.6	145.9
    */
    EXPECT_NEAR(157.9, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(156.8, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(156.4, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(154.9, hgt[7][0](0, 4), 0.1);
    EXPECT_NEAR(154.3, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(150.7, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(150.8, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(145.9, hgt[7][0](4, 4), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchNcepCfsrRegular, LoadBorderLeftOn720)
{
    double xMin = 360;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 2;
    double step = 0.5;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-cfsr/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_CFSR", "pgbh/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   153.9	151.8	150.9	150.8	151.1
    |   149.6	148.3	148.0	149.5	150.6
    |   148.0	146.7	149.8	158.7	165.2
    |   151.0	151.9	156.6	166.0	169.5
    |   165.0	165.8	167.7	171.3	170.1
    */
    EXPECT_NEAR(153.9, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(151.8, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(150.9, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(151.1, hgt[0][0](0, 4), 0.1);
    EXPECT_NEAR(149.6, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(148.0, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(165.0, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(170.1, hgt[0][0](4, 4), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    |   157.9	156.8	156.4	155.6	154.9
    |   154.3	153.6	153.3	154.7	155.9
    |   150.7	147.6	147.9	152.4	155.7
    |   147.4	145.7	146.3	148.8	146.7
    |   150.8	150.8	150.9	150.6	145.9
    */
    EXPECT_NEAR(157.9, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(156.8, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(156.4, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(154.9, hgt[7][0](0, 4), 0.1);
    EXPECT_NEAR(154.3, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(150.7, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(150.8, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(145.9, hgt[7][0](4, 4), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchNcepCfsrRegular, LoadBorderRight)
{
    double xMin = -3;
    double xWidth = 3;
    double yMin = 35;
    double yWidth = 2;
    double step = 0.5;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-cfsr/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_CFSR", "pgbh/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    174.8	169.0	161.9	159.3	158.2	156.8   |   153.9
    159.4	157.8	156.4	156.1	154.4	151.7   |   149.6
    156.4	156.4	154.9	152.6	150.4	149.2   |   148.0
    155.9	153.7	150.3	148.7	149.0	150.5   |   151.0
    156.3	152.3	150.7	155.2	162.2	164.2   |   165.0
    */
    EXPECT_NEAR(174.8, hgt[0][0](0, 0), 0.1);
    EXPECT_NEAR(169.0, hgt[0][0](0, 1), 0.1);
    EXPECT_NEAR(161.9, hgt[0][0](0, 2), 0.1);
    EXPECT_NEAR(156.8, hgt[0][0](0, 5), 0.1);
    EXPECT_NEAR(153.9, hgt[0][0](0, 6), 0.1);
    EXPECT_NEAR(159.4, hgt[0][0](1, 0), 0.1);
    EXPECT_NEAR(156.4, hgt[0][0](2, 0), 0.1);
    EXPECT_NEAR(156.3, hgt[0][0](4, 0), 0.1);
    EXPECT_NEAR(165.0, hgt[0][0](4, 6), 0.1);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    169.2	163.9	158.3	158.0	158.3	159.0   |   157.9
    155.7	154.5	154.1	154.9	155.5	155.1   |   154.3
    152.3	153.1	152.8	152.3	152.1	152.2   |   150.7
    152.0	151.8	150.0	148.9	148.5	148.7   |   147.4
    149.7	150.2	150.5	151.8	152.7	151.3   |   150.8
    */
    EXPECT_NEAR(169.2, hgt[7][0](0, 0), 0.1);
    EXPECT_NEAR(163.9, hgt[7][0](0, 1), 0.1);
    EXPECT_NEAR(158.3, hgt[7][0](0, 2), 0.1);
    EXPECT_NEAR(159.0, hgt[7][0](0, 5), 0.1);
    EXPECT_NEAR(157.9, hgt[7][0](0, 6), 0.1);
    EXPECT_NEAR(155.7, hgt[7][0](1, 0), 0.1);
    EXPECT_NEAR(152.3, hgt[7][0](2, 0), 0.1);
    EXPECT_NEAR(149.7, hgt[7][0](4, 0), 0.1);
    EXPECT_NEAR(150.8, hgt[7][0](4, 6), 0.1);

    wxDELETE(predictor);
}

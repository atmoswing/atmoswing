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
#include "asPredictor.h"
#include "asAreaCompGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorNcepR2General, LoadEasy)
{
    double xMin = 10;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    60.0 	43.0	29.0	20.0	20.0
    107.0	96.0	83.0	75.0	72.0
    140.0	132.0	123.0	116.0	113.0
    */
    EXPECT_FLOAT_EQ(60, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(43, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(29, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(20, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(20, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(107, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(140, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(113, hgt[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    57.0	52.0	46.0	37.0	26.0
    90.0	83.0	77.0	68.0	60.0
    117.0	111.0	105.0	99.0	94.0
    */
    EXPECT_FLOAT_EQ(57, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(52, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(46, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(37, hgt[1][0](0, 3));
    EXPECT_FLOAT_EQ(26, hgt[1][0](0, 4));
    EXPECT_FLOAT_EQ(90, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(117, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(94, hgt[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    172.0	174.0	169.0	160.0	149.0
    188.0	196.0	192.0	178.0	156.0
    204.0	214.0	212.0	194.0	166.0
    */
    EXPECT_FLOAT_EQ(172, hgt[11][0](0, 0));
    EXPECT_FLOAT_EQ(174, hgt[11][0](0, 1));
    EXPECT_FLOAT_EQ(169, hgt[11][0](0, 2));
    EXPECT_FLOAT_EQ(160, hgt[11][0](0, 3));
    EXPECT_FLOAT_EQ(149, hgt[11][0](0, 4));
    EXPECT_FLOAT_EQ(188, hgt[11][0](1, 0));
    EXPECT_FLOAT_EQ(204, hgt[11][0](2, 0));
    EXPECT_FLOAT_EQ(166, hgt[11][0](2, 4));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    55.0	64.0	67.0	65.0	61.0
    95.0	94.0	90.0	84.0	80.0
    124.0	117.0	111.0	107.0	107.0
    */
    EXPECT_FLOAT_EQ(55, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(64, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(67, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(65, hgt[40][0](0, 3));
    EXPECT_FLOAT_EQ(61, hgt[40][0](0, 4));
    EXPECT_FLOAT_EQ(95, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(124, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(107, hgt[40][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadComposite)
{
    double xMin = -10;
    int xPtsNb = 7;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	170.0	157.0	140.0   |   121.0	104.0	89.0
    198.0	190.0	178.0	163.0   |   147.0	133.0	123.0
    203.0	196.0	189.0	179.0   |   168.0	156.0	148.0
    */
    EXPECT_FLOAT_EQ(176, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(170, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(157, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(140, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(121, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(104, hgt[0][0](0, 5));
    EXPECT_FLOAT_EQ(89, hgt[0][0](0, 6));
    EXPECT_FLOAT_EQ(198, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(203, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(148, hgt[0][0](2, 6));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    176.0	169.0	157.0	139.0    |   119.0	98.0	79.0
    189.0	187.0	180.0	169.0    |   152.0	133.0	114.0
    197.0	196.0	194.0	188.0    |   174.0	155.0	138.0
    */
    EXPECT_FLOAT_EQ(176, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(169, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(157, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(139, hgt[1][0](0, 3));
    EXPECT_FLOAT_EQ(119, hgt[1][0](0, 4));
    EXPECT_FLOAT_EQ(98, hgt[1][0](0, 5));
    EXPECT_FLOAT_EQ(79, hgt[1][0](0, 6));
    EXPECT_FLOAT_EQ(189, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(197, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(138, hgt[1][0](2, 6));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    85.0	81.0	85.0	96.0     |   112.0	130.0	147.0
    127.0	121.0	119.0	122.0    |   132.0	144.0	159.0
    138.0	130.0	127.0	133.0    |   145.0	159.0	173.0
    */
    EXPECT_FLOAT_EQ(85, hgt[11][0](0, 0));
    EXPECT_FLOAT_EQ(81, hgt[11][0](0, 1));
    EXPECT_FLOAT_EQ(85, hgt[11][0](0, 2));
    EXPECT_FLOAT_EQ(96, hgt[11][0](0, 3));
    EXPECT_FLOAT_EQ(112, hgt[11][0](0, 4));
    EXPECT_FLOAT_EQ(130, hgt[11][0](0, 5));
    EXPECT_FLOAT_EQ(147, hgt[11][0](0, 6));
    EXPECT_FLOAT_EQ(127, hgt[11][0](1, 0));
    EXPECT_FLOAT_EQ(138, hgt[11][0](2, 0));
    EXPECT_FLOAT_EQ(173, hgt[11][0](2, 6));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    198.0	180.0	156.0	127.0   |   95.0	67.0	51.0
    224.0	210.0	191.0	166.0   |   140.0	117.0	102.0
    229.0	219.0	207.0	191.0   |   171.0	153.0	140.0
    */
    EXPECT_FLOAT_EQ(198, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(180, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(156, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(127, hgt[40][0](0, 3));
    EXPECT_FLOAT_EQ(95, hgt[40][0](0, 4));
    EXPECT_FLOAT_EQ(67, hgt[40][0](0, 5));
    EXPECT_FLOAT_EQ(51, hgt[40][0](0, 6));
    EXPECT_FLOAT_EQ(224, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(229, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(140, hgt[40][0](2, 6));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadBorderLeft)
{
    double xMin = 0;
    int xPtsNb = 3;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   121.0	104.0	89.0
    |   147.0	133.0	123.0
    |   168.0	156.0	148.0
    */
    EXPECT_FLOAT_EQ(121, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(104, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(89, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(147, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(168, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(148, hgt[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   119.0	98.0	79.0
    |   152.0	133.0	114.0
    |   174.0	155.0	138.0
    */
    EXPECT_FLOAT_EQ(119, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(98, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(79, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(152, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(174, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(138, hgt[1][0](2, 2));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    |   95.0	67.0	51.0
    |   140.0	117.0	102.0
    |   171.0	153.0	140.0
    */
    EXPECT_FLOAT_EQ(95, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(67, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(51, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(140, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(171, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(140, hgt[40][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadBorderLeftOn720)
{
    double xMin = 360;
    int xPtsNb = 3;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   121.0	104.0	89.0
    |   147.0	133.0	123.0
    |   168.0	156.0	148.0
    */
    EXPECT_FLOAT_EQ(121, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(104, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(89, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(147, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(168, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(148, hgt[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   119.0	98.0	79.0
    |   152.0	133.0	114.0
    |   174.0	155.0	138.0
    */
    EXPECT_FLOAT_EQ(119, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(98, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(79, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(152, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(174, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(138, hgt[1][0](2, 2));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    |   95.0	67.0	51.0
    |   140.0	117.0	102.0
    |   171.0	153.0	140.0
    */
    EXPECT_FLOAT_EQ(95, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(67, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(51, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(140, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(171, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(140, hgt[40][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadBorderRight)
{
    double xMin = 350;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	170.0	157.0	140.0   |   121.0
    198.0	190.0	178.0	163.0   |   147.0
    203.0	196.0	189.0	179.0   |   168.0
    */
    EXPECT_FLOAT_EQ(176, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(170, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(157, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(140, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(121, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(198, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(203, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(168, hgt[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    176.0	169.0	157.0	139.0    |   119.0
    189.0	187.0	180.0	169.0    |   152.0
    197.0	196.0	194.0	188.0    |   174.0
    */
    EXPECT_FLOAT_EQ(176, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(169, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(157, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(139, hgt[1][0](0, 3));
    EXPECT_FLOAT_EQ(119, hgt[1][0](0, 4));
    EXPECT_FLOAT_EQ(189, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(197, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(174, hgt[1][0](2, 4));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    198.0	180.0	156.0	127.0   |   95.0
    224.0	210.0	191.0	166.0   |   140.0
    229.0	219.0	207.0	191.0   |   171.0
    */
    EXPECT_FLOAT_EQ(198, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(180, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(156, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(127, hgt[40][0](0, 3));
    EXPECT_FLOAT_EQ(95, hgt[40][0](0, 4));
    EXPECT_FLOAT_EQ(224, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(229, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(171, hgt[40][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadCompositeStepLon)
{
    double xMin = -10;
    int xPtsNb = 7;
    double yMin = 35;
    int yPtsNb = 3;
    double steplon = 5;
    double steplat = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	157.0	|   121.0	89.0
    198.0	178.0	|   147.0	123.0
    203.0	189.0	|   168.0	148.0
    subset of :
    176.0	170.0	157.0	140.0   |   121.0	104.0	89.0
    198.0	190.0	178.0	163.0   |   147.0	133.0	123.0
    203.0	196.0	189.0	179.0   |   168.0	156.0	148.0
    */
    EXPECT_FLOAT_EQ(176, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(157, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(121, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(89, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(198, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(203, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(148, hgt[0][0](2, 3));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    176.0	157.0	|   119.0	79.0
    189.0	180.0	|   152.0	114.0
    197.0	194.0	|   174.0	138.0
    */
    EXPECT_FLOAT_EQ(176, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(157, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(119, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(79, hgt[1][0](0, 3));
    EXPECT_FLOAT_EQ(189, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(197, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(138, hgt[1][0](2, 3));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    198.0	156.0	|   95.0	51.0
    224.0	191.0	|   140.0	102.0
    229.0	207.0	|   171.0	140.0
    */
    EXPECT_FLOAT_EQ(198, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(156, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(95, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(51, hgt[40][0](0, 3));
    EXPECT_FLOAT_EQ(224, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(229, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(140, hgt[40][0](2, 3));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadCompositeStepLonMoved)
{
    double xMin = -7.5;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double steplon = 5;
    double steplat = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    170.0	140.0   |   104.0
    190.0	163.0   |   133.0
    196.0	179.0   |   156.0
    */
    EXPECT_FLOAT_EQ(170, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(140, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(104, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(190, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(196, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(156, hgt[0][0](2, 2));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    180.0	127.0   |   67.0
    210.0	166.0   |   117.0
    219.0	191.0   |   153.0
    */
    EXPECT_FLOAT_EQ(180, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(127, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(67, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(210, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(219, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(153, hgt[40][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadCompositeStepLonLat)
{
    double xMin = -10;
    int xPtsNb = 4;
    double yMin = 35;
    int yPtsNb = 2;
    double steplon = 5;
    double steplat = 5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	157.0	|   121.0	89.0
    203.0	189.0	|   168.0	148.0
    */
    EXPECT_FLOAT_EQ(176, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(157, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(121, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(89, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(203, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(148, hgt[0][0](1, 3));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    176.0	157.0	|   119.0	79.0
    197.0	194.0	|   174.0	138.0
    */
    EXPECT_FLOAT_EQ(176, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(157, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(119, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(79, hgt[1][0](0, 3));
    EXPECT_FLOAT_EQ(197, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(138, hgt[1][0](1, 3));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    198.0	156.0	|   95.0	51.0
    229.0	207.0	|   171.0	140.0
    */
    EXPECT_FLOAT_EQ(198, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(156, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(95, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(51, hgt[40][0](0, 3));
    EXPECT_FLOAT_EQ(229, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(140, hgt[40][0](1, 3));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2General, LoadCompositeStepLonLatTime)
{
    double xMin = -10;
    int xPtsNb = 4;
    double yMin = 35;
    int yPtsNb = 2;
    double steplon = 5;
    double steplat = 5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 24;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	157.0	|   121.0	89.0
    203.0	189.0	|   168.0	148.0
    */
    EXPECT_FLOAT_EQ(176, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(157, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(121, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(89, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(203, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(148, hgt[0][0](1, 3));

    /* Values time step 10 (horizontal=Lon, vertical=Lat)
    198.0	156.0	|   95.0	51.0
    229.0	207.0	|   171.0	140.0
    */
    EXPECT_FLOAT_EQ(198, hgt[10][0](0, 0));
    EXPECT_FLOAT_EQ(156, hgt[10][0](0, 1));
    EXPECT_FLOAT_EQ(95, hgt[10][0](0, 2));
    EXPECT_FLOAT_EQ(51, hgt[10][0](0, 3));
    EXPECT_FLOAT_EQ(229, hgt[10][0](1, 0));
    EXPECT_FLOAT_EQ(140, hgt[10][0](1, 3));

    wxDELETE(area);
    wxDELETE(predictor);
}
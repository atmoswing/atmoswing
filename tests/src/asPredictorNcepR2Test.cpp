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

#include "asAreaGrid.h"
#include "asAreaRegGrid.h"
#include "asPredictor.h"
#include "asTimeArray.h"

TEST(PredictorNcepR2, GetCorrectPredictors) {
    asPredictor *predictor;

    predictor = asPredictor::GetInstance("NCEP_R2", "pressure/air", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "pressure/rhum", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "pressure/omega", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "pressure/uwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "pressure/vwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "pressure/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "surface/pr_wtr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "surface/pres", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "surface/slp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/shum", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/tmax2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/tmin2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/sktmp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/tmp0-10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/soilw0-10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilMoisture);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/soilw10-200", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilMoisture);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/tmp10-200", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/uwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/vwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/weasd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SnowWaterEquivalent);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/cprat", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitationRate);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/dlwrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/dswrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/gflux", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/lhtfl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/pevpr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialEvaporation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/prate", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitationRate);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/shtfl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/tcdc", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::CloudCover);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/uflx", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MomentumFlux);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/ugwd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GravityWaveStress);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/ulwrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/uswrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/vflx", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MomentumFlux);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R2", "gauss/vgwd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GravityWaveStress);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2, LoadEasy) {
    double xMin = 10;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, RegularLoadEasy) {
    double xMin = 10;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 5;
    double step = 2.5;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

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

    wxDELETE(predictor);
}

TEST(PredictorNcepR2, GetMinMaxValues) {
    double xMin = 10;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 5;
    double step = 2.5;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    EXPECT_FLOAT_EQ(-7, predictor->GetMinValue());
    EXPECT_FLOAT_EQ(286, predictor->GetMaxValue());

    wxDELETE(predictor);
}

TEST(PredictorNcepR2, LoadWithNegativeVals) {
    double xMin = -10;
    int xPtsNb = 7;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, LoadBorderLeft) {
    double xMin = 0;
    int xPtsNb = 3;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, LoadBorderLeftOn720) {
    double xMin = 360;
    int xPtsNb = 3;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, LoadBorderRight) {
    double xMin = 350;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, LoadWithNegativeValsStepLon) {
    double xMin = -10;
    int xPtsNb = 7;
    double yMin = 35;
    int yPtsNb = 3;
    double steplon = 5;
    double steplat = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, LoadWithNegativeValsStepLonMoved) {
    double xMin = -7.5;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double steplon = 5;
    double steplat = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, LoadWithNegativeValsStepLonLat) {
    double xMin = -10;
    int xPtsNb = 4;
    double yMin = 35;
    int yPtsNb = 2;
    double steplon = 5;
    double steplat = 5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, LoadWithNegativeValsStepLonLatTime) {
    double xMin = -10;
    int xPtsNb = 4;
    double yMin = 35;
    int yPtsNb = 2;
    double steplon = 5;
    double steplat = 5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 24;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

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

TEST(PredictorNcepR2, RegularLoadWithNegativeValsStepLonLatTime) {
    double xMin = -10;
    double xWidth = 15;
    double yMin = 35;
    double yWidth = 5;
    double steplon = 5;
    double steplat = 5;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, steplon, yMin, yWidth, steplat);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 11, 00, 00);
    double timeStep = 24;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

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

    wxDELETE(predictor);
}

TEST(PredictorNcepR2, GaussianLoadEasy) {
    double xMin = 7.5;
    int xPtsNb = 5;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 2;
    asAreaGrid *area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "gaussian_grid/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    286.44	286.15	287.11	288.86	290.08
    283.91	282.92	283.69	285.07	286.79
    278.96	279.84	280.42	281.24	282.51
    */
    EXPECT_FLOAT_EQ(286.44f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(286.15f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(287.11f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(288.86f, air[0][0](0, 3));
    EXPECT_FLOAT_EQ(290.08f, air[0][0](0, 4));
    EXPECT_FLOAT_EQ(283.91f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(278.96f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(282.51f, air[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    293.04	292.96	293.09	290.29	290.60
    289.10	288.84	290.33	292.02	293.44
    286.25	286.60	289.12	291.63	292.75
    */
    EXPECT_FLOAT_EQ(293.04f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(292.96f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(293.09f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(290.29f, air[1][0](0, 3));
    EXPECT_FLOAT_EQ(290.60f, air[1][0](0, 4));
    EXPECT_FLOAT_EQ(289.10f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(286.25f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(292.75f, air[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    279.43	277.99	279.23	284.24	285.07
    280.17	279.24	281.57	282.47	282.28
    278.08	281.46	283.77	284.54	284.39
    */
    EXPECT_FLOAT_EQ(279.43f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(277.99f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(279.23f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(284.24f, air[11][0](0, 3));
    EXPECT_FLOAT_EQ(285.07f, air[11][0](0, 4));
    EXPECT_FLOAT_EQ(280.17f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(278.08f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(284.39f, air[11][0](2, 4));

    /* Values time step 20 (horizontal=Lon, vertical=Lat)
    281.81	283.26	286.18	288.59	289.69
    281.82	282.76	284.36	284.05	283.94
    282.53	284.64	283.24	279.87	278.18
    */
    EXPECT_FLOAT_EQ(281.81f, air[20][0](0, 0));
    EXPECT_FLOAT_EQ(283.26f, air[20][0](0, 1));
    EXPECT_FLOAT_EQ(286.18f, air[20][0](0, 2));
    EXPECT_FLOAT_EQ(288.59f, air[20][0](0, 3));
    EXPECT_FLOAT_EQ(289.69f, air[20][0](0, 4));
    EXPECT_FLOAT_EQ(281.82f, air[20][0](1, 0));
    EXPECT_FLOAT_EQ(282.53f, air[20][0](2, 0));
    EXPECT_FLOAT_EQ(278.18f, air[20][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2, GaussianLoadWithNegativeVals) {
    double xMin = -7.5;
    int xPtsNb = 7;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 2;
    asAreaGrid *area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "gaussian_grid/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    278.10	278.82	285.20	284.96   |   284.46	285.92	287.61
    277.15	272.98	274.92	279.08   |   282.23	283.46	284.65
    273.57	272.72	272.38	272.51   |   275.49	273.53	271.54
    */
    EXPECT_FLOAT_EQ(278.10f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(278.82f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(285.20f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(284.96f, air[0][0](0, 3));
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 4));
    EXPECT_FLOAT_EQ(285.92f, air[0][0](0, 5));
    EXPECT_FLOAT_EQ(287.61f, air[0][0](0, 6));
    EXPECT_FLOAT_EQ(277.15f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(273.57f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.54f, air[0][0](2, 6));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    291.76	292.21	289.61	288.81   |   289.55	290.82	292.27
    291.99	291.38	290.35	289.85   |   288.82	289.76	290.56
    293.69	292.93	291.72	289.46   |   288.36	288.09	287.94
    */
    EXPECT_FLOAT_EQ(291.76f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(292.21f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(289.61f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.81f, air[1][0](0, 3));
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 4));
    EXPECT_FLOAT_EQ(290.82f, air[1][0](0, 5));
    EXPECT_FLOAT_EQ(292.27f, air[1][0](0, 6));
    EXPECT_FLOAT_EQ(291.99f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(293.69f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(287.94f, air[1][0](2, 6));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    281.45	280.02	286.29	285.97   |   284.87	284.87	284.68
    280.27	283.96	285.08	284.46   |   284.12	284.58	285.24
    283.65	285.85	284.79	283.07   |   281.65	282.45	282.25
    */
    EXPECT_FLOAT_EQ(281.45f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(280.02f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(286.29f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(285.97f, air[11][0](0, 3));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 4));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 5));
    EXPECT_FLOAT_EQ(284.68f, air[11][0](0, 6));
    EXPECT_FLOAT_EQ(280.27f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(283.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(282.25f, air[11][0](2, 6));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2, GaussianLoadBorderLeft) {
    double xMin = 0;
    int xPtsNb = 3;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 2;
    asAreaGrid *area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "gaussian_grid/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   284.46	285.92	287.61
    |   282.23	283.46	284.65
    |   275.49	273.53	271.54
    */
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(285.92f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(287.61f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(282.23f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(275.49f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.54f, air[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.55	290.82	292.27
    |   288.82	289.76	290.56
    |   288.36	288.09	287.94
    */
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(290.82f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(292.27f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.82f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(288.36f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(287.94f, air[1][0](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   284.87	284.87	284.68
    |   284.12	284.58	285.24
    |   281.65	282.45	282.25
    */
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(284.68f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(284.12f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(281.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(282.25f, air[11][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2, GaussianLoadBorderLeftOn720) {
    double xMin = 360;
    int xPtsNb = 3;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 2;
    asAreaGrid *area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "gaussian_grid/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   284.46	285.92	287.61
    |   282.23	283.46	284.65
    |   275.49	273.53	271.54
    */
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(285.92f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(287.61f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(282.23f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(275.49f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.54f, air[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.55	290.82	292.27
    |   288.82	289.76	290.56
    |   288.36	288.09	287.94
    */
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(290.82f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(292.27f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.82f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(288.36f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(287.94f, air[1][0](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   284.87	284.87	284.68
    |   284.12	284.58	285.24
    |   281.65	282.45	282.25
    */
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(284.68f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(284.12f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(281.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(282.25f, air[11][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR2, GaussianLoadBorderRight) {
    double xMin = 352.5;
    int xPtsNb = 5;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 2;
    asAreaGrid *area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R2", "gaussian_grid/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    278.10	278.82	285.20	284.96   |   284.46
    277.15	272.98	274.92	279.08   |   282.23
    273.57	272.72	272.38	272.51   |   275.49
    */
    EXPECT_FLOAT_EQ(278.10f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(278.82f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(285.20f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(284.96f, air[0][0](0, 3));
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 4));
    EXPECT_FLOAT_EQ(277.15f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(273.57f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(275.49f, air[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    291.76	292.21	289.61	288.81   |   289.55
    291.99	291.38	290.35	289.85   |   288.82
    293.69	292.93	291.72	289.46   |   288.36
    */
    EXPECT_FLOAT_EQ(291.76f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(292.21f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(289.61f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.81f, air[1][0](0, 3));
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 4));
    EXPECT_FLOAT_EQ(291.99f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(293.69f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(288.36f, air[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    281.45	280.02	286.29	285.97   |   284.87
    280.27	283.96	285.08	284.46   |   284.12
    283.65	285.85	284.79	283.07   |   281.65
    */
    EXPECT_FLOAT_EQ(281.45f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(280.02f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(286.29f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(285.97f, air[11][0](0, 3));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 4));
    EXPECT_FLOAT_EQ(280.27f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(283.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(281.65f, air[11][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

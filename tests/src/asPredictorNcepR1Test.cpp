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

#include <gtest/gtest.h>
#include <wx/filename.h>

#include "asAreaGrid.h"
#include "asAreaRegGrid.h"
#include "asPredictor.h"
#include "asTimeArray.h"

TEST(PredictorNcepR1, GetCorrectPredictors) {
    asPredictor* predictor;

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/air", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/rhum", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/shum", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/omega", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/uwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/vwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "pressure/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/air", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/lftx", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SurfaceLiftedIndex);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/lftx4", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SurfaceLiftedIndex);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/omega", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/pottmp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/pr_wtr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/pres", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/rhum", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/slp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/uwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "surface/vwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/pevpr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialEvaporation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/shum", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/soilw0-10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilMoisture);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/soilw10-200", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilMoisture);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/sktmp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/tmp0-10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/tmp10-200", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/tmp300", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SoilTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/uwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/vwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/weasd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SnowWaterEquivalent);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/tmax2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/tmin2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/cfnlf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/cfnsf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/cprat", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitationRate);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/csdlf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/csdsf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/csusf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/dlwrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/dswrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/gflux", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/lhtfl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/nbdsf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/nddsf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/nlwrs", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/nswrs", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/prate", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitationRate);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/shtfl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/uflx", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MomentumFlux);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/ugwd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GravityWaveStress);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/ulwrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/uswrf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/vbdsf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/vddsf", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/vflx", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MomentumFlux);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NCEP_R1", "gauss/vgwd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GravityWaveStress);
    wxDELETE(predictor);
}

TEST(PredictorNcepR1, LoadEasy) {
    double xMin = 10;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	175.0	170.0	162.0	151.0
    185.0	180.0	173.0	162.0	144.0
    193.0	183.0	174.0	163.0	143.0
    */
    EXPECT_FLOAT_EQ(176, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(175, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(170, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(162, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(151, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(185, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(193, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(143, hgt[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    159.0	156.0	151.0	146.0	141.0
    171.0	162.0	151.0	140.0	129.0
    182.0	168.0	154.0	143.0	130.0
    */
    EXPECT_FLOAT_EQ(159, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(156, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(151, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(146, hgt[1][0](0, 3));
    EXPECT_FLOAT_EQ(141, hgt[1][0](0, 4));
    EXPECT_FLOAT_EQ(171, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(182, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(130, hgt[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    121.0	104.0	98.0	102.0	114.0
    141.0	125.0	115.0	112.0	116.0
    158.0	147.0	139.0	133.0	131.0
    */
    EXPECT_FLOAT_EQ(121, hgt[11][0](0, 0));
    EXPECT_FLOAT_EQ(104, hgt[11][0](0, 1));
    EXPECT_FLOAT_EQ(98, hgt[11][0](0, 2));
    EXPECT_FLOAT_EQ(102, hgt[11][0](0, 3));
    EXPECT_FLOAT_EQ(114, hgt[11][0](0, 4));
    EXPECT_FLOAT_EQ(141, hgt[11][0](1, 0));
    EXPECT_FLOAT_EQ(158, hgt[11][0](2, 0));
    EXPECT_FLOAT_EQ(131, hgt[11][0](2, 4));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    -2.0	12.0	24.0	33.0	40.0
    -1.0	11.0	24.0	36.0	49.0
    6.0	    16.0	29.0	46.0	65.0
    */
    EXPECT_FLOAT_EQ(-2, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(12, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(24, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(33, hgt[40][0](0, 3));
    EXPECT_FLOAT_EQ(40, hgt[40][0](0, 4));
    EXPECT_FLOAT_EQ(-1, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(6, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(65, hgt[40][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR1, RegularLoadEasy) {
    double xMin = 10;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 5;
    double step = 2.5;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	175.0	170.0	162.0	151.0
    185.0	180.0	173.0	162.0	144.0
    193.0	183.0	174.0	163.0	143.0
    */
    EXPECT_FLOAT_EQ(176, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(175, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(170, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(162, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(151, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(185, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(193, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(143, hgt[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    159.0	156.0	151.0	146.0	141.0
    171.0	162.0	151.0	140.0	129.0
    182.0	168.0	154.0	143.0	130.0
    */
    EXPECT_FLOAT_EQ(159, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(156, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(151, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(146, hgt[1][0](0, 3));
    EXPECT_FLOAT_EQ(141, hgt[1][0](0, 4));
    EXPECT_FLOAT_EQ(171, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(182, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(130, hgt[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    121.0	104.0	98.0	102.0	114.0
    141.0	125.0	115.0	112.0	116.0
    158.0	147.0	139.0	133.0	131.0
    */
    EXPECT_FLOAT_EQ(121, hgt[11][0](0, 0));
    EXPECT_FLOAT_EQ(104, hgt[11][0](0, 1));
    EXPECT_FLOAT_EQ(98, hgt[11][0](0, 2));
    EXPECT_FLOAT_EQ(102, hgt[11][0](0, 3));
    EXPECT_FLOAT_EQ(114, hgt[11][0](0, 4));
    EXPECT_FLOAT_EQ(141, hgt[11][0](1, 0));
    EXPECT_FLOAT_EQ(158, hgt[11][0](2, 0));
    EXPECT_FLOAT_EQ(131, hgt[11][0](2, 4));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    -2.0	12.0	24.0	33.0	40.0
    -1.0	11.0	24.0	36.0	49.0
    6.0	    16.0	29.0	46.0	65.0
    */
    EXPECT_FLOAT_EQ(-2, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(12, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(24, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(33, hgt[40][0](0, 3));
    EXPECT_FLOAT_EQ(40, hgt[40][0](0, 4));
    EXPECT_FLOAT_EQ(-1, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(6, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(65, hgt[40][0](2, 4));

    wxDELETE(predictor);
}

TEST(PredictorNcepR1, RegularGetMinMaxValues) {
    double xMin = 10;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 5;
    double step = 2.5;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    EXPECT_FLOAT_EQ(-3, predictor->GetMinValue());
    EXPECT_FLOAT_EQ(199, predictor->GetMaxValue());

    wxDELETE(predictor);
}

TEST(PredictorNcepR1, LoadBorderLeft) {
    double xMin = 0;
    int xPtsNb = 3;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   175.0	171.0	171.0
    |   185.0	182.0	184.0
    |   195.0	196.0	199.0
    */
    EXPECT_FLOAT_EQ(175, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(171, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(171, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(185, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(195, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(199, hgt[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   170.0	163.0	160.0
    |   179.0	175.0	175.0
    |   190.0	189.0	191.0
    */
    EXPECT_FLOAT_EQ(170, hgt[1][0](0, 0));
    EXPECT_FLOAT_EQ(163, hgt[1][0](0, 1));
    EXPECT_FLOAT_EQ(160, hgt[1][0](0, 2));
    EXPECT_FLOAT_EQ(179, hgt[1][0](1, 0));
    EXPECT_FLOAT_EQ(190, hgt[1][0](2, 0));
    EXPECT_FLOAT_EQ(191, hgt[1][0](2, 2));

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    |   4.0	    -5.0	-12.0
    |   -2.0	-7.0	-11.0
    |   7.0	    1.0	    -2.0
    */
    EXPECT_FLOAT_EQ(4, hgt[40][0](0, 0));
    EXPECT_FLOAT_EQ(-5, hgt[40][0](0, 1));
    EXPECT_FLOAT_EQ(-12, hgt[40][0](0, 2));
    EXPECT_FLOAT_EQ(-2, hgt[40][0](1, 0));
    EXPECT_FLOAT_EQ(7, hgt[40][0](2, 0));
    EXPECT_FLOAT_EQ(-2, hgt[40][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR1, SetData) {
    double xMin = 0;
    int xPtsNb = 4;
    double yMin = 35;
    int yPtsNb = 2;
    double steplon = 5;
    double steplat = 5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 5, 00, 00);
    double timeStep = 24;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f newdata(5, va2f(1, a2f::Zero(1, 4)));

    newdata[0][0] << 1, 2, 3, 4;
    newdata[1][0] << 11, 12, 13, 14;
    newdata[2][0] << 21, 22, 23, 24;
    newdata[3][0] << 31, 32, 33, 34;
    newdata[4][0] << 41, 42, 43, 44;

    predictor->SetData(newdata);

    EXPECT_FLOAT_EQ(1, predictor->GetLatPtsnb());
    EXPECT_FLOAT_EQ(4, predictor->GetLonPtsnb());
    EXPECT_FLOAT_EQ(1, predictor->GetData()[0][0](0, 0));
    EXPECT_FLOAT_EQ(2, predictor->GetData()[0][0](0, 1));
    EXPECT_FLOAT_EQ(4, predictor->GetData()[0][0](0, 3));
    EXPECT_FLOAT_EQ(14, predictor->GetData()[1][0](0, 3));
    EXPECT_FLOAT_EQ(41, predictor->GetData()[4][0](0, 0));
    EXPECT_FLOAT_EQ(44, predictor->GetData()[4][0](0, 3));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR1, GaussianLoadEasy) {
    double xMin = 7.5;
    int xPtsNb = 5;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    asAreaGrid* area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, 0));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    276.9	279.6	282.7	286.7	288.5
    276.2	275.5	276.9	280.4	283.7
    271.3	270.8	273.9	275.6	277.7
    */
    EXPECT_FLOAT_EQ(276.9f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(279.6f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(282.7f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(286.7f, air[0][0](0, 3));
    EXPECT_FLOAT_EQ(288.5f, air[0][0](0, 4));
    EXPECT_FLOAT_EQ(276.2f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(271.3f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(277.7f, air[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    293.4	292.9	291.9	289.2	288.8
    293.6	292.9	291.8	291.0	290.8
    291.6	291.5	290.8	290.1	290.1
    */
    EXPECT_FLOAT_EQ(293.4f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(292.9f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(291.9f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(289.2f, air[1][0](0, 3));
    EXPECT_FLOAT_EQ(288.8f, air[1][0](0, 4));
    EXPECT_FLOAT_EQ(293.6f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(291.6f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(290.1f, air[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    282.1	282.0	283.2	289.4	290.3
    281.5	277.1	275.9	279.6	282.1
    274.5	272.8	275.5	277.8	278.7
    */
    EXPECT_FLOAT_EQ(282.1f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(282.0f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(283.2f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(289.4f, air[11][0](0, 3));
    EXPECT_FLOAT_EQ(290.3f, air[11][0](0, 4));
    EXPECT_FLOAT_EQ(281.5f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(274.5f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(278.7f, air[11][0](2, 4));

    /* Values time step 20 (horizontal=Lon, vertical=Lat)
    269.0	273.2	280.2	285.6	288.0
    270.6	268.1	271.2	278.9	282.4
    272.7	268.3	267.1	271.3	276.8
    */
    EXPECT_FLOAT_EQ(269.0f, air[20][0](0, 0));
    EXPECT_FLOAT_EQ(273.2f, air[20][0](0, 1));
    EXPECT_FLOAT_EQ(280.2f, air[20][0](0, 2));
    EXPECT_FLOAT_EQ(285.6f, air[20][0](0, 3));
    EXPECT_FLOAT_EQ(288.0f, air[20][0](0, 4));
    EXPECT_FLOAT_EQ(270.6f, air[20][0](1, 0));
    EXPECT_FLOAT_EQ(272.7f, air[20][0](2, 0));
    EXPECT_FLOAT_EQ(276.8f, air[20][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR1, GaussianLoadBorderLeft) {
    double xMin = 0;
    int xPtsNb = 3;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    asAreaGrid* area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, 0));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   272.5	271.7	274.8
    |   271.8	271.0	274.3
    |   278.1	275.7	271.5
    */
    EXPECT_FLOAT_EQ(272.5f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(271.7f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(274.8f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(271.8f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(278.1f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.5f, air[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.8	291.1	292.0
    |   292.4	293.4	293.6
    |   293.2	292.2	291.6
    */
    EXPECT_FLOAT_EQ(289.8f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(291.1f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(292.0f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(292.4f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(293.2f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(291.6f, air[1][0](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   277.9	277.9 	278.9
    |   279.7	279.1	280.5
    |   282.0	280.3	278.6
    */
    EXPECT_FLOAT_EQ(277.9f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(277.9f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(278.9f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(279.7f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(282.0f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(278.6f, air[11][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR1, GaussianLoadBorderLeftOn720) {
    double xMin = 360;
    int xPtsNb = 3;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    asAreaGrid* area = asAreaGrid::GetInstance(xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, 0));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   272.5	271.7	274.8
    |   271.8	271.0	274.3
    |   278.1	275.7	271.5
    */
    EXPECT_FLOAT_EQ(272.5f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(271.7f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(274.8f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(271.8f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(278.1f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.5f, air[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.8	291.1	292.0
    |   292.4	293.4	293.6
    |   293.2	292.2	291.6
    */
    EXPECT_FLOAT_EQ(289.8f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(291.1f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(292.0f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(292.4f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(293.2f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(291.6f, air[1][0](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   277.9	277.9 	278.9
    |   279.7	279.1	280.5
    |   282.0	280.3	278.6
    */
    EXPECT_FLOAT_EQ(277.9f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(277.9f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(278.9f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(279.7f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(282.0f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(278.6f, air[11][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorNcepR1, GaussianSetData) {
    double xMin = -7.5;
    int xPtsNb = 4;
    double yMin = 29.523;
    int yPtsNb = 2;
    double steplon = 0;
    double steplat = 0;
    asAreaGrid* area = asAreaGrid::GetInstance(xMin, xPtsNb, steplon, yMin, yPtsNb, steplat);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 5, 00, 00);
    double timeStep = 24;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor* predictor = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray, 0));

    vva2f newdata(5, va2f(1, a2f::Zero(1, 4)));

    newdata[0][0] << 1, 2, 3, 4;
    newdata[1][0] << 11, 12, 13, 14;
    newdata[2][0] << 21, 22, 23, 24;
    newdata[3][0] << 31, 32, 33, 34;
    newdata[4][0] << 41, 42, 43, 44;

    predictor->SetData(newdata);

    EXPECT_FLOAT_EQ(1, predictor->GetLatPtsnb());
    EXPECT_FLOAT_EQ(4, predictor->GetLonPtsnb());
    EXPECT_FLOAT_EQ(1, predictor->GetData()[0][0](0, 0));
    EXPECT_FLOAT_EQ(2, predictor->GetData()[0][0](0, 1));
    EXPECT_FLOAT_EQ(4, predictor->GetData()[0][0](0, 3));
    EXPECT_FLOAT_EQ(14, predictor->GetData()[1][0](0, 3));
    EXPECT_FLOAT_EQ(41, predictor->GetData()[4][0](0, 0));
    EXPECT_FLOAT_EQ(44, predictor->GetData()[4][0](0, 3));

    wxDELETE(area);
    wxDELETE(predictor);
}

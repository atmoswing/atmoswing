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
#include <wx/filename.h>

#include "asAreaGrid.h"
#include "asPredictorOper.h"
#include "asTimeArray.h"

TEST(PredictorOperCustomMeteoFvg, GetCorrectPredictors) {
    asPredictorOper *predictor;

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "DP500925");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Other);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "LRT700500");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Other);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "LRT850500");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Other);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "LRTE700500");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Other);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "LRTE850500");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Other);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB500850");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MaximumBuoyancy);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB500925");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MaximumBuoyancy);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB700925");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MaximumBuoyancy);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB850500");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MaximumBuoyancy);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "thetaES");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialTemperature);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "thetaE");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialTemperature);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "vflux");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MomentumFlux);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "uflux");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MomentumFlux);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "2t_sfc");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "10u_sfc");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "10v_sfc");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "cp_sfc");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "msl_sfc");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "tp_sfc");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "q");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "gh");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "t");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "w");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "r");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "u");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "v");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomMeteoFvg, LoadSingleDay) {
    vwxs filePaths;
    filePaths.push_back(wxFileName::GetCwd() + "/files/data-custom-meteo-fvg-operational/MB500925.2020062106.grib");

    asTimeArray dates(asTime::GetMJD(2020, 6, 21, 06), asTime::GetMJD(2020, 6, 21, 06), 6, "Simple");
    dates.Init();

    double xMin = 6;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 5;
    double step = 0.125;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB500925");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filePaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f data = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time 06h (horizontal=Lon, vertical=Lat)
    Extracted:
    10.919	9.278	7.830	7.512	7.608	6.612
    9.954	9.131	8.042	7.555	7.725	7.921
    8.371	8.455	8.475	8.526	9.071	9.790
    8.760	8.659	8.982	8.683	9.494	10.531
    9.836	10.344	10.228	9.200	9.229	9.848
    */
    EXPECT_NEAR(10.919, data[0][0](0, 0), 0.002);
    EXPECT_NEAR(9.278, data[0][0](0, 1), 0.002);
    EXPECT_NEAR(7.830, data[0][0](0, 2), 0.002);
    EXPECT_NEAR(7.512, data[0][0](0, 3), 0.002);
    EXPECT_NEAR(7.608, data[0][0](0, 4), 0.002);
    EXPECT_NEAR(6.612, data[0][0](0, 5), 0.002);
    EXPECT_NEAR(9.954, data[0][0](1, 0), 0.002);
    EXPECT_NEAR(8.371, data[0][0](2, 0), 0.002);
    EXPECT_NEAR(9.836, data[0][0](4, 0), 0.002);
    EXPECT_NEAR(9.848, data[0][0](4, 5), 0.002);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomMeteoFvg, LoadLastTimeStep) {
    vwxs filePaths;
    filePaths.push_back(wxFileName::GetCwd() + "/files/data-custom-meteo-fvg-operational/MB500925.2020062190.grib");

    asTimeArray dates(asTime::GetMJD(2020, 6, 24, 18), asTime::GetMJD(2020, 6, 24, 18), 6, "Simple");
    dates.Init();

    double xMin = 6;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 5;
    double step = 0.125;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB500925");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filePaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f data = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time 90h (horizontal=Lon, vertical=Lat)
    Extracted:
    -7.667	-9.976	-13.864	-15.870	-14.057	-10.746
    -12.172	-13.010	-14.806	-14.507	-10.971	-8.071
    -14.181	-13.320	-12.670	-12.046	-9.670	-7.123
    -15.720	-12.759	-9.788	-8.628	-8.952	-8.637
    -14.759	-11.781	-7.036	-5.073	-7.305	-9.857
    */
    EXPECT_NEAR(-7.667, data[0][0](0, 0), 0.002);
    EXPECT_NEAR(-9.976, data[0][0](0, 1), 0.002);
    EXPECT_NEAR(-13.864, data[0][0](0, 2), 0.002);
    EXPECT_NEAR(-15.870, data[0][0](0, 3), 0.002);
    EXPECT_NEAR(-14.057, data[0][0](0, 4), 0.002);
    EXPECT_NEAR(-10.746, data[0][0](0, 5), 0.002);
    EXPECT_NEAR(-12.172, data[0][0](1, 0), 0.002);
    EXPECT_NEAR(-14.181, data[0][0](2, 0), 0.002);
    EXPECT_NEAR(-14.759, data[0][0](4, 0), 0.002);
    EXPECT_NEAR(-9.857, data[0][0](4, 5), 0.002);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomMeteoFvg, LoadFullTimeArray) {
    vwxs filePaths;
    filePaths.push_back(wxFileName::GetCwd() + "/files/data-custom-meteo-fvg-operational/MB500925.2020062106.grib");
    filePaths.push_back(wxFileName::GetCwd() + "/files/data-custom-meteo-fvg-operational/MB500925.2020062190.grib");

    asTimeArray dates(asTime::GetMJD(2020, 6, 21, 06), asTime::GetMJD(2020, 6, 24, 18), 6, "Simple");
    dates.Init();

    double xMin = 6;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 5;
    double step = 0.125;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB500925");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filePaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f data = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    10.919	9.278	7.830	7.512	7.608	6.612
    9.954	9.131	8.042	7.555	7.725	7.921
    8.371	8.455	8.475	8.526	9.071	9.790
    8.760	8.659	8.982	8.683	9.494	10.531
    9.836	10.344	10.228	9.200	9.229	9.848
    */
    EXPECT_NEAR(10.919, data[0][0](0, 0), 0.002);
    EXPECT_NEAR(9.278, data[0][0](0, 1), 0.002);
    EXPECT_NEAR(7.830, data[0][0](0, 2), 0.002);
    EXPECT_NEAR(7.512, data[0][0](0, 3), 0.002);
    EXPECT_NEAR(7.608, data[0][0](0, 4), 0.002);
    EXPECT_NEAR(6.612, data[0][0](0, 5), 0.002);
    EXPECT_NEAR(9.954, data[0][0](1, 0), 0.002);
    EXPECT_NEAR(8.371, data[0][0](2, 0), 0.002);
    EXPECT_NEAR(9.836, data[0][0](4, 0), 0.002);
    EXPECT_NEAR(9.848, data[0][0](4, 5), 0.002);

    /* Values time 90h (horizontal=Lon, vertical=Lat)
    Extracted:
    -7.667	-9.976	-13.864	-15.870	-14.057	-10.746
    -12.172	-13.010	-14.806	-14.507	-10.971	-8.071
    -14.181	-13.320	-12.670	-12.046	-9.670	-7.123
    -15.720	-12.759	-9.788	-8.628	-8.952	-8.637
    -14.759	-11.781	-7.036	-5.073	-7.305	-9.857
    */
    EXPECT_NEAR(-7.667, data[14][0](0, 0), 0.002);
    EXPECT_NEAR(-9.976, data[14][0](0, 1), 0.002);
    EXPECT_NEAR(-13.864, data[14][0](0, 2), 0.002);
    EXPECT_NEAR(-15.870, data[14][0](0, 3), 0.002);
    EXPECT_NEAR(-14.057, data[14][0](0, 4), 0.002);
    EXPECT_NEAR(-10.746, data[14][0](0, 5), 0.002);
    EXPECT_NEAR(-12.172, data[14][0](1, 0), 0.002);
    EXPECT_NEAR(-14.181, data[14][0](2, 0), 0.002);
    EXPECT_NEAR(-14.759, data[14][0](4, 0), 0.002);
    EXPECT_NEAR(-9.857, data[14][0](4, 5), 0.002);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomMeteoFvg, StandardizeWithProvidedMeanAndSd) {
    vwxs filePaths;
    filePaths.push_back(wxFileName::GetCwd() + "/files/data-custom-meteo-fvg-operational/MB500925.2020062106.grib");

    asTimeArray dates(asTime::GetMJD(2020, 6, 21, 06), asTime::GetMJD(2020, 6, 21, 06), 6, "Simple");
    dates.Init();

    double xMin = 6;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 5;
    double step = 0.125;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("Custom_MeteoFVG_Forecast", "MB500925");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filePaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    ASSERT_TRUE(predictor->StandardizeData(4.858158690266188, 7.742704912270005));

    vva2f data = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time 06h (horizontal=Lon, vertical=Lat)
    Extracted:
    0.7828	0.5708	0.3838	0.3428	0.3552	0.2265
    0.6581	0.5519	0.4112	0.3483	0.3703	0.3956
    0.4537	0.4645	0.4671	0.4737	0.5441	0.6370
    0.5039	0.4909	0.5326	0.4940	0.5987	0.7327
    0.6429	0.7085	0.6935	0.5608	0.5645	0.6445
    */
    EXPECT_NEAR(0.7828, data[0][0](0, 0), 0.002);
    EXPECT_NEAR(0.5708, data[0][0](0, 1), 0.002);
    EXPECT_NEAR(0.3838, data[0][0](0, 2), 0.002);
    EXPECT_NEAR(0.3428, data[0][0](0, 3), 0.002);
    EXPECT_NEAR(0.3552, data[0][0](0, 4), 0.002);
    EXPECT_NEAR(0.2265, data[0][0](0, 5), 0.002);
    EXPECT_NEAR(0.6581, data[0][0](1, 0), 0.002);
    EXPECT_NEAR(0.4537, data[0][0](2, 0), 0.002);
    EXPECT_NEAR(0.6429, data[0][0](4, 0), 0.002);
    EXPECT_NEAR(0.6445, data[0][0](4, 5), 0.002);

    wxDELETE(area);
    wxDELETE(predictor);
}
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
 * Portions Copyright 2018 Pascal Horton, University of Bern.
 */

#include <gtest/gtest.h>
#include <wx/filename.h>

#include "asAreaGenGrid.h"
#include "asPredictorProj.h"
#include "asTimeArray.h"

TEST(PredictorProjCmip5, GetCorrectPredictors) {
    asPredictorProj* predictor;

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "u", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "v", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "slp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "rh", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "sh", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "huss", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "pr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "prc", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "tas", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "tasmax", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "tasmin", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);
}

TEST(PredictorProjCmip5, LoadEasy) {
    double xMin = 3.375;
    double xWidth = 3;
    double yMin = 75.7;
    double yWidth = 2;
    asAreaGenGrid area(xMin, xWidth, yMin, yWidth, 0);

    double start = asTime::GetMJD(2006, 1, 1, 12, 00);
    double end = asTime::GetMJD(2006, 1, 1, 12, 00);
    double timeStepHours = 24;
    asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-cmip5/");

    asPredictorProj* predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "pr", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    ASSERT_TRUE(predictor->Load(area, timearray, 0));

    vva2f pr = predictor->GetData();
    // pr[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    3.7753E-07	7.6832E-07	7.8179E-07	2.9757E-07
    1.1153E-06	5.6990E-06	2.2500E-05	3.9894E-05
    1.7723E-05	2.5831E-05	3.3057E-05	3.7672E-05
    */

    EXPECT_NEAR(3.7753E-07, pr[0][0](0, 0), 1E-11);
    EXPECT_NEAR(7.6832E-07, pr[0][0](0, 1), 1E-11);
    EXPECT_NEAR(7.8179E-07, pr[0][0](0, 2), 1E-11);
    EXPECT_NEAR(2.9757E-07, pr[0][0](0, 3), 1E-11);
    EXPECT_NEAR(1.1153E-06, pr[0][0](1, 0), 1E-10);
    EXPECT_NEAR(1.7723E-05, pr[0][0](2, 0), 1E-9);
    EXPECT_NEAR(3.7672E-05, pr[0][0](2, 3), 1E-9);

    wxDELETE(predictor);
}

TEST(PredictorProjCmip5, LoadBorderLeft) {
    double xMin = 0;
    double xWidth = 2;
    double yMin = 75.5;
    double yWidth = 2;
    asAreaGenGrid area(xMin, xWidth, yMin, yWidth, 0);

    double start = asTime::GetMJD(2006, 1, 1, 12, 00);
    double end = asTime::GetMJD(2006, 1, 1, 12, 00);
    double timeStepHours = 24;
    asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-cmip5/");

    asPredictorProj* predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "pr", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    ASSERT_TRUE(predictor->Load(area, timearray, 0));

    vva2f pr = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   1.6006E-07	2.0217E-07	3.0069E-07
    |   4.4013E-07	5.9736E-07	5.9959E-07
    |   8.9397E-06	9.3129E-06	1.1299E-05
    */

    EXPECT_NEAR(1.6006E-07, pr[0][0](0, 0), 1E-11);
    EXPECT_NEAR(2.0217E-07, pr[0][0](0, 1), 1E-11);
    EXPECT_NEAR(3.0069E-07, pr[0][0](0, 2), 1E-11);
    EXPECT_NEAR(4.4013E-07, pr[0][0](1, 0), 1E-11);
    EXPECT_NEAR(8.9397E-06, pr[0][0](2, 0), 1E-10);
    EXPECT_NEAR(1.1299E-05, pr[0][0](2, 2), 1E-9);

    wxDELETE(predictor);
}

TEST(PredictorProjCmip5, LoadWithPressureLevels) {
    double xMin = 0;
    double xWidth = 2;
    double yMin = 1.683;
    double yWidth = 1;
    float level = 850;
    asAreaGenGrid area(xMin, xWidth, yMin, yWidth);

    double start = asTime::GetMJD(2097, 1, 1, 12, 00);
    double end = asTime::GetMJD(2097, 1, 3, 12, 00);
    double timeStepHours = 24;
    asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-cmip5/");

    asPredictorProj* predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "zg", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f z = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1535.979	1535.357	1534.495
    1535.463	1534.927	1534.169
    */

    EXPECT_NEAR(1535.979, z[0][0](0, 0), 1E-3);
    EXPECT_NEAR(1535.357, z[0][0](0, 1), 1E-3);
    EXPECT_NEAR(1534.495, z[0][0](0, 2), 1E-3);
    EXPECT_NEAR(1535.463, z[0][0](1, 0), 1E-3);
    EXPECT_NEAR(1534.169, z[0][0](1, 2), 1E-3);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    1527.969	1527.435	1527.026
    1527.722	1527.429	1527.128
    */

    EXPECT_NEAR(1527.969, z[2][0](0, 0), 1E-3);
    EXPECT_NEAR(1527.435, z[2][0](0, 1), 1E-3);
    EXPECT_NEAR(1527.026, z[2][0](0, 2), 1E-3);
    EXPECT_NEAR(1527.722, z[2][0](1, 0), 1E-3);
    EXPECT_NEAR(1527.128, z[2][0](1, 2), 1E-3);

    wxDELETE(predictor);
}

TEST(PredictorProjCmip5, LoadOver2Years) {
    double xMin = 0;
    double xWidth = 2;
    double yMin = 1.683;
    double yWidth = 1;
    float level = 850;
    asAreaGenGrid area(xMin, xWidth, yMin, yWidth);

    double start = asTime::GetMJD(2097, 12, 25, 12, 00);
    double end = asTime::GetMJD(2098, 1, 10, 12, 00);
    double timeStepHours = 24;
    asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-cmip5/");

    asPredictorProj* predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "zg", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f z = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat) 25.12.2097
    1522.466	1522.246	1521.932
    1521.509	1521.456	1521.242
    */

    EXPECT_NEAR(1522.466, z[0][0](0, 0), 1E-3);
    EXPECT_NEAR(1522.246, z[0][0](0, 1), 1E-3);
    EXPECT_NEAR(1521.932, z[0][0](0, 2), 1E-3);
    EXPECT_NEAR(1521.509, z[0][0](1, 0), 1E-3);
    EXPECT_NEAR(1521.242, z[0][0](1, 2), 1E-3);

    /* Values time step 6 (horizontal=Lon, vertical=Lat) 31.12.2097
    1545.887	1545.936	1546.071
    1545.491	1545.513	1545.718
    */

    EXPECT_NEAR(1545.887, z[6][0](0, 0), 1E-3);
    EXPECT_NEAR(1545.936, z[6][0](0, 1), 1E-3);
    EXPECT_NEAR(1546.071, z[6][0](0, 2), 1E-3);
    EXPECT_NEAR(1545.491, z[6][0](1, 0), 1E-3);
    EXPECT_NEAR(1545.718, z[6][0](1, 2), 1E-3);

    /* Values time step 7 (horizontal=Lon, vertical=Lat) 01.01.2098
    1551.731	1551.931	1551.975
    1551.601	1551.694	1551.742
    */

    EXPECT_NEAR(1551.731, z[7][0](0, 0), 1E-3);
    EXPECT_NEAR(1551.931, z[7][0](0, 1), 1E-3);
    EXPECT_NEAR(1551.975, z[7][0](0, 2), 1E-3);
    EXPECT_NEAR(1551.601, z[7][0](1, 0), 1E-3);
    EXPECT_NEAR(1551.742, z[7][0](1, 2), 1E-3);

    /* Values time step 16 (horizontal=Lon, vertical=Lat) 10.01.2098
    1530.864	1530.571	1529.905
    1530.402	1530.241	1529.734
    */

    EXPECT_NEAR(1530.864, z[16][0](0, 0), 1E-3);
    EXPECT_NEAR(1530.571, z[16][0](0, 1), 1E-3);
    EXPECT_NEAR(1529.905, z[16][0](0, 2), 1E-3);
    EXPECT_NEAR(1530.402, z[16][0](1, 0), 1E-3);
    EXPECT_NEAR(1529.734, z[16][0](1, 2), 1E-3);

    wxDELETE(predictor);
}

TEST(PredictorProjCmip5, LoadAnotherModel) {
    double xMin = 0.01;
    double xWidth = 2.8;
    double yMin = 0.701;
    double yWidth = 2;
    float level = 500;
    asAreaGenGrid area(xMin, xWidth, yMin, yWidth);

    double start = asTime::GetMJD(2096, 12, 25, 00, 00);
    double end = asTime::GetMJD(2097, 1, 10, 00, 00);
    double timeStepHours = 6;
    asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-cmip5/");

    asPredictorProj* predictor = asPredictorProj::GetInstance("CMIP5", "CNRM-CM5", "rcp85", "ua", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f ua = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat) 25.12.2096, 00:00
    -8.808	-8.929	-8.980
    -7.923	-8.043	-7.953
    */

    EXPECT_NEAR(-8.808, ua[0][0](0, 0), 1E-3);
    EXPECT_NEAR(-8.929, ua[0][0](0, 1), 1E-3);
    EXPECT_NEAR(-8.980, ua[0][0](0, 2), 1E-3);
    EXPECT_NEAR(-7.923, ua[0][0](1, 0), 1E-3);
    EXPECT_NEAR(-7.953, ua[0][0](1, 2), 1E-3);

    /* Values time step 28 (horizontal=Lon, vertical=Lat) 31.12.2096, 24:00
    -8.262	-8.479	-8.472
    -8.564	-9.909	-10.521
    */

    EXPECT_NEAR(-8.262, ua[28][0](0, 0), 1E-3);
    EXPECT_NEAR(-8.479, ua[28][0](0, 1), 1E-3);
    EXPECT_NEAR(-8.472, ua[28][0](0, 2), 1E-3);
    EXPECT_NEAR(-8.564, ua[28][0](1, 0), 1E-3);
    EXPECT_NEAR(-10.521, ua[28][0](1, 2), 1E-3);

    /* Values time step 29 (horizontal=Lon, vertical=Lat) 01.01.2097, 06:00
    -7.018	-7.084	-7.398
    -7.557	-8.015	-8.484
    */

    EXPECT_NEAR(-7.018, ua[29][0](0, 0), 1E-3);
    EXPECT_NEAR(-7.084, ua[29][0](0, 1), 1E-3);
    EXPECT_NEAR(-7.398, ua[29][0](0, 2), 1E-3);
    EXPECT_NEAR(-7.557, ua[29][0](1, 0), 1E-3);
    EXPECT_NEAR(-8.484, ua[29][0](1, 2), 1E-3);

    /* Values time step 47 (horizontal=Lon, vertical=Lat) 05.01.2097, 18:00
    -8.465	-7.094	-6.448
    -8.283	-8.084	-7.850
    */

    EXPECT_NEAR(-8.465, ua[47][0](0, 0), 1E-3);
    EXPECT_NEAR(-7.094, ua[47][0](0, 1), 1E-3);
    EXPECT_NEAR(-6.448, ua[47][0](0, 2), 1E-3);
    EXPECT_NEAR(-8.283, ua[47][0](1, 0), 1E-3);
    EXPECT_NEAR(-7.850, ua[47][0](1, 2), 1E-3);

    wxDELETE(predictor);
}
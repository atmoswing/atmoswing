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

#include "asAreaCompRegGrid.h"
#include "asPredictor.h"
#include "asTimeArray.h"

TEST(PredictorNoaa20Cr2c, GetCorrectPredictors) {
    asPredictor *predictor;

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/air", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/hgt", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/rh", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/sh", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/uwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/vwnd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pl/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "monolevel/pr_wtr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "monolevel/prmsl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "gauss/prate", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitationRate);
    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2c, LoadEasy) {
    double xMin = 10;
    double xWidth = 8;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    -104.9	-88.2	-71.4	-54.6	-37.8
    -131.4	-112.7	-92.3	-70.3	-47.4
    -145.9	-120.7	-91.6	-60.1	-28.0
    */
    EXPECT_FLOAT_EQ(-104.9f, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(-88.2f, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(-71.4f, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(-54.6f, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(-37.8f, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(-131.4f, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(-145.9f, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(-28.0f, hgt[0][0](2, 4));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    -174.2	-169.1	-163.2	-156.7	-149.6
    -124.2	-119.4	-114.1	-108.2	-101.7
    -48.3	-44.2	-39.8	-34.8	-29.0
    */
    EXPECT_FLOAT_EQ(-174.2f, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(-169.1f, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(-163.2f, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(-156.7f, hgt[4][0](0, 3));
    EXPECT_FLOAT_EQ(-149.6f, hgt[4][0](0, 4));
    EXPECT_FLOAT_EQ(-124.2f, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(-48.3f, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(-29.0f, hgt[4][0](2, 4));

    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2c, LoadComposite) {
    double xMin = -8;
    double xWidth = 12;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    -206.3	-206.5	-201.9	-193.4   |  -181.8	-168.1	-153.2
    -237.0	-235.2	-228.6	-218.4   |  -206.0	-192.4	-178.3
    -224.2	-227.3	-225.9	-221.2   |  -214.2	-205.6	-195.4
    */
    EXPECT_FLOAT_EQ(-206.3f, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(-206.5f, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(-201.9f, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(-193.4f, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(-181.8f, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(-168.1f, hgt[0][0](0, 5));
    EXPECT_FLOAT_EQ(-153.2f, hgt[0][0](0, 6));
    EXPECT_FLOAT_EQ(-237.0f, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(-224.2f, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(-195.4f, hgt[0][0](2, 6));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    -149.3	-158.6	-166.6	-173.0   |  -177.8	-180.6	-181.6
    -119.7	-124.7	-128.9	-132.2   |  -134.2	-134.7	-133.8
    -65.4	-65.1	-65.1	-64.8    |  -63.8	-62.0	-59.2
    */
    EXPECT_FLOAT_EQ(-149.3f, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(-158.6f, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(-166.6f, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(-173.0f, hgt[4][0](0, 3));
    EXPECT_FLOAT_EQ(-177.8f, hgt[4][0](0, 4));
    EXPECT_FLOAT_EQ(-180.6f, hgt[4][0](0, 5));
    EXPECT_FLOAT_EQ(-181.6f, hgt[4][0](0, 6));
    EXPECT_FLOAT_EQ(-119.7f, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(-65.4f, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(-59.2f, hgt[4][0](2, 6));

    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2c, LoadBorderLeft) {
    double xMin = 0;
    double xWidth = 4;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |  -181.8	-168.1	-153.2
    |  -206.0	-192.4	-178.3
    |  -214.2	-205.6	-195.4
    */
    EXPECT_FLOAT_EQ(-181.8f, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(-168.1f, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(-153.2f, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(-206.0f, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(-214.2f, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(-195.4f, hgt[0][0](2, 2));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    |  -177.8	-180.6	-181.6
    |  -134.2	-134.7	-133.8
    |  -63.8	-62.0	-59.2
    */
    EXPECT_FLOAT_EQ(-177.8f, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(-180.6f, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(-181.6f, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(-134.2f, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(-63.8f, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(-59.2f, hgt[4][0](2, 2));

    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2c, LoadBorderRight) {
    double xMin = 352;
    double xWidth = 8;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    -206.3	-206.5	-201.9	-193.4   |  -181.8
    -237.0	-235.2	-228.6	-218.4   |  -206.0
    -224.2	-227.3	-225.9	-221.2   |  -214.2
    */
    EXPECT_FLOAT_EQ(-206.3f, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(-206.5f, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(-201.9f, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(-193.4f, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(-181.8f, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(-237.0f, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(-224.2f, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(-214.2f, hgt[0][0](2, 4));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    -149.3	-158.6	-166.6	-173.0   |  -177.8
    -119.7	-124.7	-128.9	-132.2   |  -134.2
    -65.4	-65.1	-65.1	-64.8    |  -63.8
    */
    EXPECT_FLOAT_EQ(-149.3f, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(-158.6f, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(-166.6f, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(-173.0f, hgt[4][0](0, 3));
    EXPECT_FLOAT_EQ(-177.8f, hgt[4][0](0, 4));
    EXPECT_FLOAT_EQ(-119.7f, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(-65.4f, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(-63.8f, hgt[4][0](2, 4));

    wxDELETE(predictor);
}

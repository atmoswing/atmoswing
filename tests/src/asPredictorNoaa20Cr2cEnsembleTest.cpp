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

TEST(PredictorNoaa20Cr2cEnsemble, GetCorrectPredictors) {
    asPredictor *predictor;

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/prmsl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/pwat", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/omega500", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/rh850", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/rh9950", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/t850", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/t9950", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z200", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z500", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z1000", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "first_guess/prate", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitationRate);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "first_guess/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2cEnsemble, Load1stMember) {
    double xMin = 10;
    double xWidth = 8;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c-ensemble/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z1000", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor->Load(&area, timearray, 0));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    44.5000	50.1875	55.6875	60.8125	65.3750
    45.6875	52.8750	59.1875	64.6875	69.3750
    51.8750	57.6250	62.1875	66.1875	69.8125
    */
    EXPECT_FLOAT_EQ(44.5000, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(50.1875, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(55.6875, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(60.8125, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(65.3750, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(45.6875, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(51.8750, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(69.8125, hgt[0][0](2, 4));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    75.8125	77.3750	78.8750	80.3125	81.3750
    62.0000	62.3125	62.1875	61.8750	61.1875
    53.1250	51.8125	49.8750	47.3750	44.5000
    */
    EXPECT_FLOAT_EQ(75.8125, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(77.3750, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(78.8750, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(80.3125, hgt[4][0](0, 3));
    EXPECT_FLOAT_EQ(81.3750, hgt[4][0](0, 4));
    EXPECT_FLOAT_EQ(62.0000, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(53.1250, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(44.5000, hgt[4][0](2, 4));

    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2cEnsemble, Load3rdMember) {
    double xMin = 10;
    double xWidth = 8;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c-ensemble/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z1000", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectMember(3);

    ASSERT_TRUE(predictor->Load(&area, timearray, 0));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    40.3750	47.0000	53.6875	60.0000	65.6250
    38.3125	46.8125	54.3750	61.1250	66.6875
    46.8125	53.1875	58.1875	62.3125	65.8125
    */
    EXPECT_FLOAT_EQ(40.3750, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(47.0000, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(53.6875, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(60.0000, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(65.6250, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(38.3125, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(46.8125, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(65.8125, hgt[0][0](2, 4));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    69.8750	72.8750	76.1250	79.1875	81.8750
    63.6250	64.8750	65.8125	66.1875	66.1875
    59.5000	58.5000	56.6250	54.1250	51.1250
    */
    EXPECT_FLOAT_EQ(69.8750, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(72.8750, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(76.1250, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(79.1875, hgt[4][0](0, 3));
    EXPECT_FLOAT_EQ(81.8750, hgt[4][0](0, 4));
    EXPECT_FLOAT_EQ(63.6250, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(59.5000, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(51.1250, hgt[4][0](2, 4));

    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2cEnsemble, LoadComposite) {
    double xMin = -8;
    double xWidth = 12;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c-ensemble/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z1000", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor->Load(&area, timearray, 0));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    28.3125	 23.6250  21.3125  21.0000   |  22.3750	25.1250	29.0000
    -11.6250 -13.1875 -11.0000 -5.6250   |  1.8750	10.6875	20.0000
    -13.3750 -14.5000 -11.3750 -4.5000   |  4.8750	15.6875	26.5000
    */
    EXPECT_FLOAT_EQ(28.3125, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(23.6250, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(21.3125, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(21.0000, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(22.3750, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(25.1250, hgt[0][0](0, 5));
    EXPECT_FLOAT_EQ(29.0000, hgt[0][0](0, 6));
    EXPECT_FLOAT_EQ(-11.6250, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(-13.3750, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(26.5000, hgt[0][0](2, 6));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    87.6250	81.1875	76.6250	73.8125   |  72.3750  71.8750  72.1875
    39.5000	38.1250	40.0000	44.1250   |  48.8750  53.3750  57.1250
    4.5000	8.6250	16.6250	26.3750   |  36.0000  44.0000  49.5000
    */
    EXPECT_FLOAT_EQ(87.6250, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(81.1875, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(76.6250, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(73.8125, hgt[4][0](0, 3));
    EXPECT_FLOAT_EQ(72.3750, hgt[4][0](0, 4));
    EXPECT_FLOAT_EQ(71.8750, hgt[4][0](0, 5));
    EXPECT_FLOAT_EQ(72.1875, hgt[4][0](0, 6));
    EXPECT_FLOAT_EQ(39.5000, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(4.5000, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(49.5000, hgt[4][0](2, 6));

    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2cEnsemble, LoadBorderLeft) {
    double xMin = 0;
    double xWidth = 4;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c-ensemble/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z1000", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor->Load(&area, timearray, 0));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |  22.3750	25.1250	29.0000
    |  1.8750	10.6875	20.0000
    |  4.8750	15.6875	26.5000
    */
    EXPECT_FLOAT_EQ(22.3750, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(25.1250, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(29.0000, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(1.8750, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(4.8750, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(26.5000, hgt[0][0](2, 2));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    |  72.3750  71.8750  72.1875
    |  48.8750  53.3750  57.1250
    |  36.0000  44.0000  49.5000
    */
    EXPECT_FLOAT_EQ(72.3750, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(71.8750, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(72.1875, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(48.8750, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(36.0000, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(49.5000, hgt[4][0](2, 2));

    wxDELETE(predictor);
}

TEST(PredictorNoaa20Cr2cEnsemble, LoadBorderRight) {
    double xMin = 352;
    double xWidth = 8;
    double yMin = 70;
    double yWidth = 4;
    double step = 2;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 10, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c-ensemble/");

    asPredictor *predictor = asPredictor::GetInstance("NOAA_20CR_v2c_ens", "analysis/z1000", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor->Load(&area, timearray, 0));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    28.3125	 23.6250  21.3125  21.0000   |  22.3750
    -11.6250 -13.1875 -11.0000 -5.6250   |  1.8750
    -13.3750 -14.5000 -11.3750 -4.5000   |  4.8750
    */
    EXPECT_FLOAT_EQ(28.3125, hgt[0][0](0, 0));
    EXPECT_FLOAT_EQ(23.6250, hgt[0][0](0, 1));
    EXPECT_FLOAT_EQ(21.3125, hgt[0][0](0, 2));
    EXPECT_FLOAT_EQ(21.0000, hgt[0][0](0, 3));
    EXPECT_FLOAT_EQ(22.3750, hgt[0][0](0, 4));
    EXPECT_FLOAT_EQ(-11.6250, hgt[0][0](1, 0));
    EXPECT_FLOAT_EQ(-13.3750, hgt[0][0](2, 0));
    EXPECT_FLOAT_EQ(4.8750, hgt[0][0](2, 4));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    87.6250	81.1875	76.6250	73.8125   |  72.3750
    39.5000	38.1250	40.0000	44.1250   |  48.8750
    4.5000	8.6250	16.6250	26.3750   |  36.0000
    */
    EXPECT_FLOAT_EQ(87.6250, hgt[4][0](0, 0));
    EXPECT_FLOAT_EQ(81.1875, hgt[4][0](0, 1));
    EXPECT_FLOAT_EQ(76.6250, hgt[4][0](0, 2));
    EXPECT_FLOAT_EQ(73.8125, hgt[4][0](0, 3));
    EXPECT_FLOAT_EQ(72.3750, hgt[4][0](0, 4));
    EXPECT_FLOAT_EQ(39.5000, hgt[4][0](1, 0));
    EXPECT_FLOAT_EQ(4.5000, hgt[4][0](2, 0));
    EXPECT_FLOAT_EQ(36.0000, hgt[4][0](2, 4));

    wxDELETE(predictor);
}

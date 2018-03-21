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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#include <wx/filename.h>
#include "asPredictorArch.h"
#include "asAreaCompRegGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorArchEcmwfCera20CRegular, Load1stMember)
{
    double xMin = 3;
    double xWidth = 8;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-cera-20c/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_CERA_20C", "press/r", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->Load(&area, timearray));

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    98.0	96.6	95.1	93.6	92.5	91.9	91.9	92.4	93.3
    96.5	94.4	92.5	90.9	89.9	89.5	89.6	90.1	90.7
    97.4	97.0	97.1	97.5	98.0	98.5	98.7	98.7	98.3
    99.4	100.2	101.2	102.0	102.3	102.1	101.3	100.2	99.1
    99.1	99.4	99.6	99.7	99.4	98.8	98.2	97.8	97.8
    */
    EXPECT_NEAR(98.0, rh[0][0](0, 0), 0.1);
    EXPECT_NEAR(96.6, rh[0][0](0, 1), 0.1);
    EXPECT_NEAR(95.1, rh[0][0](0, 2), 0.1);
    EXPECT_NEAR(93.6, rh[0][0](0, 3), 0.1);
    EXPECT_NEAR(93.3, rh[0][0](0, 8), 0.1);
    EXPECT_NEAR(96.5, rh[0][0](1, 0), 0.1);
    EXPECT_NEAR(97.4, rh[0][0](2, 0), 0.1);
    EXPECT_NEAR(99.1, rh[0][0](4, 0), 0.1);
    EXPECT_NEAR(97.8, rh[0][0](4, 8), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    99.8	99.3	98.7	98.0	97.3	96.7	96.2	95.9	95.9
    96.5	95.5	94.6	94.0	93.5	93.0	92.7	92.4	92.2
    95.1	95.4	96.2	97.1	97.7	98.0	97.6	96.8	95.6
    98.7	99.5	100.5	101.3	101.6	101.4	100.4	99.0	97.4
    99.8	99.9	99.8	99.6	99.3	99.1	99.2	99.4	99.8
    */
    EXPECT_NEAR(99.8, rh[3][0](0, 0), 0.1);
    EXPECT_NEAR(99.3, rh[3][0](0, 1), 0.1);
    EXPECT_NEAR(98.7, rh[3][0](0, 2), 0.1);
    EXPECT_NEAR(98.0, rh[3][0](0, 3), 0.1);
    EXPECT_NEAR(95.9, rh[3][0](0, 8), 0.1);
    EXPECT_NEAR(96.5, rh[3][0](1, 0), 0.1);
    EXPECT_NEAR(95.1, rh[3][0](2, 0), 0.1);
    EXPECT_NEAR(99.8, rh[3][0](4, 0), 0.1);
    EXPECT_NEAR(99.8, rh[3][0](4, 8), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchEcmwfCera20CRegular, Load3rdMember)
{
    double xMin = 3;
    double xWidth = 8;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-cera-20c/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_CERA_20C", "press/r", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectMember(3);

    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->Load(&area, timearray));

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    97.9	96.5	95.1	94.0	93.3	93.3	93.8	94.8	96.0
    98.1	96.3	94.7	93.5	92.9	93.0	93.6	94.6	95.6
    98.7	98.4	98.4	98.6	99.0	99.3	99.5	99.4	99.2
    99.5	100.3	101.3	102.0	102.2	101.8	100.9	99.8	98.6
    97.7	97.7	97.9	98.1	98.2	98.2	98.2	98.4	98.9
    */
    EXPECT_NEAR(97.9, rh[0][0](0, 0), 0.1);
    EXPECT_NEAR(96.5, rh[0][0](0, 1), 0.1);
    EXPECT_NEAR(95.1, rh[0][0](0, 2), 0.1);
    EXPECT_NEAR(94.0, rh[0][0](0, 3), 0.1);
    EXPECT_NEAR(96.0, rh[0][0](0, 8), 0.1);
    EXPECT_NEAR(98.1, rh[0][0](1, 0), 0.1);
    EXPECT_NEAR(98.7, rh[0][0](2, 0), 0.1);
    EXPECT_NEAR(97.7, rh[0][0](4, 0), 0.1);
    EXPECT_NEAR(98.9, rh[0][0](4, 8), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    98.7	98.2	97.6	97.1	96.8	96.7	96.7	96.8	96.9
    97.8	97.0	96.3	95.8	95.5	95.2	94.9	94.7	94.4
    96.9	97.3	97.9	98.5	98.8	98.7	98.0	97.0	95.9
    99.3	100.2	101.1	101.6	101.6	101.0	100.0	98.7	97.6
    100.1	100.5	100.7	100.5	100.0	99.5	99.0	98.8	98.9
    */
    EXPECT_NEAR(98.7, rh[3][0](0, 0), 0.1);
    EXPECT_NEAR(98.2, rh[3][0](0, 1), 0.1);
    EXPECT_NEAR(97.6, rh[3][0](0, 2), 0.1);
    EXPECT_NEAR(97.1, rh[3][0](0, 3), 0.1);
    EXPECT_NEAR(96.9, rh[3][0](0, 8), 0.1);
    EXPECT_NEAR(97.8, rh[3][0](1, 0), 0.1);
    EXPECT_NEAR(96.9, rh[3][0](2, 0), 0.1);
    EXPECT_NEAR(100.1, rh[3][0](4, 0), 0.1);
    EXPECT_NEAR(98.9, rh[3][0](4, 8), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchEcmwfCera20CRegular, Load3Members)
{
    double xMin = 3;
    double xWidth = 8;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-cera-20c/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_CERA_20C", "press/r", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectMembers(3);

    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->Load(&area, timearray));

    ASSERT_EQ(4, predictor->GetData().size());
    ASSERT_EQ(3, predictor->GetData()[0].size());

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    // 1st member

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    98.0	96.6	95.1	93.6	92.5	91.9	91.9	92.4	93.3
    96.5	94.4	92.5	90.9	89.9	89.5	89.6	90.1	90.7
    97.4	97.0	97.1	97.5	98.0	98.5	98.7	98.7	98.3
    99.4	100.2	101.2	102.0	102.3	102.1	101.3	100.2	99.1
    99.1	99.4	99.6	99.7	99.4	98.8	98.2	97.8	97.8
    */
    EXPECT_NEAR(98.0, rh[0][0](0, 0), 0.1);
    EXPECT_NEAR(96.6, rh[0][0](0, 1), 0.1);
    EXPECT_NEAR(95.1, rh[0][0](0, 2), 0.1);
    EXPECT_NEAR(93.6, rh[0][0](0, 3), 0.1);
    EXPECT_NEAR(93.3, rh[0][0](0, 8), 0.1);
    EXPECT_NEAR(96.5, rh[0][0](1, 0), 0.1);
    EXPECT_NEAR(97.4, rh[0][0](2, 0), 0.1);
    EXPECT_NEAR(99.1, rh[0][0](4, 0), 0.1);
    EXPECT_NEAR(97.8, rh[0][0](4, 8), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    99.8	99.3	98.7	98.0	97.3	96.7	96.2	95.9	95.9
    96.5	95.5	94.6	94.0	93.5	93.0	92.7	92.4	92.2
    95.1	95.4	96.2	97.1	97.7	98.0	97.6	96.8	95.6
    98.7	99.5	100.5	101.3	101.6	101.4	100.4	99.0	97.4
    99.8	99.9	99.8	99.6	99.3	99.1	99.2	99.4	99.8
    */
    EXPECT_NEAR(99.8, rh[3][0](0, 0), 0.1);
    EXPECT_NEAR(99.3, rh[3][0](0, 1), 0.1);
    EXPECT_NEAR(98.7, rh[3][0](0, 2), 0.1);
    EXPECT_NEAR(98.0, rh[3][0](0, 3), 0.1);
    EXPECT_NEAR(95.9, rh[3][0](0, 8), 0.1);
    EXPECT_NEAR(96.5, rh[3][0](1, 0), 0.1);
    EXPECT_NEAR(95.1, rh[3][0](2, 0), 0.1);
    EXPECT_NEAR(99.8, rh[3][0](4, 0), 0.1);
    EXPECT_NEAR(99.8, rh[3][0](4, 8), 0.1);

    // 3rd member

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    97.9	96.5	95.1	94.0	93.3	93.3	93.8	94.8	96.0
    98.1	96.3	94.7	93.5	92.9	93.0	93.6	94.6	95.6
    98.7	98.4	98.4	98.6	99.0	99.3	99.5	99.4	99.2
    99.5	100.3	101.3	102.0	102.2	101.8	100.9	99.8	98.6
    97.7	97.7	97.9	98.1	98.2	98.2	98.2	98.4	98.9
    */
    EXPECT_NEAR(97.9, rh[0][2](0, 0), 0.1);
    EXPECT_NEAR(96.5, rh[0][2](0, 1), 0.1);
    EXPECT_NEAR(95.1, rh[0][2](0, 2), 0.1);
    EXPECT_NEAR(94.0, rh[0][2](0, 3), 0.1);
    EXPECT_NEAR(96.0, rh[0][2](0, 8), 0.1);
    EXPECT_NEAR(98.1, rh[0][2](1, 0), 0.1);
    EXPECT_NEAR(98.7, rh[0][2](2, 0), 0.1);
    EXPECT_NEAR(97.7, rh[0][2](4, 0), 0.1);
    EXPECT_NEAR(98.9, rh[0][2](4, 8), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    98.7	98.2	97.6	97.1	96.8	96.7	96.7	96.8	96.9
    97.8	97.0	96.3	95.8	95.5	95.2	94.9	94.7	94.4
    96.9	97.3	97.9	98.5	98.8	98.7	98.0	97.0	95.9
    99.3	100.2	101.1	101.6	101.6	101.0	100.0	98.7	97.6
    100.1	100.5	100.7	100.5	100.0	99.5	99.0	98.8	98.9
    */
    EXPECT_NEAR(98.7, rh[3][2](0, 0), 0.1);
    EXPECT_NEAR(98.2, rh[3][2](0, 1), 0.1);
    EXPECT_NEAR(97.6, rh[3][2](0, 2), 0.1);
    EXPECT_NEAR(97.1, rh[3][2](0, 3), 0.1);
    EXPECT_NEAR(96.9, rh[3][2](0, 8), 0.1);
    EXPECT_NEAR(97.8, rh[3][2](1, 0), 0.1);
    EXPECT_NEAR(96.9, rh[3][2](2, 0), 0.1);
    EXPECT_NEAR(100.1, rh[3][2](4, 0), 0.1);
    EXPECT_NEAR(98.9, rh[3][2](4, 8), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchEcmwfCera20CRegular, LoadComposite)
{
    double xMin = -4;
    double xWidth = 8;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-cera-20c/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_CERA_20C", "press/r", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor->Load(&area, timearray));

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    96.4	96.8	97.6	98.5	99.1	99.3	98.9	98.0	96.6
    97.8	98.9	99.9	100.5	100.5	99.8	98.4	96.5	94.4
    99.7	100.5	100.9	100.8	100.1	99.2	98.2	97.4	97.0
    100.6	101.0	100.9	100.4	99.6	99.1	99.0	99.4	100.2
    101.0	101.1	100.8	100.2	99.6	99.1	98.9	99.1	99.4
    */
    EXPECT_NEAR(96.4, rh[0][0](0, 0), 0.1);
    EXPECT_NEAR(96.8, rh[0][0](0, 1), 0.1);
    EXPECT_NEAR(97.6, rh[0][0](0, 2), 0.1);
    EXPECT_NEAR(98.5, rh[0][0](0, 3), 0.1);
    EXPECT_NEAR(99.1, rh[0][0](0, 4), 0.1);
    EXPECT_NEAR(96.6, rh[0][0](0, 8), 0.1);
    EXPECT_NEAR(97.8, rh[0][0](1, 0), 0.1);
    EXPECT_NEAR(99.7, rh[0][0](2, 0), 0.1);
    EXPECT_NEAR(101.0, rh[0][0](4, 0), 0.1);
    EXPECT_NEAR(99.4, rh[0][0](4, 8), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    99.0	99.2	99.5	99.9	100.2	100.3	100.2	99.8	99.3
    100.0	100.6	100.8	100.6	99.9	98.9	97.7	96.5	95.5
    98.7	99.3	99.1	98.3	97.2	96.1	95.3	95.1	95.4
    100.5	100.9	100.7	100.0	99.1	98.5	98.3	98.7	99.5
    99.8	100.1	99.9	99.6	99.4	99.4	99.6	99.8	99.9
    */
    EXPECT_NEAR(99.0, rh[3][0](0, 0), 0.1);
    EXPECT_NEAR(99.2, rh[3][0](0, 1), 0.1);
    EXPECT_NEAR(99.5, rh[3][0](0, 2), 0.1);
    EXPECT_NEAR(99.9, rh[3][0](0, 3), 0.1);
    EXPECT_NEAR(100.2, rh[3][0](0, 4), 0.1);
    EXPECT_NEAR(99.3, rh[3][0](0, 8), 0.1);
    EXPECT_NEAR(100.0, rh[3][0](1, 0), 0.1);
    EXPECT_NEAR(98.7, rh[3][0](2, 0), 0.1);
    EXPECT_NEAR(99.8, rh[3][0](4, 0), 0.1);
    EXPECT_NEAR(99.9, rh[3][0](4, 8), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchEcmwfCera20CRegular, LoadBorderLeft)
{
    double xMin = 0;
    double xWidth = 4;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-cera-20c/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_CERA_20C", "press/r", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor->Load(&area, timearray));

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    99.1	99.3	98.9	98.0	96.6
    100.5	99.8	98.4	96.5	94.4
    100.1	99.2	98.2	97.4	97.0
    99.6	99.1	99.0	99.4	100.2
    99.6	99.1	98.9	99.1	99.4
    */
    EXPECT_NEAR(99.1, rh[0][0](0, 0), 0.1);
    EXPECT_NEAR(99.3, rh[0][0](0, 1), 0.1);
    EXPECT_NEAR(98.9, rh[0][0](0, 2), 0.1);
    EXPECT_NEAR(98.0, rh[0][0](0, 3), 0.1);
    EXPECT_NEAR(96.6, rh[0][0](0, 4), 0.1);
    EXPECT_NEAR(100.5, rh[0][0](1, 0), 0.1);
    EXPECT_NEAR(100.1, rh[0][0](2, 0), 0.1);
    EXPECT_NEAR(99.6, rh[0][0](4, 0), 0.1);
    EXPECT_NEAR(99.4, rh[0][0](4, 4), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    100.2	100.3	100.2	99.8	99.3
    99.9	98.9	97.7	96.5	95.5
    97.2	96.1	95.3	95.1	95.4
    99.1	98.5	98.3	98.7	99.5
    99.4	99.4	99.6	99.8	99.9
    */
    EXPECT_NEAR(100.2, rh[3][0](0, 0), 0.1);
    EXPECT_NEAR(100.3, rh[3][0](0, 1), 0.1);
    EXPECT_NEAR(100.2, rh[3][0](0, 2), 0.1);
    EXPECT_NEAR(99.8, rh[3][0](0, 3), 0.1);
    EXPECT_NEAR(99.3, rh[3][0](0, 4), 0.1);
    EXPECT_NEAR(99.9, rh[3][0](1, 0), 0.1);
    EXPECT_NEAR(97.2, rh[3][0](2, 0), 0.1);
    EXPECT_NEAR(99.4, rh[3][0](4, 0), 0.1);
    EXPECT_NEAR(99.9, rh[3][0](4, 4), 0.1);

    wxDELETE(predictor);
}

TEST(PredictorArchEcmwfCera20CRegular, LoadBorderRight)
{
    double xMin = -4;
    double xWidth = 4;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-cera-20c/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_CERA_20C", "press/r", predictorDataDir);
    ASSERT_TRUE(predictor->IsEnsemble());
    predictor->SelectFirstMember();

    ASSERT_TRUE(predictor->Load(&area, timearray));

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    96.4	96.8	97.6	98.5	99.1
    97.8	98.9	99.9	100.5	100.5
    99.7	100.5	100.9	100.8	100.1
    100.6	101.0	100.9	100.4	99.6
    101.0	101.1	100.8	100.2	99.6
    */
    EXPECT_NEAR(96.4, rh[0][0](0, 0), 0.1);
    EXPECT_NEAR(96.8, rh[0][0](0, 1), 0.1);
    EXPECT_NEAR(97.6, rh[0][0](0, 2), 0.1);
    EXPECT_NEAR(98.5, rh[0][0](0, 3), 0.1);
    EXPECT_NEAR(99.1, rh[0][0](0, 4), 0.1);
    EXPECT_NEAR(97.8, rh[0][0](1, 0), 0.1);
    EXPECT_NEAR(99.7, rh[0][0](2, 0), 0.1);
    EXPECT_NEAR(101.0, rh[0][0](4, 0), 0.1);
    EXPECT_NEAR(99.6, rh[0][0](4, 4), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    99.0	99.2	99.5	99.9	100.2
    100.0	100.6	100.8	100.6	99.9
    98.7	99.3	99.1	98.3	97.2
    100.5	100.9	100.7	100.0	99.1
    99.8	100.1	99.9	99.6	99.4
    */
    EXPECT_NEAR(99.0, rh[3][0](0, 0), 0.1);
    EXPECT_NEAR(99.2, rh[3][0](0, 1), 0.1);
    EXPECT_NEAR(99.5, rh[3][0](0, 2), 0.1);
    EXPECT_NEAR(99.9, rh[3][0](0, 3), 0.1);
    EXPECT_NEAR(100.2, rh[3][0](0, 4), 0.1);
    EXPECT_NEAR(100.0, rh[3][0](1, 0), 0.1);
    EXPECT_NEAR(98.7, rh[3][0](2, 0), 0.1);
    EXPECT_NEAR(99.8, rh[3][0](4, 0), 0.1);
    EXPECT_NEAR(99.4, rh[3][0](4, 4), 0.1);

    wxDELETE(predictor);
}

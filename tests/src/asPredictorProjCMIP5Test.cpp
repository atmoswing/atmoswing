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

#include <wx/filename.h>
#include "asPredictorProj.h"
#include "asAreaCompGenGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorProjCMIP5, LoadEasy)
{
    double xMin = 3.375;
    double xWidth = 3;
    double yMin = 75.7;
    double yWidth = 2;
    asAreaCompGenGrid area(xMin, xWidth, yMin, yWidth, 0);

    double start = asTime::GetMJD(2006, 1, 1, 00, 00);
    double end = asTime::GetMJD(2006, 1, 1, 00, 00);
    double timeStepHours = 24;
    asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-cmip5/");

    asPredictorProj *predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "pr", predictorDataDir);

    ASSERT_TRUE(predictor != NULL);
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

TEST(PredictorProjCMIP5, LoadComposite)
{
    double xMin = -2;
    double xWidth = 4;
    double yMin = 75.5;
    double yWidth = 2;
    asAreaCompGenGrid area(xMin, xWidth, yMin, yWidth, 0);

    double start = asTime::GetMJD(2006, 1, 1, 00, 00);
    double end = asTime::GetMJD(2006, 1, 1, 00, 00);
    double timeStepHours = 24;
    asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-cmip5/");

    asPredictorProj *predictor = asPredictorProj::GetInstance("CMIP5", "MRI-CGCM3", "rcp85", "pr", predictorDataDir);

    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    ASSERT_TRUE(predictor->Load(area, timearray, 0));

    vva2f pr = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    9.3271E-07	5.4880E-07  |   1.6006E-07	2.0217E-07	3.0069E-07
    1.7317E-07	3.1338E-07  |   4.4013E-07	5.9736E-07	5.9959E-07
    6.2179E-06	7.7716E-06  |   8.9397E-06	9.3129E-06	1.1299E-05
    */

    EXPECT_NEAR(9.3271E-07, pr[0][0](0, 0), 1E-11);
    EXPECT_NEAR(5.4880E-07, pr[0][0](0, 1), 1E-11);
    EXPECT_NEAR(1.6006E-07, pr[0][0](0, 2), 1E-11);
    EXPECT_NEAR(2.0217E-07, pr[0][0](0, 3), 1E-11);
    EXPECT_NEAR(3.0069E-07, pr[0][0](0, 4), 1E-11);
    EXPECT_NEAR(1.7317E-07, pr[0][0](1, 0), 1E-11);
    EXPECT_NEAR(6.2179E-06, pr[0][0](2, 0), 1E-11);
    EXPECT_NEAR(1.1299E-05, pr[0][0](2, 4), 1E-11);

    wxDELETE(predictor);
}
/*
TEST(PredictorArchEcmwfEraIntRegular, LoadBorderLeft)
{
    /*
    double xMin = 0;
    double xWidth = 3;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_ERA_interim", "press/z",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray));
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   1074.3	1061.9	1050.4	1038.9	1029.3
    |   987.2	974.8	963.3	952.7	943.2
    |   868.5	858.0	849.4	840.7	834.0
    |   722.0	715.3	710.6	707.7	705.8
    |   567.0	566.0	567.0	570.8	574.6

    Transformed (geopotential height):
    |	109.55	108.28	107.11	105.94	104.96
    |	100.67	99.40	98.23	97.15	96.18
    |	88.56	87.49	86.61	85.73	85.04
    |	73.62	72.94	72.46	72.17	71.97
    |	57.82	57.72	57.82	58.21	58.59
    */
/*
    EXPECT_NEAR(109.55, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(108.28, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(107.11, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(104.96, hgt[0][0](0, 4), 0.01);
    EXPECT_NEAR(100.67, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(88.56, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(57.82, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(58.59, hgt[0][0](4, 4), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    |  1296.4	1279.1	1261.9	1244.7	1228.4
    |  1246.6	1230.3	1214.1	1198.7	1183.4
    |  1170.0	1157.6	1146.1	1133.7	1122.2
    |  1066.6	1059.0	1052.3	1045.6	1039.8
    |  944.1	942.2	942.2	941.3	939.3

    Transformed (geopotential height):
    |  132.20	130.43	128.68	126.92	125.26
    |  127.12	125.46	123.80	122.23	120.67
    |  119.31	118.04	116.87	115.61	114.43
    |  108.76	107.99	107.30	106.62	106.03
    |  96.27	96.08	96.08	95.99	95.78
    */
/*
    EXPECT_NEAR(132.20, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(130.43, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(128.68, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(125.26, hgt[3][0](0, 4), 0.01);
    EXPECT_NEAR(127.12, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(119.31, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(96.27, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(95.78, hgt[3][0](4, 4), 0.01);

    wxDELETE(predictor);
}

TEST(PredictorArchEcmwfEraIntRegular, LoadBorderLeftOn720)
{
    double xMin = 360;
    double xWidth = 3;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_ERA_interim", "press/z",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray));
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   1074.3	1061.9	1050.4	1038.9	1029.3
    |   987.2	974.8	963.3	952.7	943.2
    |   868.5	858.0	849.4	840.7	834.0
    |   722.0	715.3	710.6	707.7	705.8
    |   567.0	566.0	567.0	570.8	574.6

    Transformed (geopotential height):
    |	109.55	108.28	107.11	105.94	104.96
    |	100.67	99.40	98.23	97.15	96.18
    |	88.56	87.49	86.61	85.73	85.04
    |	73.62	72.94	72.46	72.17	71.97
    |	57.82	57.72	57.82	58.21	58.59
    */
/*
    EXPECT_NEAR(109.55, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(108.28, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(107.11, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(104.96, hgt[0][0](0, 4), 0.01);
    EXPECT_NEAR(100.67, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(88.56, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(57.82, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(58.59, hgt[0][0](4, 4), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    |  1296.4	1279.1	1261.9	1244.7	1228.4
    |  1246.6	1230.3	1214.1	1198.7	1183.4
    |  1170.0	1157.6	1146.1	1133.7	1122.2
    |  1066.6	1059.0	1052.3	1045.6	1039.8
    |  944.1	942.2	942.2	941.3	939.3

    Transformed (geopotential height):
    |  132.20	130.43	128.68	126.92	125.26
    |  127.12	125.46	123.80	122.23	120.67
    |  119.31	118.04	116.87	115.61	114.43
    |  108.76	107.99	107.30	106.62	106.03
    |  96.27	96.08	96.08	95.99	95.78
    */
/*
    EXPECT_NEAR(132.20, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(130.43, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(128.68, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(125.26, hgt[3][0](0, 4), 0.01);
    EXPECT_NEAR(127.12, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(119.31, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(96.27, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(95.78, hgt[3][0](4, 4), 0.01);

    wxDELETE(predictor);
}

TEST(PredictorArchEcmwfEraIntRegular, LoadBorderRight)
{
    double xMin = -3;
    double xWidth = 3;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("ECMWF_ERA_interim", "press/z",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray));
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1124.1	1112.6	1100.1	1087.7  |   1074.3
    1037.0	1024.5	1012.1	999.6   |   987.2
    916.4	903.0	890.5	879.0   |   868.5
    767.0	752.7	740.2	729.7   |   722.0
    593.8	583.3	574.6	569.9   |   567.0

    Transformed (geopotential height):
    114.63	113.45	112.18	110.91	|	109.55
    105.74	104.47	103.21	101.93	|	100.67
    93.45	92.08	90.81	89.63	|	88.56
    78.21	76.75	75.48	74.41	|	73.62
    60.55	59.48	58.59	58.11	|	57.82
    */
/*
    EXPECT_NEAR(114.63, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(113.45, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(112.18, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(110.91, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(109.55, hgt[0][0](0, 4), 0.01);
    EXPECT_NEAR(105.74, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(93.45, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(60.55, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(57.82, hgt[0][0](4, 4), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1359.6	1345.2	1328.9	1312.7   |  1296.4
    1307.9	1292.5	1277.2	1261.9   |  1246.6
    1222.7	1208.3	1195.9	1182.5   |  1170.0
    1104.0	1092.5	1082.9	1074.3   |  1066.6
    958.5	952.7	948.9	946.0    |  944.1

    Transformed (geopotential height):
    138.64	137.17	135.51	133.86	|	132.20
    133.37	131.80	130.24	128.68	|	127.12
    124.68	123.21	121.95	120.58	|	119.31
    112.58	111.40	110.43	109.55	|	108.76
    97.74	97.15	96.76	96.47	|	96.27
    */
/*
    EXPECT_NEAR(138.64, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(137.17, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(135.51, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(133.86, hgt[3][0](0, 3), 0.01);
    EXPECT_NEAR(132.20, hgt[3][0](0, 4), 0.01);
    EXPECT_NEAR(133.37, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(124.68, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(97.74, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(96.27, hgt[3][0](4, 4), 0.01);

    wxDELETE(predictor);
}
*/

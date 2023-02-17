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

#include "asAreaRegGrid.h"
#include "asPredictor.h"
#include "asTimeArray.h"

TEST(PredictorEcmwfEra20C, GetCorrectPredictors) {
    asPredictor* predictor;

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "pl/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "pl/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "pl/r", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "pl/w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "pl/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "surf/tcw", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::TotalColumnWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "surf/tp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "surf/msl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "surf/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);
}

TEST(PredictorEcmwfEra20C, LoadEasy) {
    double xMin = 3;
    double xWidth = 8;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-20c/");

    asPredictor* predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "pressure_level/z", predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1263.5	1255.9	1247.4	1240.8	1234.2	1228.5	1225.7	1224.7	1226.6
    1123.7	1117.0	1109.5	1102.9	1097.2	1092.5	1090.6	1089.7	1092.5
    951.7	946.1	941.3	937.6	933.8	930.0	928.1	927.2	928.1
    777.0	773.2	770.4	769.4	768.5	768.5	768.5	769.4	771.3
    624.0	624.9	627.7	631.5	636.2	641.9	646.6	652.3	658.0

    Transformed (geopotential height):
    128.84	128.07	127.20	126.53	125.85	125.27	124.99	124.88	125.08
    114.59	113.90	113.14	112.46	111.88	111.40	111.21	111.12	111.40
    97.05	96.48	95.99	95.61	95.22	94.83	94.64	94.55	94.64
    79.23	78.84	78.56	78.46	78.37	78.37	78.37	78.46	78.65
    63.63	63.72	64.01	64.40	64.87	65.46	65.93	66.52	67.10
    */
    EXPECT_NEAR(128.84, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(128.07, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(127.20, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(126.53, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(125.08, hgt[0][0](0, 8), 0.01);
    EXPECT_NEAR(114.59, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(97.05, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(63.63, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(67.10, hgt[0][0](4, 8), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1453.3	1429.7	1407.0	1384.4	1362.6	1341.9	1323.0	1306.9	1292.7
    1332.4	1312.6	1291.8	1271.0	1250.2	1230.4	1213.4	1198.3	1186.0
    1198.3	1185.1	1171.8	1157.7	1144.4	1130.3	1118.0	1105.7	1095.3
    1059.4	1050.9	1042.4	1033.9	1025.4	1017.9	1011.3	1003.7	996.1
    936.6	931.0	925.3	919.6	914.9	910.2	904.5	897.9	892.2

    Transformed (geopotential height):
    148.20	145.79	143.47	141.17	138.95	136.84	134.91	133.27	131.82
    135.87	133.85	131.73	129.61	127.48	125.47	123.73	122.19	120.94
    122.19	120.85	119.49	118.05	116.70	115.26	114.00	112.75	111.69
    108.03	107.16	106.30	105.43	104.56	103.80	103.12	102.35	101.57
    95.51	94.94	94.35	93.77	93.29	92.81	92.23	91.56	90.98
    */
    EXPECT_NEAR(148.20, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(145.79, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(143.47, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(141.17, hgt[3][0](0, 3), 0.01);
    EXPECT_NEAR(131.82, hgt[3][0](0, 8), 0.01);
    EXPECT_NEAR(135.87, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(122.19, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(95.51, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(90.98, hgt[3][0](4, 8), 0.01);

    wxDELETE(predictor);
}

TEST(PredictorEcmwfEra20C, LoadBorderLeft) {
    double xMin = 0;
    double xWidth = 4;
    double yMin = 75;
    double yWidth = 4;
    double step = 1;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-20c/");

    asPredictor* predictor = asPredictor::GetInstance("ECMWF_ERA_20C", "pressure_level/z", predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   1287.1	1279.5	1271.0	1263.5	1255.9
    |   1145.4	1137.8	1130.3	1123.7	1117.0
    |   974.4	965.9	958.4	951.7	946.1
    |   800.6	791.2	782.7	777.0	773.2
    |   638.1	630.6	625.9	624.0	624.9

    Transformed (geopotential height):
    |	131.25	130.47	129.61	128.84	128.07
    |	116.80	116.02	115.26	114.59	113.90
    |	99.36	98.49	97.73	97.05	96.48
    |	81.64	80.68	79.81	79.23	78.84
    |	65.07	64.30	63.82	63.63	63.72
    */
    EXPECT_NEAR(131.25, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(130.47, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(129.61, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(128.84, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(128.07, hgt[0][0](0, 4), 0.01);
    EXPECT_NEAR(116.80, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(99.36, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(65.07, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(63.72, hgt[0][0](4, 4), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    |   1520.4	1498.7	1476.0	1453.3	1429.7
    |   1393.8	1373.0	1353.2	1332.4	1312.6
    |   1238.0	1223.8	1210.6	1198.3	1185.1
    |   1080.2	1074.5	1067.9	1059.4	1050.9
    |   944.2	943.2	940.4	936.6	931.0

    Transformed (geopotential height):
    |	155.04	152.82	150.51	148.20	145.79
    |	142.13	140.01	137.99	135.87	133.85
    |	126.24	124.79	123.45	122.19	120.85
    |	110.15	109.57	108.90	108.03	107.16
    |	96.28	96.18	95.89	95.51	94.94
    */
    EXPECT_NEAR(155.04, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(152.82, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(150.51, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(148.20, hgt[3][0](0, 3), 0.01);
    EXPECT_NEAR(145.79, hgt[3][0](0, 4), 0.01);
    EXPECT_NEAR(142.13, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(126.24, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(96.28, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(94.94, hgt[3][0](4, 4), 0.01);

    wxDELETE(predictor);
}

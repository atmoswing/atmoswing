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

TEST(PredictorEcmwfEraInt, GetCorrectPredictors) {
    asPredictor *predictor;

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/d", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Divergence);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/pv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialVorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/q", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/r", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/t", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/u", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/v", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/vo", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/w", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pl/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/d", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Divergence);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/mont", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MontgomeryPotential);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/pres", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/pv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialVorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/q", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SpecificHumidity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/u", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/v", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/vo", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vorticity);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "isentropic/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/d2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::DewpointTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/msl", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/sd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SnowWaterEquivalent);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/sst", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::SeaSurfaceTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/t2m", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/tcw", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/tcwv", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::WaterVapour);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/u10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/v10", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/tp", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Precipitation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/cape", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::CAPE);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/ie", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::MoistureFlux);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/ssr", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/ssrd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/str", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "sfa/strd", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Radiation);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pv/pt", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PotentialTemperature);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pv/pres", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pv/u", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pv/v", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pv/z", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    wxDELETE(predictor);

    predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pv/x", ".");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::ParameterUndefined);
    wxDELETE(predictor);
}

TEST(PredictorEcmwfEraInt, LoadEasy) {
    double xMin = 3;
    double xWidth = 6;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictor *predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pressure_level/z", predictorDataDir);

    ASSERT_TRUE(predictor != nullptr);
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1029.3	1019.7	1010.2	1001.6	992.0	982.4	971.9	961.4	950.8
    943.2	934.6	927.9	920.2	913.5	906.8	898.2	890.5	881.9
    834.0	827.3	822.6	816.8	812.0	808.2	803.4	799.6	795.8
    705.8	704.8	704.8	704.8	704.8	704.8	704.8	704.8	704.8
    574.6	580.4	587.1	593.8	600.5	606.2	612.9	619.6	624.4

    Transformed (geopotential height):
    104.96	103.98	103.01	102.13	101.16	100.18	99.11	98.04	96.95
    96.18	95.30	94.62	93.83	93.15	92.47	91.59	90.81	89.93
    85.04	84.36	83.88	83.29	82.80	82.41	81.92	81.54	81.15
    71.97	71.87	71.87	71.87	71.87	71.87	71.87	71.87	71.87
    58.59	59.18	59.87	60.55	61.23	61.82	62.50	63.18	63.67
    */
    EXPECT_NEAR(104.96, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(103.98, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(103.01, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(102.13, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(96.95, hgt[0][0](0, 8), 0.01);
    EXPECT_NEAR(96.18, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(85.04, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(58.59, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(63.67, hgt[0][0](4, 8), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1228.4	1211.2	1193.0	1174.8	1156.6	1137.5	1117.4	1097.3	1077.2
    1183.4	1168.1	1153.8	1139.4	1124.1	1108.8	1092.5	1076.2	1059.9
    1122.2	1110.7	1099.2	1087.7	1076.2	1064.7	1052.3	1039.8	1028.4
    1039.8	1032.2	1025.5	1017.8	1010.2	1001.6	993.9	986.2	977.6
    939.3	937.4	933.6	929.8	925.0	920.2	916.4	910.6	905.8

    Transformed (geopotential height):
    125.26	123.51	121.65	119.80	117.94	115.99	113.94	111.89	109.84
    120.67	119.11	117.65	116.19	114.63	113.07	111.40	109.74	108.08
    114.43	113.26	112.09	110.91	109.74	108.57	107.30	106.03	104.87
    106.03	105.26	104.57	103.79	103.01	102.13	101.35	100.56	99.69
    95.78	95.59	95.20	94.81	94.32	93.83	93.45	92.86	92.37
    */
    EXPECT_NEAR(125.26, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(123.51, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(121.65, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(119.80, hgt[3][0](0, 3), 0.01);
    EXPECT_NEAR(109.84, hgt[3][0](0, 8), 0.01);
    EXPECT_NEAR(120.67, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(114.43, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(95.78, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(92.37, hgt[3][0](4, 8), 0.01);

    wxDELETE(predictor);
}

TEST(PredictorEcmwfEraInt, LoadWithNegativeVals) {
    double xMin = -3;
    double xWidth = 6;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictor *predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pressure_level/z", predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1124.1	1112.6	1100.1	1087.7  |   1074.3	1061.9	1050.4	1038.9	1029.3
    1037.0	1024.5	1012.1	999.6   |   987.2	974.8	963.3	952.7	943.2
    916.4	903.0	890.5	879.0   |   868.5	858.0	849.4	840.7	834.0
    767.0	752.7	740.2	729.7   |   722.0	715.3	710.6	707.7	705.8
    593.8	583.3	574.6	569.9   |   567.0	566.0	567.0	570.8	574.6

    Transformed (geopotential height):
    114.63	113.45	112.18	110.91	|	109.55	108.28	107.11	105.94	104.96
    105.74	104.47	103.21	101.93	|	100.67	99.40	98.23	97.15	96.18
    93.45	92.08	90.81	89.63	|	88.56	87.49	86.61	85.73	85.04
    78.21	76.75	75.48	74.41	|	73.62	72.94	72.46	72.17	71.97
    60.55	59.48	58.59	58.11	|	57.82	57.72	57.82	58.21	58.59

    */
    EXPECT_NEAR(114.63, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(113.45, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(112.18, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(110.91, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(109.55, hgt[0][0](0, 4), 0.01);
    EXPECT_NEAR(104.96, hgt[0][0](0, 8), 0.01);
    EXPECT_NEAR(105.74, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(93.45, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(60.55, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(58.59, hgt[0][0](4, 8), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1359.6	1345.2	1328.9	1312.7   |  1296.4	1279.1	1261.9	1244.7	1228.4
    1307.9	1292.5	1277.2	1261.9   |  1246.6	1230.3	1214.1	1198.7	1183.4
    1222.7	1208.3	1195.9	1182.5   |  1170.0	1157.6	1146.1	1133.7	1122.2
    1104.0	1092.5	1082.9	1074.3   |  1066.6	1059.0	1052.3	1045.6	1039.8
    958.5	952.7	948.9	946.0    |  944.1	942.2	942.2	941.3	939.3

    Transformed (geopotential height):
    138.64	137.17	135.51	133.86	|	132.20	130.43	128.68	126.92	125.26
    133.37	131.80	130.24	128.68	|	127.12	125.46	123.80	122.23	120.67
    124.68	123.21	121.95	120.58	|	119.31	118.04	116.87	115.61	114.43
    112.58	111.40	110.43	109.55	|	108.76	107.99	107.30	106.62	106.03
    97.74	97.15	96.76	96.47	|	96.27	96.08	96.08	95.99	95.78

    */
    EXPECT_NEAR(138.64, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(137.17, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(135.51, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(133.86, hgt[3][0](0, 3), 0.01);
    EXPECT_NEAR(132.20, hgt[3][0](0, 4), 0.01);
    EXPECT_NEAR(125.26, hgt[3][0](0, 8), 0.01);
    EXPECT_NEAR(133.37, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(124.68, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(97.74, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(95.78, hgt[3][0](4, 8), 0.01);

    wxDELETE(predictor);
}

TEST(PredictorEcmwfEraInt, LoadBorderLeft) {
    double xMin = 0;
    double xWidth = 3;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictor *predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pressure_level/z", predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));
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

TEST(PredictorEcmwfEraInt, LoadBorderLeftOn720) {
    double xMin = 360;
    double xWidth = 3;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictor *predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pressure_level/z", predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));
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

TEST(PredictorEcmwfEraInt, LoadBorderRight) {
    double xMin = -3;
    double xWidth = 3;
    double yMin = 75;
    double yWidth = 3;
    double step = 0.75;
    float level = 1000;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-interim/");

    asPredictor *predictor = asPredictor::GetInstance("ECMWF_ERA_interim", "pressure_level/z", predictorDataDir);

    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&area, timearray, level));
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

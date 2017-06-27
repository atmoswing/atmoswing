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

#include <wx/filename.h>
#include "asDataPredictorArchive.h"
#include "asGeoAreaCompositeRegularGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(DataPredictorArchiveEcmwfEra20CRegular, LoadEasy)
{
    double Xmin = 3;
    double Xwidth = 8;
    double Ymin = 75;
    double Ywidth = 4;
    double step = 1;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-20c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C_6h", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::Geopotential);
    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->Load(&geoarea, timearray));
    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::GeopotentialHeight);

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

TEST(DataPredictorArchiveEcmwfEra20CRegular, LoadComposite)
{
    double Xmin = -4;
    double Xwidth = 8;
    double Ymin = 75;
    double Ywidth = 4;
    double step = 1;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-20c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C_6h", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&geoarea, timearray));
    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1326.7	1315.4	1305.0	1295.6  |   1287.1	1279.5	1271.0	1263.5	1255.9
    1189.8	1176.6	1165.2	1154.8  |   1145.4	1137.8	1130.3	1123.7	1117.0
    1032.0	1014.1	999.0	985.7   |   974.4	965.9	958.4	951.7	946.1
    861.1	842.2	826.1	811.9   |   800.6	791.2	782.7	777.0	773.2
    691.0	673.1	658.9	647.6   |   638.1	630.6	625.9	624.0	624.9

    Transformed (geopotential height):
    135.29	134.13	133.07	132.11	|	131.25	130.47	129.61	128.84	128.07
    121.33	119.98	118.82	117.76	|	116.80	116.02	115.26	114.59	113.90
    105.23	103.41	101.87	100.51	|	99.36	98.49	97.73	97.05	96.48
    87.81	85.88	84.24	82.79	|	81.64	80.68	79.81	79.23	78.84
    70.46	68.64	67.19	66.04	|	65.07	64.30	63.82	63.63	63.72
    */
    EXPECT_NEAR(135.29, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(134.13, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(133.07, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(132.11, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(131.25, hgt[0][0](0, 4), 0.01);
    EXPECT_NEAR(128.07, hgt[0][0](0, 8), 0.01);
    EXPECT_NEAR(121.33, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(105.23, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(70.46, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(63.72, hgt[0][0](4, 8), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1607.3	1586.5	1564.8	1543.1  |   1520.4	1498.7	1476.0	1453.3	1429.7
    1478.8	1456.2	1435.4	1413.7  |   1393.8	1373.0	1353.2	1332.4	1312.6
    1306.0	1286.1	1268.2	1252.1  |   1238.0	1223.8	1210.6	1198.3	1185.1
    1109.5	1099.1	1092.5	1085.9  |   1080.2	1074.5	1067.9	1059.4	1050.9
    929.1	932.8	937.6	941.3   |   944.2	943.2	940.4	936.6	931.0

    Transformed (geopotential height):
    163.90	161.78	159.57	157.35	|	155.04	152.82	150.51	148.20	145.79
    150.80	148.49	146.37	144.16	|	142.13	140.01	137.99	135.87	133.85
    133.17	131.15	129.32	127.68	|	126.24	124.79	123.45	122.19	120.85
    113.14	112.08	111.40	110.73	|	110.15	109.57	108.90	108.03	107.16
    94.74	95.12	95.61	95.99	|	96.28	96.18	95.89	95.51	94.94
    */
    EXPECT_NEAR(163.90, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(161.78, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(159.57, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(157.35, hgt[3][0](0, 3), 0.01);
    EXPECT_NEAR(155.04, hgt[3][0](0, 4), 0.01);
    EXPECT_NEAR(145.79, hgt[3][0](0, 8), 0.01);
    EXPECT_NEAR(150.80, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(133.17, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(94.74, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(94.94, hgt[3][0](4, 8), 0.01);

    wxDELETE(predictor);
}

TEST(DataPredictorArchiveEcmwfEra20CRegular, LoadBorderLeft)
{
    double Xmin = 0;
    double Xwidth = 4;
    double Ymin = 75;
    double Ywidth = 4;
    double step = 1;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-20c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C_6h", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&geoarea, timearray));
    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::GeopotentialHeight);

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

TEST(DataPredictorArchiveEcmwfEra20CRegular, LoadBorderRight)
{
    double Xmin = -4;
    double Xwidth = 4;
    double Ymin = 75;
    double Ywidth = 4;
    double step = 1;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1987, 9, 9, 00, 00);
    double end = asTime::GetMJD(1987, 9, 9, 18, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ecmwf-era-20c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C_6h", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::Geopotential);
    ASSERT_TRUE(predictor->Load(&geoarea, timearray));
    ASSERT_TRUE(predictor->GetParameter() == asDataPredictor::GeopotentialHeight);

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1326.7	1315.4	1305.0	1295.6  |   1287.1
    1189.8	1176.6	1165.2	1154.8  |   1145.4
    1032.0	1014.1	999.0	985.7   |   974.4
    861.1	842.2	826.1	811.9   |   800.6
    691.0	673.1	658.9	647.6   |   638.1

    Transformed (geopotential height):
    135.29	134.13	133.07	132.11	|	131.25
    121.33	119.98	118.82	117.76	|	116.80
    105.23	103.41	101.87	100.51	|	99.36
    87.81	85.88	84.24	82.79	|	81.64
    70.46	68.64	67.19	66.04	|	65.07
    */
    EXPECT_NEAR(135.29, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(134.13, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(133.07, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(132.11, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(131.25, hgt[0][0](0, 4), 0.01);
    EXPECT_NEAR(121.33, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(105.23, hgt[0][0](2, 0), 0.01);
    EXPECT_NEAR(70.46, hgt[0][0](4, 0), 0.01);
    EXPECT_NEAR(65.07, hgt[0][0](4, 4), 0.01);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1607.3	1586.5	1564.8	1543.1  |   1520.4
    1478.8	1456.2	1435.4	1413.7  |   1393.8
    1306.0	1286.1	1268.2	1252.1  |   1238.0
    1109.5	1099.1	1092.5	1085.9  |   1080.2
    929.1	932.8	937.6	941.3   |   944.2

    Transformed (geopotential height):
    163.90	161.78	159.57	157.35	|	155.04
    150.80	148.49	146.37	144.16	|	142.13
    133.17	131.15	129.32	127.68	|	126.24
    113.14	112.08	111.40	110.73	|	110.15
    94.74	95.12	95.61	95.99	|	96.28
    */
    EXPECT_NEAR(163.90, hgt[3][0](0, 0), 0.01);
    EXPECT_NEAR(161.78, hgt[3][0](0, 1), 0.01);
    EXPECT_NEAR(159.57, hgt[3][0](0, 2), 0.01);
    EXPECT_NEAR(157.35, hgt[3][0](0, 3), 0.01);
    EXPECT_NEAR(155.04, hgt[3][0](0, 4), 0.01);
    EXPECT_NEAR(150.80, hgt[3][0](1, 0), 0.01);
    EXPECT_NEAR(133.17, hgt[3][0](2, 0), 0.01);
    EXPECT_NEAR(94.74, hgt[3][0](4, 0), 0.01);
    EXPECT_NEAR(96.28, hgt[3][0](4, 4), 0.01);

    wxDELETE(predictor);
}

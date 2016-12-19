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

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor != NULL);
    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1263.5	1255.9	1247.4	1240.8	1234.2	1228.5	1225.7	1224.7	1226.6
    1123.7	1117.0	1109.5	1102.9	1097.2	1092.5	1090.6	1089.7	1092.5
    951.7	946.1	941.3	937.6	933.8	930.0	928.1	927.2	928.1
    777.0	773.2	770.4	769.4	768.5	768.5	768.5	769.4	771.3
    624.0	624.9	627.7	631.5	636.2	641.9	646.6	652.3	658.0
    */
    EXPECT_NEAR(1263.5, hgt[0](0, 0), 0.1);
    EXPECT_NEAR(1255.9, hgt[0](0, 1), 0.1);
    EXPECT_NEAR(1247.4, hgt[0](0, 2), 0.1);
    EXPECT_NEAR(1240.8, hgt[0](0, 3), 0.1);
    EXPECT_NEAR(1226.6, hgt[0](0, 8), 0.1);
    EXPECT_NEAR(1123.7, hgt[0](1, 0), 0.1);
    EXPECT_NEAR(951.7, hgt[0](2, 0), 0.1);
    EXPECT_NEAR(624.0, hgt[0](4, 0), 0.1);
    EXPECT_NEAR(658.0, hgt[0](4, 8), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1453.3	1429.7	1407.0	1384.4	1362.6	1341.9	1323.0	1306.9	1292.7
    1332.4	1312.6	1291.8	1271.0	1250.2	1230.4	1213.4	1198.3	1186.0
    1198.3	1185.1	1171.8	1157.7	1144.4	1130.3	1118.0	1105.7	1095.3
    1059.4	1050.9	1042.4	1033.9	1025.4	1017.9	1011.3	1003.7	996.1
    936.6	931.0	925.3	919.6	914.9	910.2	904.5	897.9	892.2
    */
    EXPECT_NEAR(1453.3, hgt[3](0, 0), 0.1);
    EXPECT_NEAR(1429.7, hgt[3](0, 1), 0.1);
    EXPECT_NEAR(1407.0, hgt[3](0, 2), 0.1);
    EXPECT_NEAR(1384.4, hgt[3](0, 3), 0.1);
    EXPECT_NEAR(1292.7, hgt[3](0, 8), 0.1);
    EXPECT_NEAR(1332.4, hgt[3](1, 0), 0.1);
    EXPECT_NEAR(1198.3, hgt[3](2, 0), 0.1);
    EXPECT_NEAR(936.6, hgt[3](4, 0), 0.1);
    EXPECT_NEAR(892.2, hgt[3](4, 8), 0.1);

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

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1326.7	1315.4	1305.0	1295.6  |   1287.1	1279.5	1271.0	1263.5	1255.9
    1189.8	1176.6	1165.2	1154.8  |   1145.4	1137.8	1130.3	1123.7	1117.0
    1032.0	1014.1	999.0	985.7   |   974.4	965.9	958.4	951.7	946.1
    861.1	842.2	826.1	811.9   |   800.6	791.2	782.7	777.0	773.2
    691.0	673.1	658.9	647.6   |   638.1	630.6	625.9	624.0	624.9
    */
    EXPECT_NEAR(1326.7, hgt[0](0, 0), 0.1);
    EXPECT_NEAR(1315.4, hgt[0](0, 1), 0.1);
    EXPECT_NEAR(1305.0, hgt[0](0, 2), 0.1);
    EXPECT_NEAR(1295.6, hgt[0](0, 3), 0.1);
    EXPECT_NEAR(1287.1, hgt[0](0, 4), 0.1);
    EXPECT_NEAR(1255.9, hgt[0](0, 8), 0.1);
    EXPECT_NEAR(1189.8, hgt[0](1, 0), 0.1);
    EXPECT_NEAR(1032.0, hgt[0](2, 0), 0.1);
    EXPECT_NEAR(691.0, hgt[0](4, 0), 0.1);
    EXPECT_NEAR(624.9, hgt[0](4, 8), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1607.3	1586.5	1564.8	1543.1  |   1520.4	1498.7	1476.0	1453.3	1429.7
    1478.8	1456.2	1435.4	1413.7  |   1393.8	1373.0	1353.2	1332.4	1312.6
    1306.0	1286.1	1268.2	1252.1  |   1238.0	1223.8	1210.6	1198.3	1185.1
    1109.5	1099.1	1092.5	1085.9  |   1080.2	1074.5	1067.9	1059.4	1050.9
    929.1	932.8	937.6	941.3   |   944.2	943.2	940.4	936.6	931.0
    */
    EXPECT_NEAR(1607.3, hgt[3](0, 0), 0.1);
    EXPECT_NEAR(1586.5, hgt[3](0, 1), 0.1);
    EXPECT_NEAR(1564.8, hgt[3](0, 2), 0.1);
    EXPECT_NEAR(1543.1, hgt[3](0, 3), 0.1);
    EXPECT_NEAR(1520.4, hgt[3](0, 4), 0.1);
    EXPECT_NEAR(1429.7, hgt[3](0, 8), 0.1);
    EXPECT_NEAR(1478.8, hgt[3](1, 0), 0.1);
    EXPECT_NEAR(1306.0, hgt[3](2, 0), 0.1);
    EXPECT_NEAR(929.1, hgt[3](4, 0), 0.1);
    EXPECT_NEAR(931.0, hgt[3](4, 8), 0.1);

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

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   1287.1	1279.5	1271.0	1263.5	1255.9
    |   1145.4	1137.8	1130.3	1123.7	1117.0
    |   974.4	965.9	958.4	951.7	946.1
    |   800.6	791.2	782.7	777.0	773.2
    |   638.1	630.6	625.9	624.0	624.9
    */
    EXPECT_NEAR(1287.1, hgt[0](0, 0), 0.1);
    EXPECT_NEAR(1279.5, hgt[0](0, 1), 0.1);
    EXPECT_NEAR(1271.0, hgt[0](0, 2), 0.1);
    EXPECT_NEAR(1263.5, hgt[0](0, 3), 0.1);
    EXPECT_NEAR(1255.9, hgt[0](0, 4), 0.1);
    EXPECT_NEAR(1145.4, hgt[0](1, 0), 0.1);
    EXPECT_NEAR(974.4, hgt[0](2, 0), 0.1);
    EXPECT_NEAR(638.1, hgt[0](4, 0), 0.1);
    EXPECT_NEAR(624.9, hgt[0](4, 4), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    |   1520.4	1498.7	1476.0	1453.3	1429.7
    |   1393.8	1373.0	1353.2	1332.4	1312.6
    |   1238.0	1223.8	1210.6	1198.3	1185.1
    |   1080.2	1074.5	1067.9	1059.4	1050.9
    |   944.2	943.2	940.4	936.6	931.0
    */
    EXPECT_NEAR(1520.4, hgt[3](0, 0), 0.1);
    EXPECT_NEAR(1498.7, hgt[3](0, 1), 0.1);
    EXPECT_NEAR(1476.0, hgt[3](0, 2), 0.1);
    EXPECT_NEAR(1453.3, hgt[3](0, 3), 0.1);
    EXPECT_NEAR(1429.7, hgt[3](0, 4), 0.1);
    EXPECT_NEAR(1393.8, hgt[3](1, 0), 0.1);
    EXPECT_NEAR(1238.0, hgt[3](2, 0), 0.1);
    EXPECT_NEAR(944.2, hgt[3](4, 0), 0.1);
    EXPECT_NEAR(931.0, hgt[3](4, 4), 0.1);

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

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("ECMWF_ERA_20C", "press/z",
                                                                            predictorDataDir);
    predictor->SetTimeStepHours(3);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    1326.7	1315.4	1305.0	1295.6  |   1287.1
    1189.8	1176.6	1165.2	1154.8  |   1145.4
    1032.0	1014.1	999.0	985.7   |   974.4
    861.1	842.2	826.1	811.9   |   800.6
    691.0	673.1	658.9	647.6   |   638.1
    */
    EXPECT_NEAR(1326.7, hgt[0](0, 0), 0.1);
    EXPECT_NEAR(1315.4, hgt[0](0, 1), 0.1);
    EXPECT_NEAR(1305.0, hgt[0](0, 2), 0.1);
    EXPECT_NEAR(1295.6, hgt[0](0, 3), 0.1);
    EXPECT_NEAR(1287.1, hgt[0](0, 4), 0.1);
    EXPECT_NEAR(1189.8, hgt[0](1, 0), 0.1);
    EXPECT_NEAR(1032.0, hgt[0](2, 0), 0.1);
    EXPECT_NEAR(691.0, hgt[0](4, 0), 0.1);
    EXPECT_NEAR(638.1, hgt[0](4, 4), 0.1);

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    1607.3	1586.5	1564.8	1543.1  |   1520.4
    1478.8	1456.2	1435.4	1413.7  |   1393.8
    1306.0	1286.1	1268.2	1252.1  |   1238.0
    1109.5	1099.1	1092.5	1085.9  |   1080.2
    929.1	932.8	937.6	941.3   |   944.2
    */
    EXPECT_NEAR(1607.3, hgt[3](0, 0), 0.1);
    EXPECT_NEAR(1586.5, hgt[3](0, 1), 0.1);
    EXPECT_NEAR(1564.8, hgt[3](0, 2), 0.1);
    EXPECT_NEAR(1543.1, hgt[3](0, 3), 0.1);
    EXPECT_NEAR(1520.4, hgt[3](0, 4), 0.1);
    EXPECT_NEAR(1478.8, hgt[3](1, 0), 0.1);
    EXPECT_NEAR(1306.0, hgt[3](2, 0), 0.1);
    EXPECT_NEAR(929.1, hgt[3](4, 0), 0.1);
    EXPECT_NEAR(944.2, hgt[3](4, 4), 0.1);

    wxDELETE(predictor);
}

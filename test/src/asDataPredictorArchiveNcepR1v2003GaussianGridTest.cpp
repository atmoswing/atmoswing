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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include <wx/filename.h>
#include "asDataPredictorArchive.h"
#include "asGeoAreaCompositeGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(DataPredictorArchiveNcepR1v2003Gaussian, LoadEasy)
{
    double Xmin = 7.5;
    int Xptsnb = 5;
    double Ymin = 29.523;
    int Yptsnb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb,
                                                                          step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux/air2m",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_air_2m_gauss_%d.nc");
    ASSERT_TRUE(predictor->Load(geoarea, timearray));

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    276.9	279.6	282.7	286.7	288.5
    276.2	275.5	276.9	280.4	283.7
    271.3	270.8	273.9	275.6	277.7
    */
    EXPECT_FLOAT_EQ(276.9, air[0](0, 0));
    EXPECT_FLOAT_EQ(279.6, air[0](0, 1));
    EXPECT_FLOAT_EQ(282.7, air[0](0, 2));
    EXPECT_FLOAT_EQ(286.7, air[0](0, 3));
    EXPECT_FLOAT_EQ(288.5, air[0](0, 4));
    EXPECT_FLOAT_EQ(276.2, air[0](1, 0));
    EXPECT_FLOAT_EQ(271.3, air[0](2, 0));
    EXPECT_FLOAT_EQ(277.7, air[0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    293.4	292.9	291.9	289.2	288.8
    293.6	292.9	291.8	291.0	290.8
    291.6	291.5	290.8	290.1	290.1
    */
    EXPECT_FLOAT_EQ(293.4, air[1](0, 0));
    EXPECT_FLOAT_EQ(292.9, air[1](0, 1));
    EXPECT_FLOAT_EQ(291.9, air[1](0, 2));
    EXPECT_FLOAT_EQ(289.2, air[1](0, 3));
    EXPECT_FLOAT_EQ(288.8, air[1](0, 4));
    EXPECT_FLOAT_EQ(293.6, air[1](1, 0));
    EXPECT_FLOAT_EQ(291.6, air[1](2, 0));
    EXPECT_FLOAT_EQ(290.1, air[1](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    282.1	282.0	283.2	289.4	290.3
    281.5	277.1	275.9	279.6	282.1
    274.5	272.8	275.5	277.8	278.7
    */
    EXPECT_FLOAT_EQ(282.1, air[11](0, 0));
    EXPECT_FLOAT_EQ(282.0, air[11](0, 1));
    EXPECT_FLOAT_EQ(283.2, air[11](0, 2));
    EXPECT_FLOAT_EQ(289.4, air[11](0, 3));
    EXPECT_FLOAT_EQ(290.3, air[11](0, 4));
    EXPECT_FLOAT_EQ(281.5, air[11](1, 0));
    EXPECT_FLOAT_EQ(274.5, air[11](2, 0));
    EXPECT_FLOAT_EQ(278.7, air[11](2, 4));

    /* Values time step 20 (horizontal=Lon, vertical=Lat)
    269.0	273.2	280.2	285.6	288.0
    270.6	268.1	271.2	278.9	282.4
    272.7	268.3	267.1	271.3	276.8
    */
    EXPECT_FLOAT_EQ(269.0, air[20](0, 0));
    EXPECT_FLOAT_EQ(273.2, air[20](0, 1));
    EXPECT_FLOAT_EQ(280.2, air[20](0, 2));
    EXPECT_FLOAT_EQ(285.6, air[20](0, 3));
    EXPECT_FLOAT_EQ(288.0, air[20](0, 4));
    EXPECT_FLOAT_EQ(270.6, air[20](1, 0));
    EXPECT_FLOAT_EQ(272.7, air[20](2, 0));
    EXPECT_FLOAT_EQ(276.8, air[20](2, 4));

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNcepR1v2003Gaussian, LoadComposite)
{
    double Xmin = -7.5;
    int Xptsnb = 7;
    double Ymin = 29.523;
    int Yptsnb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb,
                                                                          step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux/air2m",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_air_2m_gauss_%d.nc");
    ASSERT_TRUE(predictor->Load(geoarea, timearray));

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    280.5	272.0	271.9	274.5   |   272.5	271.7	274.8
    276.9	271.4	273.0	273.5   |   271.8	271.0	274.3
    277.0	277.5	278.5	279.1   |   278.1	275.7	271.5
    */
    EXPECT_FLOAT_EQ(280.5, air[0](0, 0));
    EXPECT_FLOAT_EQ(272.0, air[0](0, 1));
    EXPECT_FLOAT_EQ(271.9, air[0](0, 2));
    EXPECT_FLOAT_EQ(274.5, air[0](0, 3));
    EXPECT_FLOAT_EQ(272.5, air[0](0, 4));
    EXPECT_FLOAT_EQ(271.7, air[0](0, 5));
    EXPECT_FLOAT_EQ(274.8, air[0](0, 6));
    EXPECT_FLOAT_EQ(276.9, air[0](1, 0));
    EXPECT_FLOAT_EQ(277.0, air[0](2, 0));
    EXPECT_FLOAT_EQ(271.5, air[0](2, 6));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    289.5	288.5	286.9	287.7   |   289.8	291.1	292.0
    291.6	290.7	290.3	291.2   |   292.4	293.4	293.6
    292.8	292.6	293.4 	293.8   |   293.2	292.2	291.6
    */
    EXPECT_FLOAT_EQ(289.5, air[1](0, 0));
    EXPECT_FLOAT_EQ(288.5, air[1](0, 1));
    EXPECT_FLOAT_EQ(286.9, air[1](0, 2));
    EXPECT_FLOAT_EQ(287.7, air[1](0, 3));
    EXPECT_FLOAT_EQ(289.8, air[1](0, 4));
    EXPECT_FLOAT_EQ(291.1, air[1](0, 5));
    EXPECT_FLOAT_EQ(292.0, air[1](0, 6));
    EXPECT_FLOAT_EQ(291.6, air[1](1, 0));
    EXPECT_FLOAT_EQ(292.8, air[1](2, 0));
    EXPECT_FLOAT_EQ(291.6, air[1](2, 6));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    284.1	279.6	279.5	279.3   |   277.9	277.9 	278.9
    277.4	275.0	277.6	280.1   |   279.7	279.1	280.5
    278.4	280.8	283.2 	284.4   |   282. 0	280.3	278.6
    */
    EXPECT_FLOAT_EQ(284.1, air[11](0, 0));
    EXPECT_FLOAT_EQ(279.6, air[11](0, 1));
    EXPECT_FLOAT_EQ(279.5, air[11](0, 2));
    EXPECT_FLOAT_EQ(279.3, air[11](0, 3));
    EXPECT_FLOAT_EQ(277.9, air[11](0, 4));
    EXPECT_FLOAT_EQ(277.9, air[11](0, 5));
    EXPECT_FLOAT_EQ(278.9, air[11](0, 6));
    EXPECT_FLOAT_EQ(277.4, air[11](1, 0));
    EXPECT_FLOAT_EQ(278.4, air[11](2, 0));
    EXPECT_FLOAT_EQ(278.6, air[11](2, 6));

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNcepR1v2003Gaussian, LoadBorderLeft)
{
    double Xmin = 0;
    int Xptsnb = 3;
    double Ymin = 29.523;
    int Yptsnb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb,
                                                                          step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux/air2m",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_air_2m_gauss_%d.nc");
    ASSERT_TRUE(predictor->Load(geoarea, timearray));

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   272.5	271.7	274.8
    |   271.8	271.0	274.3
    |   278.1	275.7	271.5
    */
    EXPECT_FLOAT_EQ(272.5, air[0](0, 0));
    EXPECT_FLOAT_EQ(271.7, air[0](0, 1));
    EXPECT_FLOAT_EQ(274.8, air[0](0, 2));
    EXPECT_FLOAT_EQ(271.8, air[0](1, 0));
    EXPECT_FLOAT_EQ(278.1, air[0](2, 0));
    EXPECT_FLOAT_EQ(271.5, air[0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.8	291.1	292.0
    |   292.4	293.4	293.6
    |   293.2	292.2	291.6
    */
    EXPECT_FLOAT_EQ(289.8, air[1](0, 0));
    EXPECT_FLOAT_EQ(291.1, air[1](0, 1));
    EXPECT_FLOAT_EQ(292.0, air[1](0, 2));
    EXPECT_FLOAT_EQ(292.4, air[1](1, 0));
    EXPECT_FLOAT_EQ(293.2, air[1](2, 0));
    EXPECT_FLOAT_EQ(291.6, air[1](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   277.9	277.9 	278.9
    |   279.7	279.1	280.5
    |   282.0	280.3	278.6
    */
    EXPECT_FLOAT_EQ(277.9, air[11](0, 0));
    EXPECT_FLOAT_EQ(277.9, air[11](0, 1));
    EXPECT_FLOAT_EQ(278.9, air[11](0, 2));
    EXPECT_FLOAT_EQ(279.7, air[11](1, 0));
    EXPECT_FLOAT_EQ(282.0, air[11](2, 0));
    EXPECT_FLOAT_EQ(278.6, air[11](2, 2));

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNcepR1v2003Gaussian, LoadBorderLeftOn720)
{
    double Xmin = 360;
    int Xptsnb = 3;
    double Ymin = 29.523;
    int Yptsnb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb,
                                                                          step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux/air2m",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_air_2m_gauss_%d.nc");
    ASSERT_TRUE(predictor->Load(geoarea, timearray));

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   272.5	271.7	274.8
    |   271.8	271.0	274.3
    |   278.1	275.7	271.5
    */
    EXPECT_FLOAT_EQ(272.5, air[0](0, 0));
    EXPECT_FLOAT_EQ(271.7, air[0](0, 1));
    EXPECT_FLOAT_EQ(274.8, air[0](0, 2));
    EXPECT_FLOAT_EQ(271.8, air[0](1, 0));
    EXPECT_FLOAT_EQ(278.1, air[0](2, 0));
    EXPECT_FLOAT_EQ(271.5, air[0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.8	291.1	292.0
    |   292.4	293.4	293.6
    |   293.2	292.2	291.6
    */
    EXPECT_FLOAT_EQ(289.8, air[1](0, 0));
    EXPECT_FLOAT_EQ(291.1, air[1](0, 1));
    EXPECT_FLOAT_EQ(292.0, air[1](0, 2));
    EXPECT_FLOAT_EQ(292.4, air[1](1, 0));
    EXPECT_FLOAT_EQ(293.2, air[1](2, 0));
    EXPECT_FLOAT_EQ(291.6, air[1](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   277.9	277.9 	278.9
    |   279.7	279.1	280.5
    |   282.0	280.3	278.6
    */
    EXPECT_FLOAT_EQ(277.9, air[11](0, 0));
    EXPECT_FLOAT_EQ(277.9, air[11](0, 1));
    EXPECT_FLOAT_EQ(278.9, air[11](0, 2));
    EXPECT_FLOAT_EQ(279.7, air[11](1, 0));
    EXPECT_FLOAT_EQ(282.0, air[11](2, 0));
    EXPECT_FLOAT_EQ(278.6, air[11](2, 2));

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNcepR1v2003Gaussian, LoadBorderRight)
{
    double Xmin = 352.5;
    int Xptsnb = 5;
    double Ymin = 29.523;
    int Yptsnb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb,
                                                                          step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 6, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux/air2m",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_air_2m_gauss_%d.nc");
    ASSERT_TRUE(predictor->Load(geoarea, timearray));

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    280.5	272.0	271.9	274.5   |   272.5
    276.9	271.4	273.0	273.5   |   271.8
    277.0	277.5	278.5	279.1   |   278.1
    */
    EXPECT_FLOAT_EQ(280.5, air[0](0, 0));
    EXPECT_FLOAT_EQ(272.0, air[0](0, 1));
    EXPECT_FLOAT_EQ(271.9, air[0](0, 2));
    EXPECT_FLOAT_EQ(274.5, air[0](0, 3));
    EXPECT_FLOAT_EQ(272.5, air[0](0, 4));
    EXPECT_FLOAT_EQ(276.9, air[0](1, 0));
    EXPECT_FLOAT_EQ(277.0, air[0](2, 0));
    EXPECT_FLOAT_EQ(278.1, air[0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    289.5	288.5	286.9	287.7   |   289.8
    291.6	290.7	290.3	291.2   |   292.4
    292.8	292.6	293.4 	293.8   |   293.2
    */
    EXPECT_FLOAT_EQ(289.5, air[1](0, 0));
    EXPECT_FLOAT_EQ(288.5, air[1](0, 1));
    EXPECT_FLOAT_EQ(286.9, air[1](0, 2));
    EXPECT_FLOAT_EQ(287.7, air[1](0, 3));
    EXPECT_FLOAT_EQ(289.8, air[1](0, 4));
    EXPECT_FLOAT_EQ(291.6, air[1](1, 0));
    EXPECT_FLOAT_EQ(292.8, air[1](2, 0));
    EXPECT_FLOAT_EQ(293.2, air[1](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    284.1	279.6	279.5	279.3   |   277.9
    277.4	275.0	277.6	280.1   |   279.7
    278.4	280.8	283.2 	284.4   |   282.0
    */
    EXPECT_FLOAT_EQ(284.1, air[11](0, 0));
    EXPECT_FLOAT_EQ(279.6, air[11](0, 1));
    EXPECT_FLOAT_EQ(279.5, air[11](0, 2));
    EXPECT_FLOAT_EQ(279.3, air[11](0, 3));
    EXPECT_FLOAT_EQ(277.9, air[11](0, 4));
    EXPECT_FLOAT_EQ(277.4, air[11](1, 0));
    EXPECT_FLOAT_EQ(278.4, air[11](2, 0));
    EXPECT_FLOAT_EQ(282.0, air[11](2, 4));

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNcepR1v2003Gaussian, SetData)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 29.523;
    int Yptsnb = 2;
    double steplon = 0;
    double steplat = 0;
    double level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, steplon, Ymin, Yptsnb,
                                                                          steplat, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 5, 00, 00);
    double timestephours = 24;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux/air2m",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_air_2m_gauss_%d.nc");
    ASSERT_TRUE(predictor->Load(geoarea, timearray));

    Array2DFloat tmp;
    tmp.resize(1, 4);
    VArray2DFloat newdata;

    tmp << 1, 2, 3, 4;
    newdata.push_back(tmp);
    tmp << 11, 12, 13, 14;
    newdata.push_back(tmp);
    tmp << 21, 22, 23, 24;
    newdata.push_back(tmp);
    tmp << 31, 32, 33, 34;
    newdata.push_back(tmp);
    tmp << 41, 42, 43, 44;
    newdata.push_back(tmp);

    predictor->SetData(newdata);

    EXPECT_FLOAT_EQ(1, predictor->GetLatPtsnb());
    EXPECT_FLOAT_EQ(4, predictor->GetLonPtsnb());
    EXPECT_FLOAT_EQ(1, predictor->GetData()[0](0, 0));
    EXPECT_FLOAT_EQ(2, predictor->GetData()[0](0, 1));
    EXPECT_FLOAT_EQ(4, predictor->GetData()[0](0, 3));
    EXPECT_FLOAT_EQ(14, predictor->GetData()[1](0, 3));
    EXPECT_FLOAT_EQ(41, predictor->GetData()[4](0, 0));
    EXPECT_FLOAT_EQ(44, predictor->GetData()[4](0, 3));

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

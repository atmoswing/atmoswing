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
#include "asPredictorArch.h"
#include "asAreaCompGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorArchNcepR2Gaussian, LoadEasy)
{
    double xMin = 7.5;
    int xPtsNb = 5;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_Reanalysis_v2", "flux/air2m",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    286.44	286.15	287.11	288.86	290.08
    283.91	282.92	283.69	285.07	286.79
    278.96	279.84	280.42	281.24	282.51
    */
    EXPECT_FLOAT_EQ(286.44f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(286.15f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(287.11f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(288.86f, air[0][0](0, 3));
    EXPECT_FLOAT_EQ(290.08f, air[0][0](0, 4));
    EXPECT_FLOAT_EQ(283.91f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(278.96f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(282.51f, air[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    293.04	292.96	293.09	290.29	290.60
    289.10	288.84	290.33	292.02	293.44
    286.25	286.60	289.12	291.63	292.75
    */
    EXPECT_FLOAT_EQ(293.04f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(292.96f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(293.09f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(290.29f, air[1][0](0, 3));
    EXPECT_FLOAT_EQ(290.60f, air[1][0](0, 4));
    EXPECT_FLOAT_EQ(289.10f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(286.25f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(292.75f, air[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    279.43	277.99	279.23	284.24	285.07
    280.17	279.24	281.57	282.47	282.28
    278.08	281.46	283.77	284.54	284.39
    */
    EXPECT_FLOAT_EQ(279.43f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(277.99f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(279.23f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(284.24f, air[11][0](0, 3));
    EXPECT_FLOAT_EQ(285.07f, air[11][0](0, 4));
    EXPECT_FLOAT_EQ(280.17f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(278.08f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(284.39f, air[11][0](2, 4));

    /* Values time step 20 (horizontal=Lon, vertical=Lat)
    281.81	283.26	286.18	288.59	289.69
    281.82	282.76	284.36	284.05	283.94
    282.53	284.64	283.24	279.87	278.18
    */
    EXPECT_FLOAT_EQ(281.81f, air[20][0](0, 0));
    EXPECT_FLOAT_EQ(283.26f, air[20][0](0, 1));
    EXPECT_FLOAT_EQ(286.18f, air[20][0](0, 2));
    EXPECT_FLOAT_EQ(288.59f, air[20][0](0, 3));
    EXPECT_FLOAT_EQ(289.69f, air[20][0](0, 4));
    EXPECT_FLOAT_EQ(281.82f, air[20][0](1, 0));
    EXPECT_FLOAT_EQ(282.53f, air[20][0](2, 0));
    EXPECT_FLOAT_EQ(278.18f, air[20][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorArchNcepR2Gaussian, LoadComposite)
{
    double xMin = -7.5;
    int xPtsNb = 7;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_Reanalysis_v2", "flux/air2m",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    278.10	278.82	285.20	284.96   |   284.46	285.92	287.61
    277.15	272.98	274.92	279.08   |   282.23	283.46	284.65
    273.57	272.72	272.38	272.51   |   275.49	273.53	271.54
    */
    EXPECT_FLOAT_EQ(278.10f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(278.82f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(285.20f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(284.96f, air[0][0](0, 3));
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 4));
    EXPECT_FLOAT_EQ(285.92f, air[0][0](0, 5));
    EXPECT_FLOAT_EQ(287.61f, air[0][0](0, 6));
    EXPECT_FLOAT_EQ(277.15f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(273.57f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.54f, air[0][0](2, 6));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    291.76	292.21	289.61	288.81   |   289.55	290.82	292.27
    291.99	291.38	290.35	289.85   |   288.82	289.76	290.56
    293.69	292.93	291.72	289.46   |   288.36	288.09	287.94
    */
    EXPECT_FLOAT_EQ(291.76f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(292.21f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(289.61f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.81f, air[1][0](0, 3));
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 4));
    EXPECT_FLOAT_EQ(290.82f, air[1][0](0, 5));
    EXPECT_FLOAT_EQ(292.27f, air[1][0](0, 6));
    EXPECT_FLOAT_EQ(291.99f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(293.69f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(287.94f, air[1][0](2, 6));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    281.45	280.02	286.29	285.97   |   284.87	284.87	284.68
    280.27	283.96	285.08	284.46   |   284.12	284.58	285.24
    283.65	285.85	284.79	283.07   |   281.65	282.45	282.25
    */
    EXPECT_FLOAT_EQ(281.45f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(280.02f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(286.29f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(285.97f, air[11][0](0, 3));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 4));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 5));
    EXPECT_FLOAT_EQ(284.68f, air[11][0](0, 6));
    EXPECT_FLOAT_EQ(280.27f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(283.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(282.25f, air[11][0](2, 6));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorArchNcepR2Gaussian, LoadBorderLeft)
{
    double xMin = 0;
    int xPtsNb = 3;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_Reanalysis_v2", "flux/air2m",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   284.46	285.92	287.61
    |   282.23	283.46	284.65
    |   275.49	273.53	271.54
    */
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(285.92f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(287.61f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(282.23f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(275.49f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.54f, air[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.55	290.82	292.27
    |   288.82	289.76	290.56
    |   288.36	288.09	287.94
    */
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(290.82f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(292.27f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.82f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(288.36f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(287.94f, air[1][0](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   284.87	284.87	284.68
    |   284.12	284.58	285.24
    |   281.65	282.45	282.25
    */
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(284.68f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(284.12f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(281.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(282.25f, air[11][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorArchNcepR2Gaussian, LoadBorderLeftOn720)
{
    double xMin = 360;
    int xPtsNb = 3;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_Reanalysis_v2", "flux/air2m",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   284.46	285.92	287.61
    |   282.23	283.46	284.65
    |   275.49	273.53	271.54
    */
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(285.92f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(287.61f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(282.23f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(275.49f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(271.54f, air[0][0](2, 2));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.55	290.82	292.27
    |   288.82	289.76	290.56
    |   288.36	288.09	287.94
    */
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(290.82f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(292.27f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.82f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(288.36f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(287.94f, air[1][0](2, 2));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   284.87	284.87	284.68
    |   284.12	284.58	285.24
    |   281.65	282.45	282.25
    */
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(284.68f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(284.12f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(281.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(282.25f, air[11][0](2, 2));

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorArchNcepR2Gaussian, LoadBorderRight)
{
    double xMin = 352.5;
    int xPtsNb = 5;
    double yMin = 29.523;
    int yPtsNb = 3;
    double step = 0;
    float level = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    double start = asTime::GetMJD(1979, 1, 1, 00, 00);
    double end = asTime::GetMJD(1979, 1, 6, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r2/");

    asPredictorArch *predictor = asPredictorArch::GetInstance("NCEP_Reanalysis_v2", "flux/air2m",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(area, timearray));

    vva2f air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    278.10	278.82	285.20	284.96   |   284.46
    277.15	272.98	274.92	279.08   |   282.23
    273.57	272.72	272.38	272.51   |   275.49
    */
    EXPECT_FLOAT_EQ(278.10f, air[0][0](0, 0));
    EXPECT_FLOAT_EQ(278.82f, air[0][0](0, 1));
    EXPECT_FLOAT_EQ(285.20f, air[0][0](0, 2));
    EXPECT_FLOAT_EQ(284.96f, air[0][0](0, 3));
    EXPECT_FLOAT_EQ(284.46f, air[0][0](0, 4));
    EXPECT_FLOAT_EQ(277.15f, air[0][0](1, 0));
    EXPECT_FLOAT_EQ(273.57f, air[0][0](2, 0));
    EXPECT_FLOAT_EQ(275.49f, air[0][0](2, 4));

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    291.76	292.21	289.61	288.81   |   289.55
    291.99	291.38	290.35	289.85   |   288.82
    293.69	292.93	291.72	289.46   |   288.36
    */
    EXPECT_FLOAT_EQ(291.76f, air[1][0](0, 0));
    EXPECT_FLOAT_EQ(292.21f, air[1][0](0, 1));
    EXPECT_FLOAT_EQ(289.61f, air[1][0](0, 2));
    EXPECT_FLOAT_EQ(288.81f, air[1][0](0, 3));
    EXPECT_FLOAT_EQ(289.55f, air[1][0](0, 4));
    EXPECT_FLOAT_EQ(291.99f, air[1][0](1, 0));
    EXPECT_FLOAT_EQ(293.69f, air[1][0](2, 0));
    EXPECT_FLOAT_EQ(288.36f, air[1][0](2, 4));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    281.45	280.02	286.29	285.97   |   284.87
    280.27	283.96	285.08	284.46   |   284.12
    283.65	285.85	284.79	283.07   |   281.65
    */
    EXPECT_FLOAT_EQ(281.45f, air[11][0](0, 0));
    EXPECT_FLOAT_EQ(280.02f, air[11][0](0, 1));
    EXPECT_FLOAT_EQ(286.29f, air[11][0](0, 2));
    EXPECT_FLOAT_EQ(285.97f, air[11][0](0, 3));
    EXPECT_FLOAT_EQ(284.87f, air[11][0](0, 4));
    EXPECT_FLOAT_EQ(280.27f, air[11][0](1, 0));
    EXPECT_FLOAT_EQ(283.65f, air[11][0](2, 0));
    EXPECT_FLOAT_EQ(281.65f, air[11][0](2, 4));

    wxDELETE(area);
    wxDELETE(predictor);
}

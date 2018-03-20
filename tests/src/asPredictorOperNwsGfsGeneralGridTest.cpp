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
#include "asPredictorOper.h"
#include "asGeoAreaCompositeGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorOperNwsGfsGeneral, LoadEasySmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 10;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9308	9305	9301	9297	9290	9285
    9312	9310	9306	9301	9294	9288
    9321	9317	9314	9310	9303	9295
    9336	9329	9325	9320	9315	9308
    */
    EXPECT_NEAR(9308, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9305, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9301, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9297, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9290, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9285, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9321, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9336, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](3, 5), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9302	9299	9297	9295	9291	9289
    9304	9302	9299	9296	9293	9290
    9314	9308	9304	9301	9297	9293
    9326	9321	9313	9308	9304	9300
    */
    EXPECT_NEAR(9302, hgt[1][0](0, 0), 0.5);
    EXPECT_NEAR(9299, hgt[1][0](0, 1), 0.5);
    EXPECT_NEAR(9297, hgt[1][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[1][0](0, 3), 0.5);
    EXPECT_NEAR(9291, hgt[1][0](0, 4), 0.5);
    EXPECT_NEAR(9289, hgt[1][0](0, 5), 0.5);
    EXPECT_NEAR(9304, hgt[1][0](1, 0), 0.5);
    EXPECT_NEAR(9314, hgt[1][0](2, 0), 0.5);
    EXPECT_NEAR(9326, hgt[1][0](3, 0), 0.5);
    EXPECT_NEAR(9300, hgt[1][0](3, 5), 0.5);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9299	9297	9295	9293	9290	9289
    9303	9301	9297	9294	9292	9290
    9311	9308	9304	9298	9294	9291
    9321	9318	9314	9307	9299	9295
    */
    EXPECT_NEAR(9299, hgt[2][0](0, 0), 0.5);
    EXPECT_NEAR(9297, hgt[2][0](0, 1), 0.5);
    EXPECT_NEAR(9295, hgt[2][0](0, 2), 0.5);
    EXPECT_NEAR(9293, hgt[2][0](0, 3), 0.5);
    EXPECT_NEAR(9290, hgt[2][0](0, 4), 0.5);
    EXPECT_NEAR(9289, hgt[2][0](0, 5), 0.5);
    EXPECT_NEAR(9303, hgt[2][0](1, 0), 0.5);
    EXPECT_NEAR(9311, hgt[2][0](2, 0), 0.5);
    EXPECT_NEAR(9321, hgt[2][0](3, 0), 0.5);
    EXPECT_NEAR(9295, hgt[2][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadEasyLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 10;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9308	9305	9301	9297	9290	9285
    9312	9310	9306	9301	9294	9288
    9321	9317	9314	9310	9303	9295
    9336	9329	9325	9320	9315	9308
    */
    EXPECT_NEAR(9308, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9305, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9301, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9297, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9290, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9285, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9321, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9336, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](3, 5), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9302	9299	9297	9295	9291	9289
    9304	9302	9299	9296	9293	9290
    9314	9308	9304	9301	9297	9293
    9326	9321	9313	9308	9304	9300
    */
    EXPECT_NEAR(9302, hgt[1][0](0, 0), 0.5);
    EXPECT_NEAR(9299, hgt[1][0](0, 1), 0.5);
    EXPECT_NEAR(9297, hgt[1][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[1][0](0, 3), 0.5);
    EXPECT_NEAR(9291, hgt[1][0](0, 4), 0.5);
    EXPECT_NEAR(9289, hgt[1][0](0, 5), 0.5);
    EXPECT_NEAR(9304, hgt[1][0](1, 0), 0.5);
    EXPECT_NEAR(9314, hgt[1][0](2, 0), 0.5);
    EXPECT_NEAR(9326, hgt[1][0](3, 0), 0.5);
    EXPECT_NEAR(9300, hgt[1][0](3, 5), 0.5);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9299	9297	9295	9293	9290	9289
    9303	9301	9297	9294	9292	9290
    9311	9308	9304	9298	9294	9291
    9321	9318	9314	9307	9299	9295
    */
    EXPECT_NEAR(9299, hgt[2][0](0, 0), 0.5);
    EXPECT_NEAR(9297, hgt[2][0](0, 1), 0.5);
    EXPECT_NEAR(9295, hgt[2][0](0, 2), 0.5);
    EXPECT_NEAR(9293, hgt[2][0](0, 3), 0.5);
    EXPECT_NEAR(9290, hgt[2][0](0, 4), 0.5);
    EXPECT_NEAR(9289, hgt[2][0](0, 5), 0.5);
    EXPECT_NEAR(9303, hgt[2][0](1, 0), 0.5);
    EXPECT_NEAR(9311, hgt[2][0](2, 0), 0.5);
    EXPECT_NEAR(9321, hgt[2][0](3, 0), 0.5);
    EXPECT_NEAR(9295, hgt[2][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9402	9373	9333	9300	9291
    9438	9421	9397	9369	9338	9320
    9451	9436	9421	9402	9379	9364
    9462	9449	9437	9423	9410	9395
    */
    EXPECT_NEAR(9422, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9333, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9291, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9438, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9451, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9462, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9395, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9402	9373	9333	9300	9291
    9438	9421	9397	9369	9338	9320
    9451	9436	9421	9402	9379	9364
    9462	9449	9437	9423	9410	9395
    */
    EXPECT_NEAR(9422, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9333, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9291, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9438, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9451, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9462, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9395, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadBorderLeftSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 0;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    EXPECT_NEAR(9333, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9291, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9301, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9369, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9423, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadBorderLeftLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 0;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    EXPECT_NEAR(9333, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9291, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9301, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9369, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9423, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadBorderLeftOn720SmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 360;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    EXPECT_NEAR(9333, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9291, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9301, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9369, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9423, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadBorderLeftOn720LargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 360;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    EXPECT_NEAR(9333, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9291, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9301, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9369, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9423, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadBorderRightSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 355;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9451	9438	9422	9402	9373	9333
    9463	9452	9438	9421	9397	9369
    9475	9465	9451	9436	9421	9402
    9485	9473	9462	9449	9437	9423
    */
    EXPECT_NEAR(9451, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9438, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9422, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9333, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9463, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9475, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9485, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9423, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadBorderRightLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 355;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double step = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9451	9438	9422	9402	9373	9333
    9463	9452	9438	9421	9397	9369
    9475	9465	9451	9436	9421	9402
    9485	9473	9462	9449	9437	9423
    */
    EXPECT_NEAR(9451, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9438, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9422, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9402, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9333, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9463, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9475, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9485, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9423, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStepLonSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double xStep = 2;
    double yStep = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9373	9300	9295	9308	9312
    9438	9397	9338	9316	9324	9324
    9451	9421	9379	9354	9350	9342
    9462	9437	9410	9385	9373	9360
    */
    EXPECT_NEAR(9422, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9438, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9451, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9462, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9360, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStepLonLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 4;
    double xStep = 2;
    double yStep = 1;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9373	9300	9295	9308	9312
    9438	9397	9338	9316	9324	9324
    9451	9421	9379	9354	9350	9342
    9462	9437	9410	9385	9373	9360
    */
    EXPECT_NEAR(9422, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9295, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9308, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9438, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9451, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9462, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(9360, hgt[0][0](3, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStepLonLatSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 3;
    double xStep = 2;
    double yStep = 3;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9400	9368	9332	9314	9312	9312
    9422	9373	9300	9295	9308	9312
    9462	9437	9410	9385	9373	9360
    */
    EXPECT_NEAR(9400, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9368, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9314, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9422, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9462, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9360, hgt[0][0](2, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStepLonLatLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    int xPtsNb = 6;
    double yMin = 35;
    int yPtsNb = 3;
    double xStep = 2;
    double yStep = 3;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9400	9368	9332	9314	9312	9312
    9422	9373	9300	9295	9308	9312
    9462	9437	9410	9385	9373	9360
    */
    EXPECT_NEAR(9400, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9368, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9314, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9312, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(9422, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9462, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9360, hgt[0][0](2, 5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStep25LonLatRoundStartSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -5;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9431	9397.5	9332	9300	9304
    9457	7536.6	9351	7444.4	9316
    9485	9455.5	9423	9390	9373
    */
    EXPECT_NEAR(9431, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9397.5, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9304, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9457, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9420.75, hgt[0][0](1, 1), 0.5);
    EXPECT_NEAR(9305.5, hgt[0][0](1, 3), 0.5);
    EXPECT_NEAR(9485, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9455.5, hgt[0][0](2, 1), 0.5);
    EXPECT_NEAR(9390, hgt[0][0](2, 3), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](2, 4), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStep25LonLatRoundStartLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -5;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9431	9397.5	9332	9300	9304
    9457	9420.75	9351	9305.5	9316
    9485	9455.5	9423	9390	9373
    */
    EXPECT_NEAR(9431, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9397.5, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9304, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(9457, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9420.75, hgt[0][0](1, 1), 0.5);
    EXPECT_NEAR(9305.5, hgt[0][0](1, 3), 0.5);
    EXPECT_NEAR(9485, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9455.5, hgt[0][0](2, 1), 0.5);
    EXPECT_NEAR(9390, hgt[0][0](2, 3), 0.5);
    EXPECT_NEAR(9373, hgt[0][0](2, 4), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStep25LonLatIrregularStartSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -2.5;
    int xPtsNb = 4;
    double yMin = 37.5;
    int yPtsNb = 2;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300	9304
    9420.75	9351	9305.5	9316
    */
    EXPECT_NEAR(9397.5, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9304, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9420.75, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9351, hgt[0][0](1, 1), 0.5);
    EXPECT_NEAR(9305.5, hgt[0][0](1, 2), 0.5);
    EXPECT_NEAR(9316, hgt[0][0](1, 3), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStep25LonLatIrregularStartLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -2.5;
    int xPtsNb = 4;
    double yMin = 37.5;
    int yPtsNb = 2;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300	9304
    9420.75	9351	9305.5	9316
    */
    EXPECT_NEAR(9397.5, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9304, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9420.75, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9351, hgt[0][0](1, 1), 0.5);
    EXPECT_NEAR(9305.5, hgt[0][0](1, 2), 0.5);
    EXPECT_NEAR(9316, hgt[0][0](1, 3), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStep25LonLatIrregularStartAndEndSmallFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -2.5;
    int xPtsNb = 3;
    double yMin = 37.5;
    int yPtsNb = 2;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300
    9420.75	9351	9305.5
    */
    EXPECT_NEAR(9397.5, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9420.75, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9351, hgt[0][0](1, 1), 0.5);
    EXPECT_NEAR(9305.5, hgt[0][0](1, 2), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsGeneral, LoadCompositeStep25LonLatIrregularStartAndEndLargeFile)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -2.5;
    int xPtsNb = 3;
    double yMin = 37.5;
    int yPtsNb = 2;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid *geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb,
                                                                          yStep, level);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(geoarea, dates));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300
    9420.75	9351	9305.5
    */
    EXPECT_NEAR(9397.5, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9332, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9300, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9420.75, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9351, hgt[0][0](1, 1), 0.5);
    EXPECT_NEAR(9305.5, hgt[0][0](1, 2), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

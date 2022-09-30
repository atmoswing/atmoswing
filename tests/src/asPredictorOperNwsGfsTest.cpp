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

#include <gtest/gtest.h>
#include <wx/filename.h>

#include "asAreaGrid.h"
#include "asAreaRegGrid.h"
#include "asPredictorOper.h"
#include "asTimeArray.h"

TEST(PredictorOperNwsGfs, GetCorrectPredictors) {
    asPredictorOper *predictor;

    predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "z");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "t");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::AirTemperature);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "w");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::VerticalVelocity);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "rh");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "u");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Uwind);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "v");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Vwind);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "pwat");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadEasySmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadEasySmallFileRegular) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 10;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 3;
    double step = 1;
    float level = 300;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadEasyLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsLargeFileRegular) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.L.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 3;
    double step = 1;
    float level = 300;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadBorderLeftSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadBorderLeftLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadBorderLeftOn720SmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadBorderLeftOn720LargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadBorderRightSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadBorderRightSmallFileRegular) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = 355;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 3;
    double step = 1;
    float level = 300;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadBorderRightLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStepLonSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStepLonLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStepLonLatSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStepLonLatSmallFileRegular) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -3;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 6;
    double xStep = 2;
    double yStep = 3;
    float level = 300;
    asAreaRegGrid area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStepLonLatLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatRoundStartSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatRoundStartSmallFileRegular) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -5;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 5;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    asAreaRegGrid area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatRoundStartLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatIrregularStartSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5   9331.51  9300.15  9303.79
    9420.71	 9350.67  9305.54  9315.97
    */

    EXPECT_NEAR(9397.5, hgt[0][0](0, 0), 0.01);
    EXPECT_NEAR(9331.51, hgt[0][0](0, 1), 0.01);
    EXPECT_NEAR(9300.15, hgt[0][0](0, 2), 0.01);
    EXPECT_NEAR(9303.79, hgt[0][0](0, 3), 0.01);
    EXPECT_NEAR(9420.71, hgt[0][0](1, 0), 0.01);
    EXPECT_NEAR(9350.67, hgt[0][0](1, 1), 0.01);
    EXPECT_NEAR(9305.54, hgt[0][0](1, 2), 0.01);
    EXPECT_NEAR(9315.97, hgt[0][0](1, 3), 0.01);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatIrregularStartLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatIrregularStartAndEndSmallFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatIrregularStartAndEndSmallFileRegular) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2010/gfs.hgt.24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011, 4, 11, 12, 00), asTime::GetMJD(2011, 4, 12, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -2.5;
    double xWidth = 5;
    double yMin = 37.5;
    double yWidth = 2.5;
    double xStep = 2.5;
    double yStep = 2.5;
    float level = 300;
    asAreaRegGrid area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfs, LoadWithNegativeValsStep25LonLatIrregularStartAndEndLargeFile) {
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
    asAreaGrid *area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

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

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperNwsGfsRegular, LoadVersion2017) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2017/2017121312.NWS_GFS_Forecast.hgt.012.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-nws-gfs/2017/2017121312.NWS_GFS_Forecast.hgt.024.grib2");

    asTimeArray dates(asTime::GetMJD(2017, 12, 14, 00, 00), asTime::GetMJD(2017, 12, 14, 12, 00), 12, "Simple");
    dates.Init();

    double xMin = 10;
    double xWidth = 5;
    double yMin = 35;
    double yWidth = 3;
    double step = 0.5;
    float level = 300;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted with Panoply:
    9321.9	9315.2	9308.8	9302.9	9297.7	9294.2	9289.7	9283.8	9281.2	9278.1	9270.4
    9327.9	9322.1	9316.3	9311.1	9306.9	9303.4	9299.1	9293.8	9289.5	9286.3	9281.5
    9334.2	9328.4	9323.7	9319.4	9315.4	9312.1	9308.5	9303.8	9301.1	9299.6	9294.0
    9340.6	9335.8	9331.7	9327.6	9324.5	9322.5	9320.4	9317.0	9313.8	9312.6	9311.0
    9347.5	9344.2	9341.0	9338.6	9336.7	9335.1	9334.0	9333.4	9332.1	9331.5	9332.1
    9357.9	9355.6	9353.6	9353.0	9352.3	9351.6	9351.4	9352.3	9353.5	9353.9	9354.4
    9370.7	9370.3	9370.0	9369.8	9369.2	9369.4	9370.5	9371.6	9372.9	9374.0	9374.8
    */

    EXPECT_NEAR(9321.9, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(9315.2, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(9308.8, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(9302.9, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(9278.1, hgt[0][0](0, 9), 0.5);
    EXPECT_NEAR(9270.4, hgt[0][0](0, 10), 0.5);
    EXPECT_NEAR(9327.9, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(9334.2, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(9370.7, hgt[0][0](6, 0), 0.5);
    EXPECT_NEAR(9374.8, hgt[0][0](6, 10), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted with Panoply:
    9370.7	9369.4	9368.7	9367.8	9366.4	9364.6	9361.4	9358.0	9358.7	9357.4	9352.8
    9379.5	9378.4	9377.6	9376.9	9375.6	9373.5	9371.2	9369.1	9367.8	9365.9	9363.8
    9387.1	9386.4	9385.9	9385.2	9384.0	9382.4	9380.5	9378.7	9378.0	9377.0	9373.9
    9392.9	9393.3	9392.7	9392.2	9391.6	9390.7	9389.3	9388.1	9387.4	9386.0	9384.2
    9396.8	9397.6	9398.1	9398.2	9398.1	9397.6	9396.9	9396.2	9395.6	9394.4	9392.8
    9401.2	9402.4	9403.4	9404.3	9404.9	9405.0	9404.6	9404.0	9403.4	9402.5	9400.9
    9410.0	9411.8	9413.4	9414.8	9415.8	9416.2	9416.1	9415.6	9414.9	9414.0	9412.5
    */

    EXPECT_NEAR(9370.7, hgt[1][0](0, 0), 0.5);
    EXPECT_NEAR(9369.4, hgt[1][0](0, 1), 0.5);
    EXPECT_NEAR(9368.7, hgt[1][0](0, 2), 0.5);
    EXPECT_NEAR(9367.8, hgt[1][0](0, 3), 0.5);
    EXPECT_NEAR(9357.4, hgt[1][0](0, 9), 0.5);
    EXPECT_NEAR(9352.8, hgt[1][0](0, 10), 0.5);
    EXPECT_NEAR(9379.5, hgt[1][0](1, 0), 0.5);
    EXPECT_NEAR(9387.1, hgt[1][0](2, 0), 0.5);
    EXPECT_NEAR(9410.0, hgt[1][0](6, 0), 0.5);
    EXPECT_NEAR(9412.5, hgt[1][0](6, 10), 0.5);

    wxDELETE(predictor);
}
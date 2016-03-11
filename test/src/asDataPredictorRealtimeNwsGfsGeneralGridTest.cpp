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

#include "include_tests.h"
#include "asDataPredictorRealtime.h"
#include "asGeoAreaCompositeGrid.h"
#include "asTimeArray.h"

#include "UnitTest++.h"

namespace
{

TEST(LoadEasySmallFile)
{
	wxPrintf("Testing GFS general realtime predictors...\n");
	
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 10;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9308	9305	9301	9297	9290	9285
    9312	9310	9306	9301	9294	9288
    9321	9317	9314	9310	9303	9295
    9336	9329	9325	9320	9315	9308
    */
    CHECK_CLOSE(9308, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9305, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9301, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9297, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9290, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9285, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9312, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9321, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9336, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9308, hgt[0](3,5), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9302	9299	9297	9295	9291	9289
    9304	9302	9299	9296	9293	9290
    9314	9308	9304	9301	9297	9293
    9326	9321	9313	9308	9304	9300
    */
    CHECK_CLOSE(9302, hgt[1](0,0), 0.5);
    CHECK_CLOSE(9299, hgt[1](0,1), 0.5);
    CHECK_CLOSE(9297, hgt[1](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[1](0,3), 0.5);
    CHECK_CLOSE(9291, hgt[1](0,4), 0.5);
    CHECK_CLOSE(9289, hgt[1](0,5), 0.5);
    CHECK_CLOSE(9304, hgt[1](1,0), 0.5);
    CHECK_CLOSE(9314, hgt[1](2,0), 0.5);
    CHECK_CLOSE(9326, hgt[1](3,0), 0.5);
    CHECK_CLOSE(9300, hgt[1](3,5), 0.5);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9299	9297	9295	9293	9290	9289
    9303	9301	9297	9294	9292	9290
    9311	9308	9304	9298	9294	9291
    9321	9318	9314	9307	9299	9295
    */
    CHECK_CLOSE(9299, hgt[2](0,0), 0.5);
    CHECK_CLOSE(9297, hgt[2](0,1), 0.5);
    CHECK_CLOSE(9295, hgt[2](0,2), 0.5);
    CHECK_CLOSE(9293, hgt[2](0,3), 0.5);
    CHECK_CLOSE(9290, hgt[2](0,4), 0.5);
    CHECK_CLOSE(9289, hgt[2](0,5), 0.5);
    CHECK_CLOSE(9303, hgt[2](1,0), 0.5);
    CHECK_CLOSE(9311, hgt[2](2,0), 0.5);
    CHECK_CLOSE(9321, hgt[2](3,0), 0.5);
    CHECK_CLOSE(9295, hgt[2](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadEasyLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 10;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9308	9305	9301	9297	9290	9285
    9312	9310	9306	9301	9294	9288
    9321	9317	9314	9310	9303	9295
    9336	9329	9325	9320	9315	9308
    */
    CHECK_CLOSE(9308, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9305, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9301, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9297, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9290, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9285, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9312, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9321, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9336, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9308, hgt[0](3,5), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9302	9299	9297	9295	9291	9289
    9304	9302	9299	9296	9293	9290
    9314	9308	9304	9301	9297	9293
    9326	9321	9313	9308	9304	9300
    */
    CHECK_CLOSE(9302, hgt[1](0,0), 0.5);
    CHECK_CLOSE(9299, hgt[1](0,1), 0.5);
    CHECK_CLOSE(9297, hgt[1](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[1](0,3), 0.5);
    CHECK_CLOSE(9291, hgt[1](0,4), 0.5);
    CHECK_CLOSE(9289, hgt[1](0,5), 0.5);
    CHECK_CLOSE(9304, hgt[1](1,0), 0.5);
    CHECK_CLOSE(9314, hgt[1](2,0), 0.5);
    CHECK_CLOSE(9326, hgt[1](3,0), 0.5);
    CHECK_CLOSE(9300, hgt[1](3,5), 0.5);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9299	9297	9295	9293	9290	9289
    9303	9301	9297	9294	9292	9290
    9311	9308	9304	9298	9294	9291
    9321	9318	9314	9307	9299	9295
    */
    CHECK_CLOSE(9299, hgt[2](0,0), 0.5);
    CHECK_CLOSE(9297, hgt[2](0,1), 0.5);
    CHECK_CLOSE(9295, hgt[2](0,2), 0.5);
    CHECK_CLOSE(9293, hgt[2](0,3), 0.5);
    CHECK_CLOSE(9290, hgt[2](0,4), 0.5);
    CHECK_CLOSE(9289, hgt[2](0,5), 0.5);
    CHECK_CLOSE(9303, hgt[2](1,0), 0.5);
    CHECK_CLOSE(9311, hgt[2](2,0), 0.5);
    CHECK_CLOSE(9321, hgt[2](3,0), 0.5);
    CHECK_CLOSE(9295, hgt[2](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -3;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9402	9373	9333	9300	9291
    9438	9421	9397	9369	9338	9320
    9451	9436	9421	9402	9379	9364
    9462	9449	9437	9423	9410	9395
    */
    CHECK_CLOSE(9422, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9402, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9373, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9333, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9291, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9438, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9451, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9462, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9395, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -3;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9402	9373	9333	9300	9291
    9438	9421	9397	9369	9338	9320
    9451	9436	9421	9402	9379	9364
    9462	9449	9437	9423	9410	9395
    */
    CHECK_CLOSE(9422, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9402, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9373, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9333, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9291, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9438, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9451, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9462, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9395, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderLeftSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 0;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    CHECK_CLOSE(9333, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9291, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9301, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9308, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9369, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9402, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9423, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9373, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderLeftLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 0;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    CHECK_CLOSE(9333, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9291, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9301, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9308, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9369, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9402, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9423, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9373, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderLeftOn720SmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 360;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    CHECK_CLOSE(9333, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9291, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9301, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9308, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9369, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9402, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9423, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9373, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderLeftOn720LargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 360;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9333	9300	9291	9295	9301	9308
    9369	9338	9320	9316	9320	9324
    9402	9379	9364	9354	9350	9350
    9423	9410	9395	9385	9379	9373
    */
    CHECK_CLOSE(9333, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9291, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9301, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9308, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9369, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9402, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9423, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9373, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderRightSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 355;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9451	9438	9422	9402	9373	9333
    9463	9452	9438	9421	9397	9369
    9475	9465	9451	9436	9421	9402
    9485	9473	9462	9449	9437	9423
    */
    CHECK_CLOSE(9451, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9438, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9422, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9402, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9373, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9333, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9463, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9475, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9485, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9423, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderRightLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = 355;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double step = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9451	9438	9422	9402	9373	9333
    9463	9452	9438	9421	9397	9369
    9475	9465	9451	9436	9421	9402
    9485	9473	9462	9449	9437	9423
    */
    CHECK_CLOSE(9451, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9438, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9422, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9402, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9373, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9333, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9463, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9475, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9485, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9423, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStepLonSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -3;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double Xstep = 2;
    double Ystep = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9373	9300	9295	9308	9312
    9438	9397	9338	9316	9324	9324
    9451	9421	9379	9354	9350	9342
    9462	9437	9410	9385	9373	9360
    */
    CHECK_CLOSE(9422, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9373, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9308, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9312, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9438, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9451, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9462, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9360, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStepLonLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -3;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 4;
    double Xstep = 2;
    double Ystep = 1;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9422	9373	9300	9295	9308	9312
    9438	9397	9338	9316	9324	9324
    9451	9421	9379	9354	9350	9342
    9462	9437	9410	9385	9373	9360
    */
    CHECK_CLOSE(9422, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9373, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9295, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9308, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9312, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9438, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9451, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9462, hgt[0](3,0), 0.5);
    CHECK_CLOSE(9360, hgt[0](3,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStepLonLatSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -3;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 3;
    double Xstep = 2;
    double Ystep = 3;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9400	9368	9332	9314	9312	9312
    9422	9373	9300	9295	9308	9312
    9462	9437	9410	9385	9373	9360
    */
    CHECK_CLOSE(9400, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9368, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9314, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9312, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9312, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9422, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9462, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9360, hgt[0](2,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStepLonLatLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -3;
    int Xptsnb = 6;
    double Ymin = 35;
    int Yptsnb = 3;
    double Xstep = 2;
    double Ystep = 3;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9400	9368	9332	9314	9312	9312
    9422	9373	9300	9295	9308	9312
    9462	9437	9410	9385	9373	9360
    */
    CHECK_CLOSE(9400, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9368, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9314, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9312, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9312, hgt[0](0,5), 0.5);
    CHECK_CLOSE(9422, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9462, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9360, hgt[0](2,5), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStep25LonLatRoundStartSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -5;
    int Xptsnb = 5;
    double Ymin = 35;
    int Yptsnb = 3;
    double Xstep = 2.5;
    double Ystep = 2.5;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9431	9397.5	9332	9300	9304
    9457	7536.6	9351	7444.4	9316
    9485	9455.5	9423	9390	9373
    */
    CHECK_CLOSE(9431, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9397.5, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9304, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9457, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9420.75, hgt[0](1,1), 0.5);
    CHECK_CLOSE(9305.5, hgt[0](1,3), 0.5);
    CHECK_CLOSE(9485, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9455.5, hgt[0](2,1), 0.5);
    CHECK_CLOSE(9390, hgt[0](2,3), 0.5);
    CHECK_CLOSE(9373, hgt[0](2,4), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStep25LonLatRoundStartLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -5;
    int Xptsnb = 5;
    double Ymin = 35;
    int Yptsnb = 3;
    double Xstep = 2.5;
    double Ystep = 2.5;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9431	9397.5	9332	9300	9304
    9457	9420.75	9351	9305.5	9316
    9485	9455.5	9423	9390	9373
    */
    CHECK_CLOSE(9431, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9397.5, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9304, hgt[0](0,4), 0.5);
    CHECK_CLOSE(9457, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9420.75, hgt[0](1,1), 0.5);
    CHECK_CLOSE(9305.5, hgt[0](1,3), 0.5);
    CHECK_CLOSE(9485, hgt[0](2,0), 0.5);
    CHECK_CLOSE(9455.5, hgt[0](2,1), 0.5);
    CHECK_CLOSE(9390, hgt[0](2,3), 0.5);
    CHECK_CLOSE(9373, hgt[0](2,4), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStep25LonLatIrregularStartSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -2.5;
    int Xptsnb = 4;
    double Ymin = 37.5;
    int Yptsnb = 2;
    double Xstep = 2.5;
    double Ystep = 2.5;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300	9304
    9420.75	9351	9305.5	9316
    */
    CHECK_CLOSE(9397.5, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9304, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9420.75, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9351, hgt[0](1,1), 0.5);
    CHECK_CLOSE(9305.5, hgt[0](1,2), 0.5);
    CHECK_CLOSE(9316, hgt[0](1,3), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStep25LonLatIrregularStartLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -2.5;
    int Xptsnb = 4;
    double Ymin = 37.5;
    int Yptsnb = 2;
    double Xstep = 2.5;
    double Ystep = 2.5;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300	9304
    9420.75	9351	9305.5	9316
    */
    CHECK_CLOSE(9397.5, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9304, hgt[0](0,3), 0.5);
    CHECK_CLOSE(9420.75, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9351, hgt[0](1,1), 0.5);
    CHECK_CLOSE(9305.5, hgt[0](1,2), 0.5);
    CHECK_CLOSE(9316, hgt[0](1,3), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStep25LonLatIrregularStartAndEndSmallFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -2.5;
    int Xptsnb = 3;
    double Ymin = 37.5;
    int Yptsnb = 2;
    double Xstep = 2.5;
    double Ystep = 2.5;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300
    9420.75	9351	9305.5
    */
    CHECK_CLOSE(9397.5, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9420.75, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9351, hgt[0](1,1), 0.5);
    CHECK_CLOSE(9305.5, hgt[0](1,2), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadCompositeStep25LonLatIrregularStartAndEndLargeFile)
{
    VectorString filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_12h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_18h_L.grib2");
    filepaths.push_back(wxFileName::GetCwd() + "/files/NWS_GFS_Forecast_hgt_24h_L.grib2");

    asTimeArray dates(asTime::GetMJD(2011,4,11,12,00), asTime::GetMJD(2011,4,12,00,00), 6, "Simple");
    dates.Init();

    double Xmin = -2.5;
    int Xptsnb = 3;
    double Ymin = 37.5;
    int Yptsnb = 2;
    double Xstep = 2.5;
    double Ystep = 2.5;
    double level = 300;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep, level);

    asDataPredictorRealtime* predictor = asDataPredictorRealtime::GetInstance("NWS_GFS_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Correct the step according to former GFS files.
    predictor->SetXaxisStep(1);
    predictor->SetYaxisStep(1);

    // Load
    bool successLoad = predictor->Load(geoarea, dates);
    CHECK_EQUAL(true, successLoad);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted from Degrib:
    9397.5	9332	9300
    9420.75	9351	9305.5
    */
    CHECK_CLOSE(9397.5, hgt[0](0,0), 0.5);
    CHECK_CLOSE(9332, hgt[0](0,1), 0.5);
    CHECK_CLOSE(9300, hgt[0](0,2), 0.5);
    CHECK_CLOSE(9420.75, hgt[0](1,0), 0.5);
    CHECK_CLOSE(9351, hgt[0](1,1), 0.5);
    CHECK_CLOSE(9305.5, hgt[0](1,2), 0.5);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}
}

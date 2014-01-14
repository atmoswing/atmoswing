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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include <wx/filename.h>

#include "include_tests.h"
#include "asDataPredictorArchive.h"
#include "asGeoAreaCompositeGrid.h"
#include "asTimeArray.h"

#include "UnitTest++.h"

namespace
{

TEST(LoadEasy)
{
    double Umin = 7.5;
    int Uptsnb = 5;
    double Vmin = 29.523;
    int Vptsnb = 3;
    double step = 0;
    double level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,6,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux_air2m", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1_air_2m_gauss_%d.nc");
    predictor->Load(geoarea, timearray);

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    276.9	279.6	282.7	286.7	288.5
    276.2	275.5	276.9	280.4	283.7
    271.3	270.8	273.9	275.6	277.7
    */
    CHECK_CLOSE(276.9, air[0](0,0), 0.0001);
    CHECK_CLOSE(279.6, air[0](0,1), 0.0001);
    CHECK_CLOSE(282.7, air[0](0,2), 0.0001);
    CHECK_CLOSE(286.7, air[0](0,3), 0.0001);
    CHECK_CLOSE(288.5, air[0](0,4), 0.0001);
    CHECK_CLOSE(276.2, air[0](1,0), 0.0001);
    CHECK_CLOSE(271.3, air[0](2,0), 0.0001);
    CHECK_CLOSE(277.7, air[0](2,4), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    293.4	292.9	291.9	289.2	288.8
    293.6	292.9	291.8	291.0	290.8
    291.6	291.5	290.8	290.1	290.1
    */
    CHECK_CLOSE(293.4, air[1](0,0), 0.0001);
    CHECK_CLOSE(292.9, air[1](0,1), 0.0001);
    CHECK_CLOSE(291.9, air[1](0,2), 0.0001);
    CHECK_CLOSE(289.2, air[1](0,3), 0.0001);
    CHECK_CLOSE(288.8, air[1](0,4), 0.0001);
    CHECK_CLOSE(293.6, air[1](1,0), 0.0001);
    CHECK_CLOSE(291.6, air[1](2,0), 0.0001);
    CHECK_CLOSE(290.1, air[1](2,4), 0.0001);

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    282.1	282.0	283.2	289.4	290.3
    281.5	277.1	275.9	279.6	282.1
    274.5	272.8	275.5	277.8	278.7
    */
    CHECK_CLOSE(282.1, air[11](0,0), 0.0001);
    CHECK_CLOSE(282.0, air[11](0,1), 0.0001);
    CHECK_CLOSE(283.2, air[11](0,2), 0.0001);
    CHECK_CLOSE(289.4, air[11](0,3), 0.0001);
    CHECK_CLOSE(290.3, air[11](0,4), 0.0001);
    CHECK_CLOSE(281.5, air[11](1,0), 0.0001);
    CHECK_CLOSE(274.5, air[11](2,0), 0.0001);
    CHECK_CLOSE(278.7, air[11](2,4), 0.0001);

    /* Values time step 20 (horizontal=Lon, vertical=Lat)
    269.0	273.2	280.2	285.6	288.0
    270.6	268.1	271.2	278.9	282.4
    272.7	268.3	267.1	271.3	276.8
    */
    CHECK_CLOSE(269.0, air[20](0,0), 0.0001);
    CHECK_CLOSE(273.2, air[20](0,1), 0.0001);
    CHECK_CLOSE(280.2, air[20](0,2), 0.0001);
    CHECK_CLOSE(285.6, air[20](0,3), 0.0001);
    CHECK_CLOSE(288.0, air[20](0,4), 0.0001);
    CHECK_CLOSE(270.6, air[20](1,0), 0.0001);
    CHECK_CLOSE(272.7, air[20](2,0), 0.0001);
    CHECK_CLOSE(276.8, air[20](2,4), 0.0001);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadComposite)
{
    double Umin = -7.5;
    int Uptsnb = 7;
    double Vmin = 29.523;
    int Vptsnb = 3;
    double step = 0;
    double level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,6,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux_air2m", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1_air_2m_gauss_%d.nc");
    predictor->Load(geoarea, timearray);

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    280.5	272.0	271.9	274.5   |   272.5	271.7	274.8
    276.9	271.4	273.0	273.5   |   271.8	271.0	274.3
    277.0	277.5	278.5	279.1   |   278.1	275.7	271.5
    */
    CHECK_CLOSE(280.5, air[0](0,0), 0.0001);
    CHECK_CLOSE(272.0, air[0](0,1), 0.0001);
    CHECK_CLOSE(271.9, air[0](0,2), 0.0001);
    CHECK_CLOSE(274.5, air[0](0,3), 0.0001);
    CHECK_CLOSE(272.5, air[0](0,4), 0.0001);
    CHECK_CLOSE(271.7, air[0](0,5), 0.0001);
    CHECK_CLOSE(274.8, air[0](0,6), 0.0001);
    CHECK_CLOSE(276.9, air[0](1,0), 0.0001);
    CHECK_CLOSE(277.0, air[0](2,0), 0.0001);
    CHECK_CLOSE(271.5, air[0](2,6), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    289.5	288.5	286.9	287.7   |   289.8	291.1	292.0
    291.6	290.7	290.3	291.2   |   292.4	293.4	293.6
    292.8	292.6	293.4 	293.8   |   293.2	292.2	291.6
    */
    CHECK_CLOSE(289.5, air[1](0,0), 0.0001);
    CHECK_CLOSE(288.5, air[1](0,1), 0.0001);
    CHECK_CLOSE(286.9, air[1](0,2), 0.0001);
    CHECK_CLOSE(287.7, air[1](0,3), 0.0001);
    CHECK_CLOSE(289.8, air[1](0,4), 0.0001);
    CHECK_CLOSE(291.1, air[1](0,5), 0.0001);
    CHECK_CLOSE(292.0, air[1](0,6), 0.0001);
    CHECK_CLOSE(291.6, air[1](1,0), 0.0001);
    CHECK_CLOSE(292.8, air[1](2,0), 0.0001);
    CHECK_CLOSE(291.6, air[1](2,6), 0.0001);

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    284.1	279.6	279.5	279.3   |   277.9	277.9 	278.9
    277.4	275.0	277.6	280.1   |   279.7	279.1	280.5
    278.4	280.8	283.2 	284.4   |   282. 0	280.3	278.6
    */
    CHECK_CLOSE(284.1, air[11](0,0), 0.0001);
    CHECK_CLOSE(279.6, air[11](0,1), 0.0001);
    CHECK_CLOSE(279.5, air[11](0,2), 0.0001);
    CHECK_CLOSE(279.3, air[11](0,3), 0.0001);
    CHECK_CLOSE(277.9, air[11](0,4), 0.0001);
    CHECK_CLOSE(277.9, air[11](0,5), 0.0001);
    CHECK_CLOSE(278.9, air[11](0,6), 0.0001);
    CHECK_CLOSE(277.4, air[11](1,0), 0.0001);
    CHECK_CLOSE(278.4, air[11](2,0), 0.0001);
    CHECK_CLOSE(278.6, air[11](2,6), 0.0001);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderLeft)
{
    double Umin = 0;
    int Uptsnb = 3;
    double Vmin = 29.523;
    int Vptsnb = 3;
    double step = 0;
    double level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,6,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux_air2m", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1_air_2m_gauss_%d.nc");
    predictor->Load(geoarea, timearray);

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   272.5	271.7	274.8
    |   271.8	271.0	274.3
    |   278.1	275.7	271.5
    */
    CHECK_CLOSE(272.5, air[0](0,0), 0.0001);
    CHECK_CLOSE(271.7, air[0](0,1), 0.0001);
    CHECK_CLOSE(274.8, air[0](0,2), 0.0001);
    CHECK_CLOSE(271.8, air[0](1,0), 0.0001);
    CHECK_CLOSE(278.1, air[0](2,0), 0.0001);
    CHECK_CLOSE(271.5, air[0](2,2), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.8	291.1	292.0
    |   292.4	293.4	293.6
    |   293.2	292.2	291.6
    */
    CHECK_CLOSE(289.8, air[1](0,0), 0.0001);
    CHECK_CLOSE(291.1, air[1](0,1), 0.0001);
    CHECK_CLOSE(292.0, air[1](0,2), 0.0001);
    CHECK_CLOSE(292.4, air[1](1,0), 0.0001);
    CHECK_CLOSE(293.2, air[1](2,0), 0.0001);
    CHECK_CLOSE(291.6, air[1](2,2), 0.0001);

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   277.9	277.9 	278.9
    |   279.7	279.1	280.5
    |   282.0	280.3	278.6
    */
    CHECK_CLOSE(277.9, air[11](0,0), 0.0001);
    CHECK_CLOSE(277.9, air[11](0,1), 0.0001);
    CHECK_CLOSE(278.9, air[11](0,2), 0.0001);
    CHECK_CLOSE(279.7, air[11](1,0), 0.0001);
    CHECK_CLOSE(282.0, air[11](2,0), 0.0001);
    CHECK_CLOSE(278.6, air[11](2,2), 0.0001);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderLeftOn720)
{
    double Umin = 360;
    int Uptsnb = 3;
    double Vmin = 29.523;
    int Vptsnb = 3;
    double step = 0;
    double level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,6,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux_air2m", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1_air_2m_gauss_%d.nc");
    predictor->Load(geoarea, timearray);

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   272.5	271.7	274.8
    |   271.8	271.0	274.3
    |   278.1	275.7	271.5
    */
    CHECK_CLOSE(272.5, air[0](0,0), 0.0001);
    CHECK_CLOSE(271.7, air[0](0,1), 0.0001);
    CHECK_CLOSE(274.8, air[0](0,2), 0.0001);
    CHECK_CLOSE(271.8, air[0](1,0), 0.0001);
    CHECK_CLOSE(278.1, air[0](2,0), 0.0001);
    CHECK_CLOSE(271.5, air[0](2,2), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   289.8	291.1	292.0
    |   292.4	293.4	293.6
    |   293.2	292.2	291.6
    */
    CHECK_CLOSE(289.8, air[1](0,0), 0.0001);
    CHECK_CLOSE(291.1, air[1](0,1), 0.0001);
    CHECK_CLOSE(292.0, air[1](0,2), 0.0001);
    CHECK_CLOSE(292.4, air[1](1,0), 0.0001);
    CHECK_CLOSE(293.2, air[1](2,0), 0.0001);
    CHECK_CLOSE(291.6, air[1](2,2), 0.0001);

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    |   277.9	277.9 	278.9
    |   279.7	279.1	280.5
    |   282.0	280.3	278.6
    */
    CHECK_CLOSE(277.9, air[11](0,0), 0.0001);
    CHECK_CLOSE(277.9, air[11](0,1), 0.0001);
    CHECK_CLOSE(278.9, air[11](0,2), 0.0001);
    CHECK_CLOSE(279.7, air[11](1,0), 0.0001);
    CHECK_CLOSE(282.0, air[11](2,0), 0.0001);
    CHECK_CLOSE(278.6, air[11](2,2), 0.0001);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(LoadBorderRight)
{
    double Umin = 352.5;
    int Uptsnb = 5;
    double Vmin = 29.523;
    int Vptsnb = 3;
    double step = 0;
    double level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,6,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux_air2m", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1_air_2m_gauss_%d.nc");
    predictor->Load(geoarea, timearray);

    VArray2DFloat air = predictor->GetData();
    // air[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    280.5	272.0	271.9	274.5   |   272.5
    276.9	271.4	273.0	273.5   |   271.8
    277.0	277.5	278.5	279.1   |   278.1
    */
    CHECK_CLOSE(280.5, air[0](0,0), 0.0001);
    CHECK_CLOSE(272.0, air[0](0,1), 0.0001);
    CHECK_CLOSE(271.9, air[0](0,2), 0.0001);
    CHECK_CLOSE(274.5, air[0](0,3), 0.0001);
    CHECK_CLOSE(272.5, air[0](0,4), 0.0001);
    CHECK_CLOSE(276.9, air[0](1,0), 0.0001);
    CHECK_CLOSE(277.0, air[0](2,0), 0.0001);
    CHECK_CLOSE(278.1, air[0](2,4), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    289.5	288.5	286.9	287.7   |   289.8
    291.6	290.7	290.3	291.2   |   292.4
    292.8	292.6	293.4 	293.8   |   293.2
    */
    CHECK_CLOSE(289.5, air[1](0,0), 0.0001);
    CHECK_CLOSE(288.5, air[1](0,1), 0.0001);
    CHECK_CLOSE(286.9, air[1](0,2), 0.0001);
    CHECK_CLOSE(287.7, air[1](0,3), 0.0001);
    CHECK_CLOSE(289.8, air[1](0,4), 0.0001);
    CHECK_CLOSE(291.6, air[1](1,0), 0.0001);
    CHECK_CLOSE(292.8, air[1](2,0), 0.0001);
    CHECK_CLOSE(293.2, air[1](2,4), 0.0001);

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    284.1	279.6	279.5	279.3   |   277.9
    277.4	275.0	277.6	280.1   |   279.7
    278.4	280.8	283.2 	284.4   |   282.0
    */
    CHECK_CLOSE(284.1, air[11](0,0), 0.0001);
    CHECK_CLOSE(279.6, air[11](0,1), 0.0001);
    CHECK_CLOSE(279.5, air[11](0,2), 0.0001);
    CHECK_CLOSE(279.3, air[11](0,3), 0.0001);
    CHECK_CLOSE(277.9, air[11](0,4), 0.0001);
    CHECK_CLOSE(277.4, air[11](1,0), 0.0001);
    CHECK_CLOSE(278.4, air[11](2,0), 0.0001);
    CHECK_CLOSE(282.0, air[11](2,4), 0.0001);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

TEST(SetData)
{
    double Umin = -7.5;
    int Uptsnb = 4;
    double Vmin = 29.523;
    int Vptsnb = 2;
    double steplon = 0;
    double steplat = 0;
    double level = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, steplon, Vmin, Vptsnb, steplat, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,5,00,00);
    double timestephours = 24;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "flux_air2m", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1_air_2m_gauss_%d.nc");
    predictor->Load(geoarea, timearray);

    Array2DFloat tmp;
    tmp.resize(1,4);
    VArray2DFloat newdata;

    tmp << 1,2,3,4;
    newdata.push_back(tmp);
    tmp << 11,12,13,14;
    newdata.push_back(tmp);
    tmp << 21,22,23,24;
    newdata.push_back(tmp);
    tmp << 31,32,33,34;
    newdata.push_back(tmp);
    tmp << 41,42,43,44;
    newdata.push_back(tmp);

    predictor->SetData(newdata);

    CHECK_CLOSE(1, predictor->GetLatPtsnb(), 0.0001);
    CHECK_CLOSE(4, predictor->GetLonPtsnb(), 0.0001);
    CHECK_CLOSE(1, predictor->GetData()[0](0,0), 0.0001);
    CHECK_CLOSE(2, predictor->GetData()[0](0,1), 0.0001);
    CHECK_CLOSE(4, predictor->GetData()[0](0,3), 0.0001);
    CHECK_CLOSE(14, predictor->GetData()[1](0,3), 0.0001);
    CHECK_CLOSE(41, predictor->GetData()[4](0,0), 0.0001);
    CHECK_CLOSE(44, predictor->GetData()[4](0,3), 0.0001);

    wxDELETE(geoarea);
    wxDELETE(predictor);
}

}

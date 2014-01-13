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
#include "asGeoAreaCompositeRegularGrid.h"
#include "asTimeArray.h"

#include "UnitTest++.h"

namespace
{

TEST(LoadEasy)
{
    double Umin = 10;
    double Uwidth = 10;
    double Vmin = 35;
    double Vwidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	175.0	170.0	162.0	151.0
    185.0	180.0	173.0	162.0	144.0
    193.0	183.0	174.0	163.0	143.0
    */
    CHECK_CLOSE(176, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(175, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(170, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(162, hgt[0](0,3), 0.0001);
    CHECK_CLOSE(151, hgt[0](0,4), 0.0001);
    CHECK_CLOSE(185, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(193, hgt[0](2,0), 0.0001);
    CHECK_CLOSE(143, hgt[0](2,4), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    159.0	156.0	151.0	146.0	141.0
    171.0	162.0	151.0	140.0	129.0
    182.0	168.0	154.0	143.0	130.0
    */
    CHECK_CLOSE(159, hgt[1](0,0), 0.0001);
    CHECK_CLOSE(156, hgt[1](0,1), 0.0001);
    CHECK_CLOSE(151, hgt[1](0,2), 0.0001);
    CHECK_CLOSE(146, hgt[1](0,3), 0.0001);
    CHECK_CLOSE(141, hgt[1](0,4), 0.0001);
    CHECK_CLOSE(171, hgt[1](1,0), 0.0001);
    CHECK_CLOSE(182, hgt[1](2,0), 0.0001);
    CHECK_CLOSE(130, hgt[1](2,4), 0.0001);

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    121.0	104.0	98.0	102.0	114.0
    141.0	125.0	115.0	112.0	116.0
    158.0	147.0	139.0	133.0	131.0
    */
    CHECK_CLOSE(121, hgt[11](0,0), 0.0001);
    CHECK_CLOSE(104, hgt[11](0,1), 0.0001);
    CHECK_CLOSE(98, hgt[11](0,2), 0.0001);
    CHECK_CLOSE(102, hgt[11](0,3), 0.0001);
    CHECK_CLOSE(114, hgt[11](0,4), 0.0001);
    CHECK_CLOSE(141, hgt[11](1,0), 0.0001);
    CHECK_CLOSE(158, hgt[11](2,0), 0.0001);
    CHECK_CLOSE(131, hgt[11](2,4), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    -2.0	12.0	24.0	33.0	40.0
    -1.0	11.0	24.0	36.0	49.0
    6.0	    16.0	29.0	46.0	65.0
    */
    CHECK_CLOSE(-2, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(12, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(24, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(33, hgt[40](0,3), 0.0001);
    CHECK_CLOSE(40, hgt[40](0,4), 0.0001);
    CHECK_CLOSE(-1, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(6, hgt[40](2,0), 0.0001);
    CHECK_CLOSE(65, hgt[40](2,4), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadComposite)
{
    double Umin = -10;
    double Uwidth = 15;
    double Vmin = 35;
    double Vwidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    187.0	191.0	189.0	183.0   |   175.0	171.0	171.0
    203.0	205.0	201.0	193.0   |   185.0	182.0	184.0
    208.0	209.0	206.0	200.0   |   195.0	196.0	199.0
    */
    CHECK_CLOSE(187, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(191, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(189, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(183, hgt[0](0,3), 0.0001);
    CHECK_CLOSE(175, hgt[0](0,4), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,5), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,6), 0.0001);
    CHECK_CLOSE(203, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(208, hgt[0](2,0), 0.0001);
    CHECK_CLOSE(199, hgt[0](2,6), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    188.0	192.0	188.0	180.0    |   170.0	163.0	160.0
    198.0	198.0	194.0	186.0    |   179.0	175.0	175.0
    202.0	202.0	200.0	195.0    |   190.0	189.0	191.0
    */
    CHECK_CLOSE(188, hgt[1](0,0), 0.0001);
    CHECK_CLOSE(192, hgt[1](0,1), 0.0001);
    CHECK_CLOSE(188, hgt[1](0,2), 0.0001);
    CHECK_CLOSE(180, hgt[1](0,3), 0.0001);
    CHECK_CLOSE(170, hgt[1](0,4), 0.0001);
    CHECK_CLOSE(163, hgt[1](0,5), 0.0001);
    CHECK_CLOSE(160, hgt[1](0,6), 0.0001);
    CHECK_CLOSE(198, hgt[1](1,0), 0.0001);
    CHECK_CLOSE(202, hgt[1](2,0), 0.0001);
    CHECK_CLOSE(191, hgt[1](2,6), 0.0001);

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    227.0	223.0	217.0	211.0    |   203.0	189.0	169.0
    217.0	209.0	203.0	200.0    |   198.0	192.0	179.0
    195.0	187.0	186.0	189.0    |   193.0	191.0	183.0
    */
    CHECK_CLOSE(227, hgt[11](0,0), 0.0001);
    CHECK_CLOSE(223, hgt[11](0,1), 0.0001);
    CHECK_CLOSE(217, hgt[11](0,2), 0.0001);
    CHECK_CLOSE(211, hgt[11](0,3), 0.0001);
    CHECK_CLOSE(203, hgt[11](0,4), 0.0001);
    CHECK_CLOSE(189, hgt[11](0,5), 0.0001);
    CHECK_CLOSE(169, hgt[11](0,6), 0.0001);
    CHECK_CLOSE(217, hgt[11](1,0), 0.0001);
    CHECK_CLOSE(195, hgt[11](2,0), 0.0001);
    CHECK_CLOSE(183, hgt[11](2,6), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    45.0	25.0	16.0	11.0    |   4.0	    -5.0	-12.0
    33.0	15.0	6.0	    2.0     |   -2.0	-7.0	-11.0
    53.0	37.0	25.0	15.0    |   7.0	    1.0	    -2.0
    */
    CHECK_CLOSE(45, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(25, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(16, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(11, hgt[40](0,3), 0.0001);
    CHECK_CLOSE(4, hgt[40](0,4), 0.0001);
    CHECK_CLOSE(-5, hgt[40](0,5), 0.0001);
    CHECK_CLOSE(-12, hgt[40](0,6), 0.0001);
    CHECK_CLOSE(33, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(53, hgt[40](2,0), 0.0001);
    CHECK_CLOSE(-2, hgt[40](2,6), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadBorderLeft)
{
    double Umin = 0;
    double Uwidth = 5;
    double Vmin = 35;
    double Vwidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   175.0	171.0	171.0
    |   185.0	182.0	184.0
    |   195.0	196.0	199.0
    */
    CHECK_CLOSE(175, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(185, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(195, hgt[0](2,0), 0.0001);
    CHECK_CLOSE(199, hgt[0](2,2), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   170.0	163.0	160.0
    |   179.0	175.0	175.0
    |   190.0	189.0	191.0
    */
    CHECK_CLOSE(170, hgt[1](0,0), 0.0001);
    CHECK_CLOSE(163, hgt[1](0,1), 0.0001);
    CHECK_CLOSE(160, hgt[1](0,2), 0.0001);
    CHECK_CLOSE(179, hgt[1](1,0), 0.0001);
    CHECK_CLOSE(190, hgt[1](2,0), 0.0001);
    CHECK_CLOSE(191, hgt[1](2,2), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    |   4.0	    -5.0	-12.0
    |   -2.0	-7.0	-11.0
    |   7.0	    1.0	    -2.0
    */
    CHECK_CLOSE(4, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(-5, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(-12, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(-2, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(7, hgt[40](2,0), 0.0001);
    CHECK_CLOSE(-2, hgt[40](2,2), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadBorderLeftOn720)
{
    double Umin = 360;
    double Uwidth = 5;
    double Vmin = 35;
    double Vwidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |   175.0	171.0	171.0
    |   185.0	182.0	184.0
    |   195.0	196.0	199.0
    */
    CHECK_CLOSE(175, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(185, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(195, hgt[0](2,0), 0.0001);
    CHECK_CLOSE(199, hgt[0](2,2), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    |   170.0	163.0	160.0
    |   179.0	175.0	175.0
    |   190.0	189.0	191.0
    */
    CHECK_CLOSE(170, hgt[1](0,0), 0.0001);
    CHECK_CLOSE(163, hgt[1](0,1), 0.0001);
    CHECK_CLOSE(160, hgt[1](0,2), 0.0001);
    CHECK_CLOSE(179, hgt[1](1,0), 0.0001);
    CHECK_CLOSE(190, hgt[1](2,0), 0.0001);
    CHECK_CLOSE(191, hgt[1](2,2), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    |   4.0	    -5.0	-12.0
    |   -2.0	-7.0	-11.0
    |   7.0	    1.0	    -2.0
    */
    CHECK_CLOSE(4, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(-5, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(-12, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(-2, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(7, hgt[40](2,0), 0.0001);
    CHECK_CLOSE(-2, hgt[40](2,2), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadBorderRight)
{
    double Umin = 350;
    double Uwidth = 10;
    double Vmin = 35;
    double Vwidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    187.0	191.0	189.0	183.0   |   175.0
    203.0	205.0	201.0	193.0   |   185.0
    208.0	209.0	206.0	200.0   |   195.0
    */
    CHECK_CLOSE(187, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(191, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(189, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(183, hgt[0](0,3), 0.0001);
    CHECK_CLOSE(175, hgt[0](0,4), 0.0001);
    CHECK_CLOSE(203, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(208, hgt[0](2,0), 0.0001);
    CHECK_CLOSE(195, hgt[0](2,4), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    188.0	192.0	188.0	180.0    |   170.0
    198.0	198.0	194.0	186.0    |   179.0
    202.0	202.0	200.0	195.0    |   190.0
    */
    CHECK_CLOSE(188, hgt[1](0,0), 0.0001);
    CHECK_CLOSE(192, hgt[1](0,1), 0.0001);
    CHECK_CLOSE(188, hgt[1](0,2), 0.0001);
    CHECK_CLOSE(180, hgt[1](0,3), 0.0001);
    CHECK_CLOSE(170, hgt[1](0,4), 0.0001);
    CHECK_CLOSE(198, hgt[1](1,0), 0.0001);
    CHECK_CLOSE(202, hgt[1](2,0), 0.0001);
    CHECK_CLOSE(190, hgt[1](2,4), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    45.0	25.0	16.0	11.0    |   4.0
    33.0	15.0	6.0	    2.0     |   -2.0
    53.0	37.0	25.0	15.0    |   7.0
    */
    CHECK_CLOSE(45, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(25, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(16, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(11, hgt[40](0,3), 0.0001);
    CHECK_CLOSE(4, hgt[40](0,4), 0.0001);
    CHECK_CLOSE(33, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(53, hgt[40](2,0), 0.0001);
    CHECK_CLOSE(7, hgt[40](2,4), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadCompositeStepLon)
{
    double Umin = -10;
    double Uwidth = 15;
    double Vmin = 35;
    double Vwidth = 5;
    double steplon = 5;
    double steplat = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, steplon, Vmin, Vwidth, steplat, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    187.0	189.0   |   175.0	171.0
    203.0	201.0   |   185.0	184.0
    208.0	206.0   |   195.0	199.0
    subset of :
    187.0	191.0	189.0	183.0   |   175.0	171.0	171.0
    203.0	205.0	201.0	193.0   |   185.0	182.0	184.0
    208.0	209.0	206.0	200.0   |   195.0	196.0	199.0
    */
    CHECK_CLOSE(187, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(189, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(175, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,3), 0.0001);
    CHECK_CLOSE(203, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(208, hgt[0](2,0), 0.0001);
    CHECK_CLOSE(199, hgt[0](2,3), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    188.0	188.0	|   170.0	160.0
    198.0	194.0	|   179.0	175.0
    202.0	200.0	|   190.0	191.0
    */
    CHECK_CLOSE(188, hgt[1](0,0), 0.0001);
    CHECK_CLOSE(188, hgt[1](0,1), 0.0001);
    CHECK_CLOSE(170, hgt[1](0,2), 0.0001);
    CHECK_CLOSE(160, hgt[1](0,3), 0.0001);
    CHECK_CLOSE(198, hgt[1](1,0), 0.0001);
    CHECK_CLOSE(202, hgt[1](2,0), 0.0001);
    CHECK_CLOSE(191, hgt[1](2,3), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    45.0	16.0	|   4.0	    -12.0
    33.0	6.0	    |   -2.0	-11.0
    53.0	25.0	|   7.0	    -2.0
    */
    CHECK_CLOSE(45, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(16, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(4, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(-12, hgt[40](0,3), 0.0001);
    CHECK_CLOSE(33, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(53, hgt[40](2,0), 0.0001);
    CHECK_CLOSE(-2, hgt[40](2,3), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadCompositeStepLonMoved)
{
    double Umin = -7.5;
    double Uwidth = 10;
    double Vmin = 35;
    double Vwidth = 5;
    double steplon = 5;
    double steplat = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, steplon, Vmin, Vwidth, steplat, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    191.0	183.0   |   171.0
    205.0	193.0   |   182.0
    209.0	200.0   |   196.0
    */
    CHECK_CLOSE(191, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(183, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(205, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(209, hgt[0](2,0), 0.0001);
    CHECK_CLOSE(196, hgt[0](2,2), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    25.0	11.0    |   -5.0
    15.0	2.0     |   -7.0
    37.0	15.0    |    1.0
    */
    CHECK_CLOSE(25, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(11, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(-5, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(15, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(37, hgt[40](2,0), 0.0001);
    CHECK_CLOSE(1, hgt[40](2,2), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadCompositeStepLonLat)
{
    double Umin = -10;
    double Uwidth = 15;
    double Vmin = 35;
    double Vwidth = 5;
    double steplon = 5;
    double steplat = 5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, steplon, Vmin, Vwidth, steplat, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    187.0	189.0   |   175.0	171.0
    208.0	206.0   |   195.0	199.0
    */
    CHECK_CLOSE(187, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(189, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(175, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,3), 0.0001);
    CHECK_CLOSE(208, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(199, hgt[0](1,3), 0.0001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    188.0	188.0	|   170.0	160.0
    202.0	200.0	|   190.0	191.0
    */
    CHECK_CLOSE(188, hgt[1](0,0), 0.0001);
    CHECK_CLOSE(188, hgt[1](0,1), 0.0001);
    CHECK_CLOSE(170, hgt[1](0,2), 0.0001);
    CHECK_CLOSE(160, hgt[1](0,3), 0.0001);
    CHECK_CLOSE(202, hgt[1](1,0), 0.0001);
    CHECK_CLOSE(191, hgt[1](1,3), 0.0001);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    45.0	16.0	|   4.0	    -12.0
    53.0	25.0	|   7.0	    -2.0
    */
    CHECK_CLOSE(45, hgt[40](0,0), 0.0001);
    CHECK_CLOSE(16, hgt[40](0,1), 0.0001);
    CHECK_CLOSE(4, hgt[40](0,2), 0.0001);
    CHECK_CLOSE(-12, hgt[40](0,3), 0.0001);
    CHECK_CLOSE(53, hgt[40](1,0), 0.0001);
    CHECK_CLOSE(-2, hgt[40](1,3), 0.0001);

    wxDELETE(predictor);
}

TEST(LoadCompositeStepLonLatTime)
{
    double Umin = -10;
    double Uwidth = 15;
    double Vmin = 35;
    double Vwidth = 5;
    double steplon = 5;
    double steplat = 5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, steplon, Vmin, Vwidth, steplat, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 24;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    187.0	189.0   |   175.0	171.0
    208.0	206.0   |   195.0	199.0
    */
    CHECK_CLOSE(187, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(189, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(175, hgt[0](0,2), 0.0001);
    CHECK_CLOSE(171, hgt[0](0,3), 0.0001);
    CHECK_CLOSE(208, hgt[0](1,0), 0.0001);
    CHECK_CLOSE(199, hgt[0](1,3), 0.0001);

    /* Values time step 10 (horizontal=Lon, vertical=Lat)
    45.0	16.0	|   4.0	    -12.0
    53.0	25.0	|   7.0	    -2.0
    */
    CHECK_CLOSE(45, hgt[10](0,0), 0.0001);
    CHECK_CLOSE(16, hgt[10](0,1), 0.0001);
    CHECK_CLOSE(4, hgt[10](0,2), 0.0001);
    CHECK_CLOSE(-12, hgt[10](0,3), 0.0001);
    CHECK_CLOSE(53, hgt[10](1,0), 0.0001);
    CHECK_CLOSE(-2, hgt[10](1,3), 0.0001);

    wxDELETE(predictor);
}

TEST(SetData)
{
    double Umin = -10;
    double Uwidth = 15;
    double Vmin = 35;
    double Vwidth = 5;
    double steplon = 5;
    double steplat = 5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, steplon, Vmin, Vwidth, steplat, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,5,00,00);
    double timestephours = 24;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->Load(&geoarea, timearray);

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

    wxDELETE(predictor);
}

}

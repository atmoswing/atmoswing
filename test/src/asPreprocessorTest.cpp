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

#include "wx/filename.h"

#include "include_tests.h"
#include "asDataPredictorArchive.h"
#include "asPreprocessor.h"
#include "asGeoAreaCompositeRegularGrid.h"
#include "asTimeArray.h"

#include "gtest/gtest.h"


TEST(Preprocessor, Gradients)
{
	wxPrintf("Testing the preprocessor...\n");
	
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", false);

    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 35;
    double Ywidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    CHECK_CLOSE(10, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(20, geoarea.GetXmax(), 0.01);
    CHECK_CLOSE(35, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(40, geoarea.GetYmax(), 0.01);
    CHECK_CLOSE(5, geoarea.GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(3, geoarea.GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(2.5, geoarea.GetXstep(), 0.01);
    CHECK_CLOSE(2.5, geoarea.GetYstep(), 0.01);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_hgt_%d.nc");
    predictor->Load(&geoarea, timearray);

    ASSERT_EQ(5, predictor->GetLonPtsnb());
    ASSERT_EQ(3, predictor->GetLatPtsnb());
    VArray2DFloat arrayData = predictor->GetData();
    CHECK_CLOSE(176.0, arrayData[0](0,0), 0.01);

	std::vector < asDataPredictorArchive* > vdata;
    vdata.push_back(predictor);

    wxString method = "Gradients";
    asDataPredictorArchive* gradients = new asDataPredictorArchive(*predictor);
    asPreprocessor::Preprocess(vdata, method, gradients);

    VArray2DFloat hgt = gradients->GetData();

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	175.0	170.0	162.0	151.0
    185.0	180.0	173.0	162.0	144.0
    193.0	183.0	174.0	163.0	143.0

    Gradients
    0	9
    1	5
    2	3
    3	0
    4	-7
    5	8
    6	3
    7	1
    8	1
    9	-1
    10	-1
    11	-5
    12	-8
    13	-11
    14	-5
    15	-7
    16	-11
    17	-18
    18	-10
    19	-9
    20	-11
    21	-20
    */
    /* Alignement changed
    CHECK_CLOSE(9, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(5, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(-7, hgt[0](0,4), 0.0001);
    CHECK_CLOSE(8, hgt[0](0,5), 0.0001);
    CHECK_CLOSE(-1, hgt[0](0,10), 0.0001);
    CHECK_CLOSE(-5, hgt[0](0,14), 0.0001);
    CHECK_CLOSE(-18, hgt[0](0,17), 0.0001);
    CHECK_CLOSE(-20, hgt[0](0,21), 0.0001);
    */
    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    121.0	104.0	98.0	102.0	114.0
    141.0	125.0	115.0	112.0	116.0
    158.0	147.0	139.0	133.0	131.0

    Gradients
    0	20
    1	21
    2	17
    3	10
    4	2
    5	17
    6	22
    7	24
    8	21
    9	15
    10	-17
    11	-6
    12	4
    13	12
    14	-16
    15	-10
    16	-3
    17	4
    18	-11
    19	-8
    20	-6
    21	-2
    */
    /*
    CHECK_CLOSE(20, hgt[11](0,0), 0.0001);
    CHECK_CLOSE(21, hgt[11](0,1), 0.0001);
    CHECK_CLOSE(17, hgt[11](0,5), 0.0001);
    CHECK_CLOSE(15, hgt[11](0,9), 0.0001);
    CHECK_CLOSE(-17, hgt[11](0,10), 0.0001);
    CHECK_CLOSE(12, hgt[11](0,13), 0.0001);
    CHECK_CLOSE(-16, hgt[11](0,14), 0.0001);
    CHECK_CLOSE(-2, hgt[11](0,21), 0.0001);
    */
    wxDELETE(gradients);
    wxDELETE(predictor);
}

TEST(Preprocessor, GradientsMultithreading)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);

    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 35;
    double Ywidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    CHECK_CLOSE(10, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(20, geoarea.GetXmax(), 0.01);
    CHECK_CLOSE(35, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(40, geoarea.GetYmax(), 0.01);
    CHECK_CLOSE(5, geoarea.GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(3, geoarea.GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(2.5, geoarea.GetXstep(), 0.01);
    CHECK_CLOSE(2.5, geoarea.GetYstep(), 0.01);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt", predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_hgt_%d.nc");
    predictor->Load(&geoarea, timearray);

    ASSERT_EQ(5, predictor->GetLonPtsnb());
    ASSERT_EQ(3, predictor->GetLatPtsnb());
    VArray2DFloat arrayData = predictor->GetData();
    CHECK_CLOSE(176.0, arrayData[0](0,0), 0.01);

	std::vector < asDataPredictorArchive* > vdata;
    vdata.push_back(predictor);

    wxString method = "Gradients";
    asDataPredictorArchive* gradients = new asDataPredictorArchive(*predictor);
    asPreprocessor::Preprocess(vdata, method, gradients);

    VArray2DFloat hgt = gradients->GetData();

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    176.0	175.0	170.0	162.0	151.0
    185.0	180.0	173.0	162.0	144.0
    193.0	183.0	174.0	163.0	143.0

    Gradients
    0	9
    1	5
    2	3
    3	0
    4	-7
    5	8
    6	3
    7	1
    8	1
    9	-1
    10	-1
    11	-5
    12	-8
    13	-11
    14	-5
    15	-7
    16	-11
    17	-18
    18	-10
    19	-9
    20	-11
    21	-20
    */
    /*
    CHECK_CLOSE(9, hgt[0](0,0), 0.0001);
    CHECK_CLOSE(5, hgt[0](0,1), 0.0001);
    CHECK_CLOSE(-7, hgt[0](0,4), 0.0001);
    CHECK_CLOSE(8, hgt[0](0,5), 0.0001);
    CHECK_CLOSE(-1, hgt[0](0,10), 0.0001);
    CHECK_CLOSE(-5, hgt[0](0,14), 0.0001);
    CHECK_CLOSE(-18, hgt[0](0,17), 0.0001);
    CHECK_CLOSE(-20, hgt[0](0,21), 0.0001);
    */
    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    121.0	104.0	98.0	102.0	114.0
    141.0	125.0	115.0	112.0	116.0
    158.0	147.0	139.0	133.0	131.0

    Gradients
    0	20
    1	21
    2	17
    3	10
    4	2
    5	17
    6	22
    7	24
    8	21
    9	15
    10	-17
    11	-6
    12	4
    13	12
    14	-16
    15	-10
    16	-3
    17	4
    18	-11
    19	-8
    20	-6
    21	-2
    */
    /*
    CHECK_CLOSE(20, hgt[11](0,0), 0.0001);
    CHECK_CLOSE(21, hgt[11](0,1), 0.0001);
    CHECK_CLOSE(17, hgt[11](0,5), 0.0001);
    CHECK_CLOSE(15, hgt[11](0,9), 0.0001);
    CHECK_CLOSE(-17, hgt[11](0,10), 0.0001);
    CHECK_CLOSE(12, hgt[11](0,13), 0.0001);
    CHECK_CLOSE(-16, hgt[11](0,14), 0.0001);
    CHECK_CLOSE(-2, hgt[11](0,21), 0.0001);
    */
    wxDELETE(gradients);
    wxDELETE(predictor);
}

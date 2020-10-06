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

#include "asAreaCompRegGrid.h"
#include "asPredictor.h"
#include "asPreprocessor.h"
#include "asTimeArray.h"
#include "wx/filename.h"

TEST(Preprocessor, Gradients) {
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", false);

    double xMin = 10;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 5;
    double step = 2.5;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2003/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    EXPECT_EQ(5, area.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area.GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(2.5, area.GetXstep());
    EXPECT_DOUBLE_EQ(2.5, area.GetYstep());
    EXPECT_EQ(5, predictor->GetLonPtsnb());
    EXPECT_EQ(3, predictor->GetLatPtsnb());
    vva2f arrayData = predictor->GetData();
    EXPECT_FLOAT_EQ(176.0, arrayData[0][0](0, 0));

    std::vector<asPredictor *> vdata;
    vdata.push_back(predictor);

    wxString method = "SimpleGradients";
    asPredictor *gradients = new asPredictor(*predictor);
    asPreprocessor::Preprocess(vdata, method, gradients);

    vva2f grads = gradients->GetData();

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
    10 (0)
    11 (0)
    12 (0)
    13 (0)
    14 (0)
    15	-1
    16	-5
    17	-8
    18	-11
    19 (0)
    20	-5
    21	-7
    22	-11
    23	-18
    24 (0)
    25	-10
    26	-9
    27	-11
    28	-20
    */

    EXPECT_DOUBLE_EQ(9, grads[0][0](0, 0));
    EXPECT_DOUBLE_EQ(5, grads[0][0](0, 1));
    EXPECT_DOUBLE_EQ(3, grads[0][0](0, 2));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 3));
    EXPECT_DOUBLE_EQ(-7, grads[0][0](0, 4));
    EXPECT_DOUBLE_EQ(8, grads[0][0](0, 5));
    EXPECT_DOUBLE_EQ(3, grads[0][0](0, 6));
    EXPECT_DOUBLE_EQ(1, grads[0][0](0, 7));
    EXPECT_DOUBLE_EQ(1, grads[0][0](0, 8));
    EXPECT_DOUBLE_EQ(-1, grads[0][0](0, 9));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 10));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 11));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 12));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 13));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 14));
    EXPECT_DOUBLE_EQ(-1, grads[0][0](0, 15));
    EXPECT_DOUBLE_EQ(-5, grads[0][0](0, 16));
    EXPECT_DOUBLE_EQ(-8, grads[0][0](0, 17));
    EXPECT_DOUBLE_EQ(-11, grads[0][0](0, 18));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 19));
    EXPECT_DOUBLE_EQ(-5, grads[0][0](0, 20));
    EXPECT_DOUBLE_EQ(-7, grads[0][0](0, 21));
    EXPECT_DOUBLE_EQ(-11, grads[0][0](0, 22));
    EXPECT_DOUBLE_EQ(-18, grads[0][0](0, 23));
    EXPECT_DOUBLE_EQ(0, grads[0][0](0, 24));
    EXPECT_DOUBLE_EQ(-10, grads[0][0](0, 25));
    EXPECT_DOUBLE_EQ(-9, grads[0][0](0, 26));
    EXPECT_DOUBLE_EQ(-11, grads[0][0](0, 27));
    EXPECT_DOUBLE_EQ(-20, grads[0][0](0, 28));

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
    10 (0)
    11 (0)
    12 (0)
    13 (0)
    14 (0)
    15	-17
    16	-6
    17	4
    18	12
    19 (0)
    20	-16
    21	-10
    22	-3
    23	4
    24 (0)
    25	-11
    26	-8
    27	-6
    28	-2
    */

    EXPECT_DOUBLE_EQ(20, grads[11][0](0, 0));
    EXPECT_DOUBLE_EQ(21, grads[11][0](0, 1));
    EXPECT_DOUBLE_EQ(17, grads[11][0](0, 5));
    EXPECT_DOUBLE_EQ(15, grads[11][0](0, 9));
    EXPECT_DOUBLE_EQ(-17, grads[11][0](0, 15));
    EXPECT_DOUBLE_EQ(12, grads[11][0](0, 18));
    EXPECT_DOUBLE_EQ(-16, grads[11][0](0, 20));
    EXPECT_DOUBLE_EQ(-2, grads[11][0](0, 28));

    wxDELETE(gradients);
    wxDELETE(predictor);
}

TEST(Preprocessor, GradientsMultithreading) {
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", true);

    double xMin = 10;
    double xWidth = 10;
    double yMin = 35;
    double yWidth = 5;
    double step = 2.5;
    float level = 1000;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timeStep = 6;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2003/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_R1", "pressure/hgt", predictorDataDir);

    ASSERT_TRUE(predictor->Load(&area, timearray, level));

    EXPECT_EQ(5, area.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area.GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(2.5, area.GetXstep());
    EXPECT_DOUBLE_EQ(2.5, area.GetYstep());
    EXPECT_EQ(5, predictor->GetLonPtsnb());
    EXPECT_EQ(3, predictor->GetLatPtsnb());
    vva2f arrayData = predictor->GetData();
    EXPECT_FLOAT_EQ(176.0, arrayData[0][0](0, 0));

    std::vector<asPredictor *> vdata;
    vdata.push_back(predictor);

    wxString method = "SimpleGradients";
    asPredictor *gradients = new asPredictor(*predictor);
    asPreprocessor::Preprocess(vdata, method, gradients);

    vva2f hgt = gradients->GetData();

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
    10 (0)
    11 (0)
    12 (0)
    13 (0)
    14 (0)
    15	-1
    16	-5
    17	-8
    18	-11
    19 (0)
    20	-5
    21	-7
    22	-11
    23	-18
    24 (0)
    25	-10
    26	-9
    27	-11
    28	-20
    */

    EXPECT_DOUBLE_EQ(9, hgt[0][0](0, 0));
    EXPECT_DOUBLE_EQ(5, hgt[0][0](0, 1));
    EXPECT_DOUBLE_EQ(-7, hgt[0][0](0, 4));
    EXPECT_DOUBLE_EQ(8, hgt[0][0](0, 5));
    EXPECT_DOUBLE_EQ(-1, hgt[0][0](0, 15));
    EXPECT_DOUBLE_EQ(-5, hgt[0][0](0, 20));
    EXPECT_DOUBLE_EQ(-18, hgt[0][0](0, 23));
    EXPECT_DOUBLE_EQ(-20, hgt[0][0](0, 28));

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
    10 (0)
    11 (0)
    12 (0)
    13 (0)
    14 (0)
    15	-17
    16	-6
    17	4
    18	12
    19 (0)
    20	-16
    21	-10
    22	-3
    23	4
    24 (0)
    25	-11
    26	-8
    27	-6
    28	-2
    */

    EXPECT_DOUBLE_EQ(20, hgt[11][0](0, 0));
    EXPECT_DOUBLE_EQ(21, hgt[11][0](0, 1));
    EXPECT_DOUBLE_EQ(17, hgt[11][0](0, 5));
    EXPECT_DOUBLE_EQ(15, hgt[11][0](0, 9));
    EXPECT_DOUBLE_EQ(-17, hgt[11][0](0, 15));
    EXPECT_DOUBLE_EQ(12, hgt[11][0](0, 18));
    EXPECT_DOUBLE_EQ(-16, hgt[11][0](0, 20));
    EXPECT_DOUBLE_EQ(-2, hgt[11][0](0, 28));

    wxDELETE(gradients);
    wxDELETE(predictor);
}

TEST(Preprocessor, Addition) {
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", false);

    asAreaCompGrid *area = asAreaCompGrid::GetInstance(0, 5, 0, 60, 3, 0);

    asTimeArray timearray1(asTime::GetMJD(1960, 1, 1, 00), asTime::GetMJD(1960, 1, 5, 00), 24, asTimeArray::Simple);
    timearray1.Init();
    asTimeArray timearray2(asTime::GetMJD(1960, 1, 1, 06), asTime::GetMJD(1960, 1, 5, 06), 24, asTimeArray::Simple);
    timearray2.Init();
    asTimeArray timearray3(asTime::GetMJD(1960, 1, 1, 12), asTime::GetMJD(1960, 1, 5, 12), 24, asTimeArray::Simple);
    timearray3.Init();

    wxString dir = wxFileName::GetCwd();
    dir.Append("/files/data-ncep-r1/v2003/");

    asPredictor *predictor1 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);
    asPredictor *predictor2 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);
    asPredictor *predictor3 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);

    ASSERT_TRUE(predictor1->Load(area, timearray1, 0));
    EXPECT_EQ(5, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area->GetYaxisCompositePtsnb(0));
    ASSERT_TRUE(predictor2->Load(area, timearray2, 0));
    EXPECT_EQ(5, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area->GetYaxisCompositePtsnb(0));
    ASSERT_TRUE(predictor3->Load(area, timearray3, 0));
    EXPECT_EQ(5, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area->GetYaxisCompositePtsnb(0));

    EXPECT_EQ(5, predictor1->GetLonPtsnb());
    EXPECT_EQ(3, predictor1->GetLatPtsnb());

    std::vector<asPredictor *> vdata;
    vdata.push_back(predictor1);
    vdata.push_back(predictor2);
    vdata.push_back(predictor3);

    wxString method = "Addition";
    asPredictor *addition = new asPredictor(*predictor1);
    asPreprocessor::Preprocess(vdata, method, addition);

    vva2f adds = addition->GetData();

    /* Values day 1, 00h
    278.3	279.2	279.9	279.9	278.9
    280.1	280.6	280.3	278.9	276.4
    280.9	281.0	280.2	278.8	274.7

    Values day 1, 06h
    279.2	279.7	279.9	279.5	278.2
    280.5	280.7	279.9	278.3	275.4
    281.1	281.1	280.2	278.5	274.3

    Values day 1, 12h
    280.1	280.6	280.6	279.6	277.7
    280.5	280.1	278.9	276.8	272.8
    280.9	280.4	278.9	276.1	273.7

    Sum
    837.6	839.5	840.4	839	    834.8
    841.1	841.4	839.1	834	    824.6
    842.9	842.5	839.3	833.4	822.7
    */

    EXPECT_NEAR(837.6, adds[0][0](0, 0), 0.05);
    EXPECT_NEAR(839.5, adds[0][0](0, 1), 0.05);
    EXPECT_NEAR(840.4, adds[0][0](0, 2), 0.05);
    EXPECT_NEAR(839.0, adds[0][0](0, 3), 0.05);
    EXPECT_NEAR(834.8, adds[0][0](0, 4), 0.05);
    EXPECT_NEAR(841.1, adds[0][0](1, 0), 0.05);
    EXPECT_NEAR(842.9, adds[0][0](2, 0), 0.05);
    EXPECT_NEAR(822.7, adds[0][0](2, 4), 0.05);

    /* Values day 5, 00h
    279.7	280.5	280.9	280.5	279.3
    280.5	280.5	279.6	277.9	275.2
    280.7	280.3	279.1	276.6	273.9

    Values day 5, 06h
    279.6	280.2	280.2	279.4	278.0
    280.7	280.4	279.2	277.2	274.0
    280.5	280.0	278.7	276.0	273.7

    Values day 5, 12h
    279.0	279.7	279.9	279.2	277.8
    280.0	279.8	278.6	276.6	273.2
    279.9	279.3	277.9	274.8	273.1

    Sum
    838.3	840.4	841	    839.1	835.1
    841.2	840.7	837.4	831.7	822.4
    841.1	839.6	835.7	827.4	820.7
    */

    EXPECT_NEAR(838.3, adds[4][0](0, 0), 0.05);
    EXPECT_NEAR(840.4, adds[4][0](0, 1), 0.05);
    EXPECT_NEAR(841.0, adds[4][0](0, 2), 0.05);
    EXPECT_NEAR(839.1, adds[4][0](0, 3), 0.05);
    EXPECT_NEAR(835.1, adds[4][0](0, 4), 0.05);
    EXPECT_NEAR(841.2, adds[4][0](1, 0), 0.05);
    EXPECT_NEAR(841.1, adds[4][0](2, 0), 0.05);
    EXPECT_NEAR(820.7, adds[4][0](2, 4), 0.05);

    wxDELETE(area);
    wxDELETE(addition);
    wxDELETE(predictor1);
    wxDELETE(predictor2);
    wxDELETE(predictor3);
}

TEST(Preprocessor, Average) {
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", false);

    asAreaCompGrid *area = asAreaCompGrid::GetInstance(0, 5, 0, 60, 3, 0);

    asTimeArray timearray1(asTime::GetMJD(1960, 1, 1, 00), asTime::GetMJD(1960, 1, 5, 00), 24, asTimeArray::Simple);
    timearray1.Init();
    asTimeArray timearray2(asTime::GetMJD(1960, 1, 1, 06), asTime::GetMJD(1960, 1, 5, 06), 24, asTimeArray::Simple);
    timearray2.Init();
    asTimeArray timearray3(asTime::GetMJD(1960, 1, 1, 12), asTime::GetMJD(1960, 1, 5, 12), 24, asTimeArray::Simple);
    timearray3.Init();

    wxString dir = wxFileName::GetCwd();
    dir.Append("/files/data-ncep-r1/v2003/");

    asPredictor *predictor1 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);
    asPredictor *predictor2 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);
    asPredictor *predictor3 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);

    ASSERT_TRUE(predictor1->Load(area, timearray1, 0));
    ASSERT_TRUE(predictor2->Load(area, timearray2, 0));
    ASSERT_TRUE(predictor3->Load(area, timearray3, 0));

    EXPECT_EQ(5, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, predictor1->GetLonPtsnb());
    EXPECT_EQ(3, predictor1->GetLatPtsnb());

    std::vector<asPredictor *> vdata;
    vdata.push_back(predictor1);
    vdata.push_back(predictor2);
    vdata.push_back(predictor3);

    wxString method = "Average";
    asPredictor *average = new asPredictor(*predictor1);
    asPreprocessor::Preprocess(vdata, method, average);

    vva2f avg = average->GetData();

    /* Values day 1, 00h
    278.3	279.2	279.9	279.9	278.9
    280.1	280.6	280.3	278.9	276.4
    280.9	281.0	280.2	278.8	274.7

    Values day 1, 06h
    279.2	279.7	279.9	279.5	278.2
    280.5	280.7	279.9	278.3	275.4
    281.1	281.1	280.2	278.5	274.3

    Values day 1, 12h
    280.1	280.6	280.6	279.6	277.7
    280.5	280.1	278.9	276.8	272.8
    280.9	280.4	278.9	276.1	273.7

    Average
    279.2	279.8	280.1	279.7	278.3
    280.4	280.5	279.7	278.0	274.9
    281.0	280.8	279.8	277.8	274.2
    */

    EXPECT_NEAR(279.2, avg[0][0](0, 0), 0.05);
    EXPECT_NEAR(279.8, avg[0][0](0, 1), 0.05);
    EXPECT_NEAR(280.1, avg[0][0](0, 2), 0.05);
    EXPECT_NEAR(279.7, avg[0][0](0, 3), 0.05);
    EXPECT_NEAR(278.3, avg[0][0](0, 4), 0.05);
    EXPECT_NEAR(280.4, avg[0][0](1, 0), 0.05);
    EXPECT_NEAR(281.0, avg[0][0](2, 0), 0.05);
    EXPECT_NEAR(274.2, avg[0][0](2, 4), 0.05);

    /* Values day 5, 00h
    279.7	280.5	280.9	280.5	279.3
    280.5	280.5	279.6	277.9	275.2
    280.7	280.3	279.1	276.6	273.9

    Values day 5, 06h
    279.6	280.2	280.2	279.4	278.0
    280.7	280.4	279.2	277.2	274.0
    280.5	280.0	278.7	276.0	273.7

    Values day 5, 12h
    279.0	279.7	279.9	279.2	277.8
    280.0	279.8	278.6	276.6	273.2
    279.9	279.3	277.9	274.8	273.1

    Average
    279.4	280.1	280.3	279.7	278.4
    280.4	280.2	279.1	277.2	274.1
    280.4	279.9	278.6	275.8	273.6
    */

    EXPECT_NEAR(279.4, avg[4][0](0, 0), 0.05);
    EXPECT_NEAR(280.1, avg[4][0](0, 1), 0.05);
    EXPECT_NEAR(280.3, avg[4][0](0, 2), 0.05);
    EXPECT_NEAR(279.7, avg[4][0](0, 3), 0.05);
    EXPECT_NEAR(278.4, avg[4][0](0, 4), 0.05);
    EXPECT_NEAR(280.4, avg[4][0](1, 0), 0.05);
    EXPECT_NEAR(280.4, avg[4][0](2, 0), 0.05);
    EXPECT_NEAR(273.6, avg[4][0](2, 4), 0.05);

    wxDELETE(area);
    wxDELETE(average);
    wxDELETE(predictor1);
    wxDELETE(predictor2);
    wxDELETE(predictor3);
}

TEST(Preprocessor, Difference) {
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", false);

    asAreaCompGrid *area = asAreaCompGrid::GetInstance(0, 5, 0, 60, 3, 0);

    asTimeArray timearray1(asTime::GetMJD(1960, 1, 1, 00), asTime::GetMJD(1960, 1, 5, 00), 24, asTimeArray::Simple);
    timearray1.Init();
    asTimeArray timearray2(asTime::GetMJD(1960, 1, 1, 06), asTime::GetMJD(1960, 1, 5, 06), 24, asTimeArray::Simple);
    timearray2.Init();

    wxString dir = wxFileName::GetCwd();
    dir.Append("/files/data-ncep-r1/v2003/");

    asPredictor *predictor1 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);
    asPredictor *predictor2 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);

    ASSERT_TRUE(predictor1->Load(area, timearray1, 0));
    ASSERT_TRUE(predictor2->Load(area, timearray2, 0));

    EXPECT_EQ(5, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, predictor1->GetLonPtsnb());
    EXPECT_EQ(3, predictor1->GetLatPtsnb());

    std::vector<asPredictor *> vdata;
    vdata.push_back(predictor1);
    vdata.push_back(predictor2);

    wxString method = "Difference";
    asPredictor *difference = new asPredictor(*predictor1);
    asPreprocessor::Preprocess(vdata, method, difference);

    vva2f diffs = difference->GetData();

    /* Values day 1, 00h
    278.3	279.2	279.9	279.9	278.9
    280.1	280.6	280.3	278.9	276.4
    280.9	281.0	280.2	278.8	274.7

    Values day 1, 06h
    279.2	279.7	279.9	279.5	278.2
    280.5	280.7	279.9	278.3	275.4
    281.1	281.1	280.2	278.5	274.3

    Diff
    -0.9	-0.5	0	    0.4	    0.7
    -0.4	-0.1	0.4	    0.6	    1
    -0.2	-0.1	0	    0.3	    0.4
    */

    EXPECT_NEAR(-0.9, diffs[0][0](0, 0), 0.05);
    EXPECT_NEAR(-0.5, diffs[0][0](0, 1), 0.05);
    EXPECT_NEAR(0, diffs[0][0](0, 2), 0.05);
    EXPECT_NEAR(0.4, diffs[0][0](0, 3), 0.05);
    EXPECT_NEAR(0.7, diffs[0][0](0, 4), 0.05);
    EXPECT_NEAR(-0.4, diffs[0][0](1, 0), 0.05);
    EXPECT_NEAR(-0.2, diffs[0][0](2, 0), 0.05);
    EXPECT_NEAR(0.4, diffs[0][0](2, 4), 0.05);

    /* Values day 5, 00h
    279.7	280.5	280.9	280.5	279.3
    280.5	280.5	279.6	277.9	275.2
    280.7	280.3	279.1	276.6	273.9

    Values day 5, 06h
    279.6	280.2	280.2	279.4	278.0
    280.7	280.4	279.2	277.2	274.0
    280.5	280.0	278.7	276.0	273.7

    Diff
    0.1	    0.3	    0.7	    1.1	    1.3
    -0.2	0.1	    0.4	    0.7	    1.2
    0.2	    0.3	    0.4	    0.6	    0.2
    */

    EXPECT_NEAR(0.1, diffs[4][0](0, 0), 0.05);
    EXPECT_NEAR(0.3, diffs[4][0](0, 1), 0.05);
    EXPECT_NEAR(0.7, diffs[4][0](0, 2), 0.05);
    EXPECT_NEAR(1.1, diffs[4][0](0, 3), 0.05);
    EXPECT_NEAR(1.3, diffs[4][0](0, 4), 0.05);
    EXPECT_NEAR(-0.2, diffs[4][0](1, 0), 0.05);
    EXPECT_NEAR(0.2, diffs[4][0](2, 0), 0.05);
    EXPECT_NEAR(0.2, diffs[4][0](2, 4), 0.05);

    wxDELETE(area);
    wxDELETE(difference);
    wxDELETE(predictor1);
    wxDELETE(predictor2);
}

TEST(Preprocessor, Multiplication) {
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", false);

    asAreaCompGrid *area = asAreaCompGrid::GetInstance(0, 5, 0, 60, 3, 0);

    asTimeArray timearray1(asTime::GetMJD(1960, 1, 1, 00), asTime::GetMJD(1960, 1, 5, 00), 24, asTimeArray::Simple);
    timearray1.Init();
    asTimeArray timearray2(asTime::GetMJD(1960, 1, 1, 06), asTime::GetMJD(1960, 1, 5, 06), 24, asTimeArray::Simple);
    timearray2.Init();

    wxString dir = wxFileName::GetCwd();
    dir.Append("/files/data-ncep-r1/v2003/");

    asPredictor *predictor1 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);
    asPredictor *predictor2 = asPredictor::GetInstance("NCEP_R1", "surface_gauss/air", dir);

    ASSERT_TRUE(predictor1->Load(area, timearray1, 0));
    ASSERT_TRUE(predictor2->Load(area, timearray2, 0));

    EXPECT_EQ(5, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(3, area->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, predictor1->GetLonPtsnb());
    EXPECT_EQ(3, predictor1->GetLatPtsnb());

    std::vector<asPredictor *> vdata;
    vdata.push_back(predictor1);
    vdata.push_back(predictor2);

    wxString method = "Multiplication";
    asPredictor *multiplication = new asPredictor(*predictor1);
    asPreprocessor::Preprocess(vdata, method, multiplication);

    vva2f multi = multiplication->GetData();

    /* Values day 1, 00h
    278.3	279.2	279.9	279.9	278.9
    280.1	280.6	280.3	278.9	276.4
    280.9	281.0	280.2	278.8	274.7

    Values day 1, 06h
    279.2	279.7	279.9	279.5	278.2
    280.5	280.7	279.9	278.3	275.4
    281.1	281.1	280.2	278.5	274.3

    Diff
    77701.36	78092.24	78344.01	78232.05	77589.98
    78568.05	78764.42	78455.97	77617.87	76120.56
    78960.99	78989.1	    78512.04	77645.8	    75350.21
    */

    EXPECT_NEAR(77701.36, multi[0][0](0, 0), 0.05);
    EXPECT_NEAR(78092.24, multi[0][0](0, 1), 0.05);
    EXPECT_NEAR(78344.01, multi[0][0](0, 2), 0.05);
    EXPECT_NEAR(78232.05, multi[0][0](0, 3), 0.05);
    EXPECT_NEAR(77589.98, multi[0][0](0, 4), 0.05);
    EXPECT_NEAR(78568.05, multi[0][0](1, 0), 0.05);
    EXPECT_NEAR(78960.99, multi[0][0](2, 0), 0.05);
    EXPECT_NEAR(75350.21, multi[0][0](2, 4), 0.05);

    /* Values day 5, 00h
    279.7	280.5	280.9	280.5	279.3
    280.5	280.5	279.6	277.9	275.2
    280.7	280.3	279.1	276.6	273.9

    Values day 5, 06h
    279.6	280.2	280.2	279.4	278.0
    280.7	280.4	279.2	277.2	274.0
    280.5	280.0	278.7	276.0	273.7

    Diff
    78204.12	78596.1	    78708.18	78371.7	    77645.4
    78736.35	78652.2	    78064.32	77033.88	75404.8
    78736.35	78484	    77785.17	76341.6	    74966.43
    */

    EXPECT_NEAR(78204.12, multi[4][0](0, 0), 0.05);
    EXPECT_NEAR(78596.1, multi[4][0](0, 1), 0.05);
    EXPECT_NEAR(78708.18, multi[4][0](0, 2), 0.05);
    EXPECT_NEAR(78371.7, multi[4][0](0, 3), 0.05);
    EXPECT_NEAR(77645.4, multi[4][0](0, 4), 0.05);
    EXPECT_NEAR(78736.35, multi[4][0](1, 0), 0.05);
    EXPECT_NEAR(78736.35, multi[4][0](2, 0), 0.05);
    EXPECT_NEAR(74966.43, multi[4][0](2, 4), 0.05);

    wxDELETE(area);
    wxDELETE(multiplication);
    wxDELETE(predictor1);
    wxDELETE(predictor2);
}
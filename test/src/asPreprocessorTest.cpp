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
#include "asDataPredictorArchive.h"
#include "asPreprocessor.h"
#include "asGeoAreaCompositeRegularGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(Preprocessor, Gradients)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Processing/AllowMultithreading", false);

    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 35;
    double Ywidth = 5;
    double step = 2.5;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    EXPECT_DOUBLE_EQ(10, geoarea.GetXmin());
    EXPECT_DOUBLE_EQ(20, geoarea.GetXmax());
    EXPECT_DOUBLE_EQ(35, geoarea.GetYmin());
    EXPECT_DOUBLE_EQ(40, geoarea.GetYmax());
    EXPECT_DOUBLE_EQ(5, geoarea.GetXaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(3, geoarea.GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(2.5, geoarea.GetXstep());
    EXPECT_DOUBLE_EQ(2.5, geoarea.GetYstep());

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_hgt_%d.nc");
    predictor->Load(&geoarea, timearray);

    EXPECT_EQ(5, predictor->GetLonPtsnb());
    EXPECT_EQ(3, predictor->GetLatPtsnb());
    VArray2DFloat arrayData = predictor->GetData();
    EXPECT_FLOAT_EQ(176.0, arrayData[0](0, 0));

    std::vector<asDataPredictorArchive *> vdata;
    vdata.push_back(predictor);

    wxString method = "Gradients";
    asDataPredictorArchive *gradients = new asDataPredictorArchive(*predictor);
    asPreprocessor::Preprocess(vdata, method, gradients);

    VArray2DFloat grads = gradients->GetData();

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

    EXPECT_DOUBLE_EQ(9, grads[0](0, 0));
    EXPECT_DOUBLE_EQ(5, grads[0](0, 1));
    EXPECT_DOUBLE_EQ(3, grads[0](0, 2));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 3));
    EXPECT_DOUBLE_EQ(-7, grads[0](0, 4));
    EXPECT_DOUBLE_EQ(8, grads[0](0, 5));
    EXPECT_DOUBLE_EQ(3, grads[0](0, 6));
    EXPECT_DOUBLE_EQ(1, grads[0](0, 7));
    EXPECT_DOUBLE_EQ(1, grads[0](0, 8));
    EXPECT_DOUBLE_EQ(-1, grads[0](0, 9));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 10));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 11));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 12));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 13));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 14));
    EXPECT_DOUBLE_EQ(-1, grads[0](0, 15));
    EXPECT_DOUBLE_EQ(-5, grads[0](0, 16));
    EXPECT_DOUBLE_EQ(-8, grads[0](0, 17));
    EXPECT_DOUBLE_EQ(-11, grads[0](0, 18));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 19));
    EXPECT_DOUBLE_EQ(-5, grads[0](0, 20));
    EXPECT_DOUBLE_EQ(-7, grads[0](0, 21));
    EXPECT_DOUBLE_EQ(-11, grads[0](0, 22));
    EXPECT_DOUBLE_EQ(-18, grads[0](0, 23));
    EXPECT_DOUBLE_EQ(0, grads[0](0, 24));
    EXPECT_DOUBLE_EQ(-10, grads[0](0, 25));
    EXPECT_DOUBLE_EQ(-9, grads[0](0, 26));
    EXPECT_DOUBLE_EQ(-11, grads[0](0, 27));
    EXPECT_DOUBLE_EQ(-20, grads[0](0, 28));

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

    EXPECT_DOUBLE_EQ(20, grads[11](0, 0));
    EXPECT_DOUBLE_EQ(21, grads[11](0, 1));
    EXPECT_DOUBLE_EQ(17, grads[11](0, 5));
    EXPECT_DOUBLE_EQ(15, grads[11](0, 9));
    EXPECT_DOUBLE_EQ(-17, grads[11](0, 15));
    EXPECT_DOUBLE_EQ(12, grads[11](0, 18));
    EXPECT_DOUBLE_EQ(-16, grads[11](0, 20));
    EXPECT_DOUBLE_EQ(-2, grads[11](0, 28));

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
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    EXPECT_DOUBLE_EQ(10, geoarea.GetXmin());
    EXPECT_DOUBLE_EQ(20, geoarea.GetXmax());
    EXPECT_DOUBLE_EQ(35, geoarea.GetYmin());
    EXPECT_DOUBLE_EQ(40, geoarea.GetYmax());
    EXPECT_DOUBLE_EQ(5, geoarea.GetXaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(3, geoarea.GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(2.5, geoarea.GetXstep());
    EXPECT_DOUBLE_EQ(2.5, geoarea.GetYstep());

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "hgt",
                                                                            predictorDataDir);

    predictor->SetFileNamePattern("NCEP_Reanalysis_v1(2003)_hgt_%d.nc");
    predictor->Load(&geoarea, timearray);

    EXPECT_EQ(5, predictor->GetLonPtsnb());
    EXPECT_EQ(3, predictor->GetLatPtsnb());
    VArray2DFloat arrayData = predictor->GetData();
    EXPECT_FLOAT_EQ(176.0, arrayData[0](0, 0));

    std::vector<asDataPredictorArchive *> vdata;
    vdata.push_back(predictor);

    wxString method = "Gradients";
    asDataPredictorArchive *gradients = new asDataPredictorArchive(*predictor);
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

    EXPECT_DOUBLE_EQ(9, hgt[0](0, 0));
    EXPECT_DOUBLE_EQ(5, hgt[0](0, 1));
    EXPECT_DOUBLE_EQ(-7, hgt[0](0, 4));
    EXPECT_DOUBLE_EQ(8, hgt[0](0, 5));
    EXPECT_DOUBLE_EQ(-1, hgt[0](0, 15));
    EXPECT_DOUBLE_EQ(-5, hgt[0](0, 20));
    EXPECT_DOUBLE_EQ(-18, hgt[0](0, 23));
    EXPECT_DOUBLE_EQ(-20, hgt[0](0, 28));

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

    EXPECT_DOUBLE_EQ(20, hgt[11](0, 0));
    EXPECT_DOUBLE_EQ(21, hgt[11](0, 1));
    EXPECT_DOUBLE_EQ(17, hgt[11](0, 5));
    EXPECT_DOUBLE_EQ(15, hgt[11](0, 9));
    EXPECT_DOUBLE_EQ(-17, hgt[11](0, 15));
    EXPECT_DOUBLE_EQ(12, hgt[11](0, 18));
    EXPECT_DOUBLE_EQ(-16, hgt[11](0, 20));
    EXPECT_DOUBLE_EQ(-2, hgt[11](0, 28));

    wxDELETE(gradients);
    wxDELETE(predictor);
}

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
#include "asDataPredictorArchive.h"
#include "asGeoAreaCompositeRegularGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(DataPredictorArchiveNoaa20Cr2cRegular, LoadEasy)
{
    double Xmin = 10;
    double Xwidth = 8;
    double Ymin = 70;
    double Ywidth = 4;
    double step = 2;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NOAA_20CR_v2c", "press/hgt",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    -104.9	-88.2	-71.4	-54.6	-37.8
    -131.4	-112.7	-92.3	-70.3	-47.4
    -145.9	-120.7	-91.6	-60.1	-28.0
    */
    EXPECT_FLOAT_EQ(-104.9, hgt[0](0, 0));
    EXPECT_FLOAT_EQ(-88.2, hgt[0](0, 1));
    EXPECT_FLOAT_EQ(-71.4, hgt[0](0, 2));
    EXPECT_FLOAT_EQ(-54.6, hgt[0](0, 3));
    EXPECT_FLOAT_EQ(-37.8, hgt[0](0, 4));
    EXPECT_FLOAT_EQ(-131.4, hgt[0](1, 0));
    EXPECT_FLOAT_EQ(-145.9, hgt[0](2, 0));
    EXPECT_FLOAT_EQ(-28.0, hgt[0](2, 4));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    -174.2	-169.1	-163.2	-156.7	-149.6
    -124.2	-119.4	-114.1	-108.2	-101.7
    -48.3	-44.2	-39.8	-34.8	-29.0
    */
    EXPECT_FLOAT_EQ(-174.2, hgt[4](0, 0));
    EXPECT_FLOAT_EQ(-169.1, hgt[4](0, 1));
    EXPECT_FLOAT_EQ(-163.2, hgt[4](0, 2));
    EXPECT_FLOAT_EQ(-156.7, hgt[4](0, 3));
    EXPECT_FLOAT_EQ(-149.6, hgt[4](0, 4));
    EXPECT_FLOAT_EQ(-124.2, hgt[4](1, 0));
    EXPECT_FLOAT_EQ(-48.3, hgt[4](2, 0));
    EXPECT_FLOAT_EQ(-29.0, hgt[4](2, 4));

    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNoaa20Cr2cRegular, LoadComposite)
{
    double Xmin = -8;
    double Xwidth = 12;
    double Ymin = 70;
    double Ywidth = 4;
    double step = 2;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NOAA_20CR_v2c", "press/hgt",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    -206.3	-206.5	-201.9	-193.4   |  -181.8	-168.1	-153.2
    -237.0	-235.2	-228.6	-218.4   |  -206.0	-192.4	-178.3
    -224.2	-227.3	-225.9	-221.2   |  -214.2	-205.6	-195.4
    */
    EXPECT_FLOAT_EQ(-206.3, hgt[0](0, 0));
    EXPECT_FLOAT_EQ(-206.5, hgt[0](0, 1));
    EXPECT_FLOAT_EQ(-201.9, hgt[0](0, 2));
    EXPECT_FLOAT_EQ(-193.4, hgt[0](0, 3));
    EXPECT_FLOAT_EQ(-181.8, hgt[0](0, 4));
    EXPECT_FLOAT_EQ(-168.1, hgt[0](0, 5));
    EXPECT_FLOAT_EQ(-153.2, hgt[0](0, 6));
    EXPECT_FLOAT_EQ(-237.0, hgt[0](1, 0));
    EXPECT_FLOAT_EQ(-224.2, hgt[0](2, 0));
    EXPECT_FLOAT_EQ(-195.4, hgt[0](2, 6));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    -149.3	-158.6	-166.6	-173.0   |  -177.8	-180.6	-181.6
    -119.7	-124.7	-128.9	-132.2   |  -134.2	-134.7	-133.8
    -65.4	-65.1	-65.1	-64.8    |  -63.8	-62.0	-59.2
    */
    EXPECT_FLOAT_EQ(-149.3, hgt[4](0, 0));
    EXPECT_FLOAT_EQ(-158.6, hgt[4](0, 1));
    EXPECT_FLOAT_EQ(-166.6, hgt[4](0, 2));
    EXPECT_FLOAT_EQ(-173.0, hgt[4](0, 3));
    EXPECT_FLOAT_EQ(-177.8, hgt[4](0, 4));
    EXPECT_FLOAT_EQ(-180.6, hgt[4](0, 5));
    EXPECT_FLOAT_EQ(-181.6, hgt[4](0, 6));
    EXPECT_FLOAT_EQ(-119.7, hgt[4](1, 0));
    EXPECT_FLOAT_EQ(-65.4, hgt[4](2, 0));
    EXPECT_FLOAT_EQ(-59.2, hgt[4](2, 6));

    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNoaa20Cr2cRegular, LoadBorderLeft)
{
    double Xmin = 0;
    double Xwidth = 4;
    double Ymin = 70;
    double Ywidth = 4;
    double step = 2;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NOAA_20CR_v2c", "press/hgt",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    |  -181.8	-168.1	-153.2
    |  -206.0	-192.4	-178.3
    |  -214.2	-205.6	-195.4
    */
    EXPECT_FLOAT_EQ(-181.8, hgt[0](0, 0));
    EXPECT_FLOAT_EQ(-168.1, hgt[0](0, 1));
    EXPECT_FLOAT_EQ(-153.2, hgt[0](0, 2));
    EXPECT_FLOAT_EQ(-206.0, hgt[0](1, 0));
    EXPECT_FLOAT_EQ(-214.2, hgt[0](2, 0));
    EXPECT_FLOAT_EQ(-195.4, hgt[0](2, 2));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    |  -177.8	-180.6	-181.6
    |  -134.2	-134.7	-133.8
    |  -63.8	-62.0	-59.2
    */
    EXPECT_FLOAT_EQ(-177.8, hgt[4](0, 0));
    EXPECT_FLOAT_EQ(-180.6, hgt[4](0, 1));
    EXPECT_FLOAT_EQ(-181.6, hgt[4](0, 2));
    EXPECT_FLOAT_EQ(-134.2, hgt[4](1, 0));
    EXPECT_FLOAT_EQ(-63.8, hgt[4](2, 0));
    EXPECT_FLOAT_EQ(-59.2, hgt[4](2, 2));

    wxDELETE(predictor);
}

TEST(DataPredictorArchiveNoaa20Cr2cRegular, LoadBorderRight)
{
    double Xmin = 352;
    double Xwidth = 8;
    double Ymin = 70;
    double Ywidth = 4;
    double step = 2;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 2, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-noaa-20crv2c/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NOAA_20CR_v2c", "press/hgt",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(&geoarea, timearray));

    VArray2DFloat hgt = predictor->GetData();
    // hgt[time](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    -206.3	-206.5	-201.9	-193.4   |  -181.8
    -237.0	-235.2	-228.6	-218.4   |  -206.0
    -224.2	-227.3	-225.9	-221.2   |  -214.2
    */
    EXPECT_FLOAT_EQ(-206.3, hgt[0](0, 0));
    EXPECT_FLOAT_EQ(-206.5, hgt[0](0, 1));
    EXPECT_FLOAT_EQ(-201.9, hgt[0](0, 2));
    EXPECT_FLOAT_EQ(-193.4, hgt[0](0, 3));
    EXPECT_FLOAT_EQ(-181.8, hgt[0](0, 4));
    EXPECT_FLOAT_EQ(-237.0, hgt[0](1, 0));
    EXPECT_FLOAT_EQ(-224.2, hgt[0](2, 0));
    EXPECT_FLOAT_EQ(-214.2, hgt[0](2, 4));

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    -149.3	-158.6	-166.6	-173.0   |  -177.8
    -119.7	-124.7	-128.9	-132.2   |  -134.2
    -65.4	-65.1	-65.1	-64.8    |  -63.8
    */
    EXPECT_FLOAT_EQ(-149.3, hgt[4](0, 0));
    EXPECT_FLOAT_EQ(-158.6, hgt[4](0, 1));
    EXPECT_FLOAT_EQ(-166.6, hgt[4](0, 2));
    EXPECT_FLOAT_EQ(-173.0, hgt[4](0, 3));
    EXPECT_FLOAT_EQ(-177.8, hgt[4](0, 4));
    EXPECT_FLOAT_EQ(-119.7, hgt[4](1, 0));
    EXPECT_FLOAT_EQ(-65.4, hgt[4](2, 0));
    EXPECT_FLOAT_EQ(-63.8, hgt[4](2, 4));

    wxDELETE(predictor);
}

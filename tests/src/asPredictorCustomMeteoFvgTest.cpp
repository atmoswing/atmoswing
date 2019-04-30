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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include <wx/filename.h>
#include "asPredictor.h"
#include "asAreaCompGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorCustomMeteoFvg, LoadSingleDay)
{
    asTimeArray dates(asTime::GetMJD(2011, 7, 18, 06), asTime::GetMJD(2011, 7, 18, 06), 6, "Simple");
    dates.Init();

    double xMin = -2;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 4;
    double step = 0.25;
    float level = 500;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-custom-meteo-fvg/");

    asPredictor *predictor = asPredictor::GetInstance("Custom_MeteoFVG_IFS", "gh_500", predictorDataDir);
    wxASSERT(predictor);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    5589.2	5587.8	5586.5	5585.3	5583.9	5583.7
    5596.8	5595.4	5594.0	5592.7	5591.3	5591.0
    5604.4	5603.0	5601.5	5600.2	5598.7	5598.4
    5612.0	5610.5	5609.0	5607.5	5606.0	5605.8
    */
    EXPECT_NEAR(5589.2, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(5587.8, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(5586.5, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(5585.3, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(5583.9, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(5583.7, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(5596.8, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(5604.4, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(5612.0, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(5605.8, hgt[0][0](3, 5), 0.5);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorCustomMeteoFvg, LoadSecondTimeStep)
{
    asTimeArray dates(asTime::GetMJD(2011, 7, 18, 12), asTime::GetMJD(2011, 7, 18, 12), 6, "Simple");
    dates.Init();

    double xMin = -2;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 4;
    double step = 0.25;
    float level = 500;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-custom-meteo-fvg/");

    asPredictor *predictor = asPredictor::GetInstance("Custom_MeteoFVG_IFS", "gh_500", predictorDataDir);
    wxASSERT(predictor);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    5624.2	5622.1	5620.1	5617.9	5615.9	5614.7
    5632.1	5629.9	5627.7	5625.6	5623.3	5622.1
    5640.1	5637.7	5635.4	5633.1	5630.8	5629.4
    5647.9	5645.6	5643.1	5640.7	5638.2	5636.8
    */
    EXPECT_NEAR(5624.2, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(5622.1, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(5620.1, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(5617.9, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(5615.9, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(5614.7, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(5632.1, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(5640.1, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(5647.9, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(5636.8, hgt[0][0](3, 5), 0.5);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorCustomMeteoFvg, LoadFullTimeArray)
{
    asTimeArray dates(asTime::GetMJD(2011, 7, 18, 6), asTime::GetMJD(2011, 7, 19, 24), 6, "Simple");
    dates.Init();

    double xMin = -2;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 4;
    double step = 0.25;
    float level = 500;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-custom-meteo-fvg/");

    asPredictor *predictor = asPredictor::GetInstance("Custom_MeteoFVG_IFS", "gh_500", predictorDataDir);
    wxASSERT(predictor);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    5589.2	5587.8	5586.5	5585.3	5583.9	5583.7
    5596.8	5595.4	5594.0	5592.7	5591.3	5591.0
    5604.4	5603.0	5601.5	5600.2	5598.7	5598.4
    5612.0	5610.5	5609.0	5607.5	5606.0	5605.8
    */
    EXPECT_NEAR(5589.2, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(5587.8, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(5586.5, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(5585.3, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(5583.9, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(5583.7, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(5596.8, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(5604.4, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(5612.0, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(5605.8, hgt[0][0](3, 5), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted:
    5624.2	5622.1	5620.1	5617.9	5615.9	5614.7
    5632.1	5629.9	5627.7	5625.6	5623.3	5622.1
    5640.1	5637.7	5635.4	5633.1	5630.8	5629.4
    5647.9	5645.6	5643.1	5640.7	5638.2	5636.8
    */
    EXPECT_NEAR(5624.2, hgt[1][0](0, 0), 0.5);
    EXPECT_NEAR(5622.1, hgt[1][0](0, 1), 0.5);
    EXPECT_NEAR(5620.1, hgt[1][0](0, 2), 0.5);
    EXPECT_NEAR(5617.9, hgt[1][0](0, 3), 0.5);
    EXPECT_NEAR(5615.9, hgt[1][0](0, 4), 0.5);
    EXPECT_NEAR(5614.7, hgt[1][0](0, 5), 0.5);
    EXPECT_NEAR(5632.1, hgt[1][0](1, 0), 0.5);
    EXPECT_NEAR(5640.1, hgt[1][0](2, 0), 0.5);
    EXPECT_NEAR(5647.9, hgt[1][0](3, 0), 0.5);
    EXPECT_NEAR(5636.8, hgt[1][0](3, 5), 0.5);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    Extracted:
    5691.9	5687.7	5683.3	5679.1	5674.8	5670.7
    5697.4	5693.1	5688.7	5684.3	5679.9	5675.8
    5702.9	5698.4	5693.9	5689.4	5684.9	5680.8
    5708.4	5703.8	5699.3	5694.7	5690.1	5685.9
    */
    EXPECT_NEAR(5691.9, hgt[7][0](0, 0), 0.5);
    EXPECT_NEAR(5687.7, hgt[7][0](0, 1), 0.5);
    EXPECT_NEAR(5683.3, hgt[7][0](0, 2), 0.5);
    EXPECT_NEAR(5679.1, hgt[7][0](0, 3), 0.5);
    EXPECT_NEAR(5674.8, hgt[7][0](0, 4), 0.5);
    EXPECT_NEAR(5670.7, hgt[7][0](0, 5), 0.5);
    EXPECT_NEAR(5697.4, hgt[7][0](1, 0), 0.5);
    EXPECT_NEAR(5702.9, hgt[7][0](2, 0), 0.5);
    EXPECT_NEAR(5708.4, hgt[7][0](3, 0), 0.5);
    EXPECT_NEAR(5685.9, hgt[7][0](3, 5), 0.5);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorCustomMeteoFvg, LoadPacked)
{
    asTimeArray dates(asTime::GetMJD(2015, 5, 26, 6), asTime::GetMJD(2015, 5, 30, 24), 6, "Simple");
    dates.Init();

    double xMin = -2;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 4;
    double step = 0.25;
    float level = 500;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-custom-meteo-fvg-packed/");

    asPredictor *predictor = asPredictor::GetInstance("Custom_MeteoFVG_IFS_packed", "gh_500", predictorDataDir);
    wxASSERT(predictor);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    5589.2	5587.8	5586.5	5585.3	5583.9	5583.7
    5596.8	5595.4	5594.0	5592.7	5591.3	5591.0
    5604.4	5603.0	5601.5	5600.2	5598.7	5598.4
    5612.0	5610.5	5609.0	5607.5	5606.0	5605.8
    */
    EXPECT_NEAR(5589.2, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(5587.8, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(5586.5, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(5585.3, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(5583.9, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(5583.7, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(5596.8, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(5604.4, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(5612.0, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(5605.8, hgt[0][0](3, 5), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted:
    5624.2	5622.1	5620.1	5617.9	5615.9	5614.7
    5632.1	5629.9	5627.7	5625.6	5623.3	5622.1
    5640.1	5637.7	5635.4	5633.1	5630.8	5629.4
    5647.9	5645.6	5643.1	5640.7	5638.2	5636.8
    */
    EXPECT_NEAR(5624.2, hgt[1][0](0, 0), 0.5);
    EXPECT_NEAR(5622.1, hgt[1][0](0, 1), 0.5);
    EXPECT_NEAR(5620.1, hgt[1][0](0, 2), 0.5);
    EXPECT_NEAR(5617.9, hgt[1][0](0, 3), 0.5);
    EXPECT_NEAR(5615.9, hgt[1][0](0, 4), 0.5);
    EXPECT_NEAR(5614.7, hgt[1][0](0, 5), 0.5);
    EXPECT_NEAR(5632.1, hgt[1][0](1, 0), 0.5);
    EXPECT_NEAR(5640.1, hgt[1][0](2, 0), 0.5);
    EXPECT_NEAR(5647.9, hgt[1][0](3, 0), 0.5);
    EXPECT_NEAR(5636.8, hgt[1][0](3, 5), 0.5);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    Extracted:
    5691.9	5687.7	5683.3	5679.1	5674.8	5670.7
    5697.4	5693.1	5688.7	5684.3	5679.9	5675.8
    5702.9	5698.4	5693.9	5689.4	5684.9	5680.8
    5708.4	5703.8	5699.3	5694.7	5690.1	5685.9
    */
    EXPECT_NEAR(5691.9, hgt[7][0](0, 0), 0.5);
    EXPECT_NEAR(5687.7, hgt[7][0](0, 1), 0.5);
    EXPECT_NEAR(5683.3, hgt[7][0](0, 2), 0.5);
    EXPECT_NEAR(5679.1, hgt[7][0](0, 3), 0.5);
    EXPECT_NEAR(5674.8, hgt[7][0](0, 4), 0.5);
    EXPECT_NEAR(5670.7, hgt[7][0](0, 5), 0.5);
    EXPECT_NEAR(5697.4, hgt[7][0](1, 0), 0.5);
    EXPECT_NEAR(5702.9, hgt[7][0](2, 0), 0.5);
    EXPECT_NEAR(5708.4, hgt[7][0](3, 0), 0.5);
    EXPECT_NEAR(5685.9, hgt[7][0](3, 5), 0.5);

    wxDELETE(area);
    wxDELETE(predictor);
}

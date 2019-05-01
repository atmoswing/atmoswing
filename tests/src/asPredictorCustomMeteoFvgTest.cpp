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
    5757.2	5753.8	5750.4	5747.2	5743.8	5740.5
    5758.1	5754.8	5751.6	5748.2	5744.9	5741.6
    5759.1	5755.8	5752.6	5749.3	5746.1	5742.8
    5759.9	5756.8	5753.6	5750.4	5747.2	5743.9
    */
    EXPECT_NEAR(5757.2, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(5753.8, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(5750.4, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(5747.2, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(5743.8, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(5740.5, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(5758.1, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(5759.1, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(5759.9, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(5743.9, hgt[0][0](3, 5), 0.5);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted:
    5766.0	5763.2	5760.6	5757.8	5755.2	5752.3
    5765.5	5762.7	5760.0	5757.2	5754.5	5751.7
    5765.1	5762.2	5759.5	5756.6	5753.8	5751.0
    5764.6	5761.7	5758.8	5756.0	5753.1	5750.3
    */
    EXPECT_NEAR(5766.0, hgt[1][0](0, 0), 0.5);
    EXPECT_NEAR(5763.2, hgt[1][0](0, 1), 0.5);
    EXPECT_NEAR(5760.6, hgt[1][0](0, 2), 0.5);
    EXPECT_NEAR(5757.8, hgt[1][0](0, 3), 0.5);
    EXPECT_NEAR(5755.2, hgt[1][0](0, 4), 0.5);
    EXPECT_NEAR(5752.3, hgt[1][0](0, 5), 0.5);
    EXPECT_NEAR(5765.5, hgt[1][0](1, 0), 0.5);
    EXPECT_NEAR(5765.1, hgt[1][0](2, 0), 0.5);
    EXPECT_NEAR(5764.6, hgt[1][0](3, 0), 0.5);
    EXPECT_NEAR(5750.3, hgt[1][0](3, 5), 0.5);

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    Extracted:
    5753.0	5752.0	5750.9	5749.9	5748.8	5747.6
    5756.4	5755.4	5754.3	5753.3	5752.2	5751.0
    5759.8	5758.8	5757.7	5756.6	5755.5	5754.3
    5763.1	5762.1	5761.0	5760.0	5758.9	5757.7
    */
    EXPECT_NEAR(5753.0, hgt[7][0](0, 0), 0.5);
    EXPECT_NEAR(5752.0, hgt[7][0](0, 1), 0.5);
    EXPECT_NEAR(5750.9, hgt[7][0](0, 2), 0.5);
    EXPECT_NEAR(5749.9, hgt[7][0](0, 3), 0.5);
    EXPECT_NEAR(5748.8, hgt[7][0](0, 4), 0.5);
    EXPECT_NEAR(5747.6, hgt[7][0](0, 5), 0.5);
    EXPECT_NEAR(5756.4, hgt[7][0](1, 0), 0.5);
    EXPECT_NEAR(5759.8, hgt[7][0](2, 0), 0.5);
    EXPECT_NEAR(5763.1, hgt[7][0](3, 0), 0.5);
    EXPECT_NEAR(5757.7, hgt[7][0](3, 5), 0.5);

    /* Values time step 8 (horizontal=Lon, vertical=Lat)
    Extracted:
    NaN
    */
    EXPECT_TRUE(isnan(hgt[8][0](0, 0)));
    EXPECT_TRUE(isnan(hgt[8][0](0, 1)));
    EXPECT_TRUE(isnan(hgt[8][0](3, 5)));

    /* Values time step 11 (horizontal=Lon, vertical=Lat)
    Extracted:
    NaN
    */
    EXPECT_TRUE(isnan(hgt[11][0](0, 0)));
    EXPECT_TRUE(isnan(hgt[11][0](0, 1)));
    EXPECT_TRUE(isnan(hgt[11][0](3, 5)));

    /* Values time step 12 (horizontal=Lon, vertical=Lat)
    Extracted:
    5669.7	5669.9	5670.2	5670.3	5670.6	5670.8
    5674.7	5675.1	5675.3	5675.7	5675.9	5676.3
    5679.7	5680.1	5680.6	5680.9	5681.3	5681.7
    5684.7	5685.2	5685.7	5686.2	5686.7	5687.1
    */
    EXPECT_NEAR(5669.7, hgt[12][0](0, 0), 0.5);
    EXPECT_NEAR(5669.9, hgt[12][0](0, 1), 0.5);
    EXPECT_NEAR(5670.2, hgt[12][0](0, 2), 0.5);
    EXPECT_NEAR(5670.3, hgt[12][0](0, 3), 0.5);
    EXPECT_NEAR(5670.6, hgt[12][0](0, 4), 0.5);
    EXPECT_NEAR(5670.8, hgt[12][0](0, 5), 0.5);
    EXPECT_NEAR(5674.7, hgt[12][0](1, 0), 0.5);
    EXPECT_NEAR(5679.7, hgt[12][0](2, 0), 0.5);
    EXPECT_NEAR(5684.7, hgt[12][0](3, 0), 0.5);
    EXPECT_NEAR(5687.1, hgt[12][0](3, 5), 0.5);

    /* Values time step 19 (horizontal=Lon, vertical=Lat)
    Extracted:
    5714.8	5714.3	5713.9	5713.4	5713.0	5712.5
    5717.0	5716.5	5716.1	5715.6	5715.1	5714.6
    5719.3	5718.8	5718.3	5717.8	5717.4	5716.9
    5721.5	5721.0	5720.5	5720.0	5719.5	5719.0
    */
    EXPECT_NEAR(5714.8, hgt[19][0](0, 0), 0.5);
    EXPECT_NEAR(5714.3, hgt[19][0](0, 1), 0.5);
    EXPECT_NEAR(5713.9, hgt[19][0](0, 2), 0.5);
    EXPECT_NEAR(5713.4, hgt[19][0](0, 3), 0.5);
    EXPECT_NEAR(5713.0, hgt[19][0](0, 4), 0.5);
    EXPECT_NEAR(5712.5, hgt[19][0](0, 5), 0.5);
    EXPECT_NEAR(5717.0, hgt[19][0](1, 0), 0.5);
    EXPECT_NEAR(5719.3, hgt[19][0](2, 0), 0.5);
    EXPECT_NEAR(5721.5, hgt[19][0](3, 0), 0.5);
    EXPECT_NEAR(5719.0, hgt[19][0](3, 5), 0.5);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorCustomMeteoFvg, LoadPackedPartial)
{
    asTimeArray dates(asTime::GetMJD(2015, 5, 27, 6), asTime::GetMJD(2015, 5, 29, 6), 6, "Simple");
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

    /* Values time step 3 (horizontal=Lon, vertical=Lat)
    Extracted:
    5753.0	5752.0	5750.9	5749.9	5748.8	5747.6
    5756.4	5755.4	5754.3	5753.3	5752.2	5751.0
    5759.8	5758.8	5757.7	5756.6	5755.5	5754.3
    5763.1	5762.1	5761.0	5760.0	5758.9	5757.7
    */
    EXPECT_NEAR(5753.0, hgt[3][0](0, 0), 0.5);
    EXPECT_NEAR(5752.0, hgt[3][0](0, 1), 0.5);
    EXPECT_NEAR(5750.9, hgt[3][0](0, 2), 0.5);
    EXPECT_NEAR(5749.9, hgt[3][0](0, 3), 0.5);
    EXPECT_NEAR(5748.8, hgt[3][0](0, 4), 0.5);
    EXPECT_NEAR(5747.6, hgt[3][0](0, 5), 0.5);
    EXPECT_NEAR(5756.4, hgt[3][0](1, 0), 0.5);
    EXPECT_NEAR(5759.8, hgt[3][0](2, 0), 0.5);
    EXPECT_NEAR(5763.1, hgt[3][0](3, 0), 0.5);
    EXPECT_NEAR(5757.7, hgt[3][0](3, 5), 0.5);

    /* Values time step 4 (horizontal=Lon, vertical=Lat)
    Extracted:
    NaN
    */
    EXPECT_TRUE(isnan(hgt[4][0](0, 0)));
    EXPECT_TRUE(isnan(hgt[4][0](0, 1)));
    EXPECT_TRUE(isnan(hgt[4][0](3, 5)));

    /* Values time step 7 (horizontal=Lon, vertical=Lat)
    Extracted:
    NaN
    */
    EXPECT_TRUE(isnan(hgt[7][0](0, 0)));
    EXPECT_TRUE(isnan(hgt[7][0](0, 1)));
    EXPECT_TRUE(isnan(hgt[7][0](3, 5)));

    /* Values time step 8 (horizontal=Lon, vertical=Lat)
    Extracted:
    5669.7	5669.9	5670.2	5670.3	5670.6	5670.8
    5674.7	5675.1	5675.3	5675.7	5675.9	5676.3
    5679.7	5680.1	5680.6	5680.9	5681.3	5681.7
    5684.7	5685.2	5685.7	5686.2	5686.7	5687.1
    */
    EXPECT_NEAR(5669.7, hgt[8][0](0, 0), 0.5);
    EXPECT_NEAR(5669.9, hgt[8][0](0, 1), 0.5);
    EXPECT_NEAR(5670.2, hgt[8][0](0, 2), 0.5);
    EXPECT_NEAR(5670.3, hgt[8][0](0, 3), 0.5);
    EXPECT_NEAR(5670.6, hgt[8][0](0, 4), 0.5);
    EXPECT_NEAR(5670.8, hgt[8][0](0, 5), 0.5);
    EXPECT_NEAR(5674.7, hgt[8][0](1, 0), 0.5);
    EXPECT_NEAR(5679.7, hgt[8][0](2, 0), 0.5);
    EXPECT_NEAR(5684.7, hgt[8][0](3, 0), 0.5);
    EXPECT_NEAR(5687.1, hgt[8][0](3, 5), 0.5);

    wxDELETE(area);
    wxDELETE(predictor);
}
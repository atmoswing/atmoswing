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


TEST(Predictor, Standardize)
{
    double xMin = 10;
    int xPtsNb = 5;
    double yMin = 35;
    int yPtsNb = 3;
    double step = 2.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    double start = asTime::GetMJD(1960, 1, 1, 12, 00);
    double end = asTime::GetMJD(1960, 1, 10, 12, 00);
    double timeStep = 24;
    asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2014/");

    asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v1", "pressure/hgt", predictorDataDir);

    predictor->SetStandardize(true);

    ASSERT_TRUE(predictor->Load(area, timearray, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    0.99184	0.99184	0.92467	0.83512	0.72317
    1.19334	1.19334	1.03662	0.81273	0.49928
    1.26051	1.17095	0.99184	0.70078	0.32017
    */
    EXPECT_NEAR(0.99184, hgt[0][0](0, 0), 0.00001);
    EXPECT_NEAR(0.99184, hgt[0][0](0, 1), 0.00001);
    EXPECT_NEAR(0.92467, hgt[0][0](0, 2), 0.00001);
    EXPECT_NEAR(0.83512, hgt[0][0](0, 3), 0.00001);
    EXPECT_NEAR(0.72317, hgt[0][0](0, 4), 0.00001);
    EXPECT_NEAR(1.19334, hgt[0][0](1, 0), 0.00001);
    EXPECT_NEAR(1.26051, hgt[0][0](2, 0), 0.00001);
    EXPECT_NEAR(0.32017, hgt[0][0](2, 4), 0.00001);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    0.47689	0.27539	0.14105	0.07388	0.09627
    0.85751	0.67839	0.52167	0.40972	0.32017
    1.14857	1.01423	0.87990	0.79034	0.65600
    */
    EXPECT_NEAR(0.47689, hgt[1][0](0, 0), 0.00001);
    EXPECT_NEAR(0.27539, hgt[1][0](0, 1), 0.00001);
    EXPECT_NEAR(0.14105, hgt[1][0](0, 2), 0.00001);
    EXPECT_NEAR(0.07388, hgt[1][0](0, 3), 0.00001);
    EXPECT_NEAR(0.09627, hgt[1][0](0, 4), 0.00001);
    EXPECT_NEAR(0.85751, hgt[1][0](1, 0), 0.00001);
    EXPECT_NEAR(1.14857, hgt[1][0](2, 0), 0.00001);
    EXPECT_NEAR(0.65600, hgt[1][0](2, 4), 0.00001);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    -0.19479	-0.50823	-0.59779	-0.48585	-0.17240
     0.40972 	 0.11866 	-0.08284	-0.15001	-0.01567
     0.99184 	 0.76795 	 0.54406 	 0.36494	 0.29778
    */
    EXPECT_NEAR(-0.19479, hgt[2][0](0, 0), 0.00001);
    EXPECT_NEAR(-0.50823, hgt[2][0](0, 1), 0.00001);
    EXPECT_NEAR(-0.59779, hgt[2][0](0, 2), 0.00001);
    EXPECT_NEAR(-0.48585, hgt[2][0](0, 3), 0.00001);
    EXPECT_NEAR(-0.17240, hgt[2][0](0, 4), 0.00001);
    EXPECT_NEAR(0.40972, hgt[2][0](1, 0), 0.00001);
    EXPECT_NEAR(0.99184, hgt[2][0](2, 0), 0.00001);
    EXPECT_NEAR(0.29778, hgt[2][0](2, 4), 0.00001);

    wxDELETE(area);
    wxDELETE(predictor);
}

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
#include "asPredictorOper.h"
#include "asAreaCompGrid.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(PredictorOperEcmwfIfs, LoadSingleDay)
{
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-ecmwf-ifs-grib/2019-02-01_z.grib");

    asTimeArray dates(asTime::GetMJD(2019, 2, 1, 00, 00), asTime::GetMJD(2019, 2, 1, 00, 00), 6, "Simple");
    dates.Init();

    double xMin = -2;
    int xPtsNb = 6;
    double yMin = 45;
    int yPtsNb = 4;
    double step = 0.5;
    float level = 1000;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper *predictor = asPredictorOper::GetInstance("ECMWF_IFS_GRIB_Forecast", "hgt");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    -1107.0	-1059.0	-1007.2	-972.1	-959.9	-925.2
    -1025.7	-992.9	-959.3	-928.9	-906.2	-855.4
    -947.1	-930.4	-911.6	-879.4	-845.4	-787.0
    -858.9	-849.9	-839.8	-810.4	-779.9	-727.9
    */
    EXPECT_NEAR(-1107.0, hgt[0][0](0, 0), 0.5);
    EXPECT_NEAR(-1059.0, hgt[0][0](0, 1), 0.5);
    EXPECT_NEAR(-1007.2, hgt[0][0](0, 2), 0.5);
    EXPECT_NEAR(-972.1, hgt[0][0](0, 3), 0.5);
    EXPECT_NEAR(-959.9, hgt[0][0](0, 4), 0.5);
    EXPECT_NEAR(-925.2, hgt[0][0](0, 5), 0.5);
    EXPECT_NEAR(-1025.7, hgt[0][0](1, 0), 0.5);
    EXPECT_NEAR(-947.1, hgt[0][0](2, 0), 0.5);
    EXPECT_NEAR(-858.9, hgt[0][0](3, 0), 0.5);
    EXPECT_NEAR(-727.9, hgt[0][0](3, 5), 0.5);

    wxDELETE(area);
    wxDELETE(predictor);
}

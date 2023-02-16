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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#include <gtest/gtest.h>
#include <wx/filename.h>

#include "asAreaGrid.h"
#include "asPredictorOper.h"
#include "asTimeArray.h"

TEST(PredictorOperCustomVigicruesIfs, GetCorrectPredictors) {
    asPredictorOper* predictor;

    predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS_Forecast", "z");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS_Forecast", "tcwv");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::PrecipitableWater);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomVigicruesIfs, LoadSingleDay) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_Z_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 2, 00), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 5;
    double yMin = 47;
    int yPtsNb = 5;
    double step = 0.25;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS_Forecast", "z");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    56009.88   55993.51   55976.63   55960.13   55942.63
    56039.38   56024.01   56008.51   55992.51   55975.88
    56066.63   56052.38   56038.01   56023.13   56007.88
    56088.88   56075.88   56062.76   56049.38   56035.26
    56108.88   56096.63   56083.63   56070.51   56056.88
    */
    EXPECT_NEAR(56009.88 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(55993.51 / 9.80665, hgt[0][0](0, 1), 0.05);
    EXPECT_NEAR(55976.63 / 9.80665, hgt[0][0](0, 2), 0.05);
    EXPECT_NEAR(55960.13 / 9.80665, hgt[0][0](0, 3), 0.05);
    EXPECT_NEAR(55942.63 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(56039.38 / 9.80665, hgt[0][0](1, 0), 0.05);
    EXPECT_NEAR(56066.63 / 9.80665, hgt[0][0](2, 0), 0.05);
    EXPECT_NEAR(56088.88 / 9.80665, hgt[0][0](3, 0), 0.05);
    EXPECT_NEAR(56056.88 / 9.80665, hgt[0][0](4, 4), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomVigicruesIfs, LoadThirdTimeStep) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_Z_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 12), asTime::GetMJD(2023, 2, 2, 12), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 5;
    double yMin = 47;
    int yPtsNb = 5;
    double step = 0.25;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS_Forecast", "z");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    56334.75   56321.88   56309.88   56297.88   56285.38
    56355.25   56344.00   56332.63   56320.25   56307.63
    56375.25   56364.25   56352.75   56340.00   56326.88
    56393.38   56382.63   56370.88   56357.63   56343.88
    56408.25   56397.75   56385.63   56371.75   56357.88
    */
    EXPECT_NEAR(56334.75 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(56321.88 / 9.80665, hgt[0][0](0, 1), 0.05);
    EXPECT_NEAR(56309.88 / 9.80665, hgt[0][0](0, 2), 0.05);
    EXPECT_NEAR(56297.88 / 9.80665, hgt[0][0](0, 3), 0.05);
    EXPECT_NEAR(56285.38 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(56355.25 / 9.80665, hgt[0][0](1, 0), 0.05);
    EXPECT_NEAR(56375.25 / 9.80665, hgt[0][0](2, 0), 0.05);
    EXPECT_NEAR(56393.38 / 9.80665, hgt[0][0](3, 0), 0.05);
    EXPECT_NEAR(56357.88 / 9.80665, hgt[0][0](4, 4), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomVigicruesIfs, LoadFullTimeArray) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_Z_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 12, 00), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 5;
    double yMin = 47;
    int yPtsNb = 5;
    double step = 0.25;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS_Forecast", "z");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    56009.88   55993.51   55976.63   55960.13   55942.63
    56039.38   56024.01   56008.51   55992.51   55975.88
    56066.63   56052.38   56038.01   56023.13   56007.88
    56088.88   56075.88   56062.76   56049.38   56035.26
    56108.88   56096.63   56083.63   56070.51   56056.88
    */
    EXPECT_NEAR(56009.88 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(55993.51 / 9.80665, hgt[0][0](0, 1), 0.05);
    EXPECT_NEAR(55976.63 / 9.80665, hgt[0][0](0, 2), 0.05);
    EXPECT_NEAR(55960.13 / 9.80665, hgt[0][0](0, 3), 0.05);
    EXPECT_NEAR(55942.63 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(56039.38 / 9.80665, hgt[0][0](1, 0), 0.05);
    EXPECT_NEAR(56066.63 / 9.80665, hgt[0][0](2, 0), 0.05);
    EXPECT_NEAR(56088.88 / 9.80665, hgt[0][0](3, 0), 0.05);
    EXPECT_NEAR(56056.88 / 9.80665, hgt[0][0](4, 4), 0.05);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    Extracted:
    56334.75   56321.88   56309.88   56297.88   56285.38
    56355.25   56344.00   56332.63   56320.25   56307.63
    56375.25   56364.25   56352.75   56340.00   56326.88
    56393.38   56382.63   56370.88   56357.63   56343.88
    56408.25   56397.75   56385.63   56371.75   56357.88
    */
    EXPECT_NEAR(56334.75 / 9.80665, hgt[2][0](0, 0), 0.05);
    EXPECT_NEAR(56321.88 / 9.80665, hgt[2][0](0, 1), 0.05);
    EXPECT_NEAR(56309.88 / 9.80665, hgt[2][0](0, 2), 0.05);
    EXPECT_NEAR(56297.88 / 9.80665, hgt[2][0](0, 3), 0.05);
    EXPECT_NEAR(56285.38 / 9.80665, hgt[2][0](0, 4), 0.05);
    EXPECT_NEAR(56355.25 / 9.80665, hgt[2][0](1, 0), 0.05);
    EXPECT_NEAR(56375.25 / 9.80665, hgt[2][0](2, 0), 0.05);
    EXPECT_NEAR(56393.38 / 9.80665, hgt[2][0](3, 0), 0.05);
    EXPECT_NEAR(56357.88 / 9.80665, hgt[2][0](4, 4), 0.05);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    Extracted:
    53558.28   53512.40   53472.90   53441.53   53417.28
    53589.65   53546.40   53509.78   53480.65   53459.15
    53621.53   53580.15   53546.40   53520.65   53501.53
    53652.78   53613.40   53582.65   53559.90   53541.03
    53683.40   53646.28   53617.53   53595.78   53576.15
    */
    EXPECT_NEAR(53558.28 / 9.80665, hgt[40][0](0, 0), 0.05);
    EXPECT_NEAR(53512.40 / 9.80665, hgt[40][0](0, 1), 0.05);
    EXPECT_NEAR(53472.90 / 9.80665, hgt[40][0](0, 2), 0.05);
    EXPECT_NEAR(53441.53 / 9.80665, hgt[40][0](0, 3), 0.05);
    EXPECT_NEAR(53417.28 / 9.80665, hgt[40][0](0, 4), 0.05);
    EXPECT_NEAR(53589.65 / 9.80665, hgt[40][0](1, 0), 0.05);
    EXPECT_NEAR(53621.53 / 9.80665, hgt[40][0](2, 0), 0.05);
    EXPECT_NEAR(53652.78 / 9.80665, hgt[40][0](3, 0), 0.05);
    EXPECT_NEAR(53576.15 / 9.80665, hgt[40][0](4, 4), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomVigicruesIfs, LoadTotalColumnWaterVapor) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_TCWV_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 12, 00), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 5;
    double yMin = 47;
    int yPtsNb = 5;
    double step = 0.25;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS_Forecast", "tcwv");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, 0));

    vva2f tcwv = predictor->GetData();
    // tcwv[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    11.20   11.32   11.45   11.63   11.82
    10.14   10.34   10.57   10.81   11.03
     9.54    9.71    9.87    9.94    9.91
     8.85    9.22    9.46    9.44    9.41
     8.63    8.74    9.16    9.38    9.35
    */
    EXPECT_NEAR(11.20, tcwv[0][0](0, 0), 0.05);
    EXPECT_NEAR(11.32, tcwv[0][0](0, 1), 0.05);
    EXPECT_NEAR(11.45, tcwv[0][0](0, 2), 0.05);
    EXPECT_NEAR(11.63, tcwv[0][0](0, 3), 0.05);
    EXPECT_NEAR(11.82, tcwv[0][0](0, 4), 0.05);
    EXPECT_NEAR(10.14, tcwv[0][0](1, 0), 0.05);
    EXPECT_NEAR(9.54, tcwv[0][0](2, 0), 0.05);
    EXPECT_NEAR(8.85, tcwv[0][0](3, 0), 0.05);
    EXPECT_NEAR(9.35, tcwv[0][0](4, 4), 0.05);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    Extracted:
    10.74   10.76   10.80   10.76   10.75
    10.46   10.25   10.19   10.34   10.42
     9.88   10.15   10.44   10.60   10.69
    10.18   10.09    9.99   10.06   10.11
     8.98    9.07    9.15    9.23    9.54
    */
    EXPECT_NEAR(10.74, tcwv[2][0](0, 0), 0.05);
    EXPECT_NEAR(10.76, tcwv[2][0](0, 1), 0.05);
    EXPECT_NEAR(10.80, tcwv[2][0](0, 2), 0.05);
    EXPECT_NEAR(10.76, tcwv[2][0](0, 3), 0.05);
    EXPECT_NEAR(10.75, tcwv[2][0](0, 4), 0.05);
    EXPECT_NEAR(10.46, tcwv[2][0](1, 0), 0.05);
    EXPECT_NEAR(9.88, tcwv[2][0](2, 0), 0.05);
    EXPECT_NEAR(10.18, tcwv[2][0](3, 0), 0.05);
    EXPECT_NEAR(9.54, tcwv[2][0](4, 4), 0.05);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    Extracted:
    10.05   10.49   10.52   10.54   10.84
    10.03   10.51   10.75   10.70   10.72
    10.17   10.64   11.00   10.95   10.81
    10.24   10.55   10.82   10.92   10.94
    10.19   10.65   10.83   10.81   10.94
    */
    EXPECT_NEAR(10.05, tcwv[40][0](0, 0), 0.05);
    EXPECT_NEAR(10.49, tcwv[40][0](0, 1), 0.05);
    EXPECT_NEAR(10.52, tcwv[40][0](0, 2), 0.05);
    EXPECT_NEAR(10.54, tcwv[40][0](0, 3), 0.05);
    EXPECT_NEAR(10.84, tcwv[40][0](0, 4), 0.05);
    EXPECT_NEAR(10.03, tcwv[40][0](1, 0), 0.05);
    EXPECT_NEAR(10.17, tcwv[40][0](2, 0), 0.05);
    EXPECT_NEAR(10.24, tcwv[40][0](3, 0), 0.05);
    EXPECT_NEAR(10.94, tcwv[40][0](4, 4), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperCustomVigicruesIfs, LoadRelativeHumidity) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_R_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 12, 00), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 5;
    double yMin = 47;
    int yPtsNb = 5;
    double step = 0.25;
    float level = 850;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS_Forecast", "r");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    26.3   28.2   29.8   31.6   33.3
    10.0   12.7   16.1   20.0   23.2
     7.8    8.1    9.6   12.4   14.6
    10.7   10.9   11.4   12.2   12.4
    10.7   10.5   11.0   11.5   11.1
    */
    EXPECT_NEAR(26.3, rh[0][0](0, 0), 0.05);
    EXPECT_NEAR(28.2, rh[0][0](0, 1), 0.05);
    EXPECT_NEAR(29.8, rh[0][0](0, 2), 0.05);
    EXPECT_NEAR(31.6, rh[0][0](0, 3), 0.05);
    EXPECT_NEAR(33.3, rh[0][0](0, 4), 0.05);
    EXPECT_NEAR(10.0, rh[0][0](1, 0), 0.05);
    EXPECT_NEAR(7.8, rh[0][0](2, 0), 0.05);
    EXPECT_NEAR(10.7, rh[0][0](3, 0), 0.05);
    EXPECT_NEAR(11.1, rh[0][0](4, 4), 0.05);

    /* Values time step 40 (horizontal=Lon, vertical=Lat)
    Extracted:
    69.2   74.6   76.0   75.5   78.2
    70.5   75.3   76.9   75.5   76.2
    71.9   74.8   76.3   76.4   77.1
    72.2   73.4   75.9   77.7   78.0
    72.9   73.2   76.4   78.2   77.4
    */
    EXPECT_NEAR(69.2, rh[40][0](0, 0), 0.05);
    EXPECT_NEAR(74.6, rh[40][0](0, 1), 0.05);
    EXPECT_NEAR(76.0, rh[40][0](0, 2), 0.05);
    EXPECT_NEAR(75.5, rh[40][0](0, 3), 0.05);
    EXPECT_NEAR(78.2, rh[40][0](0, 4), 0.05);
    EXPECT_NEAR(70.5, rh[40][0](1, 0), 0.05);
    EXPECT_NEAR(71.9, rh[40][0](2, 0), 0.05);
    EXPECT_NEAR(72.2, rh[40][0](3, 0), 0.05);
    EXPECT_NEAR(77.4, rh[40][0](4, 4), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}
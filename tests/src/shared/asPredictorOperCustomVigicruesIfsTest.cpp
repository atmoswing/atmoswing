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

#include "asIncludes.h"
#include <wx/filename.h>

#include "asAreaGrid.h"
#include "asPredictorOper.h"
#include "asTimeArray.h"

TEST(PredictorOperCustomVigicruesIfs, GetCorrectPredictors) {
    asPredictorOper* predictor;

    predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "z");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "tcwv");
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

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "z");
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

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "z");
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
    for (int i = 0; i < 41; ++i) {
        filepaths.push_back(wxFileName::GetCwd() +
                            "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_Z_202302020000.grb");
    }

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

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "z");
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
    for (int i = 0; i < 41; ++i) {
        filepaths.push_back(wxFileName::GetCwd() +
                            "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_TCWV_202302020000.grb");
    }

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 12, 00), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 5;
    double yMin = 47;
    int yPtsNb = 5;
    double step = 0.25;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "tcwv");
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
    for (int i = 0; i < 41; ++i) {
        filepaths.push_back(wxFileName::GetCwd() +
                            "/files/data-custom-vigicrues-ifs/2023/02/02/CEP_R_202302020000.grb");
    }

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

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "r");
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

// GRIB2 files from the ECMWF open data (run of 22.10.2026 00h, lead times 0 and 6 h), cropped on the
// area by AtmoSwing-Vigicrues (DownloadEcmwfData with grib_edition: 2). They keep the open data encoding
// (CCSDS packing, longitudes starting at 354°).
static vva2f LoadGrib2(const wxString& dataId, float level) {
    vwxs filepaths;
    for (int i = 0; i < 2; ++i) {
        filepaths.push_back(wxFileName::GetCwd() + "/files/data-custom-vigicrues-ifs-grib2/2026/10/22/CEP_" +
                            dataId.Upper() + "_202610220000.grb");
    }

    asTimeArray dates(asTime::GetMJD(2026, 10, 22, 00), asTime::GetMJD(2026, 10, 22, 6), 6, "Simple");
    dates.Init();

    asAreaGrid* area = asAreaGrid::GetInstance("Regular", -6, 5, 0.25, 47, 5, 0.25);

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", dataId);
    wxASSERT(predictor);

    predictor->SetFileNames(filepaths);

    vva2f data;
    if (predictor->Load(area, dates, level)) {
        data = predictor->GetData();
    }

    wxDELETE(area);
    wxDELETE(predictor);

    return data;
}

TEST(PredictorOperCustomVigicruesIfs, LoadGrib2Geopotential) {
    vva2f hgt = LoadGrib2("z", 500);
    ASSERT_EQ(hgt.size(), 2);
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    54681.13   54673.63   54665.13   54654.88   54647.63
    54755.38   54747.38   54741.63   54732.13   54719.63
    54839.63   54828.63   54819.88   54810.38   54798.63
    54933.63   54919.13   54906.13   54895.13   54884.13
    55026.13   55013.38   54999.13   54987.63   54977.38
    */
    EXPECT_NEAR(54681.13 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(54673.63 / 9.80665, hgt[0][0](0, 1), 0.05);
    EXPECT_NEAR(54665.13 / 9.80665, hgt[0][0](0, 2), 0.05);
    EXPECT_NEAR(54654.88 / 9.80665, hgt[0][0](0, 3), 0.05);
    EXPECT_NEAR(54647.63 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(54755.38 / 9.80665, hgt[0][0](1, 0), 0.05);
    EXPECT_NEAR(54839.63 / 9.80665, hgt[0][0](2, 0), 0.05);
    EXPECT_NEAR(54933.63 / 9.80665, hgt[0][0](3, 0), 0.05);
    EXPECT_NEAR(54977.38 / 9.80665, hgt[0][0](4, 4), 0.05);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted:
    54721.13   54722.13   54723.13   54721.13   54717.63
    54786.63   54785.38   54787.88   54788.13   54784.63
    54848.38   54847.63   54853.13   54856.13   54853.63
    54920.88   54920.63   54926.88   54929.63   54928.63
    55005.13   55004.88   55007.38   55007.88   55008.13
    */
    EXPECT_NEAR(54721.13 / 9.80665, hgt[1][0](0, 0), 0.05);
    EXPECT_NEAR(54722.13 / 9.80665, hgt[1][0](0, 1), 0.05);
    EXPECT_NEAR(54786.63 / 9.80665, hgt[1][0](1, 0), 0.05);
    EXPECT_NEAR(54848.38 / 9.80665, hgt[1][0](2, 0), 0.05);
    EXPECT_NEAR(54920.88 / 9.80665, hgt[1][0](3, 0), 0.05);
    EXPECT_NEAR(55008.13 / 9.80665, hgt[1][0](4, 4), 0.05);
}

TEST(PredictorOperCustomVigicruesIfs, LoadGrib2GeopotentialOtherLevel) {
    vva2f hgt = LoadGrib2("z", 1000);
    ASSERT_EQ(hgt.size(), 2);

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    1243.58   1250.58   1260.83   1280.58   1303.08
    ...
    1474.83   1478.08   1484.08   1490.33   1495.58
    */
    EXPECT_NEAR(1243.58 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(1303.08 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(1474.83 / 9.80665, hgt[0][0](4, 0), 0.05);
    EXPECT_NEAR(1495.58 / 9.80665, hgt[0][0](4, 4), 0.05);
}

TEST(PredictorOperCustomVigicruesIfs, LoadGrib2RelativeHumidity) {
    vva2f rh = LoadGrib2("r", 850);
    ASSERT_EQ(rh.size(), 2);

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    80.46   81.46   79.46   81.46   85.46
    81.46   79.46   79.46   81.46   82.46
    79.46   80.46   80.46   81.46   83.46
    81.46   84.46   81.46   79.46   82.46
    82.46   82.46   81.46   80.46   79.46
    */
    EXPECT_NEAR(80.46, rh[0][0](0, 0), 0.05);
    EXPECT_NEAR(81.46, rh[0][0](0, 1), 0.05);
    EXPECT_NEAR(79.46, rh[0][0](0, 2), 0.05);
    EXPECT_NEAR(85.46, rh[0][0](0, 4), 0.05);
    EXPECT_NEAR(81.46, rh[0][0](1, 0), 0.05);
    EXPECT_NEAR(79.46, rh[0][0](2, 0), 0.05);
    EXPECT_NEAR(81.46, rh[0][0](3, 0), 0.05);
    EXPECT_NEAR(79.46, rh[0][0](4, 4), 0.05);
}

TEST(PredictorOperCustomVigicruesIfs, LoadGrib2TotalColumnWaterVapor) {
    vva2f tcwv = LoadGrib2("tcwv", 0);
    ASSERT_EQ(tcwv.size(), 2);

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    21.91   21.66   21.41   21.66   21.91
    21.91   21.16   21.41   21.91   21.66
    21.41   21.41   21.41   21.66   21.66
    21.91   21.66   21.41   21.41   21.41
    21.91   21.66   21.41   21.16   21.41
    */
    EXPECT_NEAR(21.91, tcwv[0][0](0, 0), 0.05);
    EXPECT_NEAR(21.66, tcwv[0][0](0, 1), 0.05);
    EXPECT_NEAR(21.41, tcwv[0][0](0, 2), 0.05);
    EXPECT_NEAR(21.91, tcwv[0][0](1, 0), 0.05);
    EXPECT_NEAR(21.41, tcwv[0][0](2, 0), 0.05);
    EXPECT_NEAR(21.91, tcwv[0][0](3, 0), 0.05);
    EXPECT_NEAR(21.41, tcwv[0][0](4, 4), 0.05);

    /* Values time step 1 (horizontal=Lon, vertical=Lat)
    Extracted:
    24.16   23.91   23.91   23.91   23.91
    24.41   24.41   24.66   24.66   24.41
    24.41   24.66   24.66   24.66   24.66
    25.16   24.91   24.91   24.66   24.66
    25.41   25.41   25.41   25.41   24.91
    */
    EXPECT_NEAR(24.16, tcwv[1][0](0, 0), 0.05);
    EXPECT_NEAR(23.91, tcwv[1][0](0, 1), 0.05);
    EXPECT_NEAR(24.41, tcwv[1][0](1, 0), 0.05);
    EXPECT_NEAR(25.16, tcwv[1][0](3, 0), 0.05);
    EXPECT_NEAR(24.91, tcwv[1][0](4, 4), 0.05);
}

TEST(PredictorOperCustomVigicruesIfs, CanFindFiles) {
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

    asPredictorOper* predictor = asPredictorOper::GetInstance("Custom_Vigicrues_IFS", "z");
    wxASSERT(predictor);

    predictor->SetPredictorsRealtimeDirectory(wxFileName::GetCwd() + "/files/data-custom-vigicrues-ifs");
    predictor->SetRunDateInUse(asTime::GetTimeFromString("2023-02-02"));
    predictor->SetLevel(500);

    // Build file names
    ASSERT_TRUE(predictor->BuildFilenamesAndUrls(0, 6, 41));

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
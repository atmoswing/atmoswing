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

TEST(PredictorOperMeteoFranceArpege, GetCorrectPredictors) {
    asPredictorOper* predictor;

    predictor = asPredictorOper::GetInstance("MF_ARPEGE_Forecast", "z");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::Geopotential);
    wxDELETE(predictor);

    predictor = asPredictorOper::GetInstance("MF_ARPEGE_Forecast", "rh");
    ASSERT_TRUE(predictor->GetParameter() == asPredictor::RelativeHumidity);
    wxDELETE(predictor);
}

TEST(PredictorOperMeteoFranceArpege, LoadSingleDay) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302020000_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 2, 00), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 11;
    double yMin = 47;
    int yPtsNb = 11;
    double step = 0.1;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("MF_ARPEGE_Forecast", "z");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    56003.6   55996.5   55989.4   55982.5   55975.5   55968.6   55961.6   55954.7   55947.9   55941.2   55934.6
    56016.9   56010.4   56003.9   55997.2   55990.6   55983.9   55977.0   55970.1   55963.4   55956.6   55950.0
    56030.4   56024.4   56018.2   56012.0   56005.6   55999.1   55992.4   55985.6   55978.9   55972.1   55965.4
    56044.5   56038.6   56032.9   56026.6   56020.4   56014.1   56007.7   56001.1   55994.6   55988.1   55981.4
    56057.9   56052.2   56046.6   56040.6   56034.6   56028.6   56022.4   56016.1   56009.6   56003.1   55996.4
    56069.6   56064.2   56058.7   56053.1   56047.2   56041.1   56035.0   56028.5   56022.1   56015.7   56009.1
    56079.5   56074.4   56069.0   56063.6   56057.9   56052.0   56046.1   56040.1   56033.7   56027.5   56021.0
    56088.6   56083.7   56078.7   56073.7   56068.6   56063.2   56057.5   56051.7   56045.6   56039.6   56033.5
    56099.6   56094.7   56089.9   56084.7   56079.5   56074.1   56068.6   56062.6   56056.9   56051.1   56045.7
    56112.6   56107.2   56101.6   56096.0   56090.2   56084.6   56079.0   56073.1   56067.9   56063.1   56058.5
    56124.0   56118.2   56112.5   56106.6   56101.0   56095.5   56090.4   56085.4   56080.5   56075.5   56069.7
    */
    EXPECT_NEAR(56003.6 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(55996.5 / 9.80665, hgt[0][0](0, 1), 0.05);
    EXPECT_NEAR(55989.4 / 9.80665, hgt[0][0](0, 2), 0.05);
    EXPECT_NEAR(55982.5 / 9.80665, hgt[0][0](0, 3), 0.05);
    EXPECT_NEAR(55975.5 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(56016.9 / 9.80665, hgt[0][0](1, 0), 0.05);
    EXPECT_NEAR(56030.4 / 9.80665, hgt[0][0](2, 0), 0.05);
    EXPECT_NEAR(56044.5 / 9.80665, hgt[0][0](3, 0), 0.05);
    EXPECT_NEAR(56069.7 / 9.80665, hgt[0][0](10, 10), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperMeteoFranceArpege, LoadThirdTimeStep) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302021200_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 12), asTime::GetMJD(2023, 2, 2, 12), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 11;
    double yMin = 47;
    int yPtsNb = 11;
    double step = 0.1;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("MF_ARPEGE_Forecast", "z");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    56347.0   56341.9   56336.8   56332.0   56327.1   56322.3   56317.0   56311.3   56305.0   56298.3   56291.3
    56355.8   56350.9   56346.4   56341.8   56336.9   56331.8   56326.1   56320.0   56313.5   56306.8   56299.8
    56364.0   56359.5   56355.1   56350.5   56345.4   56339.9   56333.9   56327.5   56320.8   56313.9   56306.8
    56371.5   56367.0   56362.5   56357.6   56352.3   56346.4   56339.9   56333.1   56326.3   56319.3   56312.4
    56378.1   56373.5   56369.0   56363.9   56358.1   56351.8   56344.9   56337.9   56331.0   56324.4   56317.9
    56384.6   56379.6   56374.6   56369.0   56362.9   56356.3   56349.5   56342.9   56336.4   56330.3   56324.1
    56390.0   56384.8   56379.1   56373.0   56366.9   56360.8   56354.5   56348.5   56342.8   56337.0   56331.1
    56394.3   56388.8   56383.0   56377.1   56371.4   56365.9   56360.4   56355.0   56349.8   56344.3   56338.4
    56398.5   56393.1   56387.8   56382.3   56377.0   56371.8   56366.8   56361.8   56356.6   56351.0   56345.0
    56403.3   56398.3   56393.3   56388.0   56382.9   56377.9   56372.8   56367.6   56362.1   56356.4   56350.3
    56408.8   56403.8   56398.6   56393.3   56387.9   56382.5   56377.1   56371.8   56366.3   56360.4   56354.3
    */
    EXPECT_NEAR(56347.0 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(56341.9 / 9.80665, hgt[0][0](0, 1), 0.05);
    EXPECT_NEAR(56336.8 / 9.80665, hgt[0][0](0, 2), 0.05);
    EXPECT_NEAR(56332.0 / 9.80665, hgt[0][0](0, 3), 0.05);
    EXPECT_NEAR(56327.1 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(56355.8 / 9.80665, hgt[0][0](1, 0), 0.05);
    EXPECT_NEAR(56364.0 / 9.80665, hgt[0][0](2, 0), 0.05);
    EXPECT_NEAR(56371.5 / 9.80665, hgt[0][0](3, 0), 0.05);
    EXPECT_NEAR(56354.3 / 9.80665, hgt[0][0](10, 10), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperMeteoFranceArpege, LoadFullTimeArray) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302020000_202302020000.grb");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302020600_202302020000.grb");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302021200_202302020000.grb");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302021800_202302020000.grb");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302030000_202302020000.grb");
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_500_202302030600_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 3, 6), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 11;
    double yMin = 47;
    int yPtsNb = 11;
    double step = 0.1;
    float level = 500;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("MF_ARPEGE_Forecast", "z");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f hgt = predictor->GetData();
    // hgt[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    56003.6   55996.5   55989.4   55982.5   55975.5   55968.6   55961.6   55954.7   55947.9   55941.2   55934.6
    56016.9   56010.4   56003.9   55997.2   55990.6   55983.9   55977.0   55970.1   55963.4   55956.6   55950.0
    56030.4   56024.4   56018.2   56012.0   56005.6   55999.1   55992.4   55985.6   55978.9   55972.1   55965.4
    56044.5   56038.6   56032.9   56026.6   56020.4   56014.1   56007.7   56001.1   55994.6   55988.1   55981.4
    56057.9   56052.2   56046.6   56040.6   56034.6   56028.6   56022.4   56016.1   56009.6   56003.1   55996.4
    56069.6   56064.2   56058.7   56053.1   56047.2   56041.1   56035.0   56028.5   56022.1   56015.7   56009.1
    56079.5   56074.4   56069.0   56063.6   56057.9   56052.0   56046.1   56040.1   56033.7   56027.5   56021.0
    56088.6   56083.7   56078.7   56073.7   56068.6   56063.2   56057.5   56051.7   56045.6   56039.6   56033.5
    56099.6   56094.7   56089.9   56084.7   56079.5   56074.1   56068.6   56062.6   56056.9   56051.1   56045.7
    56112.6   56107.2   56101.6   56096.0   56090.2   56084.6   56079.0   56073.1   56067.9   56063.1   56058.5
    56124.0   56118.2   56112.5   56106.6   56101.0   56095.5   56090.4   56085.4   56080.5   56075.5   56069.7
    */
    EXPECT_NEAR(56003.6 / 9.80665, hgt[0][0](0, 0), 0.05);
    EXPECT_NEAR(55996.5 / 9.80665, hgt[0][0](0, 1), 0.05);
    EXPECT_NEAR(55989.4 / 9.80665, hgt[0][0](0, 2), 0.05);
    EXPECT_NEAR(55982.5 / 9.80665, hgt[0][0](0, 3), 0.05);
    EXPECT_NEAR(55975.5 / 9.80665, hgt[0][0](0, 4), 0.05);
    EXPECT_NEAR(56016.9 / 9.80665, hgt[0][0](1, 0), 0.05);
    EXPECT_NEAR(56030.4 / 9.80665, hgt[0][0](2, 0), 0.05);
    EXPECT_NEAR(56044.5 / 9.80665, hgt[0][0](3, 0), 0.05);
    EXPECT_NEAR(56069.7 / 9.80665, hgt[0][0](10, 10), 0.05);

    /* Values time step 2 (horizontal=Lon, vertical=Lat)
    Extracted:
    56347.0   56341.9   56336.8   56332.0   56327.1   56322.3   56317.0   56311.3   56305.0   56298.3   56291.3
    56355.8   56350.9   56346.4   56341.8   56336.9   56331.8   56326.1   56320.0   56313.5   56306.8   56299.8
    56364.0   56359.5   56355.1   56350.5   56345.4   56339.9   56333.9   56327.5   56320.8   56313.9   56306.8
    56371.5   56367.0   56362.5   56357.6   56352.3   56346.4   56339.9   56333.1   56326.3   56319.3   56312.4
    56378.1   56373.5   56369.0   56363.9   56358.1   56351.8   56344.9   56337.9   56331.0   56324.4   56317.9
    56384.6   56379.6   56374.6   56369.0   56362.9   56356.3   56349.5   56342.9   56336.4   56330.3   56324.1
    56390.0   56384.8   56379.1   56373.0   56366.9   56360.8   56354.5   56348.5   56342.8   56337.0   56331.1
    56394.3   56388.8   56383.0   56377.1   56371.4   56365.9   56360.4   56355.0   56349.8   56344.3   56338.4
    56398.5   56393.1   56387.8   56382.3   56377.0   56371.8   56366.8   56361.8   56356.6   56351.0   56345.0
    56403.3   56398.3   56393.3   56388.0   56382.9   56377.9   56372.8   56367.6   56362.1   56356.4   56350.3
    56408.8   56403.8   56398.6   56393.3   56387.9   56382.5   56377.1   56371.8   56366.3   56360.4   56354.3
    */
    EXPECT_NEAR(56347.0 / 9.80665, hgt[2][0](0, 0), 0.05);
    EXPECT_NEAR(56341.9 / 9.80665, hgt[2][0](0, 1), 0.05);
    EXPECT_NEAR(56336.8 / 9.80665, hgt[2][0](0, 2), 0.05);
    EXPECT_NEAR(56332.0 / 9.80665, hgt[2][0](0, 3), 0.05);
    EXPECT_NEAR(56327.1 / 9.80665, hgt[2][0](0, 4), 0.05);
    EXPECT_NEAR(56355.8 / 9.80665, hgt[2][0](1, 0), 0.05);
    EXPECT_NEAR(56364.0 / 9.80665, hgt[2][0](2, 0), 0.05);
    EXPECT_NEAR(56371.5 / 9.80665, hgt[2][0](3, 0), 0.05);
    EXPECT_NEAR(56354.3 / 9.80665, hgt[2][0](10, 10), 0.05);

    /* Values time step 5 (horizontal=Lon, vertical=Lat)
    Extracted:
    56611.3   56609.9   56608.0   56606.0   56603.5   56600.8   56598.5   56596.1   56594.0   56592.0   56589.9
    56616.1   56614.1   56611.9   56609.4   56606.9   56604.1   56601.5   56599.1   56597.1   56595.1   56593.4
    56619.1   56616.8   56614.3   56612.1   56609.6   56607.0   56604.4   56602.0   56600.0   56598.3   56596.8
    56621.1   56618.8   56616.5   56614.4   56612.4   56610.1   56607.9   56605.6   56603.8   56602.3   56600.9
    56623.5   56621.1   56619.4   56617.9   56616.3   56614.5   56612.6   56610.8   56609.1   56607.6   56606.0
    56627.9   56625.9   56624.3   56623.0   56621.5   56619.8   56618.0   56616.4   56614.9   56613.1   56611.3
    56634.4   56632.3   56630.4   56628.6   56626.8   56624.9   56623.1   56621.5   56619.9   56618.1   56616.1
    56639.8   56637.6   56635.4   56633.1   56631.3   56629.4   56627.8   56626.4   56624.9   56623.4   56621.8
    56643.0   56640.8   56638.5   56636.5   56634.8   56633.4   56632.1   56631.0   56629.6   56628.4   56627.0
    56646.5   56644.0   56641.8   56639.6   56637.9   56636.3   56635.1   56634.1   56633.0   56632.0   56630.8
    56651.9   56649.3   56646.6   56644.1   56641.8   56639.6   56638.0   56636.6   56635.4   56634.4   56633.4
    */
    EXPECT_NEAR(56611.3 / 9.80665, hgt[5][0](0, 0), 0.05);
    EXPECT_NEAR(56609.9 / 9.80665, hgt[5][0](0, 1), 0.05);
    EXPECT_NEAR(56608.0 / 9.80665, hgt[5][0](0, 2), 0.05);
    EXPECT_NEAR(56606.0 / 9.80665, hgt[5][0](0, 3), 0.05);
    EXPECT_NEAR(56603.5 / 9.80665, hgt[5][0](0, 4), 0.05);
    EXPECT_NEAR(56616.1 / 9.80665, hgt[5][0](1, 0), 0.05);
    EXPECT_NEAR(56619.1 / 9.80665, hgt[5][0](2, 0), 0.05);
    EXPECT_NEAR(56621.1 / 9.80665, hgt[5][0](3, 0), 0.05);
    EXPECT_NEAR(56633.4 / 9.80665, hgt[5][0](10, 10), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}

TEST(PredictorOperMeteoFranceArpege, LoadRelativeHumidity) {
    vwxs filepaths;
    filepaths.push_back(wxFileName::GetCwd() + "/files/data-meteofrance-arpege/2023/02/02/ARP_RELATIVE_HUMIDITY__ISOBARIC_SURFACE_850_202302020000_202302020000.grb");

    asTimeArray dates(asTime::GetMJD(2023, 2, 2, 00), asTime::GetMJD(2023, 2, 2, 00), 6, "Simple");
    dates.Init();

    double xMin = -6;
    int xPtsNb = 11;
    double yMin = 47;
    int yPtsNb = 11;
    double step = 0.1;
    float level = 850;
    wxString gridType = "Regular";
    asAreaGrid* area = asAreaGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    asPredictorOper* predictor = asPredictorOper::GetInstance("MF_ARPEGE_Forecast", "r");
    wxASSERT(predictor);

    // Create file names
    predictor->SetFileNames(filepaths);

    // Load
    ASSERT_TRUE(predictor->Load(area, dates, level));

    vva2f rh = predictor->GetData();
    // rh[time][mem](lat,lon)

    /* Values time step 0 (horizontal=Lon, vertical=Lat)
    Extracted:
    29.8   31.6   33.4   35.2   37.2   39.1   40.2   39.8   38.9   37.3   35.3
    35.1   34.9   34.7   34.5   34.2   33.3   32.1   30.8   30.4   29.7   28.3
    40.7   39.1   36.1   33.2   31.3   30.3   29.2   28.5   27.7   26.4   24.6
    42.8   42.1   38.9   34.2   30.4   28.8   27.9   27.1   25.8   24.1   22.4
    40.5   40.6   38.6   34.7   30.4   27.7   26.5   25.7   24.6   23.0   21.2
    38.7   38.3   36.6   34.3   31.8   29.7   28.6   28.2   27.2   24.9   22.3
    37.2   37.3   36.5   35.1   33.5   32.2   31.8   31.7   31.2   28.9   25.2
    34.4   35.6   35.8   34.9   33.4   32.2   31.8   32.1   31.7   29.4   25.8
    30.8   31.7   32.0   31.3   29.9   28.8   28.2   27.8   27.0   24.4   20.4
    27.4   27.7   27.6   26.8   25.5   24.0   22.6   20.4   17.8   15.9   14.5
    25.8   25.4   24.9   23.9   22.5   20.9   18.9   15.8   12.7   11.2   10.5
    */
    EXPECT_NEAR(29.8, rh[0][0](0, 0), 0.05);
    EXPECT_NEAR(31.6, rh[0][0](0, 1), 0.05);
    EXPECT_NEAR(33.4, rh[0][0](0, 2), 0.05);
    EXPECT_NEAR(35.2, rh[0][0](0, 3), 0.05);
    EXPECT_NEAR(37.2, rh[0][0](0, 4), 0.05);
    EXPECT_NEAR(35.1, rh[0][0](1, 0), 0.05);
    EXPECT_NEAR(40.7, rh[0][0](2, 0), 0.05);
    EXPECT_NEAR(42.8, rh[0][0](3, 0), 0.05);
    EXPECT_NEAR(10.5, rh[0][0](10, 10), 0.05);

    wxDELETE(area);
    wxDELETE(predictor);
}
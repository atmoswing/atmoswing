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

#include "asAreaCompGrid.h"
#include "asPredictorOper.h"
#include "asTimeArray.h"
#include "gtest/gtest.h"

TEST(PredictorOperEcmwfIfs, LoadSingleDay) {
  vwxs filepaths;
  filepaths.push_back(wxFileName::GetCwd() + "/files/data-ecmwf-ifs-grib/2019-02-01_z.grib");

  asTimeArray dates(asTime::GetMJD(2019, 2, 1, 00), asTime::GetMJD(2019, 2, 1, 00), 6, "Simple");
  dates.Init();

  double xMin = -2;
  int xPtsNb = 6;
  double yMin = 45;
  int yPtsNb = 4;
  double step = 0.5;
  float level = 1000;
  wxString gridType = "Regular";
  asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

  asPredictorOper *predictor = asPredictorOper::GetInstance("ECMWF_IFS_GRIB_Forecast", "z");
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
  EXPECT_NEAR(-1107.0 / 9.80665, hgt[0][0](0, 0), 0.5);
  EXPECT_NEAR(-1059.0 / 9.80665, hgt[0][0](0, 1), 0.5);
  EXPECT_NEAR(-1007.2 / 9.80665, hgt[0][0](0, 2), 0.5);
  EXPECT_NEAR(-972.1 / 9.80665, hgt[0][0](0, 3), 0.5);
  EXPECT_NEAR(-959.9 / 9.80665, hgt[0][0](0, 4), 0.5);
  EXPECT_NEAR(-925.2 / 9.80665, hgt[0][0](0, 5), 0.5);
  EXPECT_NEAR(-1025.7 / 9.80665, hgt[0][0](1, 0), 0.5);
  EXPECT_NEAR(-947.1 / 9.80665, hgt[0][0](2, 0), 0.5);
  EXPECT_NEAR(-858.9 / 9.80665, hgt[0][0](3, 0), 0.5);
  EXPECT_NEAR(-727.9 / 9.80665, hgt[0][0](3, 5), 0.5);

  wxDELETE(area);
  wxDELETE(predictor);
}

TEST(PredictorOperEcmwfIfs, LoadSecondTimeStep) {
  vwxs filepaths;
  filepaths.push_back(wxFileName::GetCwd() + "/files/data-ecmwf-ifs-grib/2019-02-01_z.grib");

  asTimeArray dates(asTime::GetMJD(2019, 2, 1, 06), asTime::GetMJD(2019, 2, 1, 06), 6, "Simple");
  dates.Init();

  double xMin = -2;
  int xPtsNb = 6;
  double yMin = 45;
  int yPtsNb = 4;
  double step = 0.5;
  float level = 1000;
  wxString gridType = "Regular";
  asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

  asPredictorOper *predictor = asPredictorOper::GetInstance("ECMWF_IFS_GRIB_Forecast", "z");
  wxASSERT(predictor);

  // Create file names
  predictor->SetFileNames(filepaths);

  // Load
  ASSERT_TRUE(predictor->Load(area, dates, level));

  vva2f hgt = predictor->GetData();
  // hgt[time][mem](lat,lon)

  /* Values time step 0 (horizontal=Lon, vertical=Lat)
  Extracted:
  -1382.5	-1327.9	-1262.4	-1218.4	-1187.6	-1137.3
  -1340.5	-1295.0	-1223.2	-1164.2	-1129.2	-1078.5
  -1287.1	-1232.4	-1162.9	-1107.2	-1075.5	-1024.2
  -1220.5	-1161.5	-1112.4	-1068.0	-1033.8	-986.9
  */
  EXPECT_NEAR(-1382.0 / 9.80665, hgt[0][0](0, 0), 0.5);
  EXPECT_NEAR(-1327.9 / 9.80665, hgt[0][0](0, 1), 0.5);
  EXPECT_NEAR(-1262.4 / 9.80665, hgt[0][0](0, 2), 0.5);
  EXPECT_NEAR(-1218.4 / 9.80665, hgt[0][0](0, 3), 0.5);
  EXPECT_NEAR(-1187.6 / 9.80665, hgt[0][0](0, 4), 0.5);
  EXPECT_NEAR(-1137.3 / 9.80665, hgt[0][0](0, 5), 0.5);
  EXPECT_NEAR(-1340.5 / 9.80665, hgt[0][0](1, 0), 0.5);
  EXPECT_NEAR(-1287.1 / 9.80665, hgt[0][0](2, 0), 0.5);
  EXPECT_NEAR(-1220.5 / 9.80665, hgt[0][0](3, 0), 0.5);
  EXPECT_NEAR(-986.9 / 9.80665, hgt[0][0](3, 5), 0.5);

  wxDELETE(area);
  wxDELETE(predictor);
}

TEST(PredictorOperEcmwfIfs, LoadLastTimeStep) {
  vwxs filepaths;
  filepaths.push_back(wxFileName::GetCwd() + "/files/data-ecmwf-ifs-grib/2019-02-01_z.grib");

  asTimeArray dates(asTime::GetMJD(2019, 2, 2, 18), asTime::GetMJD(2019, 2, 2, 18), 6, "Simple");
  dates.Init();

  double xMin = -2;
  int xPtsNb = 6;
  double yMin = 45;
  int yPtsNb = 4;
  double step = 0.5;
  float level = 1000;
  wxString gridType = "Regular";
  asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

  asPredictorOper *predictor = asPredictorOper::GetInstance("ECMWF_IFS_GRIB_Forecast", "z");
  wxASSERT(predictor);

  // Create file names
  predictor->SetFileNames(filepaths);

  // Load
  ASSERT_TRUE(predictor->Load(area, dates, level));

  vva2f hgt = predictor->GetData();
  // hgt[time][mem](lat,lon)

  /* Values time step 0 (horizontal=Lon, vertical=Lat)
  Extracted:
  1037.8	986.4	931.6	874.8	818.0	765.6
  1030.3	979.8	932.8	876.9	822.5	780.5
  1030.2	981.8	934.3	876.8	827.8	786.0
  1042.3	987.1	944.1	898.4	846.0	793.8
  */
  EXPECT_NEAR(1037.8 / 9.80665, hgt[0][0](0, 0), 0.5);
  EXPECT_NEAR(986.4 / 9.80665, hgt[0][0](0, 1), 0.5);
  EXPECT_NEAR(931.6 / 9.80665, hgt[0][0](0, 2), 0.5);
  EXPECT_NEAR(874.8 / 9.80665, hgt[0][0](0, 3), 0.5);
  EXPECT_NEAR(818.0 / 9.80665, hgt[0][0](0, 4), 0.5);
  EXPECT_NEAR(765.6 / 9.80665, hgt[0][0](0, 5), 0.5);
  EXPECT_NEAR(1030.3 / 9.80665, hgt[0][0](1, 0), 0.5);
  EXPECT_NEAR(1030.2 / 9.80665, hgt[0][0](2, 0), 0.5);
  EXPECT_NEAR(1042.3 / 9.80665, hgt[0][0](3, 0), 0.5);
  EXPECT_NEAR(793.8 / 9.80665, hgt[0][0](3, 5), 0.5);

  wxDELETE(area);
  wxDELETE(predictor);
}

TEST(PredictorOperEcmwfIfs, LoadFullTimeArray) {
  vwxs filepaths;
  filepaths.push_back(wxFileName::GetCwd() + "/files/data-ecmwf-ifs-grib/2019-02-01_z.grib");

  asTimeArray dates(asTime::GetMJD(2019, 2, 1, 00), asTime::GetMJD(2019, 2, 2, 18), 6, "Simple");
  dates.Init();

  double xMin = -2;
  int xPtsNb = 6;
  double yMin = 45;
  int yPtsNb = 4;
  double step = 0.5;
  float level = 1000;
  wxString gridType = "Regular";
  asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

  asPredictorOper *predictor = asPredictorOper::GetInstance("ECMWF_IFS_GRIB_Forecast", "z");
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
  EXPECT_NEAR(-1107.0 / 9.80665, hgt[0][0](0, 0), 0.5);
  EXPECT_NEAR(-1059.0 / 9.80665, hgt[0][0](0, 1), 0.5);
  EXPECT_NEAR(-1007.2 / 9.80665, hgt[0][0](0, 2), 0.5);
  EXPECT_NEAR(-972.1 / 9.80665, hgt[0][0](0, 3), 0.5);
  EXPECT_NEAR(-959.9 / 9.80665, hgt[0][0](0, 4), 0.5);
  EXPECT_NEAR(-925.2 / 9.80665, hgt[0][0](0, 5), 0.5);
  EXPECT_NEAR(-1025.7 / 9.80665, hgt[0][0](1, 0), 0.5);
  EXPECT_NEAR(-947.1 / 9.80665, hgt[0][0](2, 0), 0.5);
  EXPECT_NEAR(-858.9 / 9.80665, hgt[0][0](3, 0), 0.5);
  EXPECT_NEAR(-727.9 / 9.80665, hgt[0][0](3, 5), 0.5);

  /* Values time step 1 (horizontal=Lon, vertical=Lat)
  Extracted:
  -1382.5	-1327.9	-1262.4	-1218.4	-1187.6	-1137.3
  -1340.5	-1295.0	-1223.2	-1164.2	-1129.2	-1078.5
  -1287.1	-1232.4	-1162.9	-1107.2	-1075.5	-1024.2
  -1220.5	-1161.5	-1112.4	-1068.0	-1033.8	-986.9
  */
  EXPECT_NEAR(-1382.0 / 9.80665, hgt[1][0](0, 0), 0.5);
  EXPECT_NEAR(-1327.9 / 9.80665, hgt[1][0](0, 1), 0.5);
  EXPECT_NEAR(-1262.4 / 9.80665, hgt[1][0](0, 2), 0.5);
  EXPECT_NEAR(-1218.4 / 9.80665, hgt[1][0](0, 3), 0.5);
  EXPECT_NEAR(-1187.6 / 9.80665, hgt[1][0](0, 4), 0.5);
  EXPECT_NEAR(-1137.3 / 9.80665, hgt[1][0](0, 5), 0.5);
  EXPECT_NEAR(-1340.5 / 9.80665, hgt[1][0](1, 0), 0.5);
  EXPECT_NEAR(-1287.1 / 9.80665, hgt[1][0](2, 0), 0.5);
  EXPECT_NEAR(-1220.5 / 9.80665, hgt[1][0](3, 0), 0.5);
  EXPECT_NEAR(-986.9 / 9.80665, hgt[1][0](3, 5), 0.5);

  /* Values time step 7 (horizontal=Lon, vertical=Lat)
  Extracted:
  1037.8	986.4	931.6	874.8	818.0	765.6
  1030.3	979.8	932.8	876.9	822.5	780.5
  1030.2	981.8	934.3	876.8	827.8	786.0
  1042.3	987.1	944.1	898.4	846.0	793.8
  */
  EXPECT_NEAR(1037.8 / 9.80665, hgt[7][0](0, 0), 0.5);
  EXPECT_NEAR(986.4 / 9.80665, hgt[7][0](0, 1), 0.5);
  EXPECT_NEAR(931.6 / 9.80665, hgt[7][0](0, 2), 0.5);
  EXPECT_NEAR(874.8 / 9.80665, hgt[7][0](0, 3), 0.5);
  EXPECT_NEAR(818.0 / 9.80665, hgt[7][0](0, 4), 0.5);
  EXPECT_NEAR(765.6 / 9.80665, hgt[7][0](0, 5), 0.5);
  EXPECT_NEAR(1030.3 / 9.80665, hgt[7][0](1, 0), 0.5);
  EXPECT_NEAR(1030.2 / 9.80665, hgt[7][0](2, 0), 0.5);
  EXPECT_NEAR(1042.3 / 9.80665, hgt[7][0](3, 0), 0.5);
  EXPECT_NEAR(793.8 / 9.80665, hgt[7][0](3, 5), 0.5);

  wxDELETE(area);
  wxDELETE(predictor);
}

TEST(PredictorOperEcmwfIfs, LoadTimeArrayWithoutFirst) {
  vwxs filepaths;
  filepaths.push_back(wxFileName::GetCwd() + "/files/data-ecmwf-ifs-grib/2019-02-01_z.grib");

  asTimeArray dates(asTime::GetMJD(2019, 2, 1, 06), asTime::GetMJD(2019, 2, 2, 18), 6, "Simple");
  dates.Init();

  double xMin = -2;
  int xPtsNb = 6;
  double yMin = 45;
  int yPtsNb = 4;
  double step = 0.5;
  float level = 1000;
  wxString gridType = "Regular";
  asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

  asPredictorOper *predictor = asPredictorOper::GetInstance("ECMWF_IFS_GRIB_Forecast", "z");
  wxASSERT(predictor);

  // Create file names
  predictor->SetFileNames(filepaths);

  // Load
  ASSERT_TRUE(predictor->Load(area, dates, level));

  vva2f hgt = predictor->GetData();
  // hgt[time][mem](lat,lon)

  /* Values time step 1 (horizontal=Lon, vertical=Lat)
  Extracted:
  -1382.5	-1327.9	-1262.4	-1218.4	-1187.6	-1137.3
  -1340.5	-1295.0	-1223.2	-1164.2	-1129.2	-1078.5
  -1287.1	-1232.4	-1162.9	-1107.2	-1075.5	-1024.2
  -1220.5	-1161.5	-1112.4	-1068.0	-1033.8	-986.9
  */
  EXPECT_NEAR(-1382.0 / 9.80665, hgt[0][0](0, 0), 0.5);
  EXPECT_NEAR(-1327.9 / 9.80665, hgt[0][0](0, 1), 0.5);
  EXPECT_NEAR(-1262.4 / 9.80665, hgt[0][0](0, 2), 0.5);
  EXPECT_NEAR(-1218.4 / 9.80665, hgt[0][0](0, 3), 0.5);
  EXPECT_NEAR(-1187.6 / 9.80665, hgt[0][0](0, 4), 0.5);
  EXPECT_NEAR(-1137.3 / 9.80665, hgt[0][0](0, 5), 0.5);
  EXPECT_NEAR(-1340.5 / 9.80665, hgt[0][0](1, 0), 0.5);
  EXPECT_NEAR(-1287.1 / 9.80665, hgt[0][0](2, 0), 0.5);
  EXPECT_NEAR(-1220.5 / 9.80665, hgt[0][0](3, 0), 0.5);
  EXPECT_NEAR(-986.9 / 9.80665, hgt[0][0](3, 5), 0.5);

  /* Values time step 7 (horizontal=Lon, vertical=Lat)
  Extracted:
  1037.8	986.4	931.6	874.8	818.0	765.6
  1030.3	979.8	932.8	876.9	822.5	780.5
  1030.2	981.8	934.3	876.8	827.8	786.0
  1042.3	987.1	944.1	898.4	846.0	793.8
  */
  EXPECT_NEAR(1037.8 / 9.80665, hgt[6][0](0, 0), 0.5);
  EXPECT_NEAR(986.4 / 9.80665, hgt[6][0](0, 1), 0.5);
  EXPECT_NEAR(931.6 / 9.80665, hgt[6][0](0, 2), 0.5);
  EXPECT_NEAR(874.8 / 9.80665, hgt[6][0](0, 3), 0.5);
  EXPECT_NEAR(818.0 / 9.80665, hgt[6][0](0, 4), 0.5);
  EXPECT_NEAR(765.6 / 9.80665, hgt[6][0](0, 5), 0.5);
  EXPECT_NEAR(1030.3 / 9.80665, hgt[6][0](1, 0), 0.5);
  EXPECT_NEAR(1030.2 / 9.80665, hgt[6][0](2, 0), 0.5);
  EXPECT_NEAR(1042.3 / 9.80665, hgt[6][0](3, 0), 0.5);
  EXPECT_NEAR(793.8 / 9.80665, hgt[6][0](3, 5), 0.5);

  wxDELETE(area);
  wxDELETE(predictor);
}
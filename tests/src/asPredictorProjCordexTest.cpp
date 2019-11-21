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
 * Portions Copyright 2018 Pascal Horton, University of Bern.
 */

#include <wx/filename.h>

#include <gtest/gtest.h>
#include "asAreaCompGenGrid.h"
#include "asPredictorProj.h"
#include "asTimeArray.h"

TEST(PredictorProjCordex, LoadEasy) {
  wxString predictorDataDir = wxFileName::GetCwd();
  predictorDataDir.Append("/files/data-cordex/");

  asPredictorProj *predictor =
      asPredictorProj::GetInstance("CORDEX", "CNRM-CERFACS-CNRM-CM5", "rcp85", "zg500", predictorDataDir);

  double xMin = 100;
  double xWidth = 100;
  double yMin = 250;
  double yWidth = 100;
  asAreaCompGenGrid area(xMin, xWidth, yMin, yWidth, 0, predictor->IsLatLon());

  double start = asTime::GetMJD(2023, 10, 17, 00, 00);
  double end = asTime::GetMJD(2023, 10, 17, 00, 00);
  double timeStepHours = 24;
  asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
  timearray.Init();

  ASSERT_TRUE(predictor != nullptr);
  ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
  ASSERT_TRUE(predictor->Load(area, timearray, 0));

  vva2f z = predictor->GetData();
  // pr[time][mem](lat,lon)

  /* Values time step 0 (horizontal=Lon, vertical=Lat)
  5875.949	5875.210	5874.214
  5875.322	5874.633	5873.714
  5874.811	5874.221	5873.410
  */

  EXPECT_NEAR(5875.949, z[0][0](0, 0), 1E-3);
  EXPECT_NEAR(5875.210, z[0][0](0, 1), 1E-3);
  EXPECT_NEAR(5874.214, z[0][0](0, 2), 1E-3);
  EXPECT_NEAR(5875.322, z[0][0](1, 0), 1E-3);
  EXPECT_NEAR(5874.811, z[0][0](2, 0), 1E-3);
  EXPECT_NEAR(5873.410, z[0][0](2, 2), 1E-3);

  wxDELETE(predictor);
}

TEST(PredictorProjCordex, LoadOver2Files) {
  double xMin = 100;
  double xWidth = 100;
  double yMin = 250;
  double yWidth = 100;
  asAreaCompGenGrid area(xMin, xWidth, yMin, yWidth, asFLAT_ALLOWED, asPredictor::IsLatLon("CORDEX"));

  double start = asTime::GetMJD(2025, 12, 25, 00, 00);
  double end = asTime::GetMJD(2026, 1, 10, 00, 00);
  double timeStepHours = 24;
  asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
  timearray.Init();

  wxString predictorDataDir = wxFileName::GetCwd();
  predictorDataDir.Append("/files/data-cordex/");

  asPredictorProj *predictor =
      asPredictorProj::GetInstance("CORDEX", "CNRM-CERFACS-CNRM-CM5", "rcp85", "zg500", predictorDataDir);

  ASSERT_TRUE(predictor != nullptr);
  ASSERT_TRUE(predictor->GetParameter() == asPredictor::GeopotentialHeight);
  ASSERT_TRUE(predictor->Load(area, timearray, 0));

  vva2f z = predictor->GetData();
  // hgt[time][mem](lat,lon)

  /* Values time step 0 (horizontal=Lon, vertical=Lat) 25.12.2025
  5840.507	5840.260	5840.018
  5841.373	5841.219	5841.036
  5842.273	5842.142	5841.948
  */

  EXPECT_NEAR(5840.507, z[0][0](0, 0), 1E-3);
  EXPECT_NEAR(5840.260, z[0][0](0, 1), 1E-3);
  EXPECT_NEAR(5840.018, z[0][0](0, 2), 1E-3);
  EXPECT_NEAR(5841.373, z[0][0](1, 0), 1E-3);
  EXPECT_NEAR(5842.273, z[0][0](2, 0), 1E-3);
  EXPECT_NEAR(5841.948, z[0][0](2, 2), 1E-3);

  /* Values time step 6 (horizontal=Lon, vertical=Lat) 31.12.2025
  5812.036	5812.356	5812.426
  5814.760	5815.128	5815.196
  5817.457	5817.810	5817.814
  */

  EXPECT_NEAR(5812.036, z[6][0](0, 0), 1E-3);
  EXPECT_NEAR(5812.356, z[6][0](0, 1), 1E-3);
  EXPECT_NEAR(5812.426, z[6][0](0, 2), 1E-3);
  EXPECT_NEAR(5814.760, z[6][0](1, 0), 1E-3);
  EXPECT_NEAR(5817.457, z[6][0](2, 0), 1E-3);
  EXPECT_NEAR(5817.814, z[6][0](2, 2), 1E-3);

  /* Values time step 7 (horizontal=Lon, vertical=Lat) 01.01.2026
  5813.837	5815.055	5816.039
  5817.271	5818.455	5819.358
  5820.580	5821.659	5822.421
  */

  EXPECT_NEAR(5813.837, z[7][0](0, 0), 1E-3);
  EXPECT_NEAR(5815.055, z[7][0](0, 1), 1E-3);
  EXPECT_NEAR(5816.039, z[7][0](0, 2), 1E-3);
  EXPECT_NEAR(5817.271, z[7][0](1, 0), 1E-3);
  EXPECT_NEAR(5820.580, z[7][0](2, 0), 1E-3);
  EXPECT_NEAR(5822.421, z[7][0](2, 2), 1E-3);

  /* Values time step 16 (horizontal=Lon, vertical=Lat) 10.01.2026
  5797.511	5797.913	5798.130
  5802.229	5802.723	5802.962
  5806.777	5807.274	5807.468
  */

  EXPECT_NEAR(5797.511, z[16][0](0, 0), 1E-3);
  EXPECT_NEAR(5797.913, z[16][0](0, 1), 1E-3);
  EXPECT_NEAR(5798.130, z[16][0](0, 2), 1E-3);
  EXPECT_NEAR(5802.229, z[16][0](1, 0), 1E-3);
  EXPECT_NEAR(5806.777, z[16][0](2, 0), 1E-3);
  EXPECT_NEAR(5807.468, z[16][0](2, 2), 1E-3);

  wxDELETE(predictor);
}

TEST(PredictorProjCordex, LoadAnotherModel) {
  double xMin = -28.15;
  double xWidth = 0.1;
  double yMin = -19.52;
  double yWidth = 0.1;
  asAreaCompGenGrid area(xMin, xWidth, yMin, yWidth, asFLAT_ALLOWED, asPredictor::IsLatLon("CORDEX"));

  double start = asTime::GetMJD(2046, 1, 1, 00, 00);
  double end = asTime::GetMJD(2046, 1, 10, 00, 00);
  double timeStepHours = 24;
  asTimeArray timearray(start, end, timeStepHours, asTimeArray::Simple);
  timearray.Init();

  wxString predictorDataDir = wxFileName::GetCwd();
  predictorDataDir.Append("/files/data-cordex/");

  asPredictorProj *predictor =
      asPredictorProj::GetInstance("CORDEX", "ICHEC-EC-EARTH", "rcp85", "psl", predictorDataDir);

  ASSERT_TRUE(predictor != nullptr);
  ASSERT_TRUE(predictor->GetParameter() == asPredictor::Pressure);
  ASSERT_TRUE(predictor->Load(area, timearray, 0));

  vva2f psl = predictor->GetData();
  // psl[time][mem](lat,lon)

  /* Values time step 0 (horizontal=Lon, vertical=Lat) 01.01.2046
  102389.211	102396.633
  102392.961	102401.336
  */

  EXPECT_NEAR(102389.211, psl[0][0](0, 0), 1E-3);
  EXPECT_NEAR(102396.633, psl[0][0](0, 1), 1E-3);
  EXPECT_NEAR(102392.961, psl[0][0](1, 0), 1E-3);
  EXPECT_NEAR(102401.336, psl[0][0](1, 1), 1E-3);

  /* Values time step 1 (horizontal=Lon, vertical=Lat) 02.01.2046
  102346.586	102354.922
  102349.172	102358.336
  */

  EXPECT_NEAR(102346.586, psl[1][0](0, 0), 1E-3);
  EXPECT_NEAR(102354.922, psl[1][0](0, 1), 1E-3);
  EXPECT_NEAR(102349.172, psl[1][0](1, 0), 1E-3);
  EXPECT_NEAR(102358.336, psl[1][0](1, 1), 1E-3);

  /* Values time step 4 (horizontal=Lon, vertical=Lat) 05.01.2046
  102453.555	102459.594
  102462.055	102465.555
  */

  EXPECT_NEAR(102453.555, psl[4][0](0, 0), 1E-3);
  EXPECT_NEAR(102459.594, psl[4][0](0, 1), 1E-3);
  EXPECT_NEAR(102462.055, psl[4][0](1, 0), 1E-3);
  EXPECT_NEAR(102465.555, psl[4][0](1, 1), 1E-3);

  wxDELETE(predictor);
}
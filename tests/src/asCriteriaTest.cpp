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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include <asAreaCompRegGrid.h>
#include <asCriteria.h>
#include <asFileText.h>
#include <asPredictor.h>
#include <asPreprocessor.h>
#include <asTimeArray.h>

#include "gtest/gtest.h"
#include "wx/filename.h"

TEST(Criteria, S1) {
  // Get the data file
  wxString filepath = wxFileName::GetCwd();
  filepath.Append(_T("/files/criteria_S1.txt"));
  asFileText file(filepath, asFile::ReadOnly);
  file.Open();

  // Create the containers
  int lons = 9;
  int lats = 5;
  a2f refZ1000(lats, lons), candZ1000(lats, lons);
  a2f refZ500(lats, lons), candZ500(lats, lons);

  // Skip the header
  file.SkipLines(9);

  // Get target data Z1000
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refZ1000(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(137, refZ1000(0, 0));
  EXPECT_FLOAT_EQ(89, refZ1000(1, 2));
  EXPECT_FLOAT_EQ(137, refZ1000(4, 8));

  // Skip coasent
  file.SkipLines(3);

  // Get target data Z500
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refZ500(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(5426, refZ500(0, 0));
  EXPECT_FLOAT_EQ(5721, refZ500(4, 8));

  // Vectors for candidates results
  int candidatesNb = 10;
  vf checkZ1000(candidatesNb), checkZ500(candidatesNb), critS1(candidatesNb);

  // Real values for the read checks
  checkZ1000[0] = 122;
  checkZ1000[1] = 98;
  checkZ1000[2] = 104;
  checkZ1000[3] = 92;
  checkZ1000[4] = 101;
  checkZ1000[5] = 107;
  checkZ1000[6] = 84;
  checkZ1000[7] = 158;
  checkZ1000[8] = 96;
  checkZ1000[9] = 114;
  checkZ500[0] = 5618;
  checkZ500[1] = 5667;
  checkZ500[2] = 5533;
  checkZ500[3] = 5642;
  checkZ500[4] = 5614;
  checkZ500[5] = 5582;
  checkZ500[6] = 5537;
  checkZ500[7] = 5574;
  checkZ500[8] = 5729;
  checkZ500[9] = 5660;

  // Real values for the S1 checks
  critS1[0] = 38.0f;
  critS1[1] = 40.7f;
  critS1[2] = 41.4f;
  critS1[3] = 43.7f;
  critS1[4] = 45.1f;
  critS1[5] = 46.5f;
  critS1[6] = 47.8f;
  critS1[7] = 56.6f;
  critS1[8] = 61.1f;
  critS1[9] = 61.8f;

  // Instantiate the criteria
  asCriteria *criteria = asCriteria::GetInstance("S1");

  // Loop on every candidate
  for (int iCand = 0; iCand < candidatesNb; iCand++) {
    // Skip coasent
    file.SkipLines(6);

    // Get candidate data Z1000
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candZ1000(iLat, iLon) = file.GetFloat();
      }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(checkZ1000[iCand], candZ1000(4, 8));

    // Skip coasent
    file.SkipLines(3);

    // Get candidate data Z500
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candZ500(iLat, iLon) = file.GetFloat();
      }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(checkZ500[iCand], candZ500(4, 8));

    // Process S1 and check the results
    float resZ1000, resZ500, res;

    resZ1000 = criteria->Assess(refZ1000, candZ1000, refZ1000.rows(), refZ1000.cols());
    resZ500 = criteria->Assess(refZ500, candZ500, refZ500.rows(), refZ500.cols());
    res = (resZ500 + resZ1000) / 2;
    EXPECT_NEAR(critS1[iCand], res, 0.05);
  }

  wxDELETE(criteria);
}

TEST(Criteria, S1preprocessed) {
  double xMin = 10;
  double xWidth = 10;
  double yMin = 35;
  double yWidth = 5;
  double step = 2.5;
  float level = 1000;
  asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

  double start = asTime::GetMJD(1960, 1, 1, 00, 00);
  double end = asTime::GetMJD(1960, 1, 11, 00, 00);
  double timeStep = 6;
  asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
  timearray.Init();

  wxString predictorDataDir = wxFileName::GetCwd();
  predictorDataDir.Append("/files/data-ncep-r1/v2003/");

  asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v1", "pressure/hgt", predictorDataDir);

  ASSERT_TRUE(predictor->Load(&area, timearray, level));
  std::vector<asPredictor *> vdata;
  vdata.push_back(predictor);
  vva2f hgtOriginal = predictor->GetData();

  wxString method = "SimpleGradients";
  asPredictor *gradients = new asPredictor(*predictor);
  asPreprocessor::Preprocess(vdata, method, gradients);
  vva2f hgtPreproc = gradients->GetData();

  // Create the containers
  int lonsOriginal = hgtOriginal[0][0].cols();
  int latsOriginal = hgtOriginal[0][0].rows();
  a2f refOriginal(latsOriginal, lonsOriginal), candOriginal(latsOriginal, lonsOriginal);

  int lonsPreproc = hgtPreproc[0][0].cols();
  int latsPreproc = hgtPreproc[0][0].rows();
  a2f refPreproc(latsPreproc, lonsPreproc), candPreproc(latsPreproc, lonsPreproc);

  // Set target data
  refOriginal = hgtOriginal[0][0];
  refPreproc = hgtPreproc[0][0];

  // Vectors for results
  int candidatesNb = hgtOriginal.size();
  vf critS1(candidatesNb);
  EXPECT_TRUE(candidatesNb > 1);

  // Instantiate the criteria
  asCriteria *criteria = asCriteria::GetInstance("S1");
  asCriteria *criteriaGrads = asCriteria::GetInstance("S1grads");

  // Loop on every candidate
  for (int iCand = 1; iCand < candidatesNb; iCand++) {
    float S1Original, S1Preproc;

    // Get candidate data
    candOriginal = hgtOriginal[iCand][0];
    candPreproc = hgtPreproc[iCand][0];

    S1Original = criteria->Assess(refOriginal, candOriginal, candOriginal.rows(), candOriginal.cols());
    S1Preproc = criteriaGrads->Assess(refPreproc, candPreproc, candPreproc.rows(), candPreproc.cols());
    EXPECT_FLOAT_EQ(S1Original, S1Preproc);
  }

  wxDELETE(predictor);
  wxDELETE(gradients);
  wxDELETE(criteria);
  wxDELETE(criteriaGrads);
}

TEST(Criteria, S2preprocessed) {
  double xMin = 10;
  double xWidth = 10;
  double yMin = 35;
  double yWidth = 5;
  double step = 2.5;
  float level = 1000;
  asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

  double start = asTime::GetMJD(1960, 1, 1, 00, 00);
  double end = asTime::GetMJD(1960, 1, 11, 00, 00);
  double timeStep = 6;
  asTimeArray timearray(start, end, timeStep, asTimeArray::Simple);
  timearray.Init();

  wxString predictorDataDir = wxFileName::GetCwd();
  predictorDataDir.Append("/files/data-ncep-r1/v2003/");

  asPredictor *predictor = asPredictor::GetInstance("NCEP_Reanalysis_v1", "pressure/hgt", predictorDataDir);

  ASSERT_TRUE(predictor->Load(&area, timearray, level));
  std::vector<asPredictor *> vdata;
  vdata.push_back(predictor);
  vva2f hgtOriginal = predictor->GetData();

  wxString method = "SimpleCurvature";
  asPredictor *curv = new asPredictor(*predictor);
  asPreprocessor::Preprocess(vdata, method, curv);
  vva2f hgtPreproc = curv->GetData();

  // Create the containers
  int lonsOriginal = hgtOriginal[0][0].cols();
  int latsOriginal = hgtOriginal[0][0].rows();
  a2f refOriginal(latsOriginal, lonsOriginal), candOriginal(latsOriginal, lonsOriginal);

  int lonsPreproc = hgtPreproc[0][0].cols();
  int latsPreproc = hgtPreproc[0][0].rows();
  a2f refPreproc(latsPreproc, lonsPreproc), candPreproc(latsPreproc, lonsPreproc);

  // Set target data
  refOriginal = hgtOriginal[0][0];
  refPreproc = hgtPreproc[0][0];

  // Vectors for results
  int candidatesNb = hgtOriginal.size();
  vf critS1(candidatesNb);
  EXPECT_TRUE(candidatesNb > 1);

  // Instantiate the criteria
  asCriteria *criteria = asCriteria::GetInstance("S2");
  asCriteria *criteriaGrads = asCriteria::GetInstance("S2grads");

  // Loop on every candidate
  for (int iCand = 1; iCand < candidatesNb; iCand++) {
    float S2Original, S2Preproc;

    // Get candidate data
    candOriginal = hgtOriginal[iCand][0];
    candPreproc = hgtPreproc[iCand][0];

    S2Original = criteria->Assess(refOriginal, candOriginal, candOriginal.rows(), candOriginal.cols());
    S2Preproc = criteriaGrads->Assess(refPreproc, candPreproc, candPreproc.rows(), candPreproc.cols());
    EXPECT_FLOAT_EQ(S2Original, S2Preproc);
  }

  wxDELETE(predictor);
  wxDELETE(curv);
  wxDELETE(criteria);
  wxDELETE(criteriaGrads);
}

TEST(Criteria, RSE) {
  // Get the data file
  wxString filepath = wxFileName::GetCwd();
  filepath.Append(_T("/files/criteria_RMSE.txt"));
  asFileText file(filepath, asFile::ReadOnly);
  file.Open();

  // Create the containers
  int lons = 2;
  int lats = 2;
  a2f refPRWTR(lats, 2 * lons), candPRWTR(lats, 2 * lons);
  a2f refRHUM850(lats, 2 * lons), candRHUM850(lats, 2 * lons);
  a2f refMulti(lats, 2 * lons), candMulti(lats, 2 * lons);

  // Skip the header
  file.SkipLines(9);

  // Get target data PRWTR12h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refPRWTR(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(13.6f, refPRWTR(0, 0));
  EXPECT_FLOAT_EQ(20.4f, refPRWTR(1, 1));

  // Skip coasent
  file.SkipLines(3);

  // Get target data PRWTR24h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refPRWTR(iLat, iLon + lons) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(13.3f, refPRWTR(0, 2));
  EXPECT_FLOAT_EQ(18.1f, refPRWTR(1, 3));

  // Skip coasent
  file.SkipLines(3);

  // Get target data RHUM85012h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refRHUM850(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(82, refRHUM850(0, 0));
  EXPECT_FLOAT_EQ(100, refRHUM850(1, 1));

  // Skip coasent
  file.SkipLines(3);

  // Get target data RHUM85024h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refRHUM850(iLat, iLon + lons) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(100, refRHUM850(0, 2));
  EXPECT_FLOAT_EQ(96, refRHUM850(1, 3));

  // Process to the multiplication
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < 2 * lons; iLon++) {
      refMulti(iLat, iLon) = refPRWTR(iLat, iLon) * refRHUM850(iLat, iLon);
    }
  }

  // Vectors for candidates results
  int candidatesNb = 7;
  vf checkPRWTR(candidatesNb), checkRHUM850(candidatesNb), critRMSE(candidatesNb);

  // Real values for the read checks
  checkPRWTR[0] = 16.7f;
  checkPRWTR[1] = 17.4f;
  checkPRWTR[2] = 16.3f;
  checkPRWTR[3] = 16.8f;
  checkPRWTR[4] = 15.1f;
  checkPRWTR[5] = 16.7f;
  checkPRWTR[6] = 13.3f;
  checkRHUM850[0] = 100;
  checkRHUM850[1] = 100;
  checkRHUM850[2] = 97;
  checkRHUM850[3] = 100;
  checkRHUM850[4] = 98;
  checkRHUM850[5] = 88;
  checkRHUM850[6] = 83;

  // Real values for the RMSE checks
  critRMSE[0] = 648.0f;
  critRMSE[1] = 649.5f;
  critRMSE[2] = 773.3f;
  critRMSE[3] = 854.8f;
  critRMSE[4] = 1131.7f;
  critRMSE[5] = 1554.0f;
  critRMSE[6] = 1791.5f;

  // Instantiate the criteria
  asCriteria *criteria = asCriteria::GetInstance("RSE");

  // Loop on every candidate
  for (int iCand = 0; iCand < candidatesNb; iCand++) {
    // Skip coasent
    file.SkipLines(6);

    // Get candidate data PRWTR12h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candPRWTR(iLat, iLon) = file.GetFloat();
      }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(checkPRWTR[iCand], candPRWTR(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get candidate data PRWTR24h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candPRWTR(iLat, iLon + lons) = file.GetFloat();
      }
    }

    // Skip coasent
    file.SkipLines(3);

    // Get candidate data RHUM85012h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candRHUM850(iLat, iLon) = file.GetFloat();
      }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(checkRHUM850[iCand], candRHUM850(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get candidate data RHUM85024h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candRHUM850(iLat, iLon + lons) = file.GetFloat();
      }
    }

    // Process to the multiplication
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < 2 * lons; iLon++) {
        candMulti(iLat, iLon) = candPRWTR(iLat, iLon) * candRHUM850(iLat, iLon);
      }
    }

    // Process RMSE and check the results
    float res;

    res = criteria->Assess(refMulti, candMulti, refMulti.rows(), refMulti.cols());
    EXPECT_NEAR(critRMSE[iCand], res, 0.05);
  }

  wxDELETE(criteria);
}

TEST(Criteria, RMSE) {
  // Get the data file
  wxString filepath = wxFileName::GetCwd();
  filepath.Append(_T("/files/criteria_RMSE.txt"));
  asFileText file(filepath, asFile::ReadOnly);
  file.Open();

  // Create the containers
  int lons = 2;
  int lats = 2;
  a2f refPRWTR12h(lats, lons), refPRWTR24h(lats, lons), candPRWTR12h(lats, lons), candPRWTR24h(lats, lons);
  a2f refRHUM85012h(lats, lons), refRHUM85024h(lats, lons), candRHUM85012h(lats, lons), candRHUM85024h(lats, lons);
  a2f refMulti12h(lats, lons), refMulti24h(lats, lons), candMulti12h(lats, lons), candMulti24h(lats, lons);

  // Skip the header
  file.SkipLines(9);

  // Get target data PRWTR12h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refPRWTR12h(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(13.6f, refPRWTR12h(0, 0));
  EXPECT_FLOAT_EQ(20.4f, refPRWTR12h(1, 1));

  // Skip coasent
  file.SkipLines(3);

  // Get target data PRWTR24h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refPRWTR24h(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(13.3f, refPRWTR24h(0, 0));
  EXPECT_FLOAT_EQ(18.1f, refPRWTR24h(1, 1));

  // Skip coasent
  file.SkipLines(3);

  // Get target data RHUM85012h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refRHUM85012h(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(82, refRHUM85012h(0, 0));
  EXPECT_FLOAT_EQ(100, refRHUM85012h(1, 1));

  // Skip coasent
  file.SkipLines(3);

  // Get target data RHUM85024h
  for (int iLat = 0; iLat < lats; iLat++) {
    for (int iLon = 0; iLon < lons; iLon++) {
      refRHUM85024h(iLat, iLon) = file.GetFloat();
    }
  }

  // Check that the data were correctly read from the file
  EXPECT_FLOAT_EQ(100, refRHUM85024h(0, 0));
  EXPECT_FLOAT_EQ(96, refRHUM85024h(1, 1));

  // Process to the multiplication
  refMulti12h = refPRWTR12h * refRHUM85012h;
  refMulti24h = refPRWTR24h * refRHUM85024h;

  // Vectors for candidates results
  int candidatesNb = 7;
  vf checkPRWTR12h(candidatesNb), checkRHUM85012h(candidatesNb), critRMSE(candidatesNb);

  // Real values for the read checks
  checkPRWTR12h[0] = 16.7f;
  checkPRWTR12h[1] = 17.4f;
  checkPRWTR12h[2] = 16.3f;
  checkPRWTR12h[3] = 16.8f;
  checkPRWTR12h[4] = 15.1f;
  checkPRWTR12h[5] = 16.7f;
  checkPRWTR12h[6] = 13.3f;
  checkRHUM85012h[0] = 100;
  checkRHUM85012h[1] = 100;
  checkRHUM85012h[2] = 97;
  checkRHUM85012h[3] = 100;
  checkRHUM85012h[4] = 98;
  checkRHUM85012h[5] = 88;
  checkRHUM85012h[6] = 83;

  // Real values for the RMSE checks
  critRMSE[0] = 223.51f;
  critRMSE[1] = 208.97f;
  critRMSE[2] = 271.64f;
  critRMSE[3] = 302.15f;
  critRMSE[4] = 329.03f;
  critRMSE[5] = 537.73f;
  critRMSE[6] = 632.32f;

  // Instantiate the criteria
  asCriteria *criteria = asCriteria::GetInstance("RMSE");

  // Loop on every candidate
  for (int iCand = 0; iCand < candidatesNb; iCand++) {
    // Skip coasent
    file.SkipLines(6);

    // Get candidate data PRWTR12h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candPRWTR12h(iLat, iLon) = file.GetFloat();
      }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(checkPRWTR12h[iCand], candPRWTR12h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get candidate data PRWTR24h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candPRWTR24h(iLat, iLon) = file.GetFloat();
      }
    }

    // Skip coasent
    file.SkipLines(3);

    // Get candidate data RHUM85012h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candRHUM85012h(iLat, iLon) = file.GetFloat();
      }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(checkRHUM85012h[iCand], candRHUM85012h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get candidate data RHUM85024h
    for (int iLat = 0; iLat < lats; iLat++) {
      for (int iLon = 0; iLon < lons; iLon++) {
        candRHUM85024h(iLat, iLon) = file.GetFloat();
      }
    }

    // Process to the multiplication
    candMulti12h = candPRWTR12h * candRHUM85012h;
    candMulti24h = candPRWTR24h * candRHUM85024h;

    // Process RMSE and check the results
    float res12h, res24h, res;

    res12h = criteria->Assess(refMulti12h, candMulti12h, refMulti12h.rows(), refMulti12h.cols());
    res24h = criteria->Assess(refMulti24h, candMulti24h, refMulti24h.rows(), refMulti24h.cols());
    res = (res12h + res24h) / 2;
    EXPECT_NEAR(critRMSE[iCand], res, 0.05);
  }

  wxDELETE(criteria);
}

TEST(Criteria, RMSEwithNaNs) {
  // Create the containers
  int lons = 4;
  int lats = 4;
  a2f refData(lats, lons), candData(lats, lons);

  refData << 0.4903f, 0.1785f, 0.9245f, 0.4212f, 0.6166f, NaNf, 0.4496f, 0.612f, 0.531f, 0.5112f, 0.6155f, 0.4553f,
      0.2353f, 0.5602f, 0.2524f, 0.8011f;
  candData << 0.0136f, 0.2671f, 0.3951f, 0.8645f, 0.0489f, 0.0921f, 0.6901f, 0.0887f, 0.5477f, 0.0562f, 0.4862f,
      0.9309f, 0.3185f, 0.2835f, 0.5472f, NaNf;

  asCriteria *criteria = asCriteria::GetInstance("RMSE");

  float res = criteria->Assess(refData, candData, refData.rows(), refData.cols());

  EXPECT_FLOAT_EQ(0.3766817f, res);

  wxDELETE(criteria);
}

TEST(Criteria, Differences) {
  va2f refData;
  a2f dataTmp(2, 2);

  dataTmp(0, 0) = 12;
  dataTmp(0, 1) = 23;
  dataTmp(1, 0) = 42;
  dataTmp(1, 1) = 25;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = 547;
  dataTmp(0, 1) = 2364;
  dataTmp(1, 0) = 2672;
  dataTmp(1, 1) = 3256;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 0;
  dataTmp(1, 0) = 0;
  dataTmp(1, 1) = 0;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 0;
  dataTmp(1, 0) = 0;
  dataTmp(1, 1) = 0;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 5;
  dataTmp(1, 0) = 0;
  dataTmp(1, 1) = 0;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = 456;
  dataTmp(0, 1) = 456;
  dataTmp(1, 0) = 45;
  dataTmp(1, 1) = 7;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = -324;
  dataTmp(0, 1) = -345;
  dataTmp(1, 0) = -23;
  dataTmp(1, 1) = -26;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = -34;
  dataTmp(0, 1) = -45;
  dataTmp(1, 0) = 456;
  dataTmp(1, 1) = 3;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 0;
  dataTmp(1, 0) = 0;
  dataTmp(1, 1) = 0;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = 4;
  dataTmp(0, 1) = 456;
  dataTmp(1, 0) = 4;
  dataTmp(1, 1) = 783;
  refData.push_back(dataTmp);
  dataTmp(0, 0) = -345;
  dataTmp(0, 1) = -325;
  dataTmp(1, 0) = -27;
  dataTmp(1, 1) = -475;
  refData.push_back(dataTmp);

  va2f candData;
  dataTmp(0, 0) = 634;
  dataTmp(0, 1) = 234;
  dataTmp(1, 0) = 3465;
  dataTmp(1, 1) = 534;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = 7;
  dataTmp(0, 1) = 3;
  dataTmp(1, 0) = 35;
  dataTmp(1, 1) = 4;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = 54;
  dataTmp(0, 1) = 56;
  dataTmp(1, 0) = 4;
  dataTmp(1, 1) = 74;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 0;
  dataTmp(1, 0) = 0;
  dataTmp(1, 1) = 0;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 0;
  dataTmp(1, 0) = 4;
  dataTmp(1, 1) = 0;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 0;
  dataTmp(1, 0) = 0;
  dataTmp(1, 1) = 0;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = 34;
  dataTmp(0, 1) = 2;
  dataTmp(1, 0) = 235;
  dataTmp(1, 1) = 6;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = 0;
  dataTmp(0, 1) = 0;
  dataTmp(1, 0) = 0;
  dataTmp(1, 1) = 0;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = -637;
  dataTmp(0, 1) = -6;
  dataTmp(1, 0) = -67;
  dataTmp(1, 1) = 567;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = -37;
  dataTmp(0, 1) = -65;
  dataTmp(1, 0) = -4;
  dataTmp(1, 1) = -1;
  candData.push_back(dataTmp);
  dataTmp(0, 0) = -867;
  dataTmp(0, 1) = -568;
  dataTmp(1, 0) = -43;
  dataTmp(1, 1) = -348;
  candData.push_back(dataTmp);

  // SAD

  vf results(11);
  results[0] = 4765;
  results[1] = 8790;
  results[2] = 188;
  results[3] = 0;
  results[4] = 9;
  results[5] = 964;
  results[6] = 995;
  results[7] = 538;
  results[8] = 1277;
  results[9] = 1354;
  results[10] = 908;

  asCriteria *criteriaSAD = asCriteria::GetInstance("SAD");

  float res;
  for (int i = 0; i < 11; i++) {
    res = criteriaSAD->Assess(refData[i], candData[i], refData[i].rows(), refData[i].cols());
    EXPECT_FLOAT_EQ(results[i], res);
  }

  wxDELETE(criteriaSAD);

  // MD

  results[0] = 1191.25;
  results[1] = 2197.5;
  results[2] = 47;
  results[3] = 0;
  results[4] = 2.25;
  results[5] = 241;
  results[6] = 248.75;
  results[7] = 134.5;
  results[8] = 319.25;
  results[9] = 338.5;
  results[10] = 227;

  asCriteria *criteriaMD = asCriteria::GetInstance("MD");

  for (int i = 0; i < 11; i++) {
    res = criteriaMD->Assess(refData[i], candData[i], refData[i].rows(), refData[i].cols());
    EXPECT_FLOAT_EQ(results[i], res);
  }

  wxDELETE(criteriaMD);
}

TEST(Criteria, DMV) {
  // Create the containers
  int lons = 4;
  int lats = 4;
  a2f refData(lats, lons), candData(lats, lons);
  a2f refDataNaN(lats, lons), candDataNaN(lats, lons);

  refData << 0.4903f, 0.1785f, 0.9245f, 0.4212f, 0.6166f, 0.7172f, 0.4496f, 0.612f, 0.531f, 0.5112f, 0.6155f, 0.4553f,
      0.2353f, 0.5602f, 0.2524f, 0.8011f;
  candData << 0.0136f, 0.2671f, 0.3951f, 0.8645f, 0.0489f, 0.0921f, 0.6901f, 0.0887f, 0.5477f, 0.0562f, 0.4862f,
      0.9309f, 0.3185f, 0.2835f, 0.5472f, 0.7519f;

  refDataNaN << 0.4903f, 0.1785f, 0.9245f, 0.4212f, 0.6166f, NaNf, 0.4496f, 0.612f, 0.531f, 0.5112f, 0.6155f, 0.4553f,
      0.2353f, 0.5602f, 0.2524f, 0.8011f;
  candDataNaN << 0.0136f, 0.2671f, 0.3951f, 0.8645f, 0.0489f, 0.0921f, 0.6901f, 0.0887f, 0.5477f, 0.0562f, 0.4862f,
      0.9309f, 0.3185f, 0.2835f, 0.5472f, NaNf;

  asCriteria *criteria = asCriteria::GetInstance("DMV");

  float res = criteria->Assess(refData, candData, refData.rows(), refData.cols());
  float resNaN = criteria->Assess(refDataNaN, candDataNaN, refDataNaN.rows(), refDataNaN.cols());

  EXPECT_FLOAT_EQ(0.1243563f, res);
  EXPECT_FLOAT_EQ(0.09395714f, resNaN);

  wxDELETE(criteria);
}

TEST(Criteria, DSD) {
  // Create the containers
  int lons = 4;
  int lats = 4;
  a2f refData(lats, lons), candData(lats, lons);
  a2f refDataNaN(lats, lons), candDataNaN(lats, lons);

  refData << 0.4903f, 0.1785f, 0.9245f, 0.4212f, 0.6166f, 0.7172f, 0.4496f, 0.612f, 0.531f, 0.5112f, 0.6155f, 0.4553f,
      0.2353f, 0.5602f, 0.2524f, 0.8011f;
  candData << 0.0136f, 0.2671f, 0.3951f, 0.8645f, 0.0489f, 0.0921f, 0.6901f, 0.0887f, 0.5477f, 0.0562f, 0.4862f,
      0.9309f, 0.3185f, 0.2835f, 0.5472f, 0.7519f;

  refDataNaN << 0.4903f, 0.1785f, 0.9245f, 0.4212f, 0.6166f, NaNf, 0.4496f, 0.612f, 0.531f, 0.5112f, 0.6155f, 0.4553f,
      0.2353f, 0.5602f, 0.2524f, 0.8011f;
  candDataNaN << 0.0136f, 0.2671f, 0.3951f, 0.8645f, 0.0489f, 0.0921f, 0.6901f, 0.0887f, 0.5477f, 0.0562f, 0.4862f,
      0.9309f, 0.3185f, 0.2835f, 0.5472f, NaNf;

  asCriteria *criteria = asCriteria::GetInstance("DSD");

  float res = criteria->Assess(refData, candData, refData.rows(), refData.cols());
  float resNaN = criteria->Assess(refDataNaN, candDataNaN, refDataNaN.rows(), refDataNaN.cols());

  EXPECT_FLOAT_EQ(0.10311281f, res);
  EXPECT_FLOAT_EQ(0.108632276f, resNaN);

  wxDELETE(criteria);
}

TEST(Criteria, Gauss2D) {
  // Array 51x31
  a2f surf1 = asCriteria::GetGauss2D(51, 31);
  ASSERT_EQ(51, surf1.rows());
  ASSERT_EQ(31, surf1.cols());
  EXPECT_NEAR(0.027135, surf1(0, 0), 0.00001);
  EXPECT_NEAR(0.034035, surf1(0, 1), 0.00001);
  EXPECT_NEAR(0.042028, surf1(0, 2), 0.00001);
  EXPECT_NEAR(0.036048, surf1(2, 0), 0.00001);
  EXPECT_NEAR(0.027135, surf1(0, 30), 0.00001);
  EXPECT_NEAR(1, surf1(25, 15), 0.00001);
  EXPECT_NEAR(0.031368, surf1(49, 0), 0.00001);
  EXPECT_NEAR(0.027135, surf1(50, 0), 0.00001);
  EXPECT_NEAR(0.21467, surf1(41, 25), 0.00001);

  // Array 1x1
  a2f surf2 = asCriteria::GetGauss2D(1, 1);
  ASSERT_EQ(1, surf2.rows());
  ASSERT_EQ(1, surf2.cols());
  EXPECT_NEAR(1, surf2(0, 0), 0.00001);

  // Array 3x3
  a2f surf3 = asCriteria::GetGauss2D(3, 3);
  ASSERT_EQ(3, surf3.rows());
  ASSERT_EQ(3, surf3.cols());
  EXPECT_NEAR(0.36788, surf3(0, 0), 0.00001);
  EXPECT_NEAR(0.60653, surf3(1, 0), 0.00001);
  EXPECT_NEAR(1, surf3(1, 1), 0.00001);

  // Array 4x4
  a2f surf4 = asCriteria::GetGauss2D(4, 4);
  ASSERT_EQ(4, surf4.rows());
  ASSERT_EQ(4, surf4.cols());
  EXPECT_NEAR(0.23693, surf4(0, 0), 0.00001);
  EXPECT_NEAR(0.44933, surf4(1, 0), 0.00001);
  EXPECT_NEAR(0.85214, surf4(1, 1), 0.00001);
  EXPECT_NEAR(0.85214, surf4(1, 2), 0.00001);

  // Array 1x4
  a2f surf5 = asCriteria::GetGauss2D(1, 4);
  ASSERT_EQ(1, surf5.rows());
  ASSERT_EQ(4, surf5.cols());
  EXPECT_NEAR(0.48675, surf5(0, 0), 0.00001);
  EXPECT_NEAR(0.92312, surf5(0, 1), 0.00001);
}

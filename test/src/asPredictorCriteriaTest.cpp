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

#include "wx/filename.h"
#include <asPredictorCriteria.h>
#include <asGeoAreaCompositeRegularGrid.h>
#include <asDataPredictorArchive.h>
#include <asPreprocessor.h>
#include <asFileAscii.h>
#include <asTimeArray.h>
#include "gtest/gtest.h"


TEST(PredictorCriteria, ProcessS1)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/predictor_criteria_S1.txt"));
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int lons = 9;
    int lats = 5;
    Array2DFloat RefZ1000, CandZ1000;
    RefZ1000.resize(lats, lons);
    CandZ1000.resize(lats, lons);
    Array2DFloat RefZ500, CandZ500;
    RefZ500.resize(lats, lons);
    CandZ500.resize(lats, lons);

    // Skip the header
    file.SkipLines(9);

    // Get target data Z1000
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefZ1000(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(137, RefZ1000(0, 0));
    EXPECT_FLOAT_EQ(89, RefZ1000(1, 2));
    EXPECT_FLOAT_EQ(137, RefZ1000(4, 8));

    // Skip coasent
    file.SkipLines(3);

    // Get target data Z500
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefZ500(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(5426, RefZ500(0, 0));
    EXPECT_FLOAT_EQ(5721, RefZ500(4, 8));

    // Vectors for candidates results
    int candidatesNb = 10;
    VectorFloat checkZ1000, checkZ500, critS1;
    checkZ1000.resize(candidatesNb);
    checkZ500.resize(candidatesNb);
    critS1.resize(candidatesNb);

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
    asPredictorCriteria *criteria = asPredictorCriteria::GetInstance(_("S1"));

    // Loop on every candidate
    for (int i_cand = 0; i_cand < candidatesNb; i_cand++) {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data Z1000
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandZ1000(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkZ1000[i_cand], CandZ1000(4, 8));

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data Z500
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandZ500(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkZ500[i_cand], CandZ500(4, 8));

        // Process S1 and check the results
        float resZ1000, resZ500, res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, RefZ1000.rows(), RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, RefZ500.rows(), RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, RefZ1000.rows(), RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, RefZ500.rows(), RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, RefZ1000.rows(), RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, RefZ500.rows(), RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, RefZ1000.rows(), RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, RefZ500.rows(), RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);
    }

    wxDELETE(criteria);

}

TEST(PredictorCriteria, ProcessNS1)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/predictor_criteria_S1.txt"));
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int lons = 9;
    int lats = 5;
    Array2DFloat RefZ1000, CandZ1000;
    RefZ1000.resize(lats, lons);
    CandZ1000.resize(lats, lons);
    Array2DFloat RefZ500, CandZ500;
    RefZ500.resize(lats, lons);
    CandZ500.resize(lats, lons);

    // Skip the header
    file.SkipLines(9);

    // Get target data Z1000
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefZ1000(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(137, RefZ1000(0, 0));
    EXPECT_FLOAT_EQ(89, RefZ1000(1, 2));
    EXPECT_FLOAT_EQ(137, RefZ1000(4, 8));

    // Skip coasent
    file.SkipLines(3);

    // Get target data Z500
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefZ500(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(5426, RefZ500(0, 0));
    EXPECT_FLOAT_EQ(5721, RefZ500(4, 8));

    // Vectors for candidates results
    int candidatesNb = 10;
    VectorFloat checkZ1000, checkZ500, critS1;
    checkZ1000.resize(candidatesNb);
    checkZ500.resize(candidatesNb);
    critS1.resize(candidatesNb);

    // Real values for the read checks
    checkZ1000[0] = 122.0f / 200.0f;
    checkZ1000[1] = 98.0f / 200.0f;
    checkZ1000[2] = 104.0f / 200.0f;
    checkZ1000[3] = 92.0f / 200.0f;
    checkZ1000[4] = 101.0f / 200.0f;
    checkZ1000[5] = 107.0f / 200.0f;
    checkZ1000[6] = 84.0f / 200.0f;
    checkZ1000[7] = 158.0f / 200.0f;
    checkZ1000[8] = 96.0f / 200.0f;
    checkZ1000[9] = 114.0f / 200.0f;
    checkZ500[0] = 5618.0f / 200.0f;
    checkZ500[1] = 5667.0f / 200.0f;
    checkZ500[2] = 5533.0f / 200.0f;
    checkZ500[3] = 5642.0f / 200.0f;
    checkZ500[4] = 5614.0f / 200.0f;
    checkZ500[5] = 5582.0f / 200.0f;
    checkZ500[6] = 5537.0f / 200.0f;
    checkZ500[7] = 5574.0f / 200.0f;
    checkZ500[8] = 5729.0f / 200.0f;
    checkZ500[9] = 5660.0f / 200.0f;

    // Real values for the S1 checks
    critS1[0] = 38.0f / 200.0f;
    critS1[1] = 40.7f / 200.0f;
    critS1[2] = 41.4f / 200.0f;
    critS1[3] = 43.7f / 200.0f;
    critS1[4] = 45.1f / 200.0f;
    critS1[5] = 46.5f / 200.0f;
    critS1[6] = 47.8f / 200.0f;
    critS1[7] = 56.6f / 200.0f;
    critS1[8] = 61.1f / 200.0f;
    critS1[9] = 61.8f / 200.0f;

    // Instantiate the criteria
    asPredictorCriteria *criteria = asPredictorCriteria::GetInstance(_("NS1"));

    // Loop on every candidate
    for (int i_cand = 0; i_cand < candidatesNb; i_cand++) {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data Z1000
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandZ1000(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkZ1000[i_cand], CandZ1000(4, 8) / 200.0f);

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data Z500
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandZ500(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkZ500[i_cand], CandZ500(4, 8) / 200.0f);

        // Process S1 and check the results
        float resZ1000, resZ500, res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, (int) RefZ1000.rows(), (int) RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, (int) RefZ500.rows(), (int) RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, (int) RefZ1000.rows(), (int) RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, (int) RefZ500.rows(), (int) RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, (int) RefZ1000.rows(), (int) RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, (int) RefZ500.rows(), (int) RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000, (int) RefZ1000.rows(), (int) RefZ1000.cols());
        resZ500 = criteria->Assess(RefZ500, CandZ500, (int) RefZ500.rows(), (int) RefZ500.cols());
        res = (resZ500 + resZ1000) / 2;
        EXPECT_NEAR(critS1[i_cand], res, 0.05);
    }

    wxDELETE(criteria);

}

TEST(PredictorCriteria, ProcessS1preprocessed)
{
    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 35;
    double Ywidth = 5;
    double step = 2.5;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2003/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "press/hgt",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(&geoArea, timearray));
    std::vector<asDataPredictorArchive *> vdata;
    vdata.push_back(predictor);
    VArray2DFloat hgtOriginal = predictor->GetData();

    wxString method = "Gradients";
    asDataPredictorArchive *gradients = new asDataPredictorArchive(*predictor);
    asPreprocessor::Preprocess(vdata, method, gradients);
    VArray2DFloat hgtPreproc = gradients->GetData();

    // Resize the containers
    int lonsOriginal = hgtOriginal[0].cols();
    int latsOriginal = hgtOriginal[0].rows();
    Array2DFloat RefOriginal, CandOriginal;
    RefOriginal.resize(latsOriginal, lonsOriginal);
    CandOriginal.resize(latsOriginal, lonsOriginal);

    int lonsPreproc = hgtPreproc[0].cols();
    int latsPreproc = hgtPreproc[0].rows();
    Array2DFloat RefPreproc, CandPreproc;
    RefPreproc.resize(latsPreproc, lonsPreproc);
    CandPreproc.resize(latsPreproc, lonsPreproc);

    // Set target data
    RefOriginal = hgtOriginal[0];
    RefPreproc = hgtPreproc[0];

    // Vectors for results
    int candidatesNb = hgtOriginal.size();
    VectorFloat critS1;
    critS1.resize(candidatesNb);
    EXPECT_TRUE(candidatesNb > 1);

    // Instantiate the criteria
    asPredictorCriteria *criteria = asPredictorCriteria::GetInstance(_("S1"));
    asPredictorCriteria *criteriaGrads = asPredictorCriteria::GetInstance(_("S1grads"));

    // Loop on every candidate
    for (int i_cand = 1; i_cand < candidatesNb; i_cand++) {
        float S1Original, S1Preproc;

        // Get candidate data
        CandOriginal = hgtOriginal[i_cand];
        CandPreproc = hgtPreproc[i_cand];

        // Process the score
        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);
    }

    wxDELETE(predictor);
    wxDELETE(gradients);
    wxDELETE(criteria);
    wxDELETE(criteriaGrads);

}

TEST(PredictorCriteria, ProcessNS1preprocessed)
{
    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 35;
    double Ywidth = 5;
    double step = 2.5;
    float level = 1000;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step, level);

    double start = asTime::GetMJD(1960, 1, 1, 00, 00);
    double end = asTime::GetMJD(1960, 1, 11, 00, 00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    wxString predictorDataDir = wxFileName::GetCwd();
    predictorDataDir.Append("/files/data-ncep-r1/v2003/");

    asDataPredictorArchive *predictor = asDataPredictorArchive::GetInstance("NCEP_Reanalysis_v1", "press/hgt",
                                                                            predictorDataDir);

    ASSERT_TRUE(predictor->Load(&geoArea, timearray));
    std::vector<asDataPredictorArchive *> vdata;
    vdata.push_back(predictor);
    VArray2DFloat hgtOriginal = predictor->GetData();

    wxString method = "Gradients";
    asDataPredictorArchive *gradients = new asDataPredictorArchive(*predictor);
    asPreprocessor::Preprocess(vdata, method, gradients);
    VArray2DFloat hgtPreproc = gradients->GetData();

    // Resize the containers
    int lonsOriginal = hgtOriginal[0].cols();
    int latsOriginal = hgtOriginal[0].rows();
    Array2DFloat RefOriginal, CandOriginal;
    RefOriginal.resize(latsOriginal, lonsOriginal);
    CandOriginal.resize(latsOriginal, lonsOriginal);

    int lonsPreproc = hgtPreproc[0].cols();
    int latsPreproc = hgtPreproc[0].rows();
    Array2DFloat RefPreproc, CandPreproc;
    RefPreproc.resize(latsPreproc, lonsPreproc);
    CandPreproc.resize(latsPreproc, lonsPreproc);

    // Set target data
    RefOriginal = hgtOriginal[0];
    RefPreproc = hgtPreproc[0];

    // Vectors for results
    int candidatesNb = hgtOriginal.size();
    VectorFloat critS1;
    critS1.resize(candidatesNb);
    EXPECT_TRUE(candidatesNb > 1);

    // Instantiate the criteria
    asPredictorCriteria *criteria = asPredictorCriteria::GetInstance(_("NS1"));
    asPredictorCriteria *criteriaGrads = asPredictorCriteria::GetInstance(_("NS1grads"));

    // Loop on every candidate
    for (int i_cand = 1; i_cand < candidatesNb; i_cand++) {
        float S1Original, S1Preproc;

        // Get candidate data
        CandOriginal = hgtOriginal[i_cand];
        CandPreproc = hgtPreproc[i_cand];

        // Process the score
        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        S1Original = criteria->Assess(RefOriginal, CandOriginal, CandOriginal.rows(), CandOriginal.cols());
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        EXPECT_FLOAT_EQ(S1Original, S1Preproc);
    }

    wxDELETE(predictor);
    wxDELETE(gradients);
    wxDELETE(criteria);
    wxDELETE(criteriaGrads);

}

TEST(PredictorCriteria, ProcessRSE)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/predictor_criteria_RMSE.txt"));
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int lons = 2;
    int lats = 2;
    Array2DFloat RefPRWTR, CandPRWTR;
    RefPRWTR.resize(lats, 2 * lons);
    CandPRWTR.resize(lats, 2 * lons);
    Array2DFloat RefRHUM850, CandRHUM850;
    RefRHUM850.resize(lats, 2 * lons);
    CandRHUM850.resize(lats, 2 * lons);
    Array2DFloat RefMulti, CandMulti;
    RefMulti.resize(lats, 2 * lons);
    CandMulti.resize(lats, 2 * lons);

    // Skip the header
    file.SkipLines(9);

    // Get target data PRWTR12h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefPRWTR(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(13.6f, RefPRWTR(0, 0));
    EXPECT_FLOAT_EQ(20.4f, RefPRWTR(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data PRWTR24h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefPRWTR(i_lat, i_lon + lons) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(13.3f, RefPRWTR(0, 2));
    EXPECT_FLOAT_EQ(18.1f, RefPRWTR(1, 3));

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85012h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefRHUM850(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(82, RefRHUM850(0, 0));
    EXPECT_FLOAT_EQ(100, RefRHUM850(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85024h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefRHUM850(i_lat, i_lon + lons) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(100, RefRHUM850(0, 2));
    EXPECT_FLOAT_EQ(96, RefRHUM850(1, 3));

    // Process to the multiplication
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < 2 * lons; i_lon++) {
            RefMulti(i_lat, i_lon) = RefPRWTR(i_lat, i_lon) * RefRHUM850(i_lat, i_lon);
        }
    }

    // Vectors for candidates results
    int candidatesNb = 7;
    VectorFloat checkPRWTR, checkRHUM850, critRMSE;
    checkPRWTR.resize(candidatesNb);
    checkRHUM850.resize(candidatesNb);
    critRMSE.resize(candidatesNb);

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
    asPredictorCriteria *criteria = asPredictorCriteria::GetInstance(asPredictorCriteria::RSE);

    // Loop on every candidate
    for (int i_cand = 0; i_cand < candidatesNb; i_cand++) {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data PRWTR12h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandPRWTR(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkPRWTR[i_cand], CandPRWTR(1, 1));

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data PRWTR24h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandPRWTR(i_lat, i_lon + lons) = file.GetFloat();
            }
        }

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85012h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandRHUM850(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkRHUM850[i_cand], CandRHUM850(1, 1));

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85024h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandRHUM850(i_lat, i_lon + lons) = file.GetFloat();
            }
        }

        // Process to the multiplication
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < 2 * lons; i_lon++) {
                CandMulti(i_lat, i_lon) = CandPRWTR(i_lat, i_lon) * CandRHUM850(i_lat, i_lon);
            }
        }

        // Process RMSE and check the results
        float res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteria->Assess(RefMulti, CandMulti, RefMulti.rows(), RefMulti.cols());
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteria->Assess(RefMulti, CandMulti, RefMulti.rows(), RefMulti.cols());
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteria->Assess(RefMulti, CandMulti, RefMulti.rows(), RefMulti.cols());
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteria->Assess(RefMulti, CandMulti, RefMulti.rows(), RefMulti.cols());
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);
    }

    wxDELETE(criteria);

}

TEST(PredictorCriteria, ProcessRMSE)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/predictor_criteria_RMSE.txt"));
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int lons = 2;
    int lats = 2;
    Array2DFloat RefPRWTR12h, RefPRWTR24h, CandPRWTR12h, CandPRWTR24h;
    RefPRWTR12h.resize(lats, lons);
    RefPRWTR24h.resize(lats, lons);
    CandPRWTR12h.resize(lats, lons);
    CandPRWTR24h.resize(lats, lons);
    Array2DFloat RefRHUM85012h, RefRHUM85024h, CandRHUM85012h, CandRHUM85024h;
    RefRHUM85012h.resize(lats, lons);
    RefRHUM85024h.resize(lats, lons);
    CandRHUM85012h.resize(lats, lons);
    CandRHUM85024h.resize(lats, lons);
    Array2DFloat RefMulti12h, RefMulti24h, CandMulti12h, CandMulti24h;
    RefMulti12h.resize(lats, lons);
    RefMulti24h.resize(lats, lons);
    CandMulti12h.resize(lats, lons);
    CandMulti24h.resize(lats, lons);

    // Skip the header
    file.SkipLines(9);

    // Get target data PRWTR12h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefPRWTR12h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(13.6f, RefPRWTR12h(0, 0));
    EXPECT_FLOAT_EQ(20.4f, RefPRWTR12h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data PRWTR24h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefPRWTR24h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(13.3f, RefPRWTR24h(0, 0));
    EXPECT_FLOAT_EQ(18.1f, RefPRWTR24h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85012h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefRHUM85012h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(82, RefRHUM85012h(0, 0));
    EXPECT_FLOAT_EQ(100, RefRHUM85012h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85024h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefRHUM85024h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(100, RefRHUM85024h(0, 0));
    EXPECT_FLOAT_EQ(96, RefRHUM85024h(1, 1));

    // Process to the multiplication
    RefMulti12h = RefPRWTR12h * RefRHUM85012h;
    RefMulti24h = RefPRWTR24h * RefRHUM85024h;

    // Vectors for candidates results
    int candidatesNb = 7;
    VectorFloat checkPRWTR12h, checkRHUM85012h, critRMSE;
    checkPRWTR12h.resize(candidatesNb);
    checkRHUM85012h.resize(candidatesNb);
    critRMSE.resize(candidatesNb);

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
    asPredictorCriteria *criteria = asPredictorCriteria::GetInstance(_("RMSE"));

    // Loop on every candidate
    for (int i_cand = 0; i_cand < candidatesNb; i_cand++) {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data PRWTR12h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandPRWTR12h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkPRWTR12h[i_cand], CandPRWTR12h(1, 1));

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data PRWTR24h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandPRWTR24h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85012h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandRHUM85012h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkRHUM85012h[i_cand], CandRHUM85012h(1, 1));

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85024h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandRHUM85024h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Process to the multiplication
        CandMulti12h = CandPRWTR12h * CandRHUM85012h;
        CandMulti24h = CandPRWTR24h * CandRHUM85024h;

        // Process RMSE and check the results
        float res12h, res24h, res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);
    }

    wxDELETE(criteria);

}

TEST(PredictorCriteria, ProcessNRMSE)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/predictor_criteria_RMSE.txt"));
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int lons = 2;
    int lats = 2;
    Array2DFloat RefPRWTR12h, RefPRWTR24h, CandPRWTR12h, CandPRWTR24h;
    RefPRWTR12h.resize(lats, lons);
    RefPRWTR24h.resize(lats, lons);
    CandPRWTR12h.resize(lats, lons);
    CandPRWTR24h.resize(lats, lons);
    Array2DFloat RefRHUM85012h, RefRHUM85024h, CandRHUM85012h, CandRHUM85024h;
    RefRHUM85012h.resize(lats, lons);
    RefRHUM85024h.resize(lats, lons);
    CandRHUM85012h.resize(lats, lons);
    CandRHUM85024h.resize(lats, lons);
    Array2DFloat RefMulti12h, RefMulti24h, CandMulti12h, CandMulti24h;
    RefMulti12h.resize(lats, lons);
    RefMulti24h.resize(lats, lons);
    CandMulti12h.resize(lats, lons);
    CandMulti24h.resize(lats, lons);

    // Skip the header
    file.SkipLines(9);

    // Get target data PRWTR12h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefPRWTR12h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(13.6f, RefPRWTR12h(0, 0));
    EXPECT_FLOAT_EQ(20.4f, RefPRWTR12h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data PRWTR24h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefPRWTR24h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(13.3f, RefPRWTR24h(0, 0));
    EXPECT_FLOAT_EQ(18.1f, RefPRWTR24h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85012h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefRHUM85012h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(82, RefRHUM85012h(0, 0));
    EXPECT_FLOAT_EQ(100, RefRHUM85012h(1, 1));

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85024h
    for (int i_lat = 0; i_lat < lats; i_lat++) {
        for (int i_lon = 0; i_lon < lons; i_lon++) {
            RefRHUM85024h(i_lat, i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    EXPECT_FLOAT_EQ(100, RefRHUM85024h(0, 0));
    EXPECT_FLOAT_EQ(96, RefRHUM85024h(1, 1));

    // Process to the multiplication
    RefMulti12h = RefPRWTR12h * RefRHUM85012h;
    RefMulti24h = RefPRWTR24h * RefRHUM85024h;

    // Vectors for candidates results
    int candidatesNb = 7;
    VectorFloat checkPRWTR12h, checkRHUM85012h, critRMSE;
    checkPRWTR12h.resize(candidatesNb);
    checkRHUM85012h.resize(candidatesNb);
    critRMSE.resize(candidatesNb);

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

    // Real/fake values for the RMSE checks
    critRMSE[0] = 223.51f / (2053.4f - 62.1f);
    critRMSE[1] = 208.97f / (2053.4f - 62.1f);
    critRMSE[2] = 271.64f / (2053.4f - 62.1f);
    critRMSE[3] = 302.15f / (2053.4f - 62.1f);
    critRMSE[4] = 329.03f / (2053.4f - 62.1f);
    critRMSE[5] = 537.73f / (2053.4f - 62.1f);
    critRMSE[6] = 632.32f / (2053.4f - 62.1f);

    // Instantiate the criteria
    asPredictorCriteria *criteria = asPredictorCriteria::GetInstance(_("NRMSE"));
    criteria->SetDataRange(62.1, 2053.4); // fake range here...

    // Loop on every candidate
    for (int i_cand = 0; i_cand < candidatesNb; i_cand++) {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data PRWTR12h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandPRWTR12h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkPRWTR12h[i_cand], CandPRWTR12h(1, 1));

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data PRWTR24h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandPRWTR24h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85012h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandRHUM85012h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        EXPECT_FLOAT_EQ(checkRHUM85012h[i_cand], CandRHUM85012h(1, 1));

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85024h
        for (int i_lat = 0; i_lat < lats; i_lat++) {
            for (int i_lon = 0; i_lon < lons; i_lon++) {
                CandRHUM85024h(i_lat, i_lon) = file.GetFloat();
            }
        }

        // Process to the multiplication
        CandMulti12h = CandPRWTR12h * CandRHUM85012h;
        CandMulti24h = CandPRWTR24h * CandRHUM85024h;

        // Process RMSE and check the results
        float res12h, res24h, res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h, RefMulti12h.rows(), RefMulti12h.cols());
        res24h = criteria->Assess(RefMulti24h, CandMulti24h, RefMulti24h.rows(), RefMulti24h.cols());
        res = (res12h + res24h) / 2;
        EXPECT_NEAR(critRMSE[i_cand], res, 0.05);
    }

    wxDELETE(criteria);

}

TEST(PredictorCriteria, ProcessDifferences)
{
    wxConfigBase *pConfig = wxFileConfig::Get();

    VArray2DFloat RefData;
    Array2DFloat Datatmp;
    Datatmp.resize(2, 2);

    Datatmp(0, 0) = 12;
    Datatmp(0, 1) = 23;
    Datatmp(1, 0) = 42;
    Datatmp(1, 1) = 25;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = 547;
    Datatmp(0, 1) = 2364;
    Datatmp(1, 0) = 2672;
    Datatmp(1, 1) = 3256;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 0;
    Datatmp(1, 0) = 0;
    Datatmp(1, 1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 0;
    Datatmp(1, 0) = 0;
    Datatmp(1, 1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 5;
    Datatmp(1, 0) = 0;
    Datatmp(1, 1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = 456;
    Datatmp(0, 1) = 456;
    Datatmp(1, 0) = 45;
    Datatmp(1, 1) = 7;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = -324;
    Datatmp(0, 1) = -345;
    Datatmp(1, 0) = -23;
    Datatmp(1, 1) = -26;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = -34;
    Datatmp(0, 1) = -45;
    Datatmp(1, 0) = 456;
    Datatmp(1, 1) = 3;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 0;
    Datatmp(1, 0) = 0;
    Datatmp(1, 1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = 4;
    Datatmp(0, 1) = 456;
    Datatmp(1, 0) = 4;
    Datatmp(1, 1) = 783;
    RefData.push_back(Datatmp);
    Datatmp(0, 0) = -345;
    Datatmp(0, 1) = -325;
    Datatmp(1, 0) = -27;
    Datatmp(1, 1) = -475;
    RefData.push_back(Datatmp);

    VArray2DFloat CandData;
    Datatmp(0, 0) = 634;
    Datatmp(0, 1) = 234;
    Datatmp(1, 0) = 3465;
    Datatmp(1, 1) = 534;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = 7;
    Datatmp(0, 1) = 3;
    Datatmp(1, 0) = 35;
    Datatmp(1, 1) = 4;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = 54;
    Datatmp(0, 1) = 56;
    Datatmp(1, 0) = 4;
    Datatmp(1, 1) = 74;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 0;
    Datatmp(1, 0) = 0;
    Datatmp(1, 1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 0;
    Datatmp(1, 0) = 4;
    Datatmp(1, 1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 0;
    Datatmp(1, 0) = 0;
    Datatmp(1, 1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = 34;
    Datatmp(0, 1) = 2;
    Datatmp(1, 0) = 235;
    Datatmp(1, 1) = 6;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = 0;
    Datatmp(0, 1) = 0;
    Datatmp(1, 0) = 0;
    Datatmp(1, 1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = -637;
    Datatmp(0, 1) = -6;
    Datatmp(1, 0) = -67;
    Datatmp(1, 1) = 567;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = -37;
    Datatmp(0, 1) = -65;
    Datatmp(1, 0) = -4;
    Datatmp(1, 1) = -1;
    CandData.push_back(Datatmp);
    Datatmp(0, 0) = -867;
    Datatmp(0, 1) = -568;
    Datatmp(1, 0) = -43;
    Datatmp(1, 1) = -348;
    CandData.push_back(Datatmp);

    // SAD

    VectorFloat Results;
    Results.resize(11);
    Results[0] = 4765;
    Results[1] = 8790;
    Results[2] = 188;
    Results[3] = 0;
    Results[4] = 9;
    Results[5] = 964;
    Results[6] = 995;
    Results[7] = 538;
    Results[8] = 1277;
    Results[9] = 1354;
    Results[10] = 908;

    asPredictorCriteria *criteriaSAD = asPredictorCriteria::GetInstance(asPredictorCriteria::SAD);

    float res;
    for (int i = 0; i < 11; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaSAD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaSAD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaSAD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaSAD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
    }

    wxDELETE(criteriaSAD);

    // MD

    Results[0] = 1191.25;
    Results[1] = 2197.5;
    Results[2] = 47;
    Results[3] = 0;
    Results[4] = 2.25;
    Results[5] = 241;
    Results[6] = 248.75;
    Results[7] = 134.5;
    Results[8] = 319.25;
    Results[9] = 338.5;
    Results[10] = 227;

    asPredictorCriteria *criteriaMD = asPredictorCriteria::GetInstance(asPredictorCriteria::MD);

    for (int i = 0; i < 11; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
    }

    wxDELETE(criteriaMD);

    // NMD

    Results[0] = 1191.25f / 2298.0f;
    Results[1] = 2197.5f / 2298.0f;
    Results[2] = 47.0f / 2298.0f;
    Results[3] = 0.0f / 2298.0f;
    Results[4] = 2.25f / 2298.0f;
    Results[5] = 241.0f / 2298.0f;
    Results[6] = 248.75f / 2298.0f;
    Results[7] = 134.5f / 2298.0f;
    Results[8] = 319.25f / 2298.0f;
    Results[9] = 338.5f / 2298.0f;
    Results[10] = 227.0f / 2298.0f;

    asPredictorCriteria *criteriaNMD = asPredictorCriteria::GetInstance(asPredictorCriteria::NMD);
    criteriaNMD->SetDataRange(2, 2300.0);

    for (int i = 0; i < 11; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaNMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaNMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaNMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaNMD->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_FLOAT_EQ(Results[i], res);
    }

    wxDELETE(criteriaNMD);

    // MRDtoMax

    Results[0] = 0.956f;
    Results[1] = 0.9929f;
    Results[2] = 1;
    Results[3] = 0;
    Results[4] = NaNFloat;
    Results[5] = 1;
    Results[6] = 1.1098f;
    Results[7] = 1;
    Results[8] = 1;
    Results[9] = 1.3130f;
    Results[10] = 0.4173f;

    asPredictorCriteria *criteriaMRDtoMax = asPredictorCriteria::GetInstance(asPredictorCriteria::MRDtoMax);

    for (int i = 0; i < 4; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
    }

    for (int i = 5; i < 11; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.0001);
    }

    wxDELETE(criteriaMRDtoMax);

    // MRDtoMean

    Results[0] = 1.835f;
    Results[1] = 1.972f;
    Results[2] = 2;
    Results[3] = 0;
    Results[4] = NaNFloat;
    Results[5] = 2;
    Results[6] = 2.532f;
    Results[7] = 2;
    Results[8] = 2;
    Results[9] = NaNFloat;
    Results[10] = 0.543f;

    asPredictorCriteria *criteriaMRDtoMean = asPredictorCriteria::GetInstance(asPredictorCriteria::MRDtoMean);

    for (int i = 0; i < 4; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
    }

    for (int i = 5; i < 9; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
    }

    for (int i = 10; i < 11; i++) {
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asCOEFF_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
        pConfig->Write("/Processing/LinAlgebra", (int) asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i], RefData[i].rows(), RefData[i].cols());
        EXPECT_NEAR(Results[i], res, 0.001);
    }

    wxDELETE(criteriaMRDtoMean);

}

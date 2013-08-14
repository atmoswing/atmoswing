#include "wx/filename.h"

#include "include_tests.h"
#include <asPredictorCriteria.h>
#include <asCatalogPredictors.h>
#include <asGeoAreaCompositeRegularGrid.h>
#include <asDataPredictorArchive.h>
#include <asPreprocessor.h>
#include <asFileAscii.h>
#include <asTimeArray.h>

#include "UnitTest++.h"

namespace
{

TEST(ProcessS1)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/asPredictorCriteriaTestFile01.txt"));
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
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefZ1000(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(137, RefZ1000(0,0), 0.00001);
    CHECK_CLOSE(89, RefZ1000(1,2), 0.00001);
    CHECK_CLOSE(137, RefZ1000(4,8), 0.00001);

    // Skip coasent
    file.SkipLines(3);

    // Get target data Z500
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefZ500(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(5426, RefZ500(0,0), 0.00001);
    CHECK_CLOSE(5721, RefZ500(4,8), 0.00001);

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
    asPredictorCriteria* criteria = asPredictorCriteria::GetInstance(_("S1"));

    // Loop on every candidate
    for (int i_cand=0; i_cand<candidatesNb; i_cand++)
    {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data Z1000
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandZ1000(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        CHECK_CLOSE(checkZ1000[i_cand], CandZ1000(4,8), 0.00001);

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data Z500
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandZ500(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        CHECK_CLOSE(checkZ500[i_cand], CandZ500(4,8), 0.00001);

        // Process S1 and check the results
        float resZ1000, resZ500, res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000);
        resZ500 = criteria->Assess(RefZ500, CandZ500);
        res = (resZ500+resZ1000)/2;
        CHECK_CLOSE(critS1[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000);
        resZ500 = criteria->Assess(RefZ500, CandZ500);
        res = (resZ500+resZ1000)/2;
        CHECK_CLOSE(critS1[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000);
        resZ500 = criteria->Assess(RefZ500, CandZ500);
        res = (resZ500+resZ1000)/2;
        CHECK_CLOSE(critS1[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        resZ1000 = criteria->Assess(RefZ1000, CandZ1000);
        resZ500 = criteria->Assess(RefZ500, CandZ500);
        res = (resZ500+resZ1000)/2;
        CHECK_CLOSE(critS1[i_cand], res, 0.05);
    }

}

TEST(ProcessS1preprocessed)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/asDataPredictorArchiveTestFile01.xml"));

    asCatalogPredictorsArchive catalog(filepath);
    catalog.Load(_("NCEP_R-1"),_("hgt"));

    filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/"));
    catalog.SetDataPath(filepath);

    double Umin = 10;
    double Uwidth = 10;
    double Vmin = 35;
    double Vwidth = 5;
    double step = 2.5;
    double level = 1000;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step, level);

    double start = asTime::GetMJD(1960,1,1,00,00);
    double end = asTime::GetMJD(1960,1,11,00,00);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    asDataPredictorArchive data(catalog);
    data.Load(geoarea, timearray);
    vector < asDataPredictorArchive > vdata;
    vdata.push_back(data);
    VArray2DFloat hgtOriginal = data.GetData();

    wxString method = "Gradients";
    asDataPredictorArchive gradients(data);
    asPreprocessor::Preprocess(vdata, method, &gradients);
    VArray2DFloat hgtPreproc = gradients.GetData();

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
    CHECK_EQUAL(true, candidatesNb>1);

    // Instantiate the criteria
    asPredictorCriteria* criteria = asPredictorCriteria::GetInstance(_("S1"));
    asPredictorCriteria* criteriaGrads = asPredictorCriteria::GetInstance(_("S1grads"));

    float S1Original, S1Preproc;

    // Loop on every candidate
    for (int i_cand=1; i_cand<candidatesNb; i_cand++)
    {
        // Get candidate data
        CandOriginal = hgtOriginal[i_cand];
        CandPreproc = hgtPreproc[i_cand];

        // Process the score
        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        S1Original = criteria->Assess(RefOriginal, CandOriginal);
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        CHECK_CLOSE(S1Original, S1Preproc, 0.0001);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        S1Original = criteria->Assess(RefOriginal, CandOriginal);
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        CHECK_CLOSE(S1Original, S1Preproc, 0.0001);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        S1Original = criteria->Assess(RefOriginal, CandOriginal);
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        CHECK_CLOSE(S1Original, S1Preproc, 0.0001);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        S1Original = criteria->Assess(RefOriginal, CandOriginal);
        S1Preproc = criteriaGrads->Assess(RefPreproc, CandPreproc, CandPreproc.rows(), CandPreproc.cols());
        CHECK_CLOSE(S1Original, S1Preproc, 0.0001);
    }

}

TEST(ProcessRSE)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/asPredictorCriteriaTestFile02.txt"));
    asFileAscii file(filepath, asFile::ReadOnly);
    file.Open();

    // Resize the containers
    int lons = 2;
    int lats = 2;
    Array2DFloat RefPRWTR, CandPRWTR;
    RefPRWTR.resize(lats, 2*lons);
    CandPRWTR.resize(lats, 2*lons);
    Array2DFloat RefRHUM850, CandRHUM850;
    RefRHUM850.resize(lats, 2*lons);
    CandRHUM850.resize(lats, 2*lons);
    Array2DFloat RefMulti, CandMulti;
    RefMulti.resize(lats, 2*lons);
    CandMulti.resize(lats, 2*lons);

    // Skip the header
    file.SkipLines(9);

    // Get target data PRWTR12h
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefPRWTR(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(13.6, RefPRWTR(0,0), 0.00001);
    CHECK_CLOSE(20.4, RefPRWTR(1,1), 0.00001);

    // Skip coasent
    file.SkipLines(3);

    // Get target data PRWTR24h
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefPRWTR(i_lat,i_lon+lons) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(13.3, RefPRWTR(0,2), 0.00001);
    CHECK_CLOSE(18.1, RefPRWTR(1,3), 0.00001);

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85012h
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefRHUM850(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(82, RefRHUM850(0,0), 0.00001);
    CHECK_CLOSE(100, RefRHUM850(1,1), 0.00001);

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85024h
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefRHUM850(i_lat,i_lon+lons) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(100, RefRHUM850(0,2), 0.00001);
    CHECK_CLOSE(96, RefRHUM850(1,3), 0.00001);

    // Process to the multiplication
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<2*lons; i_lon++)
        {
            RefMulti(i_lat,i_lon) = RefPRWTR(i_lat,i_lon) * RefRHUM850(i_lat,i_lon);
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
    asPredictorCriteria* criteria = asPredictorCriteria::GetInstance(asPredictorCriteria::RSE);

    // Loop on every candidate
    for (int i_cand=0; i_cand<candidatesNb; i_cand++)
    {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data PRWTR12h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandPRWTR(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        CHECK_CLOSE(checkPRWTR[i_cand], CandPRWTR(1,1), 0.00001);

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data PRWTR24h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandPRWTR(i_lat,i_lon+lons) = file.GetFloat();
            }
        }

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85012h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandRHUM850(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        CHECK_CLOSE(checkRHUM850[i_cand], CandRHUM850(1,1), 0.00001);

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85024h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandRHUM850(i_lat,i_lon+lons) = file.GetFloat();
            }
        }

        // Process to the multiplication
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<2*lons; i_lon++)
            {
                CandMulti(i_lat,i_lon) = CandPRWTR(i_lat,i_lon) * CandRHUM850(i_lat,i_lon);
            }
        }

        // Process RMSE and check the results
        float res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteria->Assess(RefMulti, CandMulti);
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteria->Assess(RefMulti, CandMulti);
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteria->Assess(RefMulti, CandMulti);
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteria->Assess(RefMulti, CandMulti);
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);
    }

}

TEST(ProcessRMSE)
{
    // Get the data file
    wxString filepath = wxFileName::GetCwd();
    filepath.Append(_T("/files/asPredictorCriteriaTestFile02.txt"));
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
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefPRWTR12h(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(13.6, RefPRWTR12h(0,0), 0.00001);
    CHECK_CLOSE(20.4, RefPRWTR12h(1,1), 0.00001);

    // Skip coasent
    file.SkipLines(3);

    // Get target data PRWTR24h
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefPRWTR24h(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(13.3, RefPRWTR24h(0,0), 0.00001);
    CHECK_CLOSE(18.1, RefPRWTR24h(1,1), 0.00001);

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85012h
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefRHUM85012h(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(82, RefRHUM85012h(0,0), 0.00001);
    CHECK_CLOSE(100, RefRHUM85012h(1,1), 0.00001);

    // Skip coasent
    file.SkipLines(3);

    // Get target data RHUM85024h
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefRHUM85024h(i_lat,i_lon) = file.GetFloat();
        }
    }

    // Check that the data were correctly read from the file
    CHECK_CLOSE(100, RefRHUM85024h(0,0), 0.00001);
    CHECK_CLOSE(96, RefRHUM85024h(1,1), 0.00001);

    // Process to the multiplication
    for (int i_lat=0; i_lat<lats; i_lat++)
    {
        for (int i_lon=0; i_lon<lons; i_lon++)
        {
            RefMulti12h(i_lat,i_lon) = RefPRWTR12h(i_lat,i_lon) * RefRHUM85012h(i_lat,i_lon);
            RefMulti24h(i_lat,i_lon) = RefPRWTR24h(i_lat,i_lon) * RefRHUM85024h(i_lat,i_lon);
        }
    }

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
    asPredictorCriteria* criteria = asPredictorCriteria::GetInstance(_("RMSE"));

    // Loop on every candidate
    for (int i_cand=0; i_cand<candidatesNb; i_cand++)
    {
        // Skip coasent
        file.SkipLines(6);

        // Get candidate data PRWTR12h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandPRWTR12h(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        CHECK_CLOSE(checkPRWTR12h[i_cand], CandPRWTR12h(1,1), 0.00001);

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data PRWTR24h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandPRWTR24h(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85012h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandRHUM85012h(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Check that the data were correctly read from the file
        CHECK_CLOSE(checkRHUM85012h[i_cand], CandRHUM85012h(1,1), 0.00001);

        // Skip coasent
        file.SkipLines(3);

        // Get candidate data RHUM85024h
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandRHUM85024h(i_lat,i_lon) = file.GetFloat();
            }
        }

        // Process to the multiplication
        for (int i_lat=0; i_lat<lats; i_lat++)
        {
            for (int i_lon=0; i_lon<lons; i_lon++)
            {
                CandMulti12h(i_lat,i_lon) = CandPRWTR12h(i_lat,i_lon) * CandRHUM85012h(i_lat,i_lon);
                CandMulti24h(i_lat,i_lon) = CandPRWTR24h(i_lat,i_lon) * CandRHUM85024h(i_lat,i_lon);
            }
        }

        // Process RMSE and check the results
        float res12h, res24h, res;

        wxConfigBase *pConfig = wxFileConfig::Get();

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h);
        res24h = criteria->Assess(RefMulti24h, CandMulti24h);
        res = (res12h+res24h)/2;
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h);
        res24h = criteria->Assess(RefMulti24h, CandMulti24h);
        res = (res12h+res24h)/2;
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h);
        res24h = criteria->Assess(RefMulti24h, CandMulti24h);
        res = (res12h+res24h)/2;
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);

        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res12h = criteria->Assess(RefMulti12h, CandMulti12h);
        res24h = criteria->Assess(RefMulti24h, CandMulti24h);
        res = (res12h+res24h)/2;
        CHECK_CLOSE(critRMSE[i_cand], res, 0.05);
    }

}

TEST(ProcessDifferences)
{
    wxConfigBase *pConfig = wxFileConfig::Get();

    VArray2DFloat RefData;
    Array2DFloat Datatmp;
    Datatmp.resize(2,2);

    Datatmp(0,0) = 12;
    Datatmp(0,1) = 23;
    Datatmp(1,0) = 42;
    Datatmp(1,1) = 25;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = 547;
    Datatmp(0,1) = 2364;
    Datatmp(1,0) = 2672;
    Datatmp(1,1) = 3256;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 0;
    Datatmp(1,0) = 0;
    Datatmp(1,1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 0;
    Datatmp(1,0) = 0;
    Datatmp(1,1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 5;
    Datatmp(1,0) = 0;
    Datatmp(1,1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = 456;
    Datatmp(0,1) = 456;
    Datatmp(1,0) = 45;
    Datatmp(1,1) = 7;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = -324;
    Datatmp(0,1) = -345;
    Datatmp(1,0) = -23;
    Datatmp(1,1) = -26;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = -34;
    Datatmp(0,1) = -45;
    Datatmp(1,0) = 456;
    Datatmp(1,1) = 3;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 0;
    Datatmp(1,0) = 0;
    Datatmp(1,1) = 0;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = 4;
    Datatmp(0,1) = 456;
    Datatmp(1,0) = 4;
    Datatmp(1,1) = 783;
    RefData.push_back(Datatmp);
    Datatmp(0,0) = -345;
    Datatmp(0,1) = -325;
    Datatmp(1,0) = -27;
    Datatmp(1,1) = -475;
    RefData.push_back(Datatmp);

    VArray2DFloat CandData;
    Datatmp(0,0) = 634;
    Datatmp(0,1) = 234;
    Datatmp(1,0) = 3465;
    Datatmp(1,1) = 534;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = 7;
    Datatmp(0,1) = 3;
    Datatmp(1,0) = 35;
    Datatmp(1,1) = 4;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = 54;
    Datatmp(0,1) = 56;
    Datatmp(1,0) = 4;
    Datatmp(1,1) = 74;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 0;
    Datatmp(1,0) = 0;
    Datatmp(1,1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 0;
    Datatmp(1,0) = 4;
    Datatmp(1,1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 0;
    Datatmp(1,0) = 0;
    Datatmp(1,1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = 34;
    Datatmp(0,1) = 2;
    Datatmp(1,0) = 235;
    Datatmp(1,1) = 6;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = 0;
    Datatmp(0,1) = 0;
    Datatmp(1,0) = 0;
    Datatmp(1,1) = 0;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = -637;
    Datatmp(0,1) = -6;
    Datatmp(1,0) = -67;
    Datatmp(1,1) = 567;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = -37;
    Datatmp(0,1) = -65;
    Datatmp(1,0) = -4;
    Datatmp(1,1) = -1;
    CandData.push_back(Datatmp);
    Datatmp(0,0) = -867;
    Datatmp(0,1) = -568;
    Datatmp(1,0) = -43;
    Datatmp(1,1) = -348;
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

    asPredictorCriteria* criteriaSAD = asPredictorCriteria::GetInstance(asPredictorCriteria::SAD);

    float res;
    for (int i=0;i<11;i++)
    {
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteriaSAD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteriaSAD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteriaSAD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteriaSAD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
    }

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

    asPredictorCriteria* criteriaMD = asPredictorCriteria::GetInstance(asPredictorCriteria::MD);

    for (int i=0;i<11;i++)
    {
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteriaMD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteriaMD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteriaMD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteriaMD->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.00001);
    }

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

    asPredictorCriteria* criteriaMRDtoMax = asPredictorCriteria::GetInstance(asPredictorCriteria::MRDtoMax);

    for (int i=0;i<4;i++)
    {
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
    }

    for (int i=5;i<11;i++)
    {
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMax->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.0001);
    }

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

    asPredictorCriteria* criteriaMRDtoMean = asPredictorCriteria::GetInstance(asPredictorCriteria::MRDtoMean);

    for (int i=0;i<4;i++)
    {
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
    }

    for (int i=5;i<9;i++)
    {
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
    }

    for (int i=10;i<11;i++)
    {
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asCOEFF_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (int)asLIN_ALGEBRA_NOVAR);
        res = criteriaMRDtoMean->Assess(RefData[i], CandData[i]);
        CHECK_CLOSE(Results[i], res, 0.001);
    }

}
}

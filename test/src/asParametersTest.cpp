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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include "include_tests.h"
#include "asParameters.h"
#include "asParametersCalibration.h"
#include "asParametersForecast.h"

#include "UnitTest++.h"

namespace
{

TEST(ParametersLoadFromFile)
{
	wxString str("Testing base parameters...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_standard_read.xml");

    asParameters params;
    params.LoadFromFile(filepath);

    CHECK_EQUAL(asTime::GetMJD(1962,1,1), params.GetArchiveStart());
    CHECK_EQUAL(asTime::GetMJD(2008,12,31), params.GetArchiveEnd());
    CHECK_EQUAL(24, params.GetTimeArrayAnalogsTimeStepHours());
    CHECK_EQUAL(24, params.GetTimeArrayTargetTimeStepHours());
    CHECK_EQUAL(60, params.GetTimeArrayAnalogsIntervalDays());
    CHECK_EQUAL(60, params.GetTimeArrayAnalogsExcludeDays());
    CHECK_EQUAL(true, params.GetTimeArrayAnalogsMode().IsSameAs("DaysInterval"));
    CHECK_EQUAL(true, params.GetTimeArrayTargetMode().IsSameAs("Simple"));

    CHECK_EQUAL(true, params.GetMethodName(0).IsSameAs("Analogs"));
    CHECK_EQUAL(100, params.GetAnalogsNumber(0));

    CHECK_EQUAL(false, params.NeedsPreprocessing(0,0));
    CHECK_EQUAL(true, params.GetPredictorDatasetId(0,0).IsSameAs("NCEP_R-1"));
    CHECK_EQUAL(true, params.GetPredictorDataId(0,0).IsSameAs("hgt"));
    CHECK_EQUAL(500, params.GetPredictorLevel(0,0));
    CHECK_EQUAL(24, params.GetPredictorTimeHours(0,0));
    CHECK_EQUAL(true, params.GetPredictorGridType(0,0).IsSameAs("Regular"));
    CHECK_EQUAL(-10, params.GetPredictorUmin(0,0));
    CHECK_EQUAL(9, params.GetPredictorUptsnb(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorUstep(0,0));
    CHECK_EQUAL(30, params.GetPredictorVmin(0,0));
    CHECK_EQUAL(5, params.GetPredictorVptsnb(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorVstep(0,0));
    CHECK_EQUAL(true, params.GetPredictorCriteria(0,0).IsSameAs("S1"));
    CHECK_CLOSE(0.6, params.GetPredictorWeight(0,0), 0.0001);

    CHECK_EQUAL(true, params.NeedsPreprocessing(0,1));
    CHECK_EQUAL(true, params.GetPreprocessMethod(0,1).IsSameAs("Gradients"));
    CHECK_EQUAL(true, params.GetPreprocessDatasetId(0,1,0).IsSameAs("NCEP_R-1"));
    CHECK_EQUAL(true, params.GetPreprocessDataId(0,1,0).IsSameAs("hgt"));
    CHECK_EQUAL(1000, params.GetPreprocessLevel(0,1,0));
    CHECK_EQUAL(12, params.GetPreprocessTimeHours(0,1,0));
    CHECK_EQUAL(1000, params.GetPredictorLevel(0,1));
    CHECK_EQUAL(12, params.GetPredictorTimeHours(0,1));
    CHECK_EQUAL(-15, params.GetPredictorUmin(0,1));
    CHECK_EQUAL(11, params.GetPredictorUptsnb(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorUstep(0,1));
    CHECK_EQUAL(35, params.GetPredictorVmin(0,1));
    CHECK_EQUAL(7, params.GetPredictorVptsnb(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorVstep(0,1));
    CHECK_EQUAL(true, params.GetPredictorCriteria(0,1).IsSameAs("S1"));
    CHECK_CLOSE(0.4, params.GetPredictorWeight(0,1), 0.0001);

    CHECK_EQUAL(40, params.GetPredictandStationIds()[0]);
}

TEST(ParametersLoadFromFileMultipleIds)
{
	wxString str("Testing parameters with multiple station ids...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_standard_multiple_station_ids.xml");

    asParameters params;
    params.LoadFromFile(filepath);

    VectorInt stations = params.GetPredictandStationIds();

    CHECK_EQUAL(5, stations.size());
    CHECK_EQUAL(40, stations[0]);
    CHECK_EQUAL(41, stations[1]);
    CHECK_EQUAL(42, stations[2]);
    CHECK_EQUAL(43, stations[3]);
    CHECK_EQUAL(44, stations[4]);
}

TEST(GenerateSimpleParametersFileCalibration)
{
    // Get original parameters
    wxString paramsFilePath = wxFileName::GetCwd();
    paramsFilePath.Append("/files/");
    paramsFilePath.Append("parameters_calibration_R1_calib_period.xml");
    asParametersCalibration params;
    bool result = params.LoadFromFile(paramsFilePath);
    CHECK_EQUAL(true, result);

    // Generate simple file
    wxString tmpPath = wxFileName::CreateTempFileName("GenerateSimpleParametersFileCalibrationTest");
    result = params.GenerateSimpleParametersFile(tmpPath);
    CHECK_EQUAL(true, result);
}
}

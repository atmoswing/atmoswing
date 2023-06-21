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

#include "asPredictorsManager.h"
#include "asResultsForecast.h"

// Test fixture for the frame test
class PredictorsManager : public testing::Test {
  protected:
    void SetUp() override {
        wxString dataPath = wxFileName::GetCwd() + "/files/data-predictors-display/";

        // Load the workspace
        wxString filePath = dataPath + "workspace_example.asvw";
        workspace = new asWorkspace();
        ASSERT_TRUE(workspace->Load(filePath));
        workspace->SetForecastsDirectory(dataPath);
        workspace->AddPredictorDir("NWS_GFS", dataPath);

        // Load the forecast
        forecast = new asResultsForecast();
        forecast->SetFilePath(dataPath + "2023/06/14/2023-06-14_00.2Z-24h-GFS.nc");
        ASSERT_TRUE(forecast->Load());

        // Initialize the predictors manager
        predictorsManager = new asPredictorsManager(workspace, true);
        predictorsManager->SetForecastDate(forecast->GetLeadTimeOrigin());
        predictorsManager->SetDate(forecast->GetLeadTimeOrigin());
        predictorsManager->SetForecastTimeStepHours(forecast->GetForecastTimeStepHours());
        predictorsManager->SetDatasetIds(forecast->GetPredictorDatasetIdsOper());
        predictorsManager->SetDataIds(forecast->GetPredictorDataIdsOper());
        predictorsManager->SetLevels(forecast->GetPredictorLevels());
        predictorsManager->SetHours(forecast->GetPredictorHours());
    }

    void TearDown() override {
        wxDELETE(workspace);
        wxDELETE(forecast);
        wxDELETE(predictorsManager);
    }

    asWorkspace* workspace;
    asResultsForecast* forecast;
    asPredictorsManager* predictorsManager;
};

TEST_F(PredictorsManager, LoadDataTarget) {
    EXPECT_TRUE(predictorsManager->LoadData(0));
    predictorsManager->NeedsDataReload();
    EXPECT_TRUE(predictorsManager->LoadData(1));
}

TEST_F(PredictorsManager, LoadDataAnalog) {
    wxString dataPath = wxFileName::GetCwd() + "/files/data-ncep-r1/v2014/";
    workspace->SetForecastsDirectory(dataPath);
    workspace->AddPredictorDir("NCEP_R1", dataPath);

    vwxs datasetIds = {"NCEP_R1", "NCEP_R1"};
    vwxs dataIds = {"pressure/hgt", "pressure/hgt"};

    asPredictorsManager predictorsManagerArch(workspace, false);
    predictorsManagerArch.SetForecastDate(forecast->GetLeadTimeOrigin());
    predictorsManagerArch.SetDate(36934); // 1906-01-01
    predictorsManagerArch.SetForecastTimeStepHours(forecast->GetForecastTimeStepHours());
    predictorsManagerArch.SetDatasetIds(datasetIds);
    predictorsManagerArch.SetDataIds(dataIds);
    predictorsManagerArch.SetLevels(forecast->GetPredictorLevels());
    predictorsManagerArch.SetHours(forecast->GetPredictorHours());

    EXPECT_TRUE(predictorsManagerArch.LoadData(1));
}

TEST_F(PredictorsManager, GetLatLonAxes) {
    EXPECT_TRUE(predictorsManager->LoadData(0));
    EXPECT_EQ(149, predictorsManager->GetLongitudesNb());
    EXPECT_EQ(105, predictorsManager->GetLatitudesNb());
    EXPECT_EQ(328 - 360, predictorsManager->GetLongitudeMin());
    EXPECT_EQ(20, predictorsManager->GetLatitudeMin());
    EXPECT_EQ(72, predictorsManager->GetLatitudeMax());
    EXPECT_EQ(0.5, predictorsManager->GetLongitudeResol());
    EXPECT_EQ(-0.5, predictorsManager->GetLatitudeResol());
}

TEST_F(PredictorsManager, GetParameter) {
    EXPECT_TRUE(predictorsManager->LoadData(0));
    EXPECT_EQ(asPredictor::GeopotentialHeight, predictorsManager->GetParameter());
}

TEST_F(PredictorsManager, GetDataMinMax) {
    EXPECT_TRUE(predictorsManager->LoadData(0));
    EXPECT_FLOAT_EQ(5424.6968f, predictorsManager->GetDataMin());
    EXPECT_FLOAT_EQ(5943.0171f, predictorsManager->GetDataMax());
}

TEST_F(PredictorsManager, GetData) {
    EXPECT_TRUE(predictorsManager->LoadData(0));
    float* data = predictorsManager->GetData();
    EXPECT_FLOAT_EQ(5581.1768f, data[0]);
}

TEST_F(PredictorsManager, GetDataRow) {
    EXPECT_TRUE(predictorsManager->LoadData(0));
    float* data = predictorsManager->GetDataRow(3);
    EXPECT_FLOAT_EQ(5597.1567f, data[0]);
    EXPECT_FLOAT_EQ(5600.8770f, data[1]);
    EXPECT_FLOAT_EQ(5622.1367f, data[10]);
}

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
#include <wx/wx.h>

#include "asFramePredictors.h"
#include "asPredictorsManager.h"

// Test fixture for the frame test
class FramePredictors : public testing::Test {
  protected:
    void SetUp() override {
        wxString dataPath = wxFileName::GetCwd() + "/files/data-predictors-display/";

        // Load the workspace
        wxString filePath = dataPath + "workspace_example.asvw";
        workspace = new asWorkspace();
        ASSERT_TRUE(workspace->Load(filePath));
        workspace->SetForecastsDirectory(dataPath);
        workspace->AddPredictorDir("NWS_GFS", dataPath);
        wxString archiveDataPath = wxFileName::GetCwd() + "/files/data-ncep-r1/v2014/";
        workspace->AddPredictorDir("NCEP_R1", archiveDataPath);

        // Set up the forecast manager
        forecastManager = new asForecastManager(nullptr, workspace);
        forecastManager->Init();
        forecastManager->Open(dataPath + "2023/06/14/2023-06-14_00.2Z-24h-GFS.nc");

        // Create the frame
        frame = new asFramePredictors(nullptr, forecastManager, workspace, 0, 0);
    }

    void TearDown() override {
        wxDELETE(workspace);
        wxDELETE(forecastManager);
        frame->Destroy();
    }

    asWorkspace* workspace;
    asForecastManager* forecastManager;
    asFramePredictors* frame;
};

TEST_F(FramePredictors, Initialises) {
    // Ensure the frame is not initially shown
    EXPECT_FALSE(frame->IsShown());

    // Show the frame
    frame->Layout();
    frame->Init();
    frame->Show();

    // Check if the frame is shown
    EXPECT_TRUE(frame->IsShown());
}

TEST_F(FramePredictors, SwitchPanelRight) {
    frame->Layout();
    frame->Init();
    frame->Show();

    frame->SwitchPanelRight();

    EXPECT_FALSE(frame->GetPanelRight()->IsShown());
    EXPECT_TRUE(frame->GetPanelLeft()->IsShown());
}

TEST_F(FramePredictors, SwitchPanelLeft) {
    frame->Layout();
    frame->Init();
    frame->Show();

    frame->SwitchPanelLeft();

    EXPECT_TRUE(frame->GetPanelRight()->IsShown());
    EXPECT_FALSE(frame->GetPanelLeft()->IsShown());
}

TEST_F(FramePredictors, SwitchPanelRightAndLeft) {
    frame->Layout();
    frame->Init();
    frame->Show();

    frame->SwitchPanelRight();
    frame->SwitchPanelLeft();

    EXPECT_TRUE(frame->GetPanelRight()->IsShown());
    EXPECT_TRUE(frame->GetPanelLeft()->IsShown());
}

TEST_F(FramePredictors, SwitchPanelLeftAndRight) {
    frame->Layout();
    frame->Init();
    frame->Show();

    frame->SwitchPanelLeft();
    frame->SwitchPanelRight();

    EXPECT_TRUE(frame->GetPanelRight()->IsShown());
    EXPECT_TRUE(frame->GetPanelLeft()->IsShown());
}

TEST_F(FramePredictors, OpenLayers) {
    frame->Layout();
    frame->Init();
    frame->Show();

    // Layer path
    wxString dirData = asConfig::GetDataDir() + "share";
    if (!wxDirExists(dirData)) {
        wxFileName dirDataWxFile = wxFileName(asConfig::GetDataDir());
        dirDataWxFile.RemoveLastDir();
        dirDataWxFile.AppendDir("share");
        dirData = dirDataWxFile.GetFullPath();
    }
    if (!wxDirExists(dirData)) {
        wxFileName dirDataWxFile = wxFileName(asConfig::GetDataDir());
        dirDataWxFile.RemoveLastDir();
        dirData = dirDataWxFile.GetFullPath();
    }
    ASSERT_TRUE(wxDirExists(dirData));

    wxString gisData = dirData + DS + "atmoswing" + DS + "gis" + DS + "shapefiles";
    wxString filePath = gisData + DS + "geogrid.shp";

    wxArrayString layers;
    layers.Add(filePath);
    EXPECT_TRUE(frame->OpenLayers(layers));
}

TEST_F(FramePredictors, TriggerPredictorSelectionChange) {
    frame->Layout();
    frame->Init();
    frame->Show();

    // Set list selection to 2nd entry and trigger event
    wxListBox* listBox = frame->GetListPredictors();
    listBox->SetSelection(1);
    wxCommandEvent event(wxEVT_COMMAND_LISTBOX_SELECTED);
    event.SetId(listBox->GetId());
    event.SetInt(1);
    listBox->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    EXPECT_TRUE(frame->IsShown()); // Could not find a way to test the view update
}

TEST_F(FramePredictors, TriggerPredictorSelectionChangeWithAnalogData) {
    // Replace dataset for the analog to match existing data.
    vwxs datasetIds = {"NCEP_R1", "NCEP_R1"};
    vwxs dataIds = {"pressure/hgt", "pressure/hgt"};
    asPredictorsManager* predictorsManagerAnalog = frame->GetPredictorsManagerAnalog();
    predictorsManagerAnalog->SetDatasetIds(datasetIds);
    predictorsManagerAnalog->SetDataIds(dataIds);

    // Replace date for the analog to match existing data.
    asResultsForecast* forecast = frame->GetForecastManager()->GetForecast(0, 0);
    a1f analogDates = forecast->GetAnalogsDates(0);
    analogDates[0] = 36934; // 1906-01-01
    forecast->SetAnalogsDates(0, analogDates);

    frame->Layout();
    frame->Init();
    frame->Show();

    // Set list selection to 2nd entry and trigger event
    wxListBox* listBox = frame->GetListPredictors();
    listBox->SetSelection(1);
    wxCommandEvent event(wxEVT_COMMAND_LISTBOX_SELECTED);
    event.SetId(listBox->GetId());
    event.SetInt(1);
    listBox->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    EXPECT_TRUE(frame->IsShown()); // Could not find a way to test the view update
}

TEST_F(FramePredictors, ChangePredictorSelection) {
    frame->Layout();
    frame->Init();
    frame->Show();

    // Set list selection to 2nd entry and trigger event
    wxListBox* listBox = frame->GetListPredictors();
    listBox->SetSelection(1);
    wxCommandEvent event(wxEVT_COMMAND_LISTBOX_SELECTED);
    event.SetId(listBox->GetId());
    event.SetInt(1);
    listBox->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    // Set list selection to 1st entry and trigger event
    listBox->SetSelection(0);
    event.SetInt(0);
    listBox->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    EXPECT_TRUE(frame->IsShown()); // Could not find a way to test the view update
}

TEST_F(FramePredictors, TriggerMethodChange) {
    frame->Layout();
    frame->Init();
    frame->Show();

    // Set list selection to 2nd entry and trigger event
    wxChoice* choice = frame->GetChoiceMethod();
    choice->SetSelection(0);
    wxCommandEvent event(wxEVT_COMMAND_CHOICE_SELECTED);
    event.SetId(choice->GetId());
    event.SetInt(0);
    choice->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    EXPECT_TRUE(frame->IsShown()); // Could not find a way to test the view update
}

TEST_F(FramePredictors, TriggerForecastChange) {
    frame->Layout();
    frame->Init();
    frame->Show();

    // Set list selection to 2nd entry and trigger event
    wxChoice* choice = frame->GetChoiceForecast();
    choice->SetSelection(0);
    wxCommandEvent event(wxEVT_COMMAND_CHOICE_SELECTED);
    event.SetId(choice->GetId());
    event.SetInt(0);
    choice->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    EXPECT_TRUE(frame->IsShown()); // Could not find a way to test the view update
}

TEST_F(FramePredictors, TriggerTargetDateChange) {
    frame->Layout();
    frame->Init();
    frame->Show();

    // Set list selection to 2nd entry and trigger event
    wxChoice* choice = frame->GetChoiceTargetDates();
    choice->SetSelection(0);
    wxCommandEvent event(wxEVT_COMMAND_CHOICE_SELECTED);
    event.SetId(choice->GetId());
    event.SetInt(0);
    choice->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    EXPECT_TRUE(frame->IsShown()); // Could not find a way to test the view update
}

TEST_F(FramePredictors, TriggerAnalogDateChange) {
    frame->Layout();
    frame->Init();
    frame->Show();

    // Set list selection to 2nd entry and trigger event
    wxChoice* choice = frame->GetChoiceAnalogDates();
    choice->SetSelection(0);
    wxCommandEvent event(wxEVT_COMMAND_CHOICE_SELECTED);
    event.SetId(choice->GetId());
    event.SetInt(0);
    choice->GetEventHandler()->ProcessEvent(event);

    frame->Refresh();
    wxYield();

    EXPECT_TRUE(frame->IsShown()); // Could not find a way to test the view update
}

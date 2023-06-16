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

#include "asFrameForecaster.h"

// Test fixture for the frame test
class FrameForecaster : public testing::Test {
  protected:
    void SetUp() override {
        // Initialize wxWidgets
        wxApp::SetInstance(new wxApp);
        int argc = 0;
        wxChar** argv = NULL;
        wxEntryStart(argc, argv);
    }

    void TearDown() override {
        // Cleanup wxWidgets
        wxEntryCleanup();
    }

    asFrameForecaster* frame;
};

TEST_F(FrameForecaster, Initialises) {
    frame = new asFrameForecaster(nullptr);

    // Ensure the frame is not initially shown
    EXPECT_FALSE(frame->IsShown());

    // Show the frame
    frame->Show();

    // Check if the frame is shown
    EXPECT_TRUE(frame->IsShown());

    frame->Destroy();
}

TEST_F(FrameForecaster, LoadRecentFiles) {
    wxConfigBase* config = wxFileConfig::Get();
    wxString testsPath = wxFileName::GetCwd();
    config->Write("/Recent/file1", testsPath + "/files/batch_forecaster.xml");

    frame = new asFrameForecaster(nullptr);

    // Check if the recent files are loaded
    wxMenuItem* menuItem = frame->GetMenuBar()->FindItem(asID_MENU_RECENT);
    wxMenu* menu = menuItem->GetSubMenu();
    EXPECT_EQ(1, menu->GetMenuItemCount());

    frame->Destroy();
}

TEST_F(FrameForecaster, RemovesInexistingFiles) {
    wxConfigBase* config = wxFileConfig::Get();
    config->Write("/Recent/file1", "path/file/one");
    config->Write("/Recent/file1", "path/file/two");

    frame = new asFrameForecaster(nullptr);

    // Check if the recent files are loaded
    wxMenuItem* menuItem = frame->GetMenuBar()->FindItem(asID_MENU_RECENT);
    wxMenu* menu = menuItem->GetSubMenu();
    EXPECT_EQ(0, menu->GetMenuItemCount());

    frame->Destroy();
}

TEST_F(FrameForecaster, OpenBatchFileFromRecent) {
    wxConfigBase* config = wxFileConfig::Get();
    wxString testsPath = wxFileName::GetCwd();
    config->Write("/Recent/file1", testsPath + "/files/batch_forecaster.xml");

    frame = new asFrameForecaster(nullptr);

    // Check that there is no panel
    EXPECT_EQ(0, frame->GetPanelsManager()->GetPanelsNb());

    // Click on the first item
    wxCommandEvent event(wxEVT_COMMAND_MENU_SELECTED, wxID_FILE1);
    frame->GetEventHandler()->ProcessEvent(event);

    wxYield();

    // Check that there is one panel
    EXPECT_EQ(2, frame->GetPanelsManager()->GetPanelsNb());

    frame->Destroy();
}
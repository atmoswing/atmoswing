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
#include <wx/uiaction.h>
#include <wx/evtloop.h>

#include "asPanelForecast.h"
#include "asBatchForecasts.h"
#include "asFrameForecaster.h"
#include "asFrameStyledTextCtrl.h"

// Test fixture for the frame test
class PanelForecast : public testing::Test {
  protected:
    void SetUp() override {
        // Initialize wxWidgets
        wxApp::SetInstance(new wxApp);
        int argc = 0;
        wxChar** argv = NULL;
        wxEntryStart(argc, argv);

        // Add a recent batch file
        wxString filePath = wxFileName::GetCwd() + "/files/batch_forecaster.xml";

        // Create the frame
        frame = new asFrameForecaster(nullptr);
        asBatchForecasts batchForecasts;
        batchForecasts.Load(filePath);
        batchForecasts.SetParametersFileDirectory(wxFileName::GetCwd() + "/files");
        frame->SetBatchForecasts(batchForecasts);
        frame->OpenBatchForecasts();

        // Access to the panel
        panel = frame->GetPanelsManager()->GetPanel(0);
    }

    void TearDown() override {
        // Cleanup wxWidgets
        wxEntryCleanup();
    }

    asFrameForecaster* frame;
    asPanelForecast* panel;
};

void ProcessEvents() {
    wxEventLoop loop;
    while (loop.Pending()) {
        loop.Dispatch();
        wxMilliSleep(10);
    }
}

TEST_F(PanelForecast, Initialises) {
    frame->Show();

    // Check if the panel is shown
    EXPECT_TRUE(panel->IsShown());

    frame->Close();
}

TEST_F(PanelForecast, SetToolTipContent) {
    frame->Show();

    // Check if the tooltip is set
    wxString tooltipText = panel->GetButtonInfo()->GetToolTipText();
    EXPECT_TRUE(tooltipText.IsSameAs("XYZ region"));

    frame->Close();
}

TEST_F(PanelForecast, OnEditForecastFile) {
    wxUIActionSimulator sim;

    frame->Show();

    // Add some extra distance to take account of window decorations
    sim.MouseMove(panel->GetButtonEdit()->GetScreenPosition() + wxPoint(5, 5));
    sim.MouseClick();

    // Process the resulting button event
    wxYield();

    // Wait for the dialog to open (adjust the delay if necessary)
    wxMilliSleep(200);

    // Handle the dialog frame
    sim.MouseClick(); // Click to set focus on the app
    sim.Text("xyz.txt");
    sim.KeyDown(WXK_RETURN);
    sim.KeyUp(WXK_RETURN);

    // Process the resulting button event
    ProcessEvents();

    EXPECT_TRUE(panel->GetTextParametersFileNameValue().Len() > 0);
    EXPECT_TRUE(panel->GetTextParametersFileNameValue().IsSameAs("xyz.txt"));

    frame->Close();
}

TEST_F(PanelForecast, OnDetailsForecastFile) {
    wxUIActionSimulator sim;

    frame->Show();

    // Add some extra distance to take account of window decorations
    sim.MouseMove(panel->GetButtonDetails()->GetScreenPosition() + wxPoint(5, 5));
    sim.MouseClick();

    // Process the resulting button event
    ProcessEvents();

    // Get the new frame
    auto popUp = dynamic_cast<asFrameStyledTextCtrl*>(wxWindow::FindWindowById(asWINDOW_PARAMETERS_DETAILS));

    EXPECT_TRUE(popUp != nullptr);

    if (popUp) {
        EXPECT_TRUE(popUp->IsShown());
    }

    frame->Close();
}
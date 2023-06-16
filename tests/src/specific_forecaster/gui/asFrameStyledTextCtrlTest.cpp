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

#include "asFrameStyledTextCtrl.h"

// Test fixture for the frame test
class FrameStyledTextCtrl : public testing::Test {
  protected:
    void SetUp() override {
        // Initialize wxWidgets
        int argc = 0;
        wxApp::SetInstance(new wxApp);
        wxChar** argvApp = NULL;
        wxEntryStart(argc, argvApp);
    }

    void TearDown() override {
        // Cleanup wxWidgets
        wxEntryCleanup();
    }

    asFrameStyledTextCtrl* frame;
};

TEST_F(FrameStyledTextCtrl, Initialises) {
    frame = new asFrameStyledTextCtrl(nullptr);

    // Ensure the frame is not initially shown
    EXPECT_FALSE(frame->IsShown());

    // Show the frame
    frame->Show();

    // Check if the frame is shown
    EXPECT_TRUE(frame->IsShown());

    frame->Destroy();
}

TEST_F(FrameStyledTextCtrl, SetContent) {
    frame = new asFrameStyledTextCtrl(nullptr);

    frame->SetContent("Test");
    EXPECT_TRUE(frame->GetScintilla()->GetText().IsSameAs("Test"));

    frame->Destroy();
}

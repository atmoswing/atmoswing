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

#include "asFrameViewer.h"

// Test fixture for the frame test
class FrameViewer : public testing::Test {
  protected:
    void SetUp() override {
        // Initialize wxWidgets
        wxApp::SetInstance(new wxApp);
        wxEntryStart(0, nullptr);

        // Create the frame
        frame = new asFrameViewer(nullptr);
    }

    void TearDown() override {
        // Cleanup wxWidgets
        frame->Destroy();
        wxEntryCleanup();
    }

    asFrameViewer* frame;
};

TEST_F(FrameViewer, Initialises) {
    // Ensure the frame is not initially shown
    EXPECT_FALSE(frame->IsShown());

    // Show the frame
    frame->Show();

    // Check if the frame is shown
    EXPECT_TRUE(frame->IsShown());
}

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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#ifndef AS_FRAME_PREFERENCES_DOWNSCALER
#define AS_FRAME_PREFERENCES_DOWNSCALER

#include "AtmoSwingDownscalerGui.h"
#include "asIncludes.h"

class asFramePreferencesDownscaler : public asFramePreferencesDownscalerVirtual {
  public:
    explicit asFramePreferencesDownscaler(wxWindow* parent, wxWindowID id = asWINDOW_PREFERENCES);

  protected:
    void CloseFrame(wxCommandEvent& event) override;

    void Update() override;

    void LoadPreferences();

    void SavePreferences() const;

    void SaveAndClose(wxCommandEvent& event) override;

    void ApplyChanges(wxCommandEvent& event) override;

    void OnChangeMultithreadingCheckBox(wxCommandEvent& event) override;
};

#endif

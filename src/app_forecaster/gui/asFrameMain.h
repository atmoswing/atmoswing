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

#ifndef AS_FRAME_MAIN
#define AS_FRAME_MAIN

#include <wx/awx/led.h>

#include "AtmoswingForecasterGui.h"
#include "asBatchForecasts.h"
#include "asIncludes.h"
#include "asLogWindow.h"
#include "asMethodForecasting.h"
#include "asPanelsManagerForecasts.h"

class asFrameMain : public asFrameMainVirtual {
   public:
    explicit asFrameMain(wxWindow *parent);

    ~asFrameMain() override;

    void OnInit();

    double GetForecastDate() const;

    void SetForecastDate(double date);

   protected:
    asLogWindow *m_logWindow;
    asMethodForecasting *m_forecaster;
    awxLed *m_ledDownloading;
    awxLed *m_ledLoading;
    awxLed *m_ledProcessing;
    awxLed *m_ledSaving;
    asBatchForecasts m_batchForecasts;

    void OnOpenBatchForecasts(wxCommandEvent &event) override;

    void OnSaveBatchForecasts(wxCommandEvent &event) override;

    void OnSaveBatchForecastsAs(wxCommandEvent &event) override;

    bool SaveBatchForecasts();

    bool UpdateBatchForecasts();

    void OnNewBatchForecasts(wxCommandEvent &event) override;

    bool OpenBatchForecasts();

    void Update() override;

    void OpenFramePredictandDB(wxCommandEvent &event) override;

    void OnConfigureDirectories(wxCommandEvent &event) override;

    void OpenFramePreferences(wxCommandEvent &event) override;

    void OpenFrameAbout(wxCommandEvent &event) override;

    void OnShowLog(wxCommandEvent &event) override;

    void OnLogLevel1(wxCommandEvent &event) override;

    void OnLogLevel2(wxCommandEvent &event) override;

    void OnLogLevel3(wxCommandEvent &event) override;

    void OnStatusMethodUpdate(wxCommandEvent &event);

    void OnSetPresentDate(wxCommandEvent &event) override;

    void DisplayLogLevelMenu();

    void LaunchForecasting(wxCommandEvent &event);

    void CancelForecasting(wxCommandEvent &event);

    void AddForecast(wxCommandEvent &event) override;

    void SetPresentDate();

    void InitOverallProgress();

    void IncrementOverallProgress();

   private:
    asPanelsManagerForecasts *m_panelsManager;

    DECLARE_EVENT_TABLE()
};

#endif

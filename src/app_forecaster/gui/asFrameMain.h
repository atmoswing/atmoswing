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

#ifndef __asFrameMain__
#define __asFrameMain__

#include "AtmoswingForecasterGui.h"
#include "asIncludes.h"
#include "asLogWindow.h"
#include "asMethodForecasting.h"
#include "asPanelsManagerForecasts.h"
#include "asBatchForecasts.h"
#include <wx/awx/led.h>

class asPanelsManagerForecasts;

class asFrameMain
        : public asFrameMainVirtual
{

public:
    asFrameMain(wxWindow *parent);

    ~asFrameMain();

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

    void OnOpenBatchForecasts(wxCommandEvent &event);

    void OnSaveBatchForecasts(wxCommandEvent &event);

    void OnSaveBatchForecastsAs(wxCommandEvent &event);

    bool SaveBatchForecasts();

    bool UpdateBatchForecasts();

    void OnNewBatchForecasts(wxCommandEvent &event);

    bool OpenBatchForecasts();

    void Update();

    void OpenFramePredictandDB(wxCommandEvent &event) const;

    void OnConfigureDirectories(wxCommandEvent &event) const;

    void OpenFramePreferences(wxCommandEvent &event) const;

    void OpenFrameAbout(wxCommandEvent &event) const;

    void OnShowLog(wxCommandEvent &event) const;

    void OnLogLevel1(wxCommandEvent &event);

    void OnLogLevel2(wxCommandEvent &event);

    void OnLogLevel3(wxCommandEvent &event);

    void OnStatusMethodUpdate(wxCommandEvent &event);

    void OnSetPresentDate(wxCommandEvent &event);

    void DisplayLogLevelMenu();

    void LaunchForecasting(wxCommandEvent &event);

    void CancelForecasting(wxCommandEvent &event);

    void AddForecast(wxCommandEvent &event);

    void SetPresentDate();

    void InitOverallProgress();

    void IncrementOverallProgress();

private:
    asPanelsManagerForecasts *m_panelsManager;

DECLARE_EVENT_TABLE()

};

#endif // __asFrameMain__

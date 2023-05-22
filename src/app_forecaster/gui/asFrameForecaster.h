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

#ifndef AS_FRAME_FORECASTER
#define AS_FRAME_FORECASTER

#include <wx/awx/led.h>
#include <wx/filehistory.h>

#include "AtmoSwingForecasterGui.h"
#include "asBatchForecasts.h"
#include "asIncludes.h"
#include "asLogWindow.h"
#include "asMethodForecasting.h"
#include "asPanelsManagerForecasts.h"

#define asID_MENU_RECENT 1300

class asFrameForecaster : public asFrameForecasterVirtual {
  public:
    explicit asFrameForecaster(wxWindow* parent);

    ~asFrameForecaster() override;

    void OnInit();

    double GetForecastDate() const;

    void SetForecastDate(double date);

  protected:
    asLogWindow* m_logWindow; /**< The log window. */
    asMethodForecasting* m_forecaster; /**< The forecasting method. */
    awxLed* m_ledDownloading; /**< The LED for downloading. */
    awxLed* m_ledLoading; /**< The LED for loading. */
    awxLed* m_ledProcessing; /**< The LED for processing. */
    awxLed* m_ledSaving; /**< The LED for saving. */
    asBatchForecasts m_batchForecasts; /**< The batch forecasts object. */
    wxFileHistory* m_fileHistory; /**< The file history. */

    void OnOpenBatchForecasts(wxCommandEvent& event) override;

    /**
     * Open the batch file selected from the recent entries.
     *
     * @param event The menu event.
     */
    void OnFileHistory(wxCommandEvent& event);

    void OnSaveBatchForecasts(wxCommandEvent& event) override;

    void OnSaveBatchForecastsAs(wxCommandEvent& event) override;

    bool SaveBatchForecasts();

    bool UpdateBatchForecasts();

    void OnNewBatchForecasts(wxCommandEvent& event) override;

    bool OpenBatchForecasts();

    void Update() override;

    void OpenFramePredictandDB(wxCommandEvent& event) override;

    void OnConfigureDirectories(wxCommandEvent& event) override;

    void OpenFramePreferences(wxCommandEvent& event) override;

    void OpenFrameAbout(wxCommandEvent& event) override;

    void OnShowLog(wxCommandEvent& event) override;

    void OnLogLevel1(wxCommandEvent& event) override;

    void OnLogLevel2(wxCommandEvent& event) override;

    void OnLogLevel3(wxCommandEvent& event) override;

    void OnStatusMethodUpdate(wxCommandEvent& event);

    void OnSetPresentDate(wxCommandEvent& event) override;

    void DisplayLogLevelMenu();

    void LaunchForecasting(wxCommandEvent& event);

    void CancelForecasting(wxCommandEvent& event);

    void AddForecast(wxCommandEvent& event) override;

    void SetPresentDate();

    /**
     * Update the recent files list and remove the ones that do not exist anymore.
     */
    void UpdateRecentFiles();

    /**
     * Set the recent files list in the menu.
     */
    void SetRecentFiles();

    /**
     * Save the recent files list in the config file.
     */
    void SaveRecentFiles();

    void InitOverallProgress();

    void IncrementOverallProgress();

  private:
    asPanelsManagerForecasts* m_panelsManager; /**< The panels manager. */

    DECLARE_EVENT_TABLE()
};

#endif

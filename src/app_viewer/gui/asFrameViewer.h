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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#ifndef AS_FRAME_VIEWER_H
#define AS_FRAME_VIEWER_H

#include <wx/filehistory.h>
#include <wx/process.h>

#include "AtmoSwingViewerGui.h"
#include "asForecastManager.h"
#include "asForecastRenderer.h"
#include "asIncludes.h"
#include "asLeadTimeSwitcher.h"
#include "asLogWindow.h"
#include "asPanelSidebarAlarms.h"
#include "asPanelSidebarAnalogDates.h"
#include "asPanelSidebarCaptionForecastDots.h"
#include "asPanelSidebarCaptionForecastRing.h"
#include "asPanelSidebarForecasts.h"
#include "asPanelSidebarGisLayers.h"
#include "asPanelSidebarStationsList.h"
#include "asWorkspace.h"
#include "vroomgis.h"
#include "wx/dnd.h"

#define asID_MENU_RECENT 1301

const int as_POPUP_OFFSET = 50;
const int asID_MENU_POPUP_LAYER = wxID_HIGHEST + 2 + as_POPUP_OFFSET;

/** Implementing vroomDropFiles */
class asFrameViewer;

class vroomDropFiles : public wxFileDropTarget {
  public:
    explicit vroomDropFiles(asFrameViewer* parent);

    bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) override;

  private:
    asFrameViewer* m_loaderFrame;
};

/** Implementing forecastDropFiles */
class asFrameViewer;

class forecastDropFiles : public wxFileDropTarget {
  public:
    explicit forecastDropFiles(asFrameViewer* parent);

    bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) override;

  private:
    asFrameViewer* m_loaderFrame;
};

class asFrameViewer : public asFrameViewerVirtual {
  public:
    explicit asFrameViewer(wxWindow* parent, wxWindowID id = asWINDOW_MAIN);

    ~asFrameViewer() override;

    void Init();

    bool OpenLayers(const wxArrayString& names);

    bool OpenForecast(const wxArrayString& names);

    asWorkspace* GetWorkspace() {
        return &m_workspace;
    }

  protected:
  private:
    wxProcess* m_processForecast;
    vrLayerManager* m_layerManager;
    vrViewerLayerManager* m_viewerLayerManager;
    vrViewerDisplay* m_displayCtrl;
    wxKeyboardState m_keyBoardState;
    asForecastManager* m_forecastManager;
    asForecastRenderer* m_forecastViewer;
    asPanelSidebarGisLayers* m_panelSidebarGisLayers;
    asPanelSidebarForecasts* m_panelSidebarForecasts;
    asPanelSidebarStationsList* m_panelSidebarStationsList;
    asPanelSidebarCaptionForecastDots* m_panelSidebarCaptionForecastDots;
    asPanelSidebarAnalogDates* m_panelSidebarAnalogDates;
    asPanelSidebarCaptionForecastRing* m_panelSidebarCaptionForecastRing;
    asPanelSidebarAlarms* m_panelSidebarAlarms;
    asLeadTimeSwitcher* m_leadTimeSwitcher;
    asWorkspace m_workspace;
    wxFileHistory* m_fileHistory;
    bool m_launchedPresentForecast;

#if defined(__WIN32__)
    wxCriticalSection m_critSectionViewerLayerManager;
#endif

    void OpenForecastsFromTmpList();

    bool OpenRecentForecasts();

    void OnLoadPreviousForecast(wxCommandEvent& event) override;

    void OnLoadNextForecast(wxCommandEvent& event) override;

    void OnLoadPreviousDay(wxCommandEvent& event) override;

    void OnLoadNextDay(wxCommandEvent& event) override;

    void SwitchForecast(double increment);

    void OnOpenWorkspace(wxCommandEvent& event) override;

    void OnFileHistory(wxCommandEvent& event);

    void OnSaveWorkspace(wxCommandEvent& event) override;

    void OnSaveWorkspaceAs(wxCommandEvent& event) override;

    bool SaveWorkspace();

    void OnNewWorkspace(wxCommandEvent& event) override;

    bool OpenWorkspace(bool openRecentForecasts = true);

    void UpdateLeadTimeSwitch();

    void OpenFramePlots(wxCommandEvent& event);

    void OpenFrameGrid(wxCommandEvent& event);

    void OpenFramePredictors(wxCommandEvent& event);

    void OpenFramePredictandDB(wxCommandEvent& event) override;

    void OpenFramePreferences(wxCommandEvent& event) override;

    void OpenFrameAbout(wxCommandEvent& event) override;

    void OnLogLevel1(wxCommandEvent& event) override;

    void OnLogLevel2(wxCommandEvent& event) override;

    void OnLogLevel3(wxCommandEvent& event) override;

    void DisplayLogLevelMenu();

    void OnForecastRatioSelectionChange(wxCommandEvent& event);

    void OnForecastForecastSelectionChange(wxCommandEvent& event);

    void OnForecastForecastSelectFirst(wxCommandEvent& event);

    void OnForecastQuantileSelectionChange(wxCommandEvent& event);

    void DrawPlotStation(int stationRow);

    void OnOpenLayer(wxCommandEvent& event) override;

    void OnCloseLayer(wxCommandEvent& event) override;

    void OnOpenForecast(wxCommandEvent& event) override;

    void OnMoveLayer(wxCommandEvent& event) override;

    void OnChangeLeadTime(wxCommandEvent& event);

    void OnToolSelect(wxCommandEvent& event);

    void OnToolZoomIn(wxCommandEvent& event);

    void OnToolZoomOut(wxCommandEvent& event);

    void OnToolPan(wxCommandEvent& event);

    void OnKeyDown(wxKeyEvent& event);

    void OnKeyUp(wxKeyEvent& event);

    void OnToolAction(wxCommandEvent& event);

    void OnToolZoomToFit(wxCommandEvent& event);

    void FitExtentToForecasts();

    void OnStationSelection(wxCommandEvent& event);

    void OnForecastClear(wxCommandEvent& event);

    void OnClose(wxCloseEvent& event);

    void OnQuit(wxCommandEvent& event) override;

    void OnForecastNewAdded(wxCommandEvent& event);

    void ReloadViewerLayerManager();

    void UpdateHeaderTexts();

    void UpdatePanelCaptionAll();

    void UpdatePanelCaptionColorbar();

    void UpdatePanelAnalogDates();

    void UpdatePanelStationsList();

    void UpdateRecentFiles();

    void SetRecentFiles();

    void SaveRecentFiles();

    virtual void OnRightClick(wxMouseEvent& event) {
        event.Skip();
    }

    DECLARE_EVENT_TABLE()
};

#endif

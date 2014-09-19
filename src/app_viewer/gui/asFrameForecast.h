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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */

#ifndef ASFRAMEFORECAST_H
#define ASFRAMEFORECAST_H

#include "AtmoswingViewerGui.h"
#include "asIncludes.h"
#include "asLogWindow.h"
#include "asForecastManager.h"
#include "wx/dnd.h"
#include <wx/process.h>
#include "vroomgis.h"
#include "asForecastViewer.h"
#include "asPanelSidebarCaptionForecastRing.h"
#include "asPanelSidebarCaptionForecastDots.h"
#include "asPanelSidebarCalendar.h"
#include "asPanelSidebarGisLayers.h"
#include "asPanelSidebarForecasts.h"
#include "asPanelSidebarStationsList.h"
#include "asPanelSidebarAlarms.h"
#include "asPanelSidebarAnalogDates.h"
#include "asLeadTimeSwitcher.h"
#include "asWorkspace.h"

const int as_POPUP_OFFSET = 50;
const int asID_MENU_POPUP_LAYER = wxID_HIGHEST + 2 + as_POPUP_OFFSET;


/** Implementing vroomDropFiles */
class asFrameForecast;
class vroomDropFiles : public wxFileDropTarget
{
private:
    asFrameForecast * m_LoaderFrame;

public:
    vroomDropFiles(asFrameForecast * parent);
    virtual bool OnDropFiles(wxCoord x, wxCoord y,
                             const wxArrayString & filenames);
};


/** Implementing modelDropFiles */
class asFrameForecast;
class modelDropFiles : public wxFileDropTarget
{
private:
    asFrameForecast * m_LoaderFrame;

public:
    modelDropFiles(asFrameForecast * parent);
    virtual bool OnDropFiles(wxCoord x, wxCoord y,
                             const wxArrayString & filenames);
};


class asFrameForecast : public asFrameForecastVirtual
{
public:
    asFrameForecast(wxWindow* parent, wxWindowID id=asWINDOW_MAIN);
    virtual ~asFrameForecast();

    void Init();
    bool OpenLayers (const wxArrayString & names);
    bool OpenForecast (const wxArrayString & names);

    vrLayerManager *GetLayerManager()
    {
        return m_LayerManager;
    }

    void SetLayerManager(vrLayerManager* layerManager)
    {
        m_LayerManager = layerManager;
    }

    vrViewerLayerManager *GetViewerLayerManager()
    {
        return m_ViewerLayerManager;
    }

    void SetViewerLayerManager(vrViewerLayerManager* viewerLayerManager)
    {
        m_ViewerLayerManager = viewerLayerManager;
    }

    vrViewerDisplay *GetViewerDisplay()
    {
        return m_DisplayCtrl;
    }

    void SetViewerDisplay(vrViewerDisplay* viewerDisplay)
    {
        m_DisplayCtrl = viewerDisplay;
    }

    asForecastManager *GetForecastManager()
    {
        return m_ForecastManager;
    }

    void SetForecastManager(asForecastManager* forecastManager)
    {
        m_ForecastManager = forecastManager;
    }

    asForecastViewer *GetForecastViewer()
    {
        return m_ForecastViewer;
    }

    void SetForecastViewer(asForecastViewer* forecastViewer)
    {
        m_ForecastViewer = forecastViewer;
    }

    asWorkspace* GetWorkspace()
    {
        return &m_Workspace;
    }


protected:

private:
    wxProcess* m_ProcessForecast;
    vrLayerManager *m_LayerManager;
    vrViewerLayerManager *m_ViewerLayerManager;
    vrViewerDisplay *m_DisplayCtrl;
    wxKeyboardState m_KeyBoardState;
    asForecastManager *m_ForecastManager;
    asForecastViewer *m_ForecastViewer;
    asPanelSidebarGisLayers *m_PanelSidebarGisLayers;
    asPanelSidebarForecasts *m_PanelSidebarForecasts;
    asPanelSidebarStationsList *m_PanelSidebarStationsList;
    asPanelSidebarCaptionForecastDots *m_PanelSidebarCaptionForecastDots;
    asPanelSidebarAnalogDates *m_PanelSidebarAnalogDates;
    asPanelSidebarCaptionForecastRing *m_PanelSidebarCaptionForecastRing;
    asPanelSidebarAlarms *m_PanelSidebarAlarms;
    asLeadTimeSwitcher *m_LeadTimeSwitcher;
    asWorkspace m_Workspace;
    bool m_LaunchedPresentForecast;

    void OpenForecastsFromTmpList();
    bool OpenRecentForecasts();
    void OnOpenWorkspace( wxCommandEvent & event );
    void OnSaveWorkspace(wxCommandEvent & event);
    void OnSaveWorkspaceAs(wxCommandEvent & event);
    bool SaveWorkspace();
    void OnNewWorkspace(wxCommandEvent & event);
    bool OpenWorkspace();
    void UpdateLeadTimeSwitch();
    void LaunchForecastingNow( wxCommandEvent& event );
    void LaunchForecastingPast( wxCommandEvent& event );
    void OpenFrameForecaster( wxCommandEvent& event );
    void OpenFramePlots( wxCommandEvent& event );
    void OpenFrameGrid( wxCommandEvent& event );
    void OpenFramePreferences( wxCommandEvent& event );
    void OpenFrameAbout( wxCommandEvent& event );
    void OnLogLevel1( wxCommandEvent& event );
    void OnLogLevel2( wxCommandEvent& event );
    void OnLogLevel3( wxCommandEvent& event );
    void DisplayLogLevelMenu();
    void OnForecastRatioSelectionChange( wxCommandEvent& event );
    void OnForecastModelSelectionChange( wxCommandEvent& event );
    void OnForecastPercentileSelectionChange( wxCommandEvent& event );
    void DrawPlotStation( int station );
    void OnOpenLayer( wxCommandEvent & event );
    void OnCloseLayer( wxCommandEvent & event );
    void OnOpenForecast( wxCommandEvent & event );
    void OnMoveLayer( wxCommandEvent & event );
    void OnToolDisplayValue( wxCommandEvent & event );
    void OnChangeLeadTime( wxCommandEvent& event );
    void OnForecastProcessTerminate( wxProcessEvent &event );
    void OnToolSelect (wxCommandEvent & event);
    void OnToolZoomIn (wxCommandEvent & event);
    void OnToolZoomOut (wxCommandEvent & event);
    void OnToolPan (wxCommandEvent & event);
    void OnKeyDown(wxKeyEvent & event);
    void OnKeyUp(wxKeyEvent & event);
    void OnToolAction (wxCommandEvent & event);
    void OnToolZoomToFit (wxCommandEvent & event);
    void FitExtentToForecasts ();
    void OnStationSelection( wxCommandEvent& event );
    void OnForecastClear( wxCommandEvent &event );
    void OnClose(wxCloseEvent& event);
    void OnQuit( wxCommandEvent& event );
    void OnForecastNewAdded( wxCommandEvent& event );
    void ReloadViewerLayerManager( );
    void UpdateHeaderTexts();
    void UpdatePanelCaptionAll();
    void UpdatePanelCaptionColorbar();
    void UpdatePanelAnalogDates();
    void UpdatePanelStationsList();
    void UpdatePanelAlarms();
    #if defined (__WIN32__)
        wxCriticalSection m_CritSectionViewerLayerManager;
    #endif


    virtual void OnRightClick( wxMouseEvent& event )
    {
        event.Skip();
    }
    

    DECLARE_EVENT_TABLE()

};

#endif // ASFRAMEFORECAST_H

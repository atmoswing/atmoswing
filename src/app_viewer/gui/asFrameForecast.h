/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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

const int asID_MOVE_LAYER = wxID_HIGHEST + 1;
const int mm_POPUP_OFFSET = 50;
const int asID_MENU_POPUP_LAYER = wxID_HIGHEST + 2 + mm_POPUP_OFFSET;


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

    void OnInit();
    bool OpenLayers (const wxArrayString & names);
    void OpenDefaultLayers ();
    void OpenForecastsFromTmpList();
    bool OpenForecast (const wxArrayString & names);
    // The next ones need to be public as they are refered in the children events table.
    void OnForecastProcessTerminate( wxProcessEvent &event );
    void OnToolSelect (wxCommandEvent & event);
    void OnToolZoomIn (wxCommandEvent & event);
	void OnToolZoomOut (wxCommandEvent & event);
	void OnToolPan (wxCommandEvent & event);
	void OnKeyDown(wxKeyEvent & event);
    void OnKeyUp(wxKeyEvent & event);
    void OnToolAction (wxCommandEvent & event);
    void OnToolZoomToFit (wxCommandEvent & event);
    void OnStationSelection( wxCommandEvent& event );
    void OnForecastClear( wxCommandEvent &event );


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


protected:
    wxProcess* m_ProcessForecast;
	// vroomgis
	vrLayerManager *m_LayerManager;
	vrViewerLayerManager *m_ViewerLayerManager;
	vrViewerDisplay *m_DisplayCtrl;
	wxKeyboardState m_KeyBoardState;
	asForecastManager *m_ForecastManager;
	asForecastViewer *m_ForecastViewer;
	asPanelSidebarGisLayers *m_PanelSidebarGisLayers;
	asPanelSidebarForecasts *m_PanelSidebarForecasts;
	asPanelSidebarStationsList *m_PanelSidebarStationsList;
	bool m_LaunchedPresentForecast;
    //    void Update();
    void LaunchForecastingNow( wxCommandEvent& event );
    void LaunchForecastingPast( wxCommandEvent& event );
    void OpenFrameForecaster( wxCommandEvent& event );
    void OpenFramePlots( wxCommandEvent& event );
    void OpenFrameGrid( wxCommandEvent& event );
    void OpenFramePredictors( wxCommandEvent& event );
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
    void OnQuit( wxCommandEvent& event );
    void OnOpenLayer( wxCommandEvent & event );
    void OnCloseLayer( wxCommandEvent & event );
    void OnOpenForecast( wxCommandEvent & event );
	void OnMoveLayer( wxCommandEvent & event );
	void OnToolDisplayValue( wxCommandEvent & event );
	void ReloadViewerLayerManager( )
	#if defined (__WIN32__)
        wxCriticalSection m_CritSectionViewerLayerManager;
    #endif


	virtual void OnRightClick( wxMouseEvent& event )
	{
	    event.Skip();
    }

private:

};

#endif // ASFRAMEFORECAST_H

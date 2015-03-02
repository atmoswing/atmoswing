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
 
#include "asFrameForecast.h"

#include "AtmoswingAppViewer.h"
#if defined (__WIN32__)
    #include "asThreadsManager.h"
    #include "asThreadViewerLayerManagerReload.h"
    #include "asThreadViewerLayerManagerZoomIn.h"
    #include "asThreadViewerLayerManagerZoomOut.h"
#endif
#include "asFrameAbout.h"
#include "asFramePreferencesViewer.h"
#include "asFramePlotTimeSeries.h"
#include "asFramePlotDistributions.h"
#include "asFrameGridAnalogsValues.h"
#include "asPanelPlot.h"
#include "asResultsAnalogsForecast.h"
#include "asFileAscii.h"
#include "asFileWorkspace.h"
#include "asWizardWorkspace.h"
#include "img_bullets.h"
#include "img_toolbar.h"
#include "img_logo.h"
#include <wx/colour.h>
#include <wx/statline.h>
#include <wx/app.h>
#include <wx/event.h>
#include <wx/dir.h>
#include "vrrender.h"
#include "vrlayervector.h"


BEGIN_EVENT_TABLE(asFrameForecast, wxFrame)
    EVT_END_PROCESS(wxID_ANY, asFrameForecast::OnForecastProcessTerminate)
	EVT_CLOSE(asFrameForecast::OnClose)
    EVT_KEY_DOWN(asFrameForecast::OnKeyDown)
    EVT_KEY_UP(asFrameForecast::OnKeyUp)
    EVT_MENU(wxID_EXIT,  asFrameForecast::OnQuit)
    EVT_MENU (asID_SELECT, asFrameForecast::OnToolSelect)
    EVT_MENU (asID_ZOOM_IN, asFrameForecast::OnToolZoomIn)
    EVT_MENU (asID_ZOOM_OUT, asFrameForecast::OnToolZoomOut)
    EVT_MENU (asID_ZOOM_FIT, asFrameForecast::OnToolZoomToFit)
    EVT_MENU (asID_PAN, asFrameForecast::OnToolPan)
    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOM, asFrameForecast::OnToolAction)
    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOMOUT, asFrameForecast::OnToolAction)
    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_SELECT, asFrameForecast::OnToolAction)
    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_PAN, asFrameForecast::OnToolAction)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_OPEN_WORKSPACE, asFrameForecast::OnOpenWorkspace)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_STATION_SELECTION_CHANGED, asFrameForecast::OnStationSelection)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED, asFrameForecast::OnChangeLeadTime)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_CLEAR, asFrameForecast::OnForecastClear)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_NEW_ADDED, asFrameForecast::OnForecastNewAdded)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED, asFrameForecast::OnForecastRatioSelectionChange)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_SELECTION_CHANGED, asFrameForecast::OnForecastForecastSelectionChange)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_SELECT_FIRST, asFrameForecast::OnForecastForecastSelectFirst)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_QUANTILE_SELECTION_CHANGED, asFrameForecast::OnForecastQuantileSelectionChange)
END_EVENT_TABLE()


/* vroomDropFiles */

vroomDropFiles::vroomDropFiles(asFrameForecast * parent){
    wxASSERT(parent);
    m_loaderFrame = parent;
}


bool vroomDropFiles::OnDropFiles(wxCoord x, wxCoord y,
                                 const wxArrayString & filenames){
    if (filenames.GetCount() == 0) return false;

    m_loaderFrame->OpenLayers(filenames);
    return true;
}


/* forecastDropFiles */

forecastDropFiles::forecastDropFiles(asFrameForecast * parent){
    wxASSERT(parent);
    m_loaderFrame = parent;
}


bool forecastDropFiles::OnDropFiles(wxCoord x, wxCoord y,
                                 const wxArrayString & filenames){
    if (filenames.GetCount() == 0) return false;

    m_loaderFrame->OpenForecast(filenames);
    return true;
}


asFrameForecast::asFrameForecast( wxWindow* parent, wxWindowID id )
:
asFrameForecastVirtual( parent, id )
{
    g_silentMode = false;

    // Toolbar
    m_toolBar->AddTool( asID_OPEN, wxT("Open"), img_open, img_open, wxITEM_NORMAL, _("Open forecast"), _("Open a forecast"), NULL );
    m_toolBar->AddTool( asID_RUN, wxT("Run"), img_run, img_run, wxITEM_NORMAL, _("Run last forecast"), _("Run last forecast"), NULL );
    m_toolBar->AddTool( asID_RUN_PREVIOUS, wxT("Run previous"), img_run_history, img_run_history, wxITEM_NORMAL, _("Run previous forecasts"), _("Run all previous forecasts"), NULL );
    m_toolBar->AddSeparator();
    m_toolBar->AddTool( asID_SELECT, wxT("Select"), img_map_cursor, img_map_cursor, wxITEM_NORMAL, _("Select"), _("Select data on the map"), NULL );
    m_toolBar->AddTool( asID_ZOOM_IN, wxT("Zoom in"), img_map_zoom_in, img_map_zoom_in, wxITEM_NORMAL, _("Zoom in"), _("Zoom in"), NULL );
    m_toolBar->AddTool( asID_ZOOM_OUT, wxT("Zoom out"), img_map_zoom_out, img_map_zoom_out, wxITEM_NORMAL, _("Zoom out"), _("Zoom out"), NULL );
    m_toolBar->AddTool( asID_PAN, wxT("Pan"), img_map_move, img_map_move, wxITEM_NORMAL, _("Pan the map"), _("Move the map by panning"), NULL );
    m_toolBar->AddTool( asID_ZOOM_FIT, wxT("Fit"), img_map_fit, img_map_fit, wxITEM_NORMAL, _("Zoom to visible layers"), _("Zoom view to the full extent of all visible layers"), NULL );
    m_toolBar->AddSeparator();
    m_toolBar->AddTool( asID_FRAME_PLOTS, wxT("Open distributions plots"), img_frame_plots, img_frame_plots, wxITEM_NORMAL, _("Open distributions plots"), _("Open distributions plots"), NULL );
    m_toolBar->AddTool( asID_FRAME_GRID, wxT("Open analogs list"), img_frame_grid, img_frame_grid, wxITEM_NORMAL, _("Open analogs list"), _("Open analogs list"), NULL );
    m_toolBar->AddTool( asID_FRAME_FORECASTER, wxT("Open forecaster"), img_frame_forecaster, img_frame_forecaster, wxITEM_NORMAL, _("Open forecaster"), _("Open forecaster"), NULL );
    m_toolBar->AddSeparator();
    m_toolBar->AddTool( asID_PREFERENCES, wxT("Preferences"), img_preferences, img_preferences, wxITEM_NORMAL, _("Preferences"), _("Preferences"), NULL );
    m_toolBar->Realize();

    // VroomGIS controls
    m_displayCtrl = new vrViewerDisplay( m_panelGIS, wxID_ANY, wxColour(120,120,120));
    m_sizerGIS->Add( m_displayCtrl, 1, wxEXPAND, 5 );
    m_panelGIS->Layout();

    // Gis panel
    m_panelSidebarGisLayers = new asPanelSidebarGisLayers( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_panelSidebarGisLayers->Layout();
    m_sizerScrolledWindow->Add( m_panelSidebarGisLayers, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );
    m_panelSidebarGisLayers->SetDropTarget(new vroomDropFiles(this));

    // VroomGIS
    m_layerManager = new vrLayerManager();
    m_viewerLayerManager = new vrViewerLayerManager(m_layerManager, this, m_displayCtrl , m_panelSidebarGisLayers->GetTocCtrl());
//    m_layerManager->AllowReprojectOnTheFly(true);
    
    // Forecast manager
    m_forecastManager = new asForecastManager( this, &m_workspace);

    // Forecast viewer
    m_forecastViewer = new asForecastViewer( this, m_forecastManager, m_layerManager, m_viewerLayerManager);

    // Forecasts
    m_panelSidebarForecasts = new asPanelSidebarForecasts( m_scrolledWindowOptions, m_forecastManager, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_panelSidebarForecasts->Layout();
    m_sizerScrolledWindow->Insert(0, m_panelSidebarForecasts, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );
    m_panelSidebarForecasts->SetDropTarget(new forecastDropFiles(this));

    // Alarms
    m_panelSidebarAlarms = new asPanelSidebarAlarms( m_scrolledWindowOptions, &m_workspace, m_forecastManager, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_panelSidebarAlarms->Layout();
    m_sizerScrolledWindow->Add( m_panelSidebarAlarms, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Stations list
    m_panelSidebarStationsList = new asPanelSidebarStationsList( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_panelSidebarStationsList->Layout();
    m_sizerScrolledWindow->Add( m_panelSidebarStationsList, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );
    
    // Analog dates sidebar
    m_panelSidebarAnalogDates = new asPanelSidebarAnalogDates( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_panelSidebarAnalogDates->Layout();
    m_sizerScrolledWindow->Add( m_panelSidebarAnalogDates, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Caption panel
    m_panelSidebarCaptionForecastDots = new asPanelSidebarCaptionForecastDots( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_panelSidebarCaptionForecastDots->Layout();
    m_sizerScrolledWindow->Add( m_panelSidebarCaptionForecastDots, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Caption panel
    m_panelSidebarCaptionForecastRing = new asPanelSidebarCaptionForecastRing( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_panelSidebarCaptionForecastRing->Layout();
    m_sizerScrolledWindow->Add( m_panelSidebarCaptionForecastRing, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit( m_scrolledWindowOptions );
    Layout();

    // Lead time switcher
    m_leadTimeSwitcher = NULL;

    // Status bar
    SetStatusText(_("Welcome to AtmoSwing"));

    // Connect Events
    m_displayCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameForecast::OnRightClick ), NULL, this );
    m_displayCtrl->Connect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameForecast::OnKeyDown), NULL, this);
    m_displayCtrl->Connect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameForecast::OnKeyUp), NULL, this);
    this->Connect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePreferences ) );
    this->Connect( asID_FRAME_FORECASTER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameForecaster ) );
    this->Connect( asID_FRAME_PLOTS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePlots ) );
    this->Connect( asID_FRAME_GRID, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameGrid ) );
    this->Connect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingNow ) );
    this->Connect( asID_RUN_PREVIOUS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingPast ) );
    this->Connect( asID_OPEN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OnOpenForecast ) );

    // Process
    m_processForecast = NULL;
    m_launchedPresentForecast = false;

    // Restore frame position and size
    wxConfigBase *pConfig = wxFileConfig::Get();
    int minHeight = 450, minWidth = 800;
    int x = pConfig->Read("/MainFrame/x", 50),
        y = pConfig->Read("/MainFrame/y", 50),
        w = pConfig->Read("/MainFrame/w", minWidth),
        h = pConfig->Read("/MainFrame/h", minHeight);
    wxRect screen = wxGetClientDisplayRect();
    if (x<screen.x-10) x = screen.x;
    if (x>screen.width) x = screen.x;
    if (y<screen.y-10) y = screen.y;
    if (y>screen.height) y = screen.y;
    if (w+x>screen.width) w = screen.width-x;
    if (w<minWidth) w = minWidth;
    if (w+x>screen.width) x = screen.width-w;
    if (h+y>screen.height) h = screen.height-y;
    if (h<minHeight) h = minHeight;
    if (h+y>screen.height) y = screen.height-h;

    Move(x, y);
    SetClientSize(w, h);

    bool doMaximize;
    pConfig->Read("/MainFrame/Maximize", &doMaximize);
    Maximize(doMaximize);

    Layout();

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameForecast::~asFrameForecast()
{
    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/SidebarPanelsDisplay/Forecasts", !m_panelSidebarForecasts->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/StationsList", !m_panelSidebarStationsList->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/GisLayers", !m_panelSidebarGisLayers->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/AnalogDates", !m_panelSidebarAnalogDates->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/CaptionForecastDots", !m_panelSidebarCaptionForecastDots->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/Alarms", !m_panelSidebarAlarms->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/CaptionForecastRing", !m_panelSidebarCaptionForecastRing->IsReduced());

    // Save the frame position
    bool doMaximize = IsMaximized();
    pConfig->Write("/MainFrame/Maximize", doMaximize);
    if (!doMaximize)
    {
        int x, y, w, h;
        GetClientSize(&w, &h);
        GetPosition(&x, &y);
        pConfig->Write("/MainFrame/x", (long) x);
        pConfig->Write("/MainFrame/y", (long) y);
        pConfig->Write("/MainFrame/w", (long) w);
        pConfig->Write("/MainFrame/h", (long) h);
    }

    // Disconnect Events
    m_displayCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameForecast::OnRightClick ), NULL, this );
    m_displayCtrl->Disconnect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameForecast::OnKeyDown), NULL, this);
    m_displayCtrl->Disconnect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameForecast::OnKeyUp), NULL, this);
    this->Disconnect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePreferences ) );
    this->Disconnect( asID_FRAME_FORECASTER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameForecaster ) );
    this->Disconnect( asID_FRAME_PLOTS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePlots ) );
    this->Disconnect( asID_FRAME_GRID, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameGrid ) );
    this->Disconnect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingNow ) );
    this->Disconnect( asID_RUN_PREVIOUS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingPast ) );
    this->Disconnect( asID_OPEN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OnOpenForecast ) );

    // Don't delete m_viewerLayerManager, will be deleted by the manager
    wxDELETE(m_layerManager);
    wxDELETE(m_forecastManager);
    wxDELETE(m_forecastViewer);

    // Kill the process if still running
    if (m_processForecast!=NULL)
    {
        asLogMessage(_("Killing the forecast running process."));
        wxKillError killError = m_processForecast->Kill(m_processForecast->GetPid());
        switch (killError)
        {
        case (wxKILL_OK): // no error
            asLogMessage(_("The forecast process has been killed successfully."));
            break;
        case (wxKILL_BAD_SIGNAL): // no such signal
            asLogError(_("The forecast process couldn't be killed (bad signal)."));
            break;
        case (wxKILL_ACCESS_DENIED): // permission denied
            asLogError(_("The forecast process couldn't be killed (access denied)."));
            break;
        case (wxKILL_NO_PROCESS): //  no such process
            asLogError(_("The forecast process couldn't be killed (no process)."));
            break;
        case (wxKILL_ERROR): //  another, unspecified error
            asLogError(_("The forecast process couldn't be killed (error)."));
            break;
        }
    }
}

void asFrameForecast::Init()
{
    wxBusyCursor wait;

    // Update gui elements
    DisplayLogLevelMenu();
    
    // Open last workspace
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString workspaceFilePath = wxEmptyString;
    pConfig->Read("/Workspace/LastOpened", &workspaceFilePath);


    if(!workspaceFilePath.IsEmpty())
    {
        if (!m_workspace.Load(workspaceFilePath))
        {
            asLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
        }

        if (!OpenWorkspace())
        {
            asLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
        }
    }
    else
    {
        asWizardWorkspace wizard(this, &m_workspace);
        wizard.RunWizard(wizard.GetFirstPage());

        OpenWorkspace();
    }

    // Set the display options
    m_panelSidebarForecasts->GetForecastDisplayCtrl()->SetStringArray(m_forecastViewer->GetForecastDisplayStringArray());
    m_panelSidebarForecasts->GetQuantilesCtrl()->SetStringArray(m_forecastViewer->GetQuantilesStringArray());
    m_panelSidebarForecasts->GetForecastDisplayCtrl()->Select(m_forecastViewer->GetForecastDisplaySelection());
    m_panelSidebarForecasts->GetQuantilesCtrl()->Select(m_forecastViewer->GetQuantileSelection());

    // Reduce some panels
    bool display = true;
    pConfig->Read("/SidebarPanelsDisplay/Forecasts", &display, true);
    if (!display)
    {
        m_panelSidebarForecasts->ReducePanel();
        m_panelSidebarForecasts->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/StationsList", &display, true);
    if (!display)
    {
        m_panelSidebarStationsList->ReducePanel();
        m_panelSidebarStationsList->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/Alarms", &display, true);
    if (!display)
    {
        m_panelSidebarAlarms->ReducePanel();
        m_panelSidebarAlarms->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/GisLayers", &display, false);
    if (!display)
    {
        m_panelSidebarGisLayers->ReducePanel();
        m_panelSidebarGisLayers->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/AnalogDates", &display, true);
    if (!display)
    {
        m_panelSidebarAnalogDates->ReducePanel();
        m_panelSidebarAnalogDates->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/CaptionForecastDots", &display, true);
    if (!display)
    {
        m_panelSidebarCaptionForecastDots->ReducePanel();
        m_panelSidebarCaptionForecastDots->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/CaptionForecastRing", &display, true);
    if (!display)
    {
        m_panelSidebarCaptionForecastRing->ReducePanel();
        m_panelSidebarCaptionForecastRing->Layout();
    }

    // Set the select tool
    m_displayCtrl->SetToolDefault();

    m_scrolledWindowOptions->Layout();

    if (!g_cmdFilename.IsEmpty())
    {
        wxArrayString filePathsVect;
        filePathsVect.Add(g_cmdFilename);
        OpenForecast(filePathsVect);
    }

    Layout();
    Refresh();
}

void asFrameForecast::OnOpenWorkspace(wxCommandEvent & event)
{
    // Ask for a workspace file
    wxFileDialog openFileDialog (this, _("Select a workspace"),
                            wxEmptyString,
                            wxEmptyString,
                            "xml files (*.xml)|*.xml",
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_CHANGE_DIR);

    // If canceled
    if(openFileDialog.ShowModal()==wxID_CANCEL)
        return;

    wxBusyCursor wait;

    wxString workspaceFilePath = openFileDialog.GetPath();

    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Workspace/LastOpened", workspaceFilePath);

    // Do open the workspace
    if (!m_workspace.Load(workspaceFilePath))
    {
        asLogError(_("Failed to open the workspace file ") + workspaceFilePath);
    }

    if (!OpenWorkspace())
    {
        asLogError(_("Failed to open the workspace file ") + workspaceFilePath);
    }

}

void asFrameForecast::OnSaveWorkspace(wxCommandEvent & event)
{
    SaveWorkspace();
}

void asFrameForecast::OnSaveWorkspaceAs(wxCommandEvent & event)
{
    // Ask for a workspace file
    wxFileDialog openFileDialog (this, _("Select a path to save the workspace"),
                            wxEmptyString,
                            wxEmptyString,
                            "xml files (*.xml)|*.xml",
                            wxFD_SAVE | wxFD_CHANGE_DIR);

    // If canceled
    if(openFileDialog.ShowModal()==wxID_CANCEL)
        return;

    wxBusyCursor wait;

    wxString workspaceFilePath = openFileDialog.GetPath();
    m_workspace.SetFilePath(workspaceFilePath);

    if(SaveWorkspace())
    {
        // Save preferences
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/Workspace/LastOpened", workspaceFilePath);
    }
}

bool asFrameForecast::SaveWorkspace()
{
    // Update the GIS layers
    m_workspace.ClearLayers();
    int counter = -1;
    for (int i=0; i<m_viewerLayerManager->GetCount(); i++)
    {
        wxFileName fileName = m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName();
        wxString path = fileName.GetFullPath();

        if(!path.IsSameAs("Forecast - specific.memory") && !path.IsSameAs("Forecast - other.memory"))
        {
            counter++;
            m_workspace.AddLayer();
            m_workspace.SetLayerPath(counter, path);

            vrDRIVERS_TYPE type = m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetType();
            wxString strType;
            switch (type)
            {
                case vrDRIVER_UNKNOWN:
                    strType = "undefined";
                    break;
                case vrDRIVER_VECTOR_SHP:
                    strType = "vector";
                    break;
                case vrDRIVER_VECTOR_C2P:
                    strType = "vector";
                    break;
                case vrDRIVER_VECTOR_MEMORY:
                    strType = "undefined";
                    break;
                case vrDRIVER_RASTER_TIFF:
                    strType = "raster";
                    break;
                case vrDRIVER_RASTER_JPEG:
                    strType = "raster";
                    break;
                case vrDRIVER_RASTER_ESRIGRID:
                    strType = "raster";
                    break;
                case vrDRIVER_RASTER_C2D:
                    strType = "raster";
                    break;
                case vrDRIVER_RASTER_EASC:
                    strType = "raster";
                    break;
                case vrDRIVER_RASTER_SGRD7:
                    strType = "raster";
                    break;
                case vrDRIVER_RASTER_WMS:
                    strType = "wms";
                    break;
                case vrDRIVER_USER_DEFINED:
                    strType = "undefined";
                    break;
                default:
                    strType = "undefined";
            }
            m_workspace.SetLayerType(counter, strType);

            int transparency = m_viewerLayerManager->GetRenderer(i)->GetRender()->GetTransparency();
            m_workspace.SetLayerTransparency(counter, transparency);
            bool visible = m_viewerLayerManager->GetRenderer(i)->GetVisible();
            m_workspace.SetLayerVisibility(counter, visible);

            if (strType.IsSameAs("vector"))
            {
                vrRenderVector * vectRender = (vrRenderVector*) m_viewerLayerManager->GetRenderer(i)->GetRender();
                int lineWidth = vectRender->GetSize();
                m_workspace.SetLayerLineWidth(counter, lineWidth);
                wxColour lineColour = vectRender->GetColorPen();
                m_workspace.SetLayerLineColor(counter, lineColour);
                wxColour fillColour = vectRender->GetColorBrush();
                m_workspace.SetLayerFillColor(counter, fillColour);
                wxBrushStyle brushStyle = vectRender->GetBrushStyle();
                m_workspace.SetLayerBrushStyle(counter, brushStyle);
            }
        }
    }

    if(!m_workspace.Save())
    {
        asLogError(_("Could not save the worspace."));
        return false;
    }

    m_workspace.SetHasChanged(false);

    return true;
}

void asFrameForecast::OnNewWorkspace(wxCommandEvent & event)
{
    asWizardWorkspace wizard(this, &m_workspace);
    wizard.RunWizard(wizard.GetSecondPage());
}

bool asFrameForecast::OpenWorkspace()
{
    // GIS layers
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Enter();
    #endif

    m_viewerLayerManager->FreezeBegin();

    // Remove all layers
    for (int i = (signed) m_viewerLayerManager->GetCount()-1; i >= 0 ; i--) 
    {
        // Remove from viewer manager (TOC and Display)
        vrRenderer * renderer = m_viewerLayerManager->GetRenderer(i);
        vrLayer * layer = renderer->GetLayer();
        wxASSERT(renderer);
        m_viewerLayerManager->Remove(renderer);

        // Close layer (not used anymore);
        m_layerManager->Close(layer);
    }

    // Open new layers
    for (int i_layer=m_workspace.GetLayersNb()-1; i_layer>=0; i_layer--)
    {
        // Get attributes
        wxString path = m_workspace.GetLayerPath(i_layer);
        wxString type = m_workspace.GetLayerType(i_layer);
        int transparency = m_workspace.GetLayerTransparency(i_layer);
        bool visibility = m_workspace.GetLayerVisibility(i_layer);
        
        // Open the layers
        if (wxFileName::FileExists(path))
        {
            if (m_layerManager->Open(wxFileName(path)))
            {
                if (type.IsSameAs("raster"))
                {
                    vrRenderRaster* render = new vrRenderRaster();
                    render->SetTransparency(transparency);

                    vrLayer* layer = m_layerManager->GetLayer( wxFileName(path));
                    wxASSERT(layer);
                    m_viewerLayerManager->Add(-1, layer, render, NULL, visibility);
                }
                else if (type.IsSameAs("vector"))
                {
                    int width = m_workspace.GetLayerLineWidth(i_layer);
                    wxColour lineColor = m_workspace.GetLayerLineColor(i_layer);
                    wxColour fillColor = m_workspace.GetLayerFillColor(i_layer);
                    wxBrushStyle brushStyle = m_workspace.GetLayerBrushStyle(i_layer);

                    vrRenderVector* render = new vrRenderVector();
                    render->SetTransparency(transparency);
                    render->SetSize(width);
                    render->SetColorPen(lineColor);
                    render->SetBrushStyle(brushStyle);
                    render->SetColorBrush(fillColor);

                    vrLayer* layer = m_layerManager->GetLayer( wxFileName(path));
                    wxASSERT(layer);
                    m_viewerLayerManager->Add(-1, layer, render, NULL, visibility);
                }
                else if (type.IsSameAs("wms"))
                {
                    vrRenderRaster* render = new vrRenderRaster();
                    render->SetTransparency(transparency);

                    vrLayer* layer = m_layerManager->GetLayer( wxFileName(path));
                    wxASSERT(layer);
                    m_viewerLayerManager->Add(-1, layer, render, NULL, visibility);
                }
                else
                {
                    asLogError(wxString::Format(_("The GIS layer type %s does not correspond to allowed values."), type.c_str()));
                }
            }
            else
            {
                asLogWarning(wxString::Format(_("The file %s cound not be opened."), path.c_str()));
            }
        }
        else
        {
            asLogWarning(wxString::Format(_("The file %s cound not be found."), path.c_str()));
        }
    }

    m_viewerLayerManager->FreezeEnd();

    OpenRecentForecasts();

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit( m_scrolledWindowOptions );
    Layout();
    
    m_workspace.SetHasChanged(false);

    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Leave();
    #endif

    return true;
}

void asFrameForecast::OnClose(wxCloseEvent& event)
{
    if ( event.CanVeto() && m_workspace.HasChanged() )
    {
        if ( wxMessageBox("The workspace has not been saved... continue closing?",
                          "Please confirm",
                          wxICON_QUESTION | wxYES_NO) != wxYES )
        {
            event.Veto();
            return;
        }
    }
    
    event.Skip();
}

void asFrameForecast::OnQuit( wxCommandEvent& event )
{
    event.Skip();
}

void asFrameForecast::UpdateLeadTimeSwitch()
{
    // Required size
    int squareSize = 40;
    int width = (m_forecastManager->GetFullTargetDates().size()+1)*squareSize;
    int height = squareSize + 5;

    // Delete and recreate the panel. Cannot get it work with a resize...
    wxDELETE(m_leadTimeSwitcher);
    m_leadTimeSwitcher = new asLeadTimeSwitcher( m_panelTop, &m_workspace, m_forecastManager, wxID_ANY, wxDefaultPosition, wxSize(width,height), wxTAB_TRAVERSAL );
    m_leadTimeSwitcher->SetBackgroundColour( wxColour( 77, 77, 77 ) );
    m_leadTimeSwitcher->SetMinSize( wxSize( width, height ) );
    m_leadTimeSwitcher->Layout();
    Array1DFloat dates = m_forecastManager->GetFullTargetDates();
    m_leadTimeSwitcher->Draw(dates);
    
    m_sizerLeadTimeSwitch->Add( m_leadTimeSwitcher, 0, wxALL | wxALIGN_RIGHT, 5 );
    m_sizerLeadTimeSwitch->Layout();
    
    m_panelTop->Layout();
	m_sizerTop->Fit( m_panelTop );
    m_sizerContent->Layout();

}

void asFrameForecast::LaunchForecastingNow( wxCommandEvent& event )
{
    m_toolBar->EnableTool(asID_RUN, false);
    m_toolBar->EnableTool(asID_RUN_PREVIOUS, false);

    // Get forecaster path
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString forecasterPath = pConfig->Read("/Paths/ForecasterPath", asConfig::GetSoftDir()+"atmoswing-forecaster");

    if(forecasterPath.IsEmpty())
    {
        asLogError(_("Please set the path to the forecaster in the preferences."));
        return;
    }

    // Set option
    wxString options = wxString::Format(" -fn -ll 2 -lt file");
    forecasterPath.Append(options);
    asLogMessage(wxString::Format(_("Sending command: %s"), forecasterPath.c_str()));

    // Create a process
    if (m_processForecast!=NULL)
    {
        asLogError(_("There is already a running forecast process. Please wait."));
        return;
    }
    m_processForecast = new wxProcess(this);

    // Redirect output fluxes
    //m_processForecast->Redirect(); // CAUTION redirect causes the downloads to hang after a few ones!

    // Execute
    long processId = wxExecute(forecasterPath, wxEXEC_ASYNC, m_processForecast);

    if (processId==0) // if wxEXEC_ASYNC
    {
        asLogError(_("The forecaster could not be executed. Please check the path in the preferences."));
        wxDELETE(m_processForecast);
        return;
    }

    m_launchedPresentForecast = true;
}

void asFrameForecast::LaunchForecastingPast( wxCommandEvent& event )
{
    m_toolBar->EnableTool(asID_RUN, false);
    m_toolBar->EnableTool(asID_RUN_PREVIOUS, false);

    // Get forecaster path
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString forecasterPath = pConfig->Read("/Paths/ForecasterPath", asConfig::GetSoftDir()+"atmoswing-forecaster");

    if(forecasterPath.IsEmpty())
    {
        asLogError(_("Please set the path to the forecaster in the preferences."));
        return;
    }

    // Set option
    int nbPrevDays = m_workspace.GetTimeSeriesPlotPastDaysNb();
    wxString options = wxString::Format(" -fp %d -ll 2 -lt file", nbPrevDays);
    forecasterPath.Append(options);
    asLogMessage(wxString::Format(_("Sending command: %s"), forecasterPath.c_str()));

    // Create a process
    if (m_processForecast!=NULL)
    {
        asLogError(_("There is already a running forecast process. Please wait."));
        return;
    }
    m_processForecast = new wxProcess(this);

    // Redirect output fluxes
    //m_processForecast->Redirect(); // CAUTION redirect causes the downloads to hang after a few ones!

    // Execute
    long processId = wxExecute(forecasterPath, wxEXEC_ASYNC, m_processForecast);

    if (processId==0) // if wxEXEC_ASYNC
    {
        asLogError(_("The forecaster could not be executed. Please check the path in the preferences."));
        wxDELETE(m_processForecast);
        return;
    }
}

void asFrameForecast::OnForecastProcessTerminate( wxProcessEvent &event )
{
    m_toolBar->EnableTool(asID_RUN, true);
    m_toolBar->EnableTool(asID_RUN_PREVIOUS, true);

    asLogMessage("The forecast processing is over.");

    if (m_launchedPresentForecast)
    {
        if (m_forecastManager->GetMethodsNb()>0)
        {
            wxMessageDialog dlg(this,
                                "The forecast processing is over. Do you want to load the results? "
                                "This may close all files currently opened.",
                                "Open new forecast?",
                                wxCENTER |
                                wxNO_DEFAULT | wxYES_NO | wxCANCEL |
                                wxICON_INFORMATION);

            if ( dlg.ShowModal() == wxID_YES )
            {
                OpenForecastsFromTmpList();
            }
        }
        else
        {
            OpenForecastsFromTmpList();
        }
    }
    else
    {
        wxMessageBox(_("The forecast processing is over."));
    }

    wxDELETE(m_processForecast);
}

void asFrameForecast::OpenFrameForecaster( wxCommandEvent& event )
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString forecasterPath = pConfig->Read("/Paths/ForecasterPath", asConfig::GetSoftDir()+"atmoswing-forecaster");

    if(forecasterPath.IsEmpty())
    {
        asLogError(_("Please set the path to the forecaster in the preferences."));
        return;
    }

    // Execute
    long processId = wxExecute(forecasterPath, wxEXEC_ASYNC);

    if (processId==0) // if wxEXEC_ASYNC
    {
        asLogError(_("The forecaster could not be executed. Please check the path in the preferences."));
    }
}

void asFrameForecast::OpenFramePlots( wxCommandEvent& event )
{
    wxBusyCursor wait;

    asFramePlotDistributions* framePlot = new asFramePlotDistributions(this, m_forecastViewer->GetMethodSelection(), m_forecastViewer->GetForecastSelection(), m_forecastManager);
    framePlot->Layout();
    framePlot->Init();
    framePlot->Plot();
    framePlot->Show();
}

void asFrameForecast::OpenFrameGrid( wxCommandEvent& event )
{
    wxBusyCursor wait;

    asFrameGridAnalogsValues* frameGrid = new asFrameGridAnalogsValues(this, m_forecastViewer->GetMethodSelection(), m_forecastViewer->GetForecastSelection(), m_forecastManager);
    frameGrid->Layout();
    frameGrid->Init();
    frameGrid->Show();
}

void asFrameForecast::OpenFramePreferences( wxCommandEvent& event )
{
    wxBusyCursor wait;

    asFramePreferencesViewer* frame = new asFramePreferencesViewer(this, &m_workspace, asWINDOW_PREFERENCES);
    frame->Fit();
    frame->Show();
}

void asFrameForecast::OpenFrameAbout( wxCommandEvent& event )
{
    wxBusyCursor wait;

    asFrameAbout* frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameForecast::OnLogLevel1( wxCommandEvent& event )
{
    Log().SetLevel(1);
    m_menuLogLevel->FindItemByPosition(0)->Check(true);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::OnLogLevel2( wxCommandEvent& event )
{
    Log().SetLevel(2);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(true);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::OnLogLevel3( wxCommandEvent& event )
{
    Log().SetLevel(3);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(true);
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::DisplayLogLevelMenu()
{
    // Set log level in the menu
    int logLevel = (int)wxFileConfig::Get()->Read("/General/LogLevel", 1l);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel)
    {
    case 1:
        m_menuLogLevel->FindItemByPosition(0)->Check(true);
        Log().SetLevel(1);
        break;
    case 2:
        m_menuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
        break;
    case 3:
        m_menuLogLevel->FindItemByPosition(2)->Check(true);
        Log().SetLevel(3);
        break;
    default:
        m_menuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
    }
}

bool asFrameForecast::OpenLayers (const wxArrayString & names)
{
    // Open files
    for (unsigned int i = 0; i< names.GetCount(); i++)
    {
        if(!m_layerManager->Open(wxFileName(names.Item(i))))
        {
            asLogError(_("The layer could not be opened."));
            return false;
        }
    }

    // Get files
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Enter();
    #endif
    m_viewerLayerManager->FreezeBegin();
    for (unsigned int i = 0; i< names.GetCount(); i++)
    {
        vrLayer * layer = m_layerManager->GetLayer( wxFileName(names.Item(i)));
        wxASSERT(layer);

        // Add files to the viewer
        m_viewerLayerManager->Add(1, layer, NULL);
    }
    m_viewerLayerManager->FreezeEnd();
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Leave();
    #endif
        
    m_workspace.SetHasChanged(true);

    return true;

}

void asFrameForecast::OnOpenLayer(wxCommandEvent & event)
{
    vrDrivers drivers;
    wxFileDialog myFileDlg (this, _("Select GIS layers"),
                            wxEmptyString,
                            wxEmptyString,
                            drivers.GetWildcards(),
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_MULTIPLE | wxFD_CHANGE_DIR);

    wxArrayString pathsFileName;

    wxBusyCursor wait;

    // Try to open files
    if(myFileDlg.ShowModal()==wxID_OK)
    {
        myFileDlg.GetPaths(pathsFileName);
        wxASSERT(pathsFileName.GetCount() > 0);

        OpenLayers(pathsFileName);
    }
}

void asFrameForecast::OnCloseLayer(wxCommandEvent & event)
{
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Enter();
    #endif

    // Creates the list of layers
    wxArrayString layersName;
    for (int i = 0; i<m_viewerLayerManager->GetCount(); i++)
    {
        vrRenderer * renderer = m_viewerLayerManager->GetRenderer(i);
        wxASSERT(renderer);
        layersName.Add(renderer->GetLayer()->GetDisplayName().GetFullName());
    }

    if (layersName.IsEmpty())
    {
        asLogError(_("No layer opened, nothing to close."));
        #if defined (__WIN32__)
            m_critSectionViewerLayerManager.Leave();
        #endif
        return;
    }

    // Choice dialog box
    wxMultiChoiceDialog  choiceDlg (this, _("Select Layer(s) to close"),
                                    _("Close layer(s)"),
                                    layersName);
    if (choiceDlg.ShowModal() != wxID_OK)
    {
        #if defined (__WIN32__)
            m_critSectionViewerLayerManager.Leave();
        #endif
        return;
    }

    wxBusyCursor wait;

    // Get indices of layer to remove
    wxArrayInt layerToRemoveIndex = choiceDlg.GetSelections();
    if (layerToRemoveIndex.IsEmpty())
    {
        wxLogWarning(_("Nothing selected, no layer will be closed"));
        #if defined (__WIN32__)
            m_critSectionViewerLayerManager.Leave();
        #endif
        return;
    }

    // Remove layer(s)
    m_viewerLayerManager->FreezeBegin();
    for (int i = (signed) layerToRemoveIndex.GetCount()-1; i >= 0 ; i--) {

        // Remove from viewer manager (TOC and Display)
        vrRenderer * renderer = m_viewerLayerManager->GetRenderer(layerToRemoveIndex.Item(i));
        vrLayer * layer = renderer->GetLayer();
        wxASSERT(renderer);
        m_viewerLayerManager->Remove(renderer);

        // Close layer (not used anymore);
        m_layerManager->Close(layer);
    }

    m_viewerLayerManager->FreezeEnd();
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Leave();
    #endif

    m_workspace.SetHasChanged(true);
}

void asFrameForecast::OnOpenForecast(wxCommandEvent & event)
{
    wxFileDialog myFileDlg (this, _("Select a forecast file"),
                            wxEmptyString,
                            wxEmptyString,
                            "*.fcst",
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_CHANGE_DIR | wxFD_MULTIPLE);

    wxArrayString pathsFileName;

    wxBusyCursor wait;

    // Try to open files
    if(myFileDlg.ShowModal()==wxID_OK)
    {
        myFileDlg.GetPaths(pathsFileName);
        wxASSERT(pathsFileName.GetCount() > 0);

        OpenForecast(pathsFileName);
    }
}

void asFrameForecast::OpenForecastsFromTmpList()
{
    // Write the resulting files path into a temp file.
    wxString tempFile = asConfig::GetTempDir() + "AtmoSwingForecatsFilePaths.txt";
    asFileAscii filePaths(tempFile, asFile::ReadOnly);
    wxArrayString filePathsVect;
    filePaths.Open();
    while (!filePaths.EndOfFile())
    {
        wxString path = filePaths.GetLineContent();

        if (!path.IsEmpty())
        {
            filePathsVect.Add(path);
        }
    }
    filePaths.Close();

    OpenForecast(filePathsVect);
}

bool asFrameForecast::OpenRecentForecasts()
{
    m_forecastManager->ClearForecasts();

    wxString forecastsDirectory = m_workspace.GetForecastsDirectory();

    if (forecastsDirectory.IsEmpty())
    {
        asLogError("The directory containing the forecasts was not provided.");
        return false;
    }

    if (!wxFileName::DirExists(forecastsDirectory))
    {
        asLogError("The directory that is supposed to contain the forecasts does not exist.");
        return false;
    }

    // Get present date
    double now = asTime::NowMJD();

    // Check if today directory exists
    wxString basePath = forecastsDirectory + wxFileName::GetPathSeparator();
    wxFileName fullPath(basePath);
    fullPath.AppendDir(wxString::Format("%d", asTime::GetYear(now)));
    fullPath.AppendDir(wxString::Format("%02d", asTime::GetMonth(now)));
    fullPath.AppendDir(wxString::Format("%02d", asTime::GetDay(now)));

    // If does not exist, try the day before
    if (!fullPath.Exists())
    {
        now--;
        fullPath = wxFileName(basePath);
        fullPath.AppendDir(wxString::Format("%d", asTime::GetYear(now)));
        fullPath.AppendDir(wxString::Format("%02d", asTime::GetMonth(now)));
        fullPath.AppendDir(wxString::Format("%02d", asTime::GetDay(now)));
    }

    // If does not exist, warn the user and return
    if (!fullPath.Exists())
    {
        fullPath = wxFileName(basePath);
        asLogError(wxString::Format(_("No recent forecast was found under %s"), fullPath.GetFullPath().c_str()));
        return false;
    }
    
    // List the files in the directory
    wxArrayString files;
    wxDir::GetAllFiles (fullPath.GetFullPath(), &files);

    // Identify the most recent forecasts
    long mostRecentDate = 0;
    VectorInt mostRecentRows;
    for (int i=0; i<files.GetCount(); i++)
    {
        wxFileName fileName(files[i]);
        wxString fileDate = fileName.GetFullName().SubString(0,9);
        if (!fileDate.IsNumber())
        {
            asLogWarning(_("A file with an unconventional name was found in the forecasts directory."));
            continue;
        }

        long date;
        fileDate.ToLong(&date);

        if (date>mostRecentDate)
        {
            mostRecentDate = date;
            mostRecentRows.clear();
            mostRecentRows.push_back(i);
        }
        else if (date==mostRecentDate)
        {
            mostRecentRows.push_back(i);
        }
    }

    // Store the most recent file names
    wxArrayString recentFiles;
    for (int i=0; i<mostRecentRows.size(); i++)
    {
        recentFiles.Add(files[mostRecentRows[i]]);
    }

    // Open the forecasts
    if (!OpenForecast(recentFiles))
    {
        asLogError(_("Failed to open the forecasts."));
        return false;
    }

    FitExtentToForecasts();

    return true;
}

void asFrameForecast::OnLoadPreviousForecast( wxCommandEvent & event )
{
    SwitchForecast( -1.0/24.0 );
}

void asFrameForecast::OnLoadNextForecast( wxCommandEvent & event )
{
    SwitchForecast( 1.0/24.0 );
}

void asFrameForecast::OnLoadPreviousDay( wxCommandEvent & event )
{
    SwitchForecast( -1.0 );
}

void asFrameForecast::OnLoadNextDay( wxCommandEvent & event )
{
    SwitchForecast( 1.0 );
}

void asFrameForecast::SwitchForecast( double increment )
{
    wxBusyCursor wait;

    if (m_forecastManager->GetMethodsNb()==0)
    {
        asLogError("There is no opened forecast.");
        return;
    }

    // Get path
    wxString forecastsPath = m_forecastManager->GetFilePath(m_forecastViewer->GetMethodSelection(), m_forecastViewer->GetForecastSelection());
    wxFileName forecastFileName(forecastsPath);
    wxString fileName = forecastFileName.GetName();
    wxString partialFileName = fileName.SubString(10,fileName.size()-1);
    wxString patternFileName = "%d%02d%02d%02d";
    wxString prefixFileName = wxEmptyString;

    forecastFileName.RemoveLastDir();
    forecastFileName.RemoveLastDir();
    forecastFileName.RemoveLastDir();
    wxString forecastsBaseDirectory = forecastFileName.GetPath();
    
    if (!wxFileName::DirExists(forecastsBaseDirectory))
    {
        asLogError("The directory that is supposed to contain the forecasts does not exist.");
        return;
    }
    
    // Get date
    double date = m_forecastManager->GetLeadTimeOrigin();

    // Look for former files
    wxString basePath = forecastsBaseDirectory + wxFileName::GetPathSeparator();
    wxFileName fullPath(basePath);
    for (int i=0; i<100; i++)
    {
        date += increment;
        fullPath = wxFileName(basePath);
        fullPath.AppendDir(wxString::Format("%d", asTime::GetYear(date)));
        fullPath.AppendDir(wxString::Format("%02d", asTime::GetMonth(date)));
        fullPath.AppendDir(wxString::Format("%02d", asTime::GetDay(date)));
        prefixFileName = wxString::Format(patternFileName, asTime::GetYear(date), asTime::GetMonth(date), asTime::GetDay(date), asTime::GetHour(date));
        fullPath.SetName(prefixFileName + partialFileName);
        fullPath.SetExt("fcst");

        if (fullPath.Exists()) break;

        if (i==99)
        {
            asLogError(wxString::Format(_("No previous/next forecast was found under %s"), fullPath.GetFullPath().c_str()));
            return;
        }
    }
    
    // List the files in the directory
    wxArrayString files;
    wxDir::GetAllFiles (fullPath.GetPath(), &files);

    // Identify the corresponding forecasts
    wxArrayString accurateFiles;
    for (int i=0; i<files.GetCount(); i++)
    {
        wxFileName fileName(files[i]);
        wxString fileDate = fileName.GetFullName().SubString(0,9);

        if (fileDate.IsSameAs(prefixFileName))
        {
            accurateFiles.Add(files[i]);
        }
    }

    // Open the forecasts
    m_forecastManager->ClearForecasts();
    if (!OpenForecast(accurateFiles))
    {
        asLogError(_("Failed to open the forecasts."));
        return;
    }

    // Refresh view
    m_forecastViewer->Redraw();
    UpdateHeaderTexts();
    UpdatePanelCaptionAll();
    UpdatePanelAnalogDates();
}

bool asFrameForecast::OpenForecast (const wxArrayString & names)
{
    if (names.GetCount()==0) return false;

    Freeze();

    // Close plots
    bool continueClosing = true;
    while (continueClosing)
    {
        continueClosing = false;

        wxWindow* framePlotsTimeseries = wxWindow::FindWindowById(asWINDOW_PLOTS_TIMESERIES);
        if (framePlotsTimeseries!=NULL)
        {
            wxASSERT(framePlotsTimeseries);
            framePlotsTimeseries->SetId(0);
            framePlotsTimeseries->Destroy();
            continueClosing = true;
        }

        wxWindow* framePlotsDistributions = wxWindow::FindWindowById(asWINDOW_PLOTS_DISTRIBUTIONS);
        if (framePlotsDistributions!=NULL)
        {
            wxASSERT(framePlotsDistributions);
            framePlotsDistributions->SetId(0);
            framePlotsDistributions->Destroy();
            continueClosing = true;
        }

        wxWindow* frameGrid = wxWindow::FindWindowById(asWINDOW_GRID_ANALOGS);
        if (frameGrid!=NULL)
        {
            wxASSERT(frameGrid);
            frameGrid->SetId(0);
            frameGrid->Destroy();
            continueClosing = true;
        }

        wxGetApp().Yield();
        wxWakeUpIdle();
    }

    // Open files
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Enter();
    #endif
    for (unsigned int i = 0; i< names.GetCount(); i++)
    {
        if (i==0)
        {
            wxString dir = names.Item(i);
            wxUniChar dirSep = DS.GetChar(0);
            dir = dir.BeforeLast(dirSep);
            if (dir.Length()>10)
            {
                dir = dir.Left(dir.Length()-10);
                m_forecastManager->AddDirectoryPastForecasts(dir);
            }
        }

        bool doRefresh = false;
        if (i==names.GetCount()-1)
        {
            doRefresh = true;
        }

        bool successOpen = m_forecastManager->Open(names.Item(i), doRefresh);
        if(!successOpen)
        {
            asLogError(wxString::Format(_("The forecast file %d could not be opened (%s)."), i, names.Item(i).c_str()));
            #if defined (__WIN32__)
                m_critSectionViewerLayerManager.Leave();
            #endif
            m_viewerLayerManager->Reload();
            Thaw();
            return false;
        }
    }
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Leave();
    #endif

    UpdateLeadTimeSwitch();

    m_leadTimeSwitcher->SetLeadTime(m_forecastViewer->GetLeadTimeIndex());

    Thaw();

    return true;
}

void asFrameForecast::OnKeyDown(wxKeyEvent & event)
{
    m_keyBoardState = wxKeyboardState(event.ControlDown(),
                                      event.ShiftDown(),
                                      event.AltDown(),
                                      event.MetaDown());
    if (m_keyBoardState.GetModifiers() != wxMOD_CMD)
    {
        event.Skip();
        return;
    }

    const vrDisplayTool * tool = m_displayCtrl->GetTool();
    if (tool == NULL)
    {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_IN)
    {
        m_displayCtrl->SetToolZoomOut();
    }
    event.Skip();
}

void asFrameForecast::OnKeyUp(wxKeyEvent & event)
{
    if (m_keyBoardState.GetModifiers() != wxMOD_CMD)
    {
        event.Skip();
        return;
    }

    const vrDisplayTool * tool = m_displayCtrl->GetTool();
    if (tool == NULL)
    {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_OUT || tool->GetID() == wxID_ZOOM_IN)
    {
        m_displayCtrl->SetToolZoom();
    }
    event.Skip();
}

void asFrameForecast::OnToolSelect (wxCommandEvent & event)
{
    m_displayCtrl->SetToolDefault();
}

void asFrameForecast::OnToolZoomIn (wxCommandEvent & event)
{
    m_displayCtrl->SetToolZoom();
}

void asFrameForecast::OnToolZoomOut (wxCommandEvent & event)
{
    m_displayCtrl->SetToolZoomOut();
}

void asFrameForecast::OnToolPan (wxCommandEvent & event)
{
    m_displayCtrl->SetToolPan();
}

void asFrameForecast::OnToolZoomToFit (wxCommandEvent & event)
{
    // Fit to all layers
    // m_viewerLayerManager->ZoomToFit(true);
    // ReloadViewerLayerManager();

    // Fit to the forecasts layer
    FitExtentToForecasts();
}

void asFrameForecast::FitExtentToForecasts ()
{
    wxBusyCursor wait;

    vrLayerVector * layer = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - specific.memory"));

    if(layer!=NULL)
    {
        wxASSERT(layer);

        // Get the forecast layer extent
        vrRealRect extent;
        layer->GetExtent(extent);

        // Add a margin
        wxDouble width = extent.GetRight()-extent.GetLeft();
        wxDouble height = extent.GetTop()-extent.GetBottom();
        wxDouble marginFactor = 0.05;
        extent.SetLeft(extent.GetLeft()-marginFactor*width);
        extent.SetRight(extent.GetRight()+marginFactor*width);
        extent.SetBottom(extent.GetBottom()-marginFactor*height);
        extent.SetTop(extent.GetTop()+marginFactor*height);

        // Force new extent
        m_viewerLayerManager->InitializeExtent(extent);
    }
    else
    {
        asLogError(_("The forecasts layer was not found."));
    }

    ReloadViewerLayerManager();
}

void asFrameForecast::OnMoveLayer (wxCommandEvent & event)
{
    wxBusyCursor wait;

    // Check than more than 1 layer
    if (m_viewerLayerManager->GetCount() <= 1)
    {
        asLogError(_("Moving layer not possible with less than 2 layers"));
        return;
    }

    // Get selection
    int iOldPos = m_panelSidebarGisLayers->GetTocCtrl()->GetSelection();
    if (iOldPos == wxNOT_FOUND)
    {
        asLogError(_("No layer selected, select a layer first"));
        return;
    }

    // Contextual menu
    wxMenu posMenu;
    posMenu.SetTitle(_("Move layer to following position"));
    for (int i = 0; i<m_viewerLayerManager->GetCount(); i++)
    {
        posMenu.Append(asID_MENU_POPUP_LAYER + i,
                         wxString::Format("%d - %s",i+1,
                                          m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetDisplayName().GetFullName()));
    }
    wxPoint pos = wxGetMousePosition();

    int iNewID = GetPopupMenuSelectionFromUser(posMenu, ScreenToClient(pos));
    if (iNewID == wxID_NONE) return;

    int iNewPos = iNewID - asID_MENU_POPUP_LAYER;
    if (iNewPos == iOldPos) return;

    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Enter();
    #endif
    m_viewerLayerManager->Move(iOldPos, iNewPos);
    #if defined (__WIN32__)
        m_critSectionViewerLayerManager.Leave();
    #endif

    m_workspace.SetHasChanged(true);
}

void asFrameForecast::OnToolAction (wxCommandEvent & event)
{
    // Get event
    vrDisplayToolMessage * msg = (vrDisplayToolMessage*)event.GetClientData();
    wxASSERT(msg);

    if(msg->m_EvtType == vrEVT_TOOL_ZOOM)
    {
        // Get rectangle
        vrCoordinate * coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        coord->ConvertFromPixels(msg->m_rect, realRect);
        wxASSERT(realRect.IsOk());

        // Get fitted rectangle
        vrRealRect fittedRect =coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        // Moving view
        #if defined (__WIN32__)
            asThreadViewerLayerManagerZoomIn *thread = new asThreadViewerLayerManagerZoomIn(m_viewerLayerManager, &m_critSectionViewerLayerManager, fittedRect);
            ThreadsManager().AddThread(thread);
        #else
            m_viewerLayerManager->Zoom(fittedRect);
        #endif
    }
    else if (msg->m_EvtType == vrEVT_TOOL_ZOOMOUT)
    {
        // Get rectangle
        vrCoordinate * coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        wxASSERT(coord->ConvertFromPixels(msg->m_rect, realRect));
        coord->ConvertFromPixels(msg->m_rect, realRect);

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        // Moving view
        #if defined (__WIN32__)
            asThreadViewerLayerManagerZoomOut *thread = new asThreadViewerLayerManagerZoomOut(m_viewerLayerManager, &m_critSectionViewerLayerManager, fittedRect);
            ThreadsManager().AddThread(thread);
        #else
            m_viewerLayerManager->ZoomOut(fittedRect);
        #endif
    }
    else if (msg->m_EvtType == vrEVT_TOOL_SELECT)
    {
        // If no forecast open
        if (m_forecastManager->GetMethodsNb()==0)
        {
            wxDELETE(msg);
            return;
        }

        // Transform screen coordinates to real coordinates
        vrCoordinate * coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);
        wxPoint clickedPos = msg->m_position;
        if (clickedPos != wxDefaultPosition)
        {
            wxPoint2DDouble realClickedPos;
            coord->ConvertFromPixels(clickedPos, realClickedPos);

            // Create a polygon to select the stations
            OGRPolygon polygon;
            OGRLinearRing linRing;

            // Buffer around the clicked point. Get extent to select according to the scale.
            double ratioBuffer = 0.025; // 50px for the symbols 1000px for the display area
            vrRealRect actExtent = coord->GetExtent();
            int width = actExtent.GetSize().GetWidth();
            double bufferSize = width*ratioBuffer;

            linRing.addPoint(realClickedPos.m_x-bufferSize, realClickedPos.m_y-bufferSize);
            linRing.addPoint(realClickedPos.m_x-bufferSize, realClickedPos.m_y+bufferSize);
            linRing.addPoint(realClickedPos.m_x+bufferSize, realClickedPos.m_y+bufferSize);
            linRing.addPoint(realClickedPos.m_x+bufferSize, realClickedPos.m_y-bufferSize);
            linRing.addPoint(realClickedPos.m_x-bufferSize, realClickedPos.m_y-bufferSize);

            polygon.addRing(&linRing);

            // Get layer
            vrLayerVector * layer = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - specific.memory"));

            if(layer!=NULL)
            {
                // Search features
                wxArrayLong stationsClose;
                layer->SearchFeatures(&polygon, stationsClose);

                // Allow only one selection
                wxArrayLong station;
                if (stationsClose.Count()>0)
                {
                    station.Add(stationsClose.Item(0));
                    int stationItem = stationsClose.Item(0);
                    OGRFeature * feature = layer->GetFeature(stationItem);
                    int stationRow = (int)feature->GetFieldAsDouble(0);

                    if (stationRow>=0)
                    {
                        m_panelSidebarStationsList->GetChoiceCtrl()->Select(stationRow);
                    }
                    DrawPlotStation(stationRow);
                }
                else
                {
                    // Search on the other (not specific) forecast layer
                    vrLayerVector * layerOther = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - other.memory"));
                    if(layerOther!=NULL)
                    {
                        // Search features
                        layerOther->SearchFeatures(&polygon, stationsClose);

                        // Allow only one selection
                        if (stationsClose.Count()>0)
                        {
                            station.Add(stationsClose.Item(0));
                            int stationItem = stationsClose.Item(0);
                            OGRFeature * feature = layerOther->GetFeature(stationItem);
                            int stationRow = (int)feature->GetFieldAsDouble(0);

                            if (stationRow>=0)
                            {
                                m_panelSidebarStationsList->GetChoiceCtrl()->Select(stationRow);
                            }
                            DrawPlotStation(stationRow);
                        }
                    }
                }
                layer->SetSelectedIDs(station);

            }
            else
            {
                asLogError(_("The desired layer was not found."));
            }

            ReloadViewerLayerManager();
        }

    }
    else if (msg->m_EvtType == vrEVT_TOOL_PAN)
    {
        // Get rectangle
        vrCoordinate * coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        wxPoint movedPos = msg->m_position;
        wxPoint2DDouble movedRealPt;
        if (coord->ConvertFromPixels(movedPos, movedRealPt)==false)
        {
            asLogError(wxString::Format(_("Error converting point : %d, %d to real coordinate"),
                       movedPos.x, movedPos.y));
            wxDELETE(msg);
            return;
        }

        vrRealRect actExtent = coord->GetExtent();
        actExtent.MoveLeftTopTo(movedRealPt);
        coord->SetExtent(actExtent);
        ReloadViewerLayerManager();
    }
    else
    {
        asLogError(_("Operation not supported now. Please contact the developers."));
    }

    wxDELETE(msg);
}

void asFrameForecast::OnStationSelection( wxCommandEvent& event )
{
    wxBusyCursor wait;

    // Get selection
    int choice = event.GetInt();

    // If no forecast open
    if (m_forecastManager->GetMethodsNb()==0)
    {
        return;
    }

    // Display on the map when only the specific layer exists
    vrLayerVector * layer = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - specific.memory"));
    vrLayerVector * layerOther = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - other.memory"));
    if(layer!=NULL && layerOther==NULL)
    {
        // Set selection
        wxArrayLong station;
        station.Add(choice);
        layer->SetSelectedIDs(station);
    }

    DrawPlotStation(choice);

    ReloadViewerLayerManager();
}

void asFrameForecast::OnChangeLeadTime( wxCommandEvent& event )
{
    wxBusyCursor wait;

    Freeze();

    m_forecastViewer->ChangeLeadTime(event.GetInt());
    
    m_leadTimeSwitcher->SetLeadTime(m_forecastViewer->GetLeadTimeIndex());

    UpdatePanelAnalogDates();
    UpdatePanelCaptionAll();

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit( m_scrolledWindowOptions );
    Layout();

    Thaw();
}

void asFrameForecast::OnForecastClear( wxCommandEvent& event )
{
    if(m_panelSidebarForecasts!=NULL)
    {
        m_panelSidebarForecasts->ClearForecasts();
    }
}

void asFrameForecast::OnForecastRatioSelectionChange( wxCommandEvent& event )
{
    wxBusyCursor wait;

    m_forecastViewer->SetForecastDisplay(event.GetInt());

    UpdatePanelCaptionColorbar();
}

void asFrameForecast::OnForecastForecastSelectionChange( wxCommandEvent& event )
{
    wxBusyCursor wait;

    Freeze();

    asMessageForecastChoice* message = (asMessageForecastChoice*)event.GetClientData();

    m_forecastViewer->SetForecast(message->GetMethodRow(), message->GetForecastRow());
    
    if(m_leadTimeSwitcher)
    {
        m_leadTimeSwitcher->SetLeadTime(m_forecastViewer->GetLeadTimeIndex());
    }

    UpdateHeaderTexts();
    UpdatePanelCaptionAll();
    UpdatePanelAnalogDates();
    UpdatePanelStationsList();

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit( m_scrolledWindowOptions );
    Layout();

    wxDELETE(message);

    Thaw();
}

void asFrameForecast::OnForecastForecastSelectFirst( wxCommandEvent& event )
{
    m_panelSidebarForecasts->GetForecastsCtrl()->SelectFirst();
}

void asFrameForecast::OnForecastQuantileSelectionChange( wxCommandEvent& event )
{
    wxBusyCursor wait;

    m_forecastViewer->SetQuantile(event.GetInt());
}

void asFrameForecast::DrawPlotStation( int stationRow )
{
    wxBusyCursor wait;

    m_forecastViewer->LoadPastForecast();

    // Get data
    int methodRow = m_forecastViewer->GetMethodSelection();
    int forecastRow = m_forecastViewer->GetForecastSelection();

    if (forecastRow<0) // Aggregator
    {
        forecastRow = m_forecastManager->GetForecastRowSpecificForStationRow(methodRow, stationRow);
    }

    asFramePlotTimeSeries* framePlotStation = new asFramePlotTimeSeries(this, methodRow, forecastRow, stationRow, m_forecastManager);
    framePlotStation->Layout();
    framePlotStation->Init();
    framePlotStation->Plot();
    framePlotStation->Show();
}

void asFrameForecast::OnForecastNewAdded( wxCommandEvent& event )
{
    wxBusyCursor wait;

    m_panelSidebarForecasts->Update();

    if (event.GetString().IsSameAs("last"))
    {
        m_forecastViewer->FixForecastSelection();

        float previousDate = m_forecastViewer->GetLeadTimeDate();
        m_forecastViewer->SetLeadTimeDate(previousDate);
        
        m_panelSidebarAlarms->Update();
    }
}

void asFrameForecast::ReloadViewerLayerManager( )
{
    wxBusyCursor wait;

    m_viewerLayerManager->Reload();

    /* Not sure there is any way to make it safe with threads.
    #if defined (__WIN32__)
        asThreadViewerLayerManagerReload *thread = new asThreadViewerLayerManagerReload(m_viewerLayerManager, &m_critSectionViewerLayerManager);
        ThreadsManager().AddThread(thread);
    #else
        m_viewerLayerManager->Reload();
    #endif*/
}

void asFrameForecast::UpdateHeaderTexts()
{
    // Set header text
    wxString dateForecast = asTime::GetStringTime(m_forecastManager->GetLeadTimeOrigin(), "DD.MM.YYYY HH");
    wxString dateStr = wxString::Format(_("Forecast of the %sh"), dateForecast.c_str());
    m_staticTextForecastDate->SetLabel(dateStr);

    wxString forecastName;
    if (m_forecastViewer->GetForecastSelection()<0) {
        forecastName = m_forecastManager->GetMethodName(m_forecastViewer->GetMethodSelection());
    }
    else {
        forecastName = m_forecastManager->GetForecastName(m_forecastViewer->GetMethodSelection(), m_forecastViewer->GetForecastSelection());
    }

    m_staticTextForecast->SetLabel(forecastName);
    
    m_panelTop->Layout();
    m_panelTop->Refresh();
}

void asFrameForecast::UpdatePanelCaptionAll()
{
    if (m_forecastViewer->GetLeadTimeIndex()<m_forecastManager->GetLeadTimeLengthMax())
    {
        m_panelSidebarCaptionForecastDots->Show();
        m_panelSidebarCaptionForecastRing->Hide();

        m_panelSidebarCaptionForecastDots->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());
    }
    else
    {
        m_panelSidebarCaptionForecastDots->Hide();
        m_panelSidebarCaptionForecastRing->Show();

        m_panelSidebarCaptionForecastRing->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());
        
        int methodRow = m_forecastViewer->GetMethodSelection();
        int forecastRow = m_forecastViewer->GetForecastSelection();
        if (forecastRow<0) {
            forecastRow = 0;
        }

        asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(methodRow, forecastRow);
        Array1DFloat dates = forecast->GetTargetDates();
        m_panelSidebarCaptionForecastRing->SetDates(dates);
    }
}

void asFrameForecast::UpdatePanelCaptionColorbar()
{
    m_panelSidebarCaptionForecastDots->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());
    
    m_panelSidebarCaptionForecastRing->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());
}

void asFrameForecast::UpdatePanelAnalogDates()
{
    if (m_forecastViewer->GetLeadTimeIndex()>=m_forecastManager->GetLeadTimeLengthMax() || m_forecastViewer->GetForecastSelection()<0)
    {
        m_panelSidebarAnalogDates->Hide();
        return;
    }

    m_panelSidebarAnalogDates->Show();

    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_forecastViewer->GetMethodSelection(), m_forecastViewer->GetForecastSelection());
    Array1DFloat arrayDate = forecast->GetAnalogsDates(m_forecastViewer->GetLeadTimeIndex());
    Array1DFloat arrayCriteria = forecast->GetAnalogsCriteria(m_forecastViewer->GetLeadTimeIndex());
    m_panelSidebarAnalogDates->SetChoices(arrayDate, arrayCriteria);
}

void asFrameForecast::UpdatePanelStationsList()
{
    int methodRow = m_forecastViewer->GetMethodSelection();
    int forecastRow = m_forecastViewer->GetForecastSelection();
    if (forecastRow<0)
    {
        forecastRow = 0;
    }
    
    m_panelSidebarStationsList->Show();

    wxArrayString arrayStation = m_forecastManager->GetStationNamesWithHeights(methodRow, forecastRow);
    m_panelSidebarStationsList->SetChoices(arrayStation);
}

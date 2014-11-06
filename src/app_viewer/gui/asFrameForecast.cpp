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
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED, asFrameForecast::OnForecastModelSelectionChange)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_PERCENTILE_SELECTION_CHANGED, asFrameForecast::OnForecastPercentileSelectionChange)
END_EVENT_TABLE()


/* vroomDropFiles */

vroomDropFiles::vroomDropFiles(asFrameForecast * parent){
    wxASSERT(parent);
    m_LoaderFrame = parent;
}


bool vroomDropFiles::OnDropFiles(wxCoord x, wxCoord y,
                                 const wxArrayString & filenames){
    if (filenames.GetCount() == 0) return false;

    m_LoaderFrame->OpenLayers(filenames);
    return true;
}


/* modelDropFiles */

modelDropFiles::modelDropFiles(asFrameForecast * parent){
    wxASSERT(parent);
    m_LoaderFrame = parent;
}


bool modelDropFiles::OnDropFiles(wxCoord x, wxCoord y,
                                 const wxArrayString & filenames){
    if (filenames.GetCount() == 0) return false;

    m_LoaderFrame->OpenForecast(filenames);
    return true;
}


asFrameForecast::asFrameForecast( wxWindow* parent, wxWindowID id )
:
asFrameForecastVirtual( parent, id )
{
    g_SilentMode = false;

    // Toolbar
    m_ToolBar->AddTool( asID_OPEN, wxT("Open"), img_open, img_open, wxITEM_NORMAL, _("Open forecast"), _("Open a forecast"), NULL );
    m_ToolBar->AddTool( asID_RUN, wxT("Run"), img_run, img_run, wxITEM_NORMAL, _("Run last forecast"), _("Run last forecast"), NULL );
    m_ToolBar->AddTool( asID_RUN_PREVIOUS, wxT("Run previous"), img_run_history, img_run_history, wxITEM_NORMAL, _("Run previous forecasts"), _("Run all previous forecasts"), NULL );
    m_ToolBar->AddSeparator();
    m_ToolBar->AddTool( asID_SELECT, wxT("Select"), img_map_cursor, img_map_cursor, wxITEM_NORMAL, _("Select"), _("Select data on the map"), NULL );
    m_ToolBar->AddTool( asID_ZOOM_IN, wxT("Zoom in"), img_map_zoom_in, img_map_zoom_in, wxITEM_NORMAL, _("Zoom in"), _("Zoom in"), NULL );
    m_ToolBar->AddTool( asID_ZOOM_OUT, wxT("Zoom out"), img_map_zoom_out, img_map_zoom_out, wxITEM_NORMAL, _("Zoom out"), _("Zoom out"), NULL );
    m_ToolBar->AddTool( asID_PAN, wxT("Pan"), img_map_move, img_map_move, wxITEM_NORMAL, _("Pan the map"), _("Move the map by panning"), NULL );
    m_ToolBar->AddTool( asID_ZOOM_FIT, wxT("Fit"), img_map_fit, img_map_fit, wxITEM_NORMAL, _("Zoom to visible layers"), _("Zoom view to the full extent of all visible layers"), NULL );
    m_ToolBar->AddSeparator();
    m_ToolBar->AddTool( asID_FRAME_PLOTS, wxT("Open distributions plots"), img_frame_plots, img_frame_plots, wxITEM_NORMAL, _("Open distributions plots"), _("Open distributions plots"), NULL );
    m_ToolBar->AddTool( asID_FRAME_GRID, wxT("Open analogs list"), img_frame_grid, img_frame_grid, wxITEM_NORMAL, _("Open analogs list"), _("Open analogs list"), NULL );
    m_ToolBar->AddTool( asID_FRAME_FORECASTER, wxT("Open forecaster"), img_frame_forecaster, img_frame_forecaster, wxITEM_NORMAL, _("Open forecaster"), _("Open forecaster"), NULL );
    m_ToolBar->AddSeparator();
    m_ToolBar->AddTool( asID_PREFERENCES, wxT("Preferences"), img_preferences, img_preferences, wxITEM_NORMAL, _("Preferences"), _("Preferences"), NULL );
    m_ToolBar->Realize();

    // VroomGIS controls
    m_DisplayCtrl = new vrViewerDisplay( m_PanelGIS, wxID_ANY, wxColour(120,120,120));
    m_SizerGIS->Add( m_DisplayCtrl, 1, wxEXPAND, 5 );
    m_PanelGIS->Layout();

    // Forecasts
    m_PanelSidebarForecasts = new asPanelSidebarForecasts( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarForecasts->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarForecasts, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Alarms
    m_PanelSidebarAlarms = new asPanelSidebarAlarms( m_ScrolledWindowOptions, &m_Workspace, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarAlarms->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarAlarms, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Stations list
    m_PanelSidebarStationsList = new asPanelSidebarStationsList( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarStationsList->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarStationsList, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Gis panel
    m_PanelSidebarGisLayers = new asPanelSidebarGisLayers( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarGisLayers->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarGisLayers, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );
    
    // Analog dates sidebar
    m_PanelSidebarAnalogDates = new asPanelSidebarAnalogDates( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarAnalogDates->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarAnalogDates, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Caption panel
    m_PanelSidebarCaptionForecastDots = new asPanelSidebarCaptionForecastDots( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarCaptionForecastDots->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarCaptionForecastDots, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Caption panel
    m_PanelSidebarCaptionForecastRing = new asPanelSidebarCaptionForecastRing( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarCaptionForecastRing->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarCaptionForecastRing, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    m_ScrolledWindowOptions->Layout();
    m_SizerScrolledWindow->Fit( m_ScrolledWindowOptions );
    Layout();

    // Lead time switcher
    m_LeadTimeSwitcher = NULL;

    // Status bar
    SetStatusText(_("Welcome to AtmoSwing"));

    // Connect Events
    m_DisplayCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameForecast::OnRightClick ), NULL, this );
    m_DisplayCtrl->Connect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameForecast::OnKeyDown), NULL, this);
    m_DisplayCtrl->Connect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameForecast::OnKeyUp), NULL, this);
    this->Connect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePreferences ) );
    this->Connect( asID_FRAME_FORECASTER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameForecaster ) );
    this->Connect( asID_FRAME_PLOTS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePlots ) );
    this->Connect( asID_FRAME_GRID, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameGrid ) );
    this->Connect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingNow ) );
    this->Connect( asID_RUN_PREVIOUS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingPast ) );
    this->Connect( asID_OPEN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OnOpenForecast ) );

    // DND
    m_PanelSidebarGisLayers->SetDropTarget(new vroomDropFiles(this));
    m_PanelSidebarForecasts->SetDropTarget(new modelDropFiles(this));

    // VroomGIS
    m_LayerManager = new vrLayerManager();
    m_ViewerLayerManager = new vrViewerLayerManager(m_LayerManager, this, m_DisplayCtrl , m_PanelSidebarGisLayers->GetTocCtrl());
//    m_LayerManager->AllowReprojectOnTheFly(true);

    // Forecast manager
    m_ForecastManager = new asForecastManager(m_PanelSidebarForecasts->GetModelsCtrl(), &m_Workspace);

    // Forecast viewer
    m_ForecastViewer = new asForecastViewer( this, m_ForecastManager, m_LayerManager, m_ViewerLayerManager);

    // Process
    m_ProcessForecast = NULL;
    m_LaunchedPresentForecast = false;

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
    pConfig->Write("/SidebarPanelsDisplay/Forecasts", !m_PanelSidebarForecasts->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/StationsList", !m_PanelSidebarStationsList->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/GisLayers", !m_PanelSidebarGisLayers->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/AnalogDates", !m_PanelSidebarAnalogDates->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/CaptionForecastDots", !m_PanelSidebarCaptionForecastDots->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/Alarms", !m_PanelSidebarAlarms->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/CaptionForecastRing", !m_PanelSidebarCaptionForecastRing->IsReduced());

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
    m_DisplayCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameForecast::OnRightClick ), NULL, this );
    m_DisplayCtrl->Disconnect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameForecast::OnKeyDown), NULL, this);
    m_DisplayCtrl->Disconnect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameForecast::OnKeyUp), NULL, this);
    this->Disconnect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePreferences ) );
    this->Disconnect( asID_FRAME_FORECASTER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameForecaster ) );
    this->Disconnect( asID_FRAME_PLOTS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePlots ) );
    this->Disconnect( asID_FRAME_GRID, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameGrid ) );
    this->Disconnect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingNow ) );
    this->Disconnect( asID_RUN_PREVIOUS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::LaunchForecastingPast ) );
    this->Disconnect( asID_OPEN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OnOpenForecast ) );

    // Don't delete m_ViewerLayerManager, will be deleted by the manager
    wxDELETE(m_LayerManager);
    wxDELETE(m_ForecastManager);
    wxDELETE(m_ForecastViewer);

    // Kill the process if still running
    if (m_ProcessForecast!=NULL)
    {
        asLogMessage(_("Killing the forecast running process."));
        wxKillError killError = m_ProcessForecast->Kill(m_ProcessForecast->GetPid());
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
    // Update gui elements
    DisplayLogLevelMenu();
    
    // Open last workspace
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString workspaceFilePath = wxEmptyString;
    pConfig->Read("/Workspace/LastOpened", &workspaceFilePath);


    if(!workspaceFilePath.IsEmpty())
    {
        if (!m_Workspace.Load(workspaceFilePath))
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
        asWizardWorkspace wizard(this, &m_Workspace);
        wizard.RunWizard(wizard.GetFirstPage());

        OpenWorkspace();
    }

    // Set the display options
    m_PanelSidebarForecasts->GetForecastDisplayCtrl()->SetStringArray(m_ForecastViewer->GetForecastDisplayStringArray());
    m_PanelSidebarForecasts->GetPercentilesCtrl()->SetStringArray(m_ForecastViewer->GetPercentilesStringArray());
    m_PanelSidebarForecasts->GetForecastDisplayCtrl()->Select(m_ForecastViewer->GetForecastDisplaySelection());
    m_PanelSidebarForecasts->GetPercentilesCtrl()->Select(m_ForecastViewer->GetPercentileSelection());

    // Reduce some panels
    bool display = true;
    pConfig->Read("/SidebarPanelsDisplay/Forecasts", &display, true);
    if (!display)
    {
        m_PanelSidebarForecasts->ReducePanel();
        m_PanelSidebarForecasts->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/StationsList", &display, true);
    if (!display)
    {
        m_PanelSidebarStationsList->ReducePanel();
        m_PanelSidebarStationsList->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/Alarms", &display, true);
    if (!display)
    {
        m_PanelSidebarAlarms->ReducePanel();
        m_PanelSidebarAlarms->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/GisLayers", &display, false);
    if (!display)
    {
        m_PanelSidebarGisLayers->ReducePanel();
        m_PanelSidebarGisLayers->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/AnalogDates", &display, true);
    if (!display)
    {
        m_PanelSidebarAnalogDates->ReducePanel();
        m_PanelSidebarAnalogDates->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/CaptionForecastDots", &display, true);
    if (!display)
    {
        m_PanelSidebarCaptionForecastDots->ReducePanel();
        m_PanelSidebarCaptionForecastDots->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/CaptionForecastRing", &display, true);
    if (!display)
    {
        m_PanelSidebarCaptionForecastRing->ReducePanel();
        m_PanelSidebarCaptionForecastRing->Layout();
    }

    // Set the select tool
    m_DisplayCtrl->SetToolDefault();

    m_ScrolledWindowOptions->Layout();

    if (!g_CmdFilename.IsEmpty())
    {
        wxArrayString filePathsVect;
        filePathsVect.Add(g_CmdFilename);
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

    wxString workspaceFilePath = openFileDialog.GetPath();

    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Workspace/LastOpened", workspaceFilePath);

    // Do open the workspace
    if (!m_Workspace.Load(workspaceFilePath))
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

    wxString workspaceFilePath = openFileDialog.GetPath();
    m_Workspace.SetFilePath(workspaceFilePath);

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
    m_Workspace.ClearLayers();
    int counter = -1;
    for (int i=0; i<m_ViewerLayerManager->GetCount(); i++)
    {
        wxFileName fileName = m_ViewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName();
        wxString path = fileName.GetFullPath();

        if(!path.IsSameAs("Forecast.memory"))
        {
            counter++;
            m_Workspace.AddLayer();
            m_Workspace.SetLayerPath(counter, path);

            vrDRIVERS_TYPE type = m_ViewerLayerManager->GetRenderer(i)->GetLayer()->GetType();
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
            m_Workspace.SetLayerType(counter, strType);

            int transparency = m_ViewerLayerManager->GetRenderer(i)->GetRender()->GetTransparency();
            m_Workspace.SetLayerTransparency(counter, transparency);
            bool visible = m_ViewerLayerManager->GetRenderer(i)->GetVisible();
            m_Workspace.SetLayerVisibility(counter, visible);

            if (strType.IsSameAs("vector"))
            {
                vrRenderVector * vectRender = (vrRenderVector*) m_ViewerLayerManager->GetRenderer(i)->GetRender();
                int lineWidth = vectRender->GetSize();
                m_Workspace.SetLayerLineWidth(counter, lineWidth);
                wxColour lineColour = vectRender->GetColorPen();
                m_Workspace.SetLayerLineColor(counter, lineColour);
                wxColour fillColour = vectRender->GetColorBrush();
                m_Workspace.SetLayerFillColor(counter, fillColour);
                wxBrushStyle brushStyle = vectRender->GetBrushStyle();
                m_Workspace.SetLayerBrushStyle(counter, brushStyle);
            }
        }
    }

    if(!m_Workspace.Save())
    {
        asLogError(_("Could not save the worspace."));
        return false;
    }

    m_Workspace.SetHasChanged(false);

    return true;
}

void asFrameForecast::OnNewWorkspace(wxCommandEvent & event)
{
    asWizardWorkspace wizard(this, &m_Workspace);
    wizard.RunWizard(wizard.GetSecondPage());
}

bool asFrameForecast::OpenWorkspace()
{
    // GIS layers
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Enter();
    #endif

    m_ViewerLayerManager->FreezeBegin();

    // Remove all layers
    for (int i = (signed) m_ViewerLayerManager->GetCount()-1; i >= 0 ; i--) 
    {
        // Remove from viewer manager (TOC and Display)
        vrRenderer * renderer = m_ViewerLayerManager->GetRenderer(i);
        vrLayer * layer = renderer->GetLayer();
        wxASSERT(renderer);
        m_ViewerLayerManager->Remove(renderer);

        // Close layer (not used anymore);
        m_LayerManager->Close(layer);
    }

    // Open new layers
    for (int i_layer=m_Workspace.GetLayersNb()-1; i_layer>=0; i_layer--)
    {
        // Get attributes
        wxString path = m_Workspace.GetLayerPath(i_layer);
        wxString type = m_Workspace.GetLayerType(i_layer);
        int transparency = m_Workspace.GetLayerTransparency(i_layer);
        bool visibility = m_Workspace.GetLayerVisibility(i_layer);
        
        // Open the layers
        if (wxFileName::FileExists(path))
        {
            if (m_LayerManager->Open(wxFileName(path)))
            {
                if (type.IsSameAs("raster"))
                {
                    vrRenderRaster* render = new vrRenderRaster();
                    render->SetTransparency(transparency);

                    vrLayer* layer = m_LayerManager->GetLayer( wxFileName(path));
                    wxASSERT(layer);
                    m_ViewerLayerManager->Add(-1, layer, render, NULL, visibility);
                }
                else if (type.IsSameAs("vector"))
                {
                    int width = m_Workspace.GetLayerLineWidth(i_layer);
                    wxColour lineColor = m_Workspace.GetLayerLineColor(i_layer);
                    wxColour fillColor = m_Workspace.GetLayerFillColor(i_layer);
                    wxBrushStyle brushStyle = m_Workspace.GetLayerBrushStyle(i_layer);

                    vrRenderVector* render = new vrRenderVector();
                    render->SetTransparency(transparency);
                    render->SetSize(width);
                    render->SetColorPen(lineColor);
                    render->SetBrushStyle(brushStyle);
                    render->SetColorBrush(fillColor);

                    vrLayer* layer = m_LayerManager->GetLayer( wxFileName(path));
                    wxASSERT(layer);
                    m_ViewerLayerManager->Add(-1, layer, render, NULL, visibility);
                }
                else if (type.IsSameAs("wms"))
                {
                    vrRenderRaster* render = new vrRenderRaster();
                    render->SetTransparency(transparency);

                    vrLayer* layer = m_LayerManager->GetLayer( wxFileName(path));
                    wxASSERT(layer);
                    m_ViewerLayerManager->Add(-1, layer, render, NULL, visibility);
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

    m_ViewerLayerManager->FreezeEnd();

    OpenRecentForecasts();

    m_ScrolledWindowOptions->Layout();
    m_SizerScrolledWindow->Fit( m_ScrolledWindowOptions );
    Layout();
    
    m_Workspace.SetHasChanged(false);

    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif

    return true;
}

void asFrameForecast::OnClose(wxCloseEvent& event)
{
    if ( event.CanVeto() && m_Workspace.HasChanged() )
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
    int width = (m_ForecastManager->GetFullTargetDatesVector().size()+1)*squareSize;
    int height = squareSize + 5;

    // Delete and recreate the panel. Cannot get it work with a resize...
    wxDELETE(m_LeadTimeSwitcher);
    m_LeadTimeSwitcher = new asLeadTimeSwitcher( m_PanelTop, &m_Workspace, wxID_ANY, wxDefaultPosition, wxSize(width,height), wxTAB_TRAVERSAL );
    m_LeadTimeSwitcher->SetBackgroundColour( wxColour( 77, 77, 77 ) );
    m_LeadTimeSwitcher->SetMinSize( wxSize( width, height ) );
    m_LeadTimeSwitcher->Layout();
    Array1DFloat dates = m_ForecastManager->GetFullTargetDatesVector();
    m_LeadTimeSwitcher->Draw(dates, m_ForecastManager->GetModelsNames(), m_ForecastManager->GetCurrentForecasts());
    
    m_SizerLeadTimeSwitch->Add( m_LeadTimeSwitcher, 0, wxALL | wxALIGN_RIGHT, 5 );
    m_SizerLeadTimeSwitch->Layout();
    
    m_PanelTop->Layout();
	m_SizerTop->Fit( m_PanelTop );
    m_SizerContent->Layout();

}

void asFrameForecast::LaunchForecastingNow( wxCommandEvent& event )
{
    m_ToolBar->EnableTool(asID_RUN, false);
    m_ToolBar->EnableTool(asID_RUN_PREVIOUS, false);

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
    if (m_ProcessForecast!=NULL)
    {
        asLogError(_("There is already a running forecast process. Please wait."));
        return;
    }
    m_ProcessForecast = new wxProcess(this);

    // Redirect output fluxes
    //m_ProcessForecast->Redirect(); // CAUTION redirect causes the downloads to hang after a few ones!

    // Execute
    long processId = wxExecute(forecasterPath, wxEXEC_ASYNC, m_ProcessForecast);

    if (processId==0) // if wxEXEC_ASYNC
    {
        asLogError(_("The forecaster could not be executed. Please check the path in the preferences."));
        wxDELETE(m_ProcessForecast);
        return;
    }

    m_LaunchedPresentForecast = true;
}

void asFrameForecast::LaunchForecastingPast( wxCommandEvent& event )
{
    m_ToolBar->EnableTool(asID_RUN, false);
    m_ToolBar->EnableTool(asID_RUN_PREVIOUS, false);

    // Get forecaster path
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString forecasterPath = pConfig->Read("/Paths/ForecasterPath", asConfig::GetSoftDir()+"atmoswing-forecaster");

    if(forecasterPath.IsEmpty())
    {
        asLogError(_("Please set the path to the forecaster in the preferences."));
        return;
    }

    // Set option
    int nbPrevDays = m_Workspace.GetTimeSeriesPlotPastDaysNb();
    wxString options = wxString::Format(" -fp %d -ll 2 -lt file", nbPrevDays);
    forecasterPath.Append(options);
    asLogMessage(wxString::Format(_("Sending command: %s"), forecasterPath.c_str()));

    // Create a process
    if (m_ProcessForecast!=NULL)
    {
        asLogError(_("There is already a running forecast process. Please wait."));
        return;
    }
    m_ProcessForecast = new wxProcess(this);

    // Redirect output fluxes
    //m_ProcessForecast->Redirect(); // CAUTION redirect causes the downloads to hang after a few ones!

    // Execute
    long processId = wxExecute(forecasterPath, wxEXEC_ASYNC, m_ProcessForecast);

    if (processId==0) // if wxEXEC_ASYNC
    {
        asLogError(_("The forecaster could not be executed. Please check the path in the preferences."));
        wxDELETE(m_ProcessForecast);
        return;
    }
}

void asFrameForecast::OnForecastProcessTerminate( wxProcessEvent &event )
{
    m_ToolBar->EnableTool(asID_RUN, true);
    m_ToolBar->EnableTool(asID_RUN_PREVIOUS, true);

    asLogMessage("The forecast processing is over.");

    if (m_LaunchedPresentForecast)
    {
        if (m_ForecastManager->GetCurrentForecastsNb()>0)
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

    wxDELETE(m_ProcessForecast);
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
    asFramePlotDistributions* framePlot = new asFramePlotDistributions(this, m_ForecastViewer->GetModelSelection(), m_ForecastManager);
    framePlot->Layout();
    framePlot->Init();
    framePlot->Plot();
    framePlot->Show();
}

void asFrameForecast::OpenFrameGrid( wxCommandEvent& event )
{
    asFrameGridAnalogsValues* frameGrid = new asFrameGridAnalogsValues(this, m_ForecastViewer->GetModelSelection(), m_ForecastManager);
    frameGrid->Layout();
    frameGrid->Init();
    frameGrid->Show();
}

void asFrameForecast::OpenFramePreferences( wxCommandEvent& event )
{
    asFramePreferencesViewer* frame = new asFramePreferencesViewer(this, &m_Workspace, asWINDOW_PREFERENCES);
    frame->Fit();
    frame->Show();
}

void asFrameForecast::OpenFrameAbout( wxCommandEvent& event )
{
    asFrameAbout* frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameForecast::OnLogLevel1( wxCommandEvent& event )
{
    Log().SetLevel(1);
    m_MenuLogLevel->FindItemByPosition(0)->Check(true);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::OnLogLevel2( wxCommandEvent& event )
{
    Log().SetLevel(2);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(true);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::OnLogLevel3( wxCommandEvent& event )
{
    Log().SetLevel(3);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(true);
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::DisplayLogLevelMenu()
{
    // Set log level in the menu
    int logLevel = (int)wxFileConfig::Get()->Read("/General/LogLevel", 1l);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel)
    {
    case 1:
        m_MenuLogLevel->FindItemByPosition(0)->Check(true);
        Log().SetLevel(1);
        break;
    case 2:
        m_MenuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
        break;
    case 3:
        m_MenuLogLevel->FindItemByPosition(2)->Check(true);
        Log().SetLevel(3);
        break;
    default:
        m_MenuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
    }
}

bool asFrameForecast::OpenLayers (const wxArrayString & names)
{
    // Open files
    for (unsigned int i = 0; i< names.GetCount(); i++)
    {
        if(!m_LayerManager->Open(wxFileName(names.Item(i))))
        {
            asLogError(_("The layer could not be opened."));
            return false;
        }
    }

    // Get files
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Enter();
    #endif
    m_ViewerLayerManager->FreezeBegin();
    for (unsigned int i = 0; i< names.GetCount(); i++)
    {
        vrLayer * layer = m_LayerManager->GetLayer( wxFileName(names.Item(i)));
        wxASSERT(layer);

        // Add files to the viewer
        m_ViewerLayerManager->Add(1, layer, NULL);
    }
    m_ViewerLayerManager->FreezeEnd();
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif
        
    m_Workspace.SetHasChanged(true);

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
        m_CritSectionViewerLayerManager.Enter();
    #endif

    // Creates the list of layers
    wxArrayString layersName;
    for (int i = 0; i<m_ViewerLayerManager->GetCount(); i++)
    {
        vrRenderer * renderer = m_ViewerLayerManager->GetRenderer(i);
        wxASSERT(renderer);
        layersName.Add(renderer->GetLayer()->GetDisplayName().GetFullName());
    }

    if (layersName.IsEmpty())
    {
        asLogError(_("No layer opened, nothing to close."));
        #if defined (__WIN32__)
            m_CritSectionViewerLayerManager.Leave();
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
            m_CritSectionViewerLayerManager.Leave();
        #endif
        return;
    }

    // Get indices of layer to remove
    wxArrayInt layerToRemoveIndex = choiceDlg.GetSelections();
    if (layerToRemoveIndex.IsEmpty())
    {
        wxLogWarning(_("Nothing selected, no layer will be closed"));
        #if defined (__WIN32__)
            m_CritSectionViewerLayerManager.Leave();
        #endif
        return;
    }

    // Remove layer(s)
    m_ViewerLayerManager->FreezeBegin();
    for (int i = (signed) layerToRemoveIndex.GetCount()-1; i >= 0 ; i--) {

        // Remove from viewer manager (TOC and Display)
        vrRenderer * renderer = m_ViewerLayerManager->GetRenderer(layerToRemoveIndex.Item(i));
        vrLayer * layer = renderer->GetLayer();
        wxASSERT(renderer);
        m_ViewerLayerManager->Remove(renderer);

        // Close layer (not used anymore);
        m_LayerManager->Close(layer);
    }

    m_ViewerLayerManager->FreezeEnd();
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif

    m_Workspace.SetHasChanged(true);
}

void asFrameForecast::OnOpenForecast(wxCommandEvent & event)
{
    wxFileDialog myFileDlg (this, _("Select a forecast file"),
                            wxEmptyString,
                            wxEmptyString,
                            "*.fcst",
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_CHANGE_DIR | wxFD_MULTIPLE);

    wxArrayString pathsFileName;

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
    m_ForecastManager->ClearForecasts();

    wxString forecastsDirectory = m_Workspace.GetForecastsDirectory();

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
    if (m_ForecastManager->GetCurrentForecastsNb()==0)
    {
        asLogError("There is no forecast open.");
        return;
    }

    // Get path
    VectorString forecastsPaths = m_ForecastManager->GetFilePaths();
    int index = m_ForecastViewer->GetModelSelection();
    wxASSERT(forecastsPaths.size()>index);
    wxString forecastsPath = forecastsPaths[index];
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
    double date = m_ForecastManager->GetLeadTimeOrigin();

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
    m_ForecastManager->ClearForecasts();
    if (!OpenForecast(accurateFiles))
    {
        asLogError(_("Failed to open the forecasts."));
        return;
    }
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
        m_CritSectionViewerLayerManager.Enter();
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
                m_ForecastManager->AddDirectoryPastForecasts(dir);
            }
        }

        bool doRefresh = false;
        if (i==names.GetCount()-1)
        {
            doRefresh = true;
        }

        bool successOpen = m_ForecastManager->Open(names.Item(i), doRefresh);
        if(!successOpen)
        {
            asLogError(wxString::Format(_("The forecast file %d could not be opened (%s)."), i, names.Item(i).c_str()));
            #if defined (__WIN32__)
                m_CritSectionViewerLayerManager.Leave();
            #endif
            m_ViewerLayerManager->Reload();
            Thaw();
            return false;
        }
    }
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif

    UpdateLeadTimeSwitch();

    m_LeadTimeSwitcher->SetLeadTime(m_ForecastViewer->GetLeadTimeIndex());

    Thaw();

    return true;
}

void asFrameForecast::OnKeyDown(wxKeyEvent & event)
{
    m_KeyBoardState = wxKeyboardState(event.ControlDown(),
                                      event.ShiftDown(),
                                      event.AltDown(),
                                      event.MetaDown());
    if (m_KeyBoardState.GetModifiers() != wxMOD_CMD)
    {
        event.Skip();
        return;
    }

    const vrDisplayTool * tool = m_DisplayCtrl->GetTool();
    if (tool == NULL)
    {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_IN)
    {
        m_DisplayCtrl->SetToolZoomOut();
    }
    event.Skip();
}

void asFrameForecast::OnKeyUp(wxKeyEvent & event)
{
    if (m_KeyBoardState.GetModifiers() != wxMOD_CMD)
    {
        event.Skip();
        return;
    }

    const vrDisplayTool * tool = m_DisplayCtrl->GetTool();
    if (tool == NULL)
    {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_OUT || tool->GetID() == wxID_ZOOM_IN)
    {
        m_DisplayCtrl->SetToolZoom();
    }
    event.Skip();
}

void asFrameForecast::OnToolSelect (wxCommandEvent & event)
{
    m_DisplayCtrl->SetToolDefault();
}

void asFrameForecast::OnToolZoomIn (wxCommandEvent & event)
{
    m_DisplayCtrl->SetToolZoom();
}

void asFrameForecast::OnToolZoomOut (wxCommandEvent & event)
{
    m_DisplayCtrl->SetToolZoomOut();
}

void asFrameForecast::OnToolPan (wxCommandEvent & event)
{
    m_DisplayCtrl->SetToolPan();
}

void asFrameForecast::OnToolZoomToFit (wxCommandEvent & event)
{
    // Fit to all layers
    // m_ViewerLayerManager->ZoomToFit(true);
    // ReloadViewerLayerManager();

    // Fit to the forecasts layer
    FitExtentToForecasts();
}

void asFrameForecast::FitExtentToForecasts ()
{
    vrLayerVector * layer = (vrLayerVector*)m_LayerManager->GetLayer(_("Forecast.memory"));

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
        m_ViewerLayerManager->InitializeExtent(extent);
    }
    else
    {
        asLogError(_("The forecasts layer was not found."));
    }

    ReloadViewerLayerManager();
}

void asFrameForecast::OnMoveLayer (wxCommandEvent & event)
{
    // Check than more than 1 layer
    if (m_ViewerLayerManager->GetCount() <= 1)
    {
        asLogError(_("Moving layer not possible with less than 2 layers"));
        return;
    }

    // Get selection
    int iOldPos = m_PanelSidebarGisLayers->GetTocCtrl()->GetSelection();
    if (iOldPos == wxNOT_FOUND)
    {
        asLogError(_("No layer selected, select a layer first"));
        return;
    }

    // Contextual menu
    wxMenu posMenu;
    posMenu.SetTitle(_("Move layer to following position"));
    for (int i = 0; i<m_ViewerLayerManager->GetCount(); i++)
    {
        posMenu.Append(asID_MENU_POPUP_LAYER + i,
                         wxString::Format("%d - %s",i+1,
                                          m_ViewerLayerManager->GetRenderer(i)->GetLayer()->GetDisplayName().GetFullName()));
    }
    wxPoint pos = wxGetMousePosition();

    int iNewID = GetPopupMenuSelectionFromUser(posMenu, ScreenToClient(pos));
    if (iNewID == wxID_NONE) return;

    int iNewPos = iNewID - asID_MENU_POPUP_LAYER;
    if (iNewPos == iOldPos) return;

    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Enter();
    #endif
    m_ViewerLayerManager->Move(iOldPos, iNewPos);
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif

    m_Workspace.SetHasChanged(true);
}

void asFrameForecast::OnToolAction (wxCommandEvent & event)
{
    // Get event
    vrDisplayToolMessage * msg = (vrDisplayToolMessage*)event.GetClientData();
    wxASSERT(msg);

    if(msg->m_EvtType == vrEVT_TOOL_ZOOM)
    {
        // Get rectangle
        vrCoordinate * coord = m_ViewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        bool success = coord->ConvertFromPixels(msg->m_Rect, realRect);
        wxASSERT(success == true);

        // Get fitted rectangle
        vrRealRect fittedRect =coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        // Moving view
        #if defined (__WIN32__)
            asThreadViewerLayerManagerZoomIn *thread = new asThreadViewerLayerManagerZoomIn(m_ViewerLayerManager, &m_CritSectionViewerLayerManager, fittedRect);
            ThreadsManager().AddThread(thread);
        #else
            m_ViewerLayerManager->Zoom(fittedRect);
        #endif
    }
    else if (msg->m_EvtType == vrEVT_TOOL_ZOOMOUT)
    {
        // Get rectangle
        vrCoordinate * coord = m_ViewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        bool success = coord->ConvertFromPixels(msg->m_Rect, realRect);
        wxASSERT(success == true);

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        // Moving view
        #if defined (__WIN32__)
            asThreadViewerLayerManagerZoomOut *thread = new asThreadViewerLayerManagerZoomOut(m_ViewerLayerManager, &m_CritSectionViewerLayerManager, fittedRect);
            ThreadsManager().AddThread(thread);
        #else
            m_ViewerLayerManager->ZoomOut(fittedRect);
        #endif
    }
    else if (msg->m_EvtType == vrEVT_TOOL_SELECT)
    {
        // If no forecast open
        if (m_ForecastManager->GetModelsNb()==0)
        {
            wxDELETE(msg);
            return;
        }

        // Transform screen coordinates to real coordinates
        vrCoordinate * coord = m_ViewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);
        wxPoint clickedPos = msg->m_Position;
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
            vrLayerVector * layer = (vrLayerVector*)m_LayerManager->GetLayer(_("Forecast.memory"));

            if(layer!=NULL)
            {
                wxASSERT(layer);

                // Search features
                wxArrayLong stationsClose;
                layer->SearchFeatures(&polygon, stationsClose);

                // Allow only one selection
                wxArrayLong station;
                if (stationsClose.Count()>0)
                {
                    station.Add(stationsClose.Item(0));
                    int stationRow = stationsClose.Item(0);

                    if (stationRow>=0)
                    {
                        m_PanelSidebarStationsList->GetChoiceCtrl()->Select(stationRow);
                    }
                    DrawPlotStation(stationRow);
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
        vrCoordinate * coord = m_ViewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        wxPoint movedPos = msg->m_Position;
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
        asLogError(_("Operation not supported now"));
    }

    wxDELETE(msg);
}

void asFrameForecast::OnStationSelection( wxCommandEvent& event )
{
    // Get selection
    int choice = event.GetInt();

    // If no forecast open
    if (m_ForecastManager->GetModelsNb()==0)
    {
        return;
    }

    // Get layer to display selection
    vrLayerVector * layer = (vrLayerVector*)m_LayerManager->GetLayer(_("Forecast.memory"));

    if(layer!=NULL)
    {
        wxASSERT(layer);

        // Set selection
        wxArrayLong station;
        station.Add(choice);
        layer->SetSelectedIDs(station);
    }
    else
    {
        asLogError(_("The desired layer was not found."));
    }

    DrawPlotStation(choice);

    ReloadViewerLayerManager();

}

void asFrameForecast::OnChangeLeadTime( wxCommandEvent& event )
{
    Freeze();

    m_ForecastViewer->ChangeLeadTime(event.GetInt());
    
    m_LeadTimeSwitcher->SetLeadTime(m_ForecastViewer->GetLeadTimeIndex());

    UpdatePanelAnalogDates();
    UpdatePanelCaptionAll();

    m_ScrolledWindowOptions->Layout();
    m_SizerScrolledWindow->Fit( m_ScrolledWindowOptions );
    Layout();

    Thaw();
}

void asFrameForecast::OnForecastClear( wxCommandEvent& event )
{
    if(m_PanelSidebarForecasts!=NULL)
    {
        m_PanelSidebarForecasts->ClearForecasts();
    }
}

void asFrameForecast::OnForecastRatioSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetForecastDisplay(event.GetInt());

    UpdatePanelCaptionColorbar();
}

void asFrameForecast::OnForecastModelSelectionChange( wxCommandEvent& event )
{
    Freeze();

    m_ForecastViewer->SetModel(event.GetInt());
    m_LeadTimeSwitcher->SetLeadTime(m_ForecastViewer->GetLeadTimeIndex());

    UpdateHeaderTexts();
    UpdatePanelCaptionAll();
    UpdatePanelAnalogDates();
    UpdatePanelStationsList();

    Thaw();
}

void asFrameForecast::OnForecastPercentileSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetPercentile(event.GetInt());
}

void asFrameForecast::DrawPlotStation( int station )
{
    m_ForecastViewer->LoadPastForecast();

    asFramePlotTimeSeries* framePlotStation = new asFramePlotTimeSeries(this, m_ForecastViewer->GetModelSelection(), station, m_ForecastManager);
    framePlotStation->Layout();
    framePlotStation->Init();
    framePlotStation->Plot();
    framePlotStation->Show();
}

void asFrameForecast::OnForecastNewAdded( wxCommandEvent& event )
{
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(event.GetInt());
    m_PanelSidebarForecasts->AddForecast(forecast->GetModelName(), forecast->GetLeadTimeOriginString(), forecast->GetPredictandParameter(), forecast->GetPredictandTemporalResolution());

    if (event.GetString().IsSameAs("last"))
    {
        int modelIndex = m_ForecastViewer->GetModelSelection();
        if (modelIndex>event.GetInt() || modelIndex<0) 
        {
            modelIndex = event.GetInt();
        }
        m_ForecastViewer->SetModel(modelIndex);
        m_PanelSidebarForecasts->GetModelsCtrl()->SetSelection(modelIndex);

        UpdatePanelAlarms();
        UpdateHeaderTexts();
        UpdatePanelCaptionAll();
        UpdatePanelAnalogDates();
        UpdatePanelStationsList();
    }
}

void asFrameForecast::ReloadViewerLayerManager( )
{
    m_ViewerLayerManager->Reload();

    /* Not sure there is any way to make it safe with threads.
    #if defined (__WIN32__)
        asThreadViewerLayerManagerReload *thread = new asThreadViewerLayerManagerReload(m_ViewerLayerManager, &m_CritSectionViewerLayerManager);
        ThreadsManager().AddThread(thread);
    #else
        m_ViewerLayerManager->Reload();
    #endif*/
}

void asFrameForecast::UpdateHeaderTexts()
{
    // Set header text
    wxString dateForecast = asTime::GetStringTime(m_ForecastManager->GetLeadTimeOrigin(), "DD.MM.YYYY HH");
    wxString dateStr = wxString::Format(_("Forecast of the %sh"), dateForecast.c_str());
    m_StaticTextForecastDate->SetLabel(dateStr);

    wxString model = m_ForecastManager->GetModelName(m_ForecastViewer->GetModelSelection());
    wxString modelStr = wxString::Format(_("Model selected : %s"), model.c_str());
    m_StaticTextForecastModel->SetLabel(modelStr);

    m_PanelTop->Layout();
    m_PanelTop->Refresh();
}

void asFrameForecast::UpdatePanelCaptionAll()
{
    if (m_ForecastViewer->GetLeadTimeIndex()<m_ForecastManager->GetLeadTimeLengthMax())
    {
        m_PanelSidebarCaptionForecastDots->Show();
        m_PanelSidebarCaptionForecastRing->Hide();

        m_PanelSidebarCaptionForecastDots->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());
    }
    else
    {
        m_PanelSidebarCaptionForecastDots->Hide();
        m_PanelSidebarCaptionForecastRing->Show();

        m_PanelSidebarCaptionForecastRing->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());
        asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_ForecastViewer->GetModelSelection());
        Array1DFloat dates = forecast->GetTargetDates();
        m_PanelSidebarCaptionForecastRing->SetDates(dates);
    }
}

void asFrameForecast::UpdatePanelCaptionColorbar()
{
    m_PanelSidebarCaptionForecastDots->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());
    
    m_PanelSidebarCaptionForecastRing->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());
}

void asFrameForecast::UpdatePanelAnalogDates()
{
    if (m_ForecastViewer->GetLeadTimeIndex()>=m_ForecastManager->GetLeadTimeLengthMax())
    {
        m_PanelSidebarAnalogDates->Hide();
        return;
    }

    m_PanelSidebarAnalogDates->Show();
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_ForecastViewer->GetModelSelection());
    Array1DFloat arrayDate = forecast->GetAnalogsDates(m_ForecastViewer->GetLeadTimeIndex());
    Array1DFloat arrayCriteria = forecast->GetAnalogsCriteria(m_ForecastViewer->GetLeadTimeIndex());
    m_PanelSidebarAnalogDates->SetChoices(arrayDate, arrayCriteria);
}

void asFrameForecast::UpdatePanelStationsList()
{
    wxArrayString arrayStation = m_ForecastManager->GetStationNamesWithHeights(m_ForecastViewer->GetModelSelection());
    m_PanelSidebarStationsList->SetChoices(arrayStation);
}

void asFrameForecast::UpdatePanelAlarms()
{
    Array1DFloat datesFull = m_ForecastManager->GetFullTargetDatesVector();
    VectorString models = m_ForecastManager->GetModelsNames();
    std::vector <asResultsAnalogsForecast*> forecasts = m_ForecastManager->GetCurrentForecasts();
    m_PanelSidebarAlarms->UpdateAlarms(datesFull, models, forecasts);
}

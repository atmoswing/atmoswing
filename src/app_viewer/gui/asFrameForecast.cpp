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
#include "asFrameForecastDots.h"
#include "asFramePreferences.h"
#include "asFramePlotTimeSeries.h"
#include "asFramePlotDistributions.h"
#include "asFrameGridAnalogsValues.h"
#include "asFrameMeteorologicalSituation.h"
#include "asPanelPlot.h"
#include "asFileAscii.h"
#include "asFileWorkspace.h"
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
    m_ForecastsDirectory = wxEmptyString;

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
    m_ToolBar->AddTool( asID_FRAME_SITUATION, wxT("Display meteorological situation"), img_frame_forecaster, img_frame_forecaster, wxITEM_NORMAL, _("Display meteorological situation"), _("Display meteorological situation"), NULL );
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

    // Stations list
    m_PanelSidebarStationsList = new asPanelSidebarStationsList( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarStationsList->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarStationsList, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Gis panel
    m_PanelSidebarGisLayers = new asPanelSidebarGisLayers( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarGisLayers->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarGisLayers, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    m_ScrolledWindowOptions->Layout();
    m_SizerScrolledWindow->Fit( m_ScrolledWindowOptions );

    // Status bar
    SetStatusText(_("Welcome to AtmoSwing"));

    // Connect Events
    m_DisplayCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameForecast::OnRightClick ), NULL, this );
    m_DisplayCtrl->Connect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameForecast::OnKeyDown), NULL, this);
    m_DisplayCtrl->Connect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameForecast::OnKeyUp), NULL, this);
    this->Connect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePreferences ) );
    this->Connect( asID_FRAME_FORECASTER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameForecaster ) );
    this->Connect( asID_FRAME_SITUATION, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameMeteorologicalSituation ) );
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

    // Forecast manager
    m_ForecastManager = new asForecastManager(m_PanelSidebarForecasts->GetModelsCtrl());

    // Forecast viewer
    m_ForecastViewer = new asForecastViewer( this, m_ForecastManager, m_LayerManager, m_ViewerLayerManager);

    // Process
    m_ProcessForecast = NULL;
    m_LaunchedPresentForecast = false;

    // Restore frame position and size
    wxConfigBase *pConfig = wxFileConfig::Get();
    int minHeight = 450, minWidth = 800;
    int x = pConfig->Read("/MainFrameViewer/x", 50),
        y = pConfig->Read("/MainFrameViewer/y", 50),
        w = pConfig->Read("/MainFrameViewer/w", minWidth),
        h = pConfig->Read("/MainFrameViewer/h", minHeight);
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
    pConfig->Read("/MainFrameViewer/Maximize", &doMaximize);
    Maximize(doMaximize);

    // Get last workspace file
    m_WorkspaceFilePath = wxEmptyString;
    pConfig->Read("/Workspace/LastOpened", &m_WorkspaceFilePath);

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

    // Save the frame position
    bool doMaximize = IsMaximized();
    pConfig->Write("/MainFrameViewer/Maximize", doMaximize);
    if (!doMaximize)
    {
        int x, y, w, h;
        GetClientSize(&w, &h);
        GetPosition(&x, &y);
        pConfig->Write("/MainFrameViewer/x", (long) x);
        pConfig->Write("/MainFrameViewer/y", (long) y);
        pConfig->Write("/MainFrameViewer/w", (long) w);
        pConfig->Write("/MainFrameViewer/h", (long) h);
    }

    // Disconnect Events
    m_DisplayCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameForecast::OnRightClick ), NULL, this );
    m_DisplayCtrl->Disconnect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameForecast::OnKeyDown), NULL, this);
    m_DisplayCtrl->Disconnect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameForecast::OnKeyUp), NULL, this);
    this->Disconnect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFramePreferences ) );
    this->Disconnect( asID_FRAME_FORECASTER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameForecaster ) );
    this->Disconnect( asID_FRAME_SITUATION, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecast::OpenFrameMeteorologicalSituation ) );
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

void asFrameForecast::OnInit()
{
    // Update gui elements
    DisplayLogLevelMenu();
    
    // Open last workspace
    if(!m_WorkspaceFilePath.IsEmpty() && !OpenWorkspace())
    {
        asLogWarning(_("Failed to open the workspace file ") + m_WorkspaceFilePath);
    }

    // Set the display options
    m_PanelSidebarForecasts->GetForecastDisplayCtrl()->SetStringArray(m_ForecastViewer->GetForecastDisplayStringArray());
    m_PanelSidebarForecasts->GetPercentilesCtrl()->SetStringArray(m_ForecastViewer->GetPercentilesStringArray());
    m_PanelSidebarForecasts->GetForecastDisplayCtrl()->Select(m_ForecastViewer->GetForecastDisplaySelection());
    m_PanelSidebarForecasts->GetPercentilesCtrl()->Select(m_ForecastViewer->GetPercentileSelection());

    // Reduce some panels
    wxConfigBase *pConfig = wxFileConfig::Get();
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
    pConfig->Read("/SidebarPanelsDisplay/GisLayers", &display, false);
    if (!display)
    {
        m_PanelSidebarGisLayers->ReducePanel();
        m_PanelSidebarGisLayers->Layout();
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

    m_WorkspaceFilePath = openFileDialog.GetPath();

    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Workspace/LastOpened", m_WorkspaceFilePath);

    // Do open the workspace
    if (!OpenWorkspace())
    {
        asLogError(_("Failed to open the workspace file ") + m_WorkspaceFilePath);
    }

}

bool asFrameForecast::OpenWorkspace()
{
    // Open the file
    asFileWorkspace fileWorkspace(m_WorkspaceFilePath, asFile::ReadOnly);
    if(!fileWorkspace.Open()) return false;

    if(!fileWorkspace.GoToRootElement()) return false;

    // Get general data
    wxString coordinateSys = fileWorkspace.GetFirstElementAttributeValueText("CoordinateSys", "value");
    m_ForecastsDirectory = fileWorkspace.GetFirstElementAttributeValueText("ForecastsDirectory", "value");

    // GIS layers
    if(!fileWorkspace.GoToFirstNodeWithPath("Layer")) return false;

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
    while(true)
    {
        // Get attributes
        wxString path = fileWorkspace.GetFirstElementAttributeValueText("Path", "value");
        wxString type = fileWorkspace.GetFirstElementAttributeValueText("Type", "value");
        int transparency = fileWorkspace.GetFirstElementAttributeValueInt("Transparency", "value", 0);
        bool visibility = fileWorkspace.GetFirstElementAttributeValueBool("Visibility", "value", true);
        int width = fileWorkspace.GetFirstElementAttributeValueInt("Width", "value", 1);
        long lineColorLong = long(fileWorkspace.GetFirstElementAttributeValueInt("LineColor", "value", 0));
        wxColour lineColor;
        lineColor.SetRGB((wxUint32)lineColorLong);
        long fillColorLong = long(fileWorkspace.GetFirstElementAttributeValueInt("FillColor", "value", 0));
        wxColour fillColor;
        fillColor.SetRGB((wxUint32)fillColorLong);
        
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
                    vrRenderVector* render = new vrRenderVector();
                    render->SetTransparency(transparency);
                    render->SetSize(width);
                    render->SetColorPen(lineColor);
                    if (fillColorLong==0)
                    {
                        render->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
                    }
                    else
                    {
                        render->SetColorBrush(fillColor);
                    }

                    vrLayer* layer = m_LayerManager->GetLayer( wxFileName(path));
                    wxASSERT(layer);
                    m_ViewerLayerManager->Add(-1, layer, render, NULL, visibility);
                }
                else if (type.IsSameAs("wms"))
                {
                    asLogWarning(_("WMS layers are not yet implemented."));
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
        
        // Find the next layer
        if (!fileWorkspace.GoToNextSameNode()) break;
    }

    m_ViewerLayerManager->FreezeEnd();

    OpenRecentForecasts();

    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif

    return true;
}

/*
void asFrameForecast::Update()
{
    DisplayLogLevelMenu();
}
*/


void asFrameForecast::OnQuit( wxCommandEvent& event )
{
    event.Skip();
}

void asFrameForecast::LaunchForecastingNow( wxCommandEvent& event )
{
    m_ToolBar->EnableTool(asID_RUN, false);
    m_ToolBar->EnableTool(asID_RUN_PREVIOUS, false);

    // Get forecaster path
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString ForecasterPath = pConfig->Read("/StandardPaths/ForecasterPath", wxEmptyString);

    if(ForecasterPath.IsEmpty())
    {
        asLogError(_("Please set the path to the forecaster in the preferences."));
        return;
    }

    // Set option
    wxString options = wxString::Format(" -fn -ll 2 -lt file");
    ForecasterPath.Append(options);
    asLogMessage(wxString::Format(_("Sending command: %s"), ForecasterPath.c_str()));

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
    long processId = wxExecute(ForecasterPath, wxEXEC_ASYNC, m_ProcessForecast);

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
    wxString ForecasterPath = pConfig->Read("/StandardPaths/ForecasterPath", wxEmptyString);

    if(ForecasterPath.IsEmpty())
    {
        asLogError(_("Please set the path to the forecaster in the preferences."));
        return;
    }

    // Set option
    int nbPrevDays;
    pConfig->Read("/Plot/PastDaysNb", &nbPrevDays, 3);
    wxString options = wxString::Format(" -fp %d -ll 2 -lt file", nbPrevDays);
    ForecasterPath.Append(options);
    asLogMessage(wxString::Format(_("Sending command: %s"), ForecasterPath.c_str()));

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
    long processId = wxExecute(ForecasterPath, wxEXEC_ASYNC, m_ProcessForecast);

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
    wxString ForecasterPath = pConfig->Read("/StandardPaths/ForecasterPath", wxEmptyString);

    if(ForecasterPath.IsEmpty())
    {
        asLogError(_("Please set the path to the forecaster in the preferences."));
        return;
    }

    // Execute
    long processId = wxExecute(ForecasterPath, wxEXEC_ASYNC);

    if (processId==0) // if wxEXEC_ASYNC
    {
        asLogError(_("The forecaster could not be executed. Please check the path in the preferences."));
    }
}

void asFrameForecast::OpenFrameMeteorologicalSituation( wxCommandEvent& event )
{
    asFrameMeteorologicalSituation* frameSituation = new asFrameMeteorologicalSituation(this);
    frameSituation->Fit();
    frameSituation->Show();
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
    asFramePreferences* frame = new asFramePreferences(this, asWINDOW_PREFERENCES);
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
    wxFileConfig::Get()->Write("/Standard/LogLevelViewer", 1l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::OnLogLevel2( wxCommandEvent& event )
{
    Log().SetLevel(2);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(true);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/Standard/LogLevelViewer", 2l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::OnLogLevel3( wxCommandEvent& event )
{
    Log().SetLevel(3);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(true);
    wxFileConfig::Get()->Write("/Standard/LogLevelViewer", 3l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecast::DisplayLogLevelMenu()
{
    // Set log level in the menu
    int logLevel = (int)wxFileConfig::Get()->Read("/Standard/LogLevelViewer", 2l);
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

    if (m_ForecastsDirectory.IsEmpty())
    {
        asLogError("The directory containing the forecasts was not provided.");
        return false;
    }

    if (!wxFileName::DirExists(m_ForecastsDirectory))
    {
        asLogError("The directory that is supposed to contain the forecasts does not exist.");
        return false;
    }

    // Get present date
    double now = asTime::NowMJD();

    // Check if today directory exists
    wxString basePath = m_ForecastsDirectory + wxFileName::GetPathSeparator();
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

bool asFrameForecast::OpenForecast (const wxArrayString & names)
{
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
}

void asFrameForecast::OnForecastModelSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetModel(event.GetInt());
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

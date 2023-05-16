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

#include "AtmoswingAppViewer.h"
#include "asFramePredictandDB.h"
#include "asFrameViewer.h"

#if defined(__WIN32__)
#include "asThreadViewerLayerManagerReload.h"
#include "asThreadViewerLayerManagerZoomIn.h"
#include "asThreadViewerLayerManagerZoomOut.h"
#include "asThreadsManager.h"
#endif

#include <wx/dir.h>

#include "asBitmaps.h"
#include "asFileText.h"
#include "asFrameAbout.h"
#include "asFrameGridAnalogsValues.h"
#include "asFramePredictors.h"
#include "asFramePlotDistributions.h"
#include "asFramePlotTimeSeries.h"
#include "asFramePreferencesViewer.h"
#include "asWizardWorkspace.h"
#include "vrlayervector.h"

BEGIN_EVENT_TABLE(asFrameViewer, wxFrame)
EVT_CLOSE(asFrameViewer::OnClose)
EVT_KEY_DOWN(asFrameViewer::OnKeyDown)
EVT_KEY_UP(asFrameViewer::OnKeyUp)
EVT_MENU(wxID_EXIT, asFrameViewer::OnQuit)
EVT_MENU(asID_SELECT, asFrameViewer::OnToolSelect)
EVT_MENU(asID_ZOOM_IN, asFrameViewer::OnToolZoomIn)
EVT_MENU(asID_ZOOM_OUT, asFrameViewer::OnToolZoomOut)
EVT_MENU(asID_ZOOM_FIT, asFrameViewer::OnToolZoomToFit)
EVT_MENU(asID_PAN, asFrameViewer::OnToolPan)
EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOM, asFrameViewer::OnToolAction)
EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOMOUT, asFrameViewer::OnToolAction)
EVT_COMMAND(wxID_ANY, vrEVT_TOOL_SELECT, asFrameViewer::OnToolAction)
EVT_COMMAND(wxID_ANY, vrEVT_TOOL_PAN, asFrameViewer::OnToolAction)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_OPEN_WORKSPACE, asFrameViewer::OnOpenWorkspace)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_STATION_SELECTION_CHANGED, asFrameViewer::OnStationSelection)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED, asFrameViewer::OnChangeLeadTime)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_CLEAR, asFrameViewer::OnForecastClear)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_NEW_ADDED, asFrameViewer::OnForecastNewAdded)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED, asFrameViewer::OnForecastRatioSelectionChange)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_SELECTION_CHANGED, asFrameViewer::OnForecastForecastSelectionChange)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_SELECT_FIRST, asFrameViewer::OnForecastForecastSelectFirst)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_QUANTILE_SELECTION_CHANGED,
            asFrameViewer::OnForecastQuantileSelectionChange)
END_EVENT_TABLE()

/* vroomDropFiles */

vroomDropFiles::vroomDropFiles(asFrameViewer* parent)
    : m_loaderFrame(parent) {
    wxASSERT(parent);
}

bool vroomDropFiles::OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) {
    if (filenames.GetCount() == 0) return false;

    m_loaderFrame->OpenLayers(filenames);
    return true;
}

/* forecastDropFiles */

forecastDropFiles::forecastDropFiles(asFrameViewer* parent)
    : m_loaderFrame(parent) {
    wxASSERT(parent);
}

bool forecastDropFiles::OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) {
    if (filenames.GetCount() == 0) return false;

    m_loaderFrame->OpenForecast(filenames);
    return true;
}

asFrameViewer::asFrameViewer(wxWindow* parent, wxWindowID id)
    : asFrameViewerVirtual(parent, id) {
    g_silentMode = false;
    m_fileHistory = new wxFileHistory(9);

    // Adjust size
    int sashMinSize = m_splitterGIS->GetMinimumPaneSize();
    sashMinSize *= g_ppiScaleDc;
    m_splitterGIS->SetMinimumPaneSize(sashMinSize);

    // Menu recent
    auto menuOpenRecent = new wxMenu();
    m_menuFile->Insert(1, asID_MENU_RECENT, _("Open recent"), menuOpenRecent);

    // Toolbar
    m_toolBar->AddTool(asID_OPEN, wxT("Open"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::OPEN), wxNullBitmap,
                       wxITEM_NORMAL, _("Open forecast"), _("Open a forecast"), nullptr);
    m_toolBar->AddTool(asID_SELECT, wxT("Select"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_SELECT),
                       wxNullBitmap, wxITEM_NORMAL, _("Select"), _("Select data on the map"), nullptr);
    m_toolBar->AddTool(asID_ZOOM_IN, wxT("Zoom in"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_ZOOM_IN),
                       wxNullBitmap, wxITEM_NORMAL, _("Zoom in"), _("Zoom in"), nullptr);
    m_toolBar->AddTool(asID_ZOOM_OUT, wxT("Zoom out"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_ZOOM_OUT),
                       wxNullBitmap, wxITEM_NORMAL, _("Zoom out"), _("Zoom out"), nullptr);
    m_toolBar->AddTool(asID_PAN, wxT("Pan"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_MOVE),
                       wxNullBitmap, wxITEM_NORMAL, _("Pan the map"), _("Move the map by panning"), nullptr);
    m_toolBar->AddTool(asID_ZOOM_FIT, wxT("Fit"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_FIT),
                       wxNullBitmap, wxITEM_NORMAL, _("Zoom to visible layers"),
                       _("Zoom view to the full extent of all visible layers"), nullptr);
    m_toolBar->AddTool(asID_FRAME_PLOTS, wxT("Open distributions plots"),
                       asBitmaps::Get(asBitmaps::ID_TOOLBAR::FRAME_DISTRIBUTIONS),
                       wxNullBitmap, wxITEM_NORMAL, _("Open distributions plots"),
                       _("Open distributions plots"), nullptr);
    m_toolBar->AddTool(asID_FRAME_GRID, wxT("Open analogs list"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::FRAME_ANALOGS),
                       wxNullBitmap, wxITEM_NORMAL, _("Open analogs list"), _("Open analogs list"), nullptr);
    m_toolBar->AddTool(asID_FRAME_PREDICTORS, wxT("Open predictor maps"),
                       asBitmaps::Get(asBitmaps::ID_TOOLBAR::FRAME_PREDICTORS), wxNullBitmap,
                       wxITEM_NORMAL, _("Open predictor maps"), _("Open predictor maps"), nullptr);
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::PREFERENCES),
                       wxNullBitmap, wxITEM_NORMAL, _("Preferences"), _("Preferences"), nullptr);
    m_toolBar->Realize();

    // VroomGIS controls
    m_displayCtrl = new vrViewerDisplay(m_panelGIS, wxID_ANY, wxColour(120, 120, 120));
    m_sizerGIS->Add(m_displayCtrl, 1, wxEXPAND, 5);
    m_panelGIS->Layout();

    // Gis panel
    m_panelSidebarGisLayers = new asPanelSidebarGisLayers(m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition,
                                                          wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelSidebarGisLayers->Layout();
    m_sizerScrolledWindow->Add(m_panelSidebarGisLayers, 0, wxEXPAND, 0);
    m_panelSidebarGisLayers->SetDropTarget(new vroomDropFiles(this));

    // VroomGIS
    m_layerManager = new vrLayerManager();
    m_viewerLayerManager = new vrViewerLayerManager(m_layerManager, this, m_displayCtrl,
                                                    m_panelSidebarGisLayers->GetTocCtrl());
    //    m_layerManager->AllowReprojectOnTheFly(true);

    // Forecast manager
    m_forecastManager = new asForecastManager(this, &m_workspace);
    m_forecastManager->Init();

    // Forecast viewer
    m_forecastViewer = new asForecastRenderer(this, m_forecastManager, m_layerManager, m_viewerLayerManager);

    // Forecasts
    m_panelSidebarForecasts = new asPanelSidebarForecasts(m_scrolledWindowOptions, m_forecastManager, wxID_ANY,
                                                          wxDefaultPosition, wxDefaultSize,
                                                          wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelSidebarForecasts->Layout();
    m_sizerScrolledWindow->Insert(0, m_panelSidebarForecasts, 0, wxEXPAND, 0);
    m_panelSidebarForecasts->SetDropTarget(new forecastDropFiles(this));

    // Alarms
    m_panelSidebarAlarms = new asPanelSidebarAlarms(m_scrolledWindowOptions, &m_workspace, m_forecastManager, wxID_ANY,
                                                    wxDefaultPosition, wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelSidebarAlarms->Layout();
    m_sizerScrolledWindow->Add(m_panelSidebarAlarms, 0, wxEXPAND, 0);

    // Stations list
    m_panelSidebarStationsList = new asPanelSidebarStationsList(m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition,
                                                                wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelSidebarStationsList->Layout();
    m_sizerScrolledWindow->Add(m_panelSidebarStationsList, 0, wxEXPAND, 0);

    // Analog dates sidebar
    m_panelSidebarAnalogDates = new asPanelSidebarAnalogDates(m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition,
                                                              wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelSidebarAnalogDates->Layout();
    m_sizerScrolledWindow->Add(m_panelSidebarAnalogDates, 0, wxEXPAND, 0);

    // Caption panel
    m_panelSidebarCaptionForecastDots = new asPanelSidebarCaptionForecastDots(
        m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelSidebarCaptionForecastDots->Layout();
    m_sizerScrolledWindow->Add(m_panelSidebarCaptionForecastDots, 0, wxEXPAND, 0);

    // Caption panel
    m_panelSidebarCaptionForecastRing = new asPanelSidebarCaptionForecastRing(
        m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelSidebarCaptionForecastRing->Layout();
    m_sizerScrolledWindow->Add(m_panelSidebarCaptionForecastRing, 0, wxEXPAND, 0);

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit(m_scrolledWindowOptions);
    Layout();

    // Lead time switcher
    m_leadTimeSwitcher = nullptr;

    // Status bar
    SetStatusText(_("Welcome to AtmoSwing"));

    // Connect Events
    m_displayCtrl->Connect(wxEVT_RIGHT_DOWN, wxMouseEventHandler(asFrameViewer::OnRightClick), nullptr, this);
    m_displayCtrl->Connect(wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameViewer::OnKeyDown), nullptr, this);
    m_displayCtrl->Connect(wxEVT_KEY_UP, wxKeyEventHandler(asFrameViewer::OnKeyUp), nullptr, this);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePreferences, this, asID_PREFERENCES);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePlots, this, asID_FRAME_PLOTS);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFrameGrid, this, asID_FRAME_GRID);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePredictors, this, asID_FRAME_PREDICTORS);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OnOpenForecast, this, asID_OPEN);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePredictandDB, this, asID_DB_CREATE);
    Bind(wxEVT_COMMAND_MENU_SELECTED, &asFrameViewer::OnFileHistory, this, wxID_FILE1, wxID_FILE9);

    // Process
    m_processForecast = nullptr;
    m_launchedPresentForecast = false;

    // Restore frame position and size
    wxConfigBase* pConfig = wxFileConfig::Get();
    int minHeight = 450, minWidth = 800;
    int x = pConfig->ReadLong("/MainFrame/x", 50);
    int y = pConfig->ReadLong("/MainFrame/y", 50);
    int w = pConfig->ReadLong("/MainFrame/w", minWidth);
    int h = pConfig->ReadLong("/MainFrame/h", minHeight);
    wxRect screen = wxGetClientDisplayRect();
    if (x < screen.x - 10) x = screen.x;
    if (x > screen.width) x = screen.x;
    if (y < screen.y - 10) y = screen.y;
    if (y > screen.height) y = screen.y;
    if (w + x > screen.width) w = screen.width - x;
    if (w < minWidth) w = minWidth;
    if (w + x > screen.width) x = screen.width - w;
    if (h + y > screen.height) h = screen.height - y;
    if (h < minHeight) h = minHeight;
    if (h + y > screen.height) y = screen.height - h;

    Move(x, y);
    SetClientSize(w, h);

    Maximize(pConfig->ReadBool("/MainFrame/Maximize", false));

    Layout();

    SetRecentFiles();

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameViewer::~asFrameViewer() {

    SaveRecentFiles();

    // Save preferences
    wxConfigBase* pConfig = wxFileConfig::Get();
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
    if (!doMaximize) {
        int x, y, w, h;
        GetClientSize(&w, &h);
        GetPosition(&x, &y);
        pConfig->Write("/MainFrame/x", (long)x);
        pConfig->Write("/MainFrame/y", (long)y);
        pConfig->Write("/MainFrame/w", (long)w);
        pConfig->Write("/MainFrame/h", (long)h);
    }

    // Disconnect Events
    m_displayCtrl->Disconnect(wxEVT_RIGHT_DOWN, wxMouseEventHandler(asFrameViewer::OnRightClick), nullptr, this);
    m_displayCtrl->Disconnect(wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameViewer::OnKeyDown), nullptr, this);
    m_displayCtrl->Disconnect(wxEVT_KEY_UP, wxKeyEventHandler(asFrameViewer::OnKeyUp), nullptr, this);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePreferences, this, asID_PREFERENCES);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePlots, this, asID_FRAME_PLOTS);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFrameGrid, this, asID_FRAME_GRID);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePredictors, this, asID_FRAME_PREDICTORS);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OnOpenForecast, this, asID_OPEN);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameViewer::OpenFramePredictandDB, this, asID_DB_CREATE);
    Unbind(wxEVT_COMMAND_MENU_SELECTED, &asFrameViewer::OnFileHistory, this, wxID_FILE1, wxID_FILE9);

    // Don't delete m_viewerLayerManager, will be deleted by the manager
    wxDELETE(m_layerManager);
    wxDELETE(m_forecastManager);
    wxDELETE(m_forecastViewer);

    // Kill the process if still running
    if (m_processForecast != nullptr) {
        wxLogVerbose(_("Killing the forecast running process."));
        wxKillError killError = wxProcess::Kill(m_processForecast->GetPid());
        switch (killError) {
            case (wxKILL_OK):  // no error
                wxLogVerbose(_("The forecast process has been killed successfully."));
                break;
            case (wxKILL_BAD_SIGNAL):  // no such signal
                wxLogError(_("The forecast process couldn't be killed (bad signal)."));
                break;
            case (wxKILL_ACCESS_DENIED):  // permission denied
                wxLogError(_("The forecast process couldn't be killed (access denied)."));
                break;
            case (wxKILL_NO_PROCESS):  //  no such process
                wxLogError(_("The forecast process couldn't be killed (no process)."));
                break;
            case (wxKILL_ERROR):  //  another, unspecified error
                wxLogError(_("The forecast process couldn't be killed (error)."));
                break;
        }
    }
}

void asFrameViewer::Init() {
    wxBusyCursor wait;

    // Update gui elements
    DisplayLogLevelMenu();

    // Open last workspace
    wxConfigBase* pConfig = wxFileConfig::Get();
    wxString workspaceFilePath = pConfig->Read("/Workspace/LastOpened", wxEmptyString);

    // Check provided files
    bool forecastFilesProvided = false;
    if (!g_cmdFileName.IsEmpty()) {
        int strSize = g_cmdFileName.size();
        int strExt = g_cmdFileName.size() - 4;
        wxString ext = g_cmdFileName.SubString((size_t)(strExt - 1), (size_t)(strSize - 1));
        if (ext.IsSameAs(".asff", false) || ext.IsSameAs(".nc", false)) {
            forecastFilesProvided = true;
        } else if (ext.IsSameAs(".asvw", false)) {
            workspaceFilePath = g_cmdFileName;
        }
    }

    if (!workspaceFilePath.IsEmpty()) {
        if (!m_workspace.Load(workspaceFilePath)) {
            wxLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
        }

        if (!OpenWorkspace(!forecastFilesProvided)) {
            wxLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
        }
    } else {
        asWizardWorkspace wizard(this);
        wizard.RunWizard(wizard.GetFirstPage());

        pConfig->Read("/Workspace/LastOpened", &workspaceFilePath);

        if (!workspaceFilePath.IsEmpty()) {
            if (!m_workspace.Load(workspaceFilePath)) {
                wxLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
            }

            if (!OpenWorkspace()) {
                wxLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
            }
        }
    }

    // Set the display options
    m_panelSidebarForecasts->GetForecastDisplayCtrl()->SetStringArray(
        m_forecastViewer->GetForecastDisplayStringArray());
    m_panelSidebarForecasts->GetQuantilesCtrl()->SetStringArray(m_forecastViewer->GetQuantilesStringArray());
    m_panelSidebarForecasts->GetForecastDisplayCtrl()->Select(m_forecastViewer->GetForecastDisplaySelection());
    m_panelSidebarForecasts->GetQuantilesCtrl()->Select(m_forecastViewer->GetQuantileSelection());

    // Reduce some panels
    if (!pConfig->ReadBool("/SidebarPanelsDisplay/Forecasts", true)) {
        m_panelSidebarForecasts->ReducePanel();
        m_panelSidebarForecasts->Layout();
    }
    if (!pConfig->ReadBool("/SidebarPanelsDisplay/StationsList", true)) {
        m_panelSidebarStationsList->ReducePanel();
        m_panelSidebarStationsList->Layout();
    }
    if (!pConfig->ReadBool("/SidebarPanelsDisplay/Alarms", true)) {
        m_panelSidebarAlarms->ReducePanel();
        m_panelSidebarAlarms->Layout();
    }
    if (!pConfig->ReadBool("/SidebarPanelsDisplay/GisLayers", false)) {
        m_panelSidebarGisLayers->ReducePanel();
        m_panelSidebarGisLayers->Layout();
    }
    if (!pConfig->ReadBool("/SidebarPanelsDisplay/AnalogDates", true)) {
        m_panelSidebarAnalogDates->ReducePanel();
        m_panelSidebarAnalogDates->Layout();
    }
    if (!pConfig->ReadBool("/SidebarPanelsDisplay/CaptionForecastDots", true)) {
        m_panelSidebarCaptionForecastDots->ReducePanel();
        m_panelSidebarCaptionForecastDots->Layout();
    }
    if (!pConfig->ReadBool("/SidebarPanelsDisplay/CaptionForecastRing", true)) {
        m_panelSidebarCaptionForecastRing->ReducePanel();
        m_panelSidebarCaptionForecastRing->Layout();
    }

    // Set the select tool
    m_displayCtrl->SetToolDefault();

    m_scrolledWindowOptions->Layout();

    if (forecastFilesProvided) {
        wxArrayString filePathsVect;
        filePathsVect.Add(g_cmdFileName);
        OpenForecast(filePathsVect);
        FitExtentToForecasts();
    }

    Layout();
    Refresh();
}

void asFrameViewer::OnOpenWorkspace(wxCommandEvent& event) {
    // Ask for a workspace file
    wxFileDialog openFileDialog(this, _("Select a workspace"), wxEmptyString, wxEmptyString,
                                "AtmoSwing viewer workspace (*.asvw)|*.asvw",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_CHANGE_DIR);

    // If canceled
    if (openFileDialog.ShowModal() == wxID_CANCEL) return;

    wxBusyCursor wait;

    wxString workspaceFilePath = openFileDialog.GetPath();

    // Save preferences
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Write("/Workspace/LastOpened", workspaceFilePath);

    // Do open the workspace
    if (!m_workspace.Load(workspaceFilePath)) {
        wxLogError(_("Failed to open the workspace file ") + workspaceFilePath);
    }

    if (!OpenWorkspace()) {
        wxLogError(_("Failed to open the workspace file ") + workspaceFilePath);
    }

    m_fileHistory->AddFileToHistory(workspaceFilePath);
}

void asFrameViewer::OnFileHistory(wxCommandEvent& event) {
    int id = event.GetId() - wxID_FILE1;
    wxString workspaceFilePath = m_fileHistory->GetHistoryFile(id);

    wxBusyCursor wait;

    // Save preferences
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Write("/Workspace/LastOpened", workspaceFilePath);

    // Do open the workspace
    if (!m_workspace.Load(workspaceFilePath)) {
        wxLogError(_("Failed to open the workspace file ") + workspaceFilePath);
    }

    if (!OpenWorkspace()) {
        wxLogError(_("Failed to open the workspace file ") + workspaceFilePath);
    }
}

void asFrameViewer::OnSaveWorkspace(wxCommandEvent& event) {
    SaveWorkspace();
}

void asFrameViewer::OnSaveWorkspaceAs(wxCommandEvent& event) {
    // Ask for a workspace file
    wxFileDialog openFileDialog(this, _("Select a path to save the workspace"), wxEmptyString, wxEmptyString,
                                "AtmoSwing viewer workspace (*.asvw)|*.asvw", wxFD_SAVE | wxFD_CHANGE_DIR);

    // If canceled
    if (openFileDialog.ShowModal() == wxID_CANCEL) return;

    wxBusyCursor wait;

    wxString workspaceFilePath = openFileDialog.GetPath();
    m_workspace.SetFilePath(workspaceFilePath);

    if (SaveWorkspace()) {
        // Save preferences
        wxConfigBase* pConfig = wxFileConfig::Get();
        pConfig->Write("/Workspace/LastOpened", workspaceFilePath);
    }
}

bool asFrameViewer::SaveWorkspace() {
    // Update the GIS layers
    m_workspace.ClearLayers();
    int counter = -1;
    for (int i = 0; i < m_viewerLayerManager->GetCount(); i++) {
        wxFileName fileName = m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName();
        wxString path = fileName.GetFullPath();

        if (!path.IsSameAs(_("Forecast - specific") + ".memory") && !path.IsSameAs(_("Forecast - other") + ".memory")) {
            counter++;
            m_workspace.AddLayer();
            m_workspace.SetLayerPath(counter, path);

            vrDRIVERS_TYPE type = m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetType();
            wxString strType;
            switch (type) {
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

            if (strType.IsSameAs("vector")) {
                vrRenderVector* vectRender = (vrRenderVector*)m_viewerLayerManager->GetRenderer(i)->GetRender();
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

    if (!m_workspace.Save()) {
        wxLogError(_("Could not save the workspace."));
        return false;
    }

    m_workspace.SetHasChanged(false);

    return true;
}

void asFrameViewer::OnNewWorkspace(wxCommandEvent& event) {
    asWizardWorkspace wizard(this);
    wizard.RunWizard(wizard.GetSecondPage());

    // Open last workspace
    wxConfigBase* pConfig = wxFileConfig::Get();
    wxString workspaceFilePath = pConfig->Read("/Workspace/LastOpened", wxEmptyString);

    if (!workspaceFilePath.IsEmpty()) {
        if (!m_workspace.Load(workspaceFilePath)) {
            wxLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
        }

        if (!OpenWorkspace()) {
            wxLogWarning(_("Failed to open the workspace file ") + workspaceFilePath);
        }
    }
}

bool asFrameViewer::OpenWorkspace(bool openRecentForecasts) {
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Enter();
#endif

    wxBusyCursor wait;

    m_viewerLayerManager->FreezeBegin();

    m_forecastViewer->ResetForecastSelection();

    // Remove all layers
    for (int i = m_viewerLayerManager->GetCount() - 1; i >= 0; i--) {
        // Remove from viewer manager (TOC and Display)
        vrRenderer* renderer = m_viewerLayerManager->GetRenderer(i);
        vrLayer* layer = renderer->GetLayer();
        wxASSERT(renderer);
        m_viewerLayerManager->Remove(renderer);

        // Close layer (not used anymore);
        m_layerManager->Close(layer);
    }

    // Open new layers
    for (int iLayer = m_workspace.GetLayersNb() - 1; iLayer >= 0; iLayer--) {
        // Get attributes
        wxString path = m_workspace.GetLayerPath(iLayer);
        wxString type = m_workspace.GetLayerType(iLayer);
        int transparency = m_workspace.GetLayerTransparency(iLayer);
        bool visibility = m_workspace.GetLayerVisibility(iLayer);

        // Open the layers
        if (wxFileName::FileExists(path)) {
            if (m_layerManager->Open(wxFileName(path))) {
                if (type.IsSameAs("raster")) {
                    vrRenderRaster* render = new vrRenderRaster();
                    render->SetTransparency(transparency);

                    vrLayer* layer = m_layerManager->GetLayer(wxFileName(path));
                    wxASSERT(layer);
                    m_viewerLayerManager->Add(-1, layer, render, nullptr, visibility);
                } else if (type.IsSameAs("vector")) {
                    int width = m_workspace.GetLayerLineWidth(iLayer);
                    wxColour lineColor = m_workspace.GetLayerLineColor(iLayer);
                    wxColour fillColor = m_workspace.GetLayerFillColor(iLayer);
                    wxBrushStyle brushStyle = m_workspace.GetLayerBrushStyle(iLayer);

                    auto render = new vrRenderVector();
                    render->SetTransparency(transparency);
                    render->SetSize(width);
                    render->SetColorPen(lineColor);
                    render->SetBrushStyle(brushStyle);
                    render->SetColorBrush(fillColor);

                    vrLayer* layer = m_layerManager->GetLayer(wxFileName(path));
                    wxASSERT(layer);
                    m_viewerLayerManager->Add(-1, layer, render, nullptr, visibility);
                } else if (type.IsSameAs("wms")) {
                    auto render = new vrRenderRaster();
                    render->SetTransparency(transparency);

                    vrLayer* layer = m_layerManager->GetLayer(wxFileName(path));
                    wxASSERT(layer);
                    m_viewerLayerManager->Add(-1, layer, render, nullptr, visibility);
                } else {
                    wxLogError(_("The GIS layer type %s does not correspond to allowed values."), type);
                }
            } else {
                wxLogWarning(_("The file %s cound not be opened."), path);
            }
        } else {
            wxLogWarning(_("The file %s cound not be found."), path);
        }
    }

    m_viewerLayerManager->FreezeEnd();

    if (openRecentForecasts) {
        OpenRecentForecasts();
    }

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit(m_scrolledWindowOptions);
    Layout();

    m_workspace.SetHasChanged(false);

#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Leave();
#endif

    return true;
}

void asFrameViewer::OnClose(wxCloseEvent& event) {
    if (event.CanVeto() && m_workspace.HasChanged()) {
        if (wxMessageBox("The workspace has not been saved... continue closing?", "Please confirm",
                         wxICON_QUESTION | wxYES_NO) != wxYES) {
            event.Veto();
            return;
        }
    }

    event.Skip();
}

void asFrameViewer::OnQuit(wxCommandEvent& event) {
    event.Skip();
}

void asFrameViewer::UpdateLeadTimeSwitch() {
    // Delete and recreate the panel. Cannot get it work with a resize...
    wxDELETE(m_leadTimeSwitcher);
    m_leadTimeSwitcher = new asLeadTimeSwitcher(m_panelTop, &m_workspace, m_forecastManager, wxID_ANY,
                                                wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    m_leadTimeSwitcher->SetForecastSelection(m_forecastViewer->GetMethodSelection(),
                                             m_forecastViewer->GetForecastSelection());
    m_leadTimeSwitcher->SetBackgroundColour(wxColour(77, 77, 77));
    m_leadTimeSwitcher->Layout();
    m_leadTimeSwitcher->SetMinSize(m_leadTimeSwitcher->GetSize());
    a1f dates = m_forecastManager->GetFullTargetDates();
    m_leadTimeSwitcher->Draw(dates);

    m_sizerLeadTimeSwitch->Add(m_leadTimeSwitcher, 0, wxALL, 5);
    m_sizerLeadTimeSwitch->Layout();

    m_panelTop->Layout();
    m_sizerTop->Fit(m_panelTop);
    m_sizerContent->Layout();
}

void asFrameViewer::OpenFramePlots(wxCommandEvent& event) {
    if (m_forecastManager->HasForecasts()) {
        wxBusyCursor wait;

        auto framePlot = new asFramePlotDistributions(this, m_forecastViewer->GetMethodSelection(),
                                                       m_forecastViewer->GetForecastSelection(), m_forecastManager);

        if (g_ppiScaleDc > 1) {
            wxSize frameSize = framePlot->GetSize();
            frameSize.x *= g_ppiScaleDc;
            frameSize.y *= g_ppiScaleDc;
            framePlot->SetSize(frameSize);
        }

        framePlot->Layout();
        framePlot->Init();
        framePlot->Plot();
        framePlot->Show();
    }
}

void asFrameViewer::OpenFrameGrid(wxCommandEvent& event) {
    if (m_forecastManager->HasForecasts()) {
        wxBusyCursor wait;

        auto frameGrid = new asFrameGridAnalogsValues(this, m_forecastViewer->GetMethodSelection(),
                                                       m_forecastViewer->GetForecastSelection(), m_forecastManager);

        if (g_ppiScaleDc > 1) {
            wxSize frameSize = frameGrid->GetSize();
            frameSize.x *= g_ppiScaleDc;
            frameSize.y *= g_ppiScaleDc;
            frameGrid->SetSize(frameSize);
        }

        frameGrid->Layout();
        frameGrid->Init();
        frameGrid->Show();
    }
}

void asFrameViewer::OpenFramePredictors(wxCommandEvent& event) {
    if (m_forecastManager->HasForecasts()) {
        wxBusyCursor wait;

        auto framePredictors = new asFramePredictors(this, m_forecastManager, &m_workspace,
                                                      m_forecastViewer->GetMethodSelection(),
                                                      m_forecastViewer->GetForecastSelection());

        if (g_ppiScaleDc > 1) {
            wxSize frameSize = framePredictors->GetSize();
            frameSize.x *= g_ppiScaleDc;
            frameSize.y *= g_ppiScaleDc;
            framePredictors->SetSize(frameSize);
        }

        framePredictors->Layout();
        framePredictors->Init();
        framePredictors->Show();
    }
}

void asFrameViewer::OpenFramePredictandDB(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameViewer::OpenFramePreferences(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesViewer(this, &m_workspace, asWINDOW_PREFERENCES);
    frame->Fit();
    frame->Show();
}

void asFrameViewer::OpenFrameAbout(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameViewer::OnLogLevel1(wxCommandEvent& event) {
    Log()->SetLevel(1);
    m_menuLogLevel->FindItemByPosition(0)->Check(true);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameViewer::OnLogLevel2(wxCommandEvent& event) {
    Log()->SetLevel(2);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(true);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameViewer::OnLogLevel3(wxCommandEvent& event) {
    Log()->SetLevel(3);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(true);
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameViewer::DisplayLogLevelMenu() {
    // Set log level in the menu
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (wxFileConfig::Get()->ReadLong("/General/LogLevel", 1l)) {
        case 1:
            m_menuLogLevel->FindItemByPosition(0)->Check(true);
            Log()->SetLevel(1);
            break;
        case 2:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
            break;
        case 3:
            m_menuLogLevel->FindItemByPosition(2)->Check(true);
            Log()->SetLevel(3);
            break;
        default:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
    }
}

bool asFrameViewer::OpenLayers(const wxArrayString& names) {
    wxBusyCursor wait;

    // Open files
    for (int i = 0; i < names.GetCount(); i++) {
        if (!m_layerManager->Open(wxFileName(names.Item(i)))) {
            wxLogError(_("The layer could not be opened."));
            return false;
        }
    }

    // Get files
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Enter();
#endif
    m_viewerLayerManager->FreezeBegin();
    for (int i = 0; i < names.GetCount(); i++) {
        vrLayer* layer = m_layerManager->GetLayer(wxFileName(names.Item(i)));
        wxASSERT(layer);

        // Add files to the viewer
        m_viewerLayerManager->Add(1, layer, nullptr);
    }
    m_viewerLayerManager->FreezeEnd();
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Leave();
#endif

    m_workspace.SetHasChanged(true);

    return true;
}

void asFrameViewer::OnOpenLayer(wxCommandEvent& event) {
    vrDrivers drivers;
    wxFileDialog myFileDlg(this, _("Select GIS layers"), wxEmptyString, wxEmptyString, drivers.GetWildcards(),
                           wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_MULTIPLE | wxFD_CHANGE_DIR);

    wxArrayString pathsFileName;

    wxBusyCursor wait;

    // Try to open files
    if (myFileDlg.ShowModal() == wxID_OK) {
        myFileDlg.GetPaths(pathsFileName);
        wxASSERT(pathsFileName.GetCount() > 0);

        OpenLayers(pathsFileName);
    }
}

void asFrameViewer::OnCloseLayer(wxCommandEvent& event) {
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Enter();
#endif

    // Creates the list of layers
    wxArrayString layersName;
    for (int i = 0; i < m_viewerLayerManager->GetCount(); i++) {
        vrRenderer* renderer = m_viewerLayerManager->GetRenderer(i);
        wxASSERT(renderer);
        layersName.Add(renderer->GetLayer()->GetDisplayName().GetFullName());
    }

    if (layersName.IsEmpty()) {
        wxLogError(_("No layer opened, nothing to close."));
#if defined(__WIN32__)
        m_critSectionViewerLayerManager.Leave();
#endif
        return;
    }

    // Choice dialog box
    wxMultiChoiceDialog choiceDlg(this, _("Select Layer(s) to close."), _("Close layer(s)"), layersName);
    if (choiceDlg.ShowModal() != wxID_OK) {
#if defined(__WIN32__)
        m_critSectionViewerLayerManager.Leave();
#endif
        return;
    }

    wxBusyCursor wait;

    // Get indices of layer to remove
    wxArrayInt layerToRemoveIndex = choiceDlg.GetSelections();
    if (layerToRemoveIndex.IsEmpty()) {
        wxLogWarning(_("Nothing selected, no layer will be closed."));
#if defined(__WIN32__)
        m_critSectionViewerLayerManager.Leave();
#endif
        return;
    }

    // Remove layer(s)
    m_viewerLayerManager->FreezeBegin();
    for (int i = (int)layerToRemoveIndex.GetCount() - 1; i >= 0; i--) {
        // Remove from viewer manager (TOC and Display)
        vrRenderer* renderer = m_viewerLayerManager->GetRenderer((const int&)layerToRemoveIndex.Item((size_t)i));
        vrLayer* layer = renderer->GetLayer();
        wxASSERT(renderer);
        m_viewerLayerManager->Remove(renderer);

        // Close layer (not used anymore);
        m_layerManager->Close(layer);
    }

    m_viewerLayerManager->FreezeEnd();
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Leave();
#endif

    m_workspace.SetHasChanged(true);
}

void asFrameViewer::OnOpenForecast(wxCommandEvent& event) {
    wxFileDialog myFileDlg(
        this, _("Select a forecast file"), wxEmptyString, wxEmptyString,
        "Forecast files (*.nc)|*.nc|Former forecast files (*.asff)|*.asff|Former forecast files (*.fcst)|*.fcst",
        wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_CHANGE_DIR | wxFD_MULTIPLE);

    wxArrayString pathsFileName;

    wxBusyCursor wait;

    // Try to open files
    if (myFileDlg.ShowModal() == wxID_OK) {
        myFileDlg.GetPaths(pathsFileName);
        wxASSERT(pathsFileName.GetCount() > 0);

        OpenForecast(pathsFileName);
    }
}

void asFrameViewer::OpenForecastsFromTmpList() {
    // Write the resulting files path into a temp file.
    wxString tempFile = asConfig::GetTempDir() + "AtmoSwingForecastFilePaths.txt";
    asFileText filePaths(tempFile, asFile::ReadOnly);
    wxArrayString filePathsVect;
    if (!filePaths.Open()) {
        wxLogWarning(_("List of the forecasts not found."));
        return;
    }
    while (!filePaths.EndOfFile()) {
        wxString path = filePaths.GetNextLine();

        if (!path.IsEmpty()) {
            filePathsVect.Add(path);
        }
    }
    filePaths.Close();

    OpenForecast(filePathsVect);
}

bool asFrameViewer::OpenRecentForecasts() {
    m_forecastManager->ClearForecasts();

    wxString forecastsDirectory = m_workspace.GetForecastsDirectory();

    if (forecastsDirectory.IsEmpty()) {
        wxLogError(_("The directory containing the forecasts was not provided."));
        return false;
    }

    if (!wxFileName::DirExists(forecastsDirectory)) {
        wxLogError(_("The directory that is supposed to contain the forecasts does not exist."));
        return false;
    }

    // Get present date
    double now = asTime::NowMJD();

    // Check if today directory exists
    wxString basePath = forecastsDirectory + wxFileName::GetPathSeparator();
    wxFileName fullPath(basePath);
    fullPath.AppendDir(asStrF("%d", asTime::GetYear(now)));
    fullPath.AppendDir(asStrF("%02d", asTime::GetMonth(now)));
    fullPath.AppendDir(asStrF("%02d", asTime::GetDay(now)));

    // If does not exist, try the day before
    if (!fullPath.Exists()) {
        now--;
        fullPath = wxFileName(basePath);
        fullPath.AppendDir(asStrF("%d", asTime::GetYear(now)));
        fullPath.AppendDir(asStrF("%02d", asTime::GetMonth(now)));
        fullPath.AppendDir(asStrF("%02d", asTime::GetDay(now)));
    }

    // If does not exist, warn the user and return
    if (!fullPath.Exists()) {
        fullPath = wxFileName(basePath);
        wxLogError(_("No recent forecast was found under %s"), fullPath.GetFullPath());
        return false;
    }

    // List the files in the directory
    wxArrayString files;
    wxDir::GetAllFiles(fullPath.GetFullPath(), &files);

    // Identify the most recent forecasts
    double mostRecentDate = 0;
    vi mostRecentRows;
    for (int i = 0; i < (int)files.GetCount(); i++) {
        wxFileName fileName(files[i]);
        wxString fileDate = fileName.GetFullName().BeforeFirst('.');

        if (fileDate.Len() != 13 && fileDate.Len() != 10) {
            wxLogWarning(_("A file with an unconventional name was found in the forecasts directory."));
            continue;
        }

        double date = 0;
        try {
            date = asTime::GetTimeFromString(fileDate, YYYY_MM_DD_hh);
        } catch (runtime_error& e) {
            wxLogWarning(_("Error when parsing the date: %s"), e.what());
            continue;
        }

        if (date > mostRecentDate) {
            mostRecentDate = date;
            mostRecentRows.clear();
            mostRecentRows.push_back(i);
        } else if (date == mostRecentDate) {
            mostRecentRows.push_back(i);
        }
    }

    // Store the most recent file names
    wxArrayString recentFiles;
    for (int row : mostRecentRows) {
        recentFiles.Add(files[row]);
    }

    // Open the forecasts
    if (!OpenForecast(recentFiles)) {
        wxLogError(_("Failed to open the forecasts."));
        return false;
    }

    FitExtentToForecasts();

    return true;
}

void asFrameViewer::OnLoadPreviousForecast(wxCommandEvent& event) {
    SwitchForecast(-1.0 / 24.0);
}

void asFrameViewer::OnLoadNextForecast(wxCommandEvent& event) {
    SwitchForecast(1.0 / 24.0);
}

void asFrameViewer::OnLoadPreviousDay(wxCommandEvent& event) {
    SwitchForecast(-1.0);
}

void asFrameViewer::OnLoadNextDay(wxCommandEvent& event) {
    SwitchForecast(1.0);
}

void asFrameViewer::SwitchForecast(double increment) {
    wxBusyCursor wait;

    if (m_forecastManager->GetMethodsNb() == 0) {
        wxLogError("There is no opened forecast.");
        return;
    }

    // Get path
    wxString forecastsPath = m_forecastManager->GetFilePath(m_forecastViewer->GetMethodSelection(),
                                                            m_forecastViewer->GetForecastSelection());
    wxFileName forecastFileName(forecastsPath);
    wxString fileName = forecastFileName.GetName();
    wxString partialFileNameV2 = fileName.SubString(10, fileName.size() - 1);
    wxString partialFileNameV3 = fileName.SubString(13, fileName.size() - 1);
    wxString patternFileNameV2 = "%d%02d%02d%02d";
    wxString patternFileNameV3 = "%d-%02d-%02d_%02d";
    wxString prefixFileName = wxEmptyString;

    forecastFileName.RemoveLastDir();
    forecastFileName.RemoveLastDir();
    forecastFileName.RemoveLastDir();
    wxString forecastsBaseDirectory = forecastFileName.GetPath();

    if (!wxFileName::DirExists(forecastsBaseDirectory)) {
        wxLogError("The directory that is supposed to contain the forecasts does not exist.");
        return;
    }

    // Get date
    double date = m_forecastManager->GetLeadTimeOrigin();

    // Look for former files
    wxString basePath = forecastsBaseDirectory + wxFileName::GetPathSeparator();
    wxFileName fullPathV3(basePath);
    wxFileName fullPathV1, fullPathV2, fullPathV4;
    for (int i = 0; i < 100; i++) {
        date += increment;
        fullPathV3 = wxFileName(basePath);
        fullPathV3.AppendDir(asStrF("%d", asTime::GetYear(date)));
        fullPathV3.AppendDir(asStrF("%02d", asTime::GetMonth(date)));
        fullPathV3.AppendDir(asStrF("%02d", asTime::GetDay(date)));

        fullPathV2 = fullPathV3;

        prefixFileName = asStrF(patternFileNameV3, asTime::GetYear(date), asTime::GetMonth(date), asTime::GetDay(date),
                                asTime::GetHour(date));
        fullPathV3.SetName(prefixFileName + partialFileNameV3);

        fullPathV4 = fullPathV3;
        fullPathV4.SetExt("nc");

        if (fullPathV4.Exists()) break;

        fullPathV3.SetExt("asff");

        if (fullPathV3.Exists()) break;

        prefixFileName = asStrF(patternFileNameV2, asTime::GetYear(date), asTime::GetMonth(date), asTime::GetDay(date),
                                asTime::GetHour(date));
        fullPathV2.SetName(prefixFileName + partialFileNameV2);
        fullPathV2.SetExt("asff");

        if (fullPathV2.Exists()) break;

        fullPathV1 = fullPathV2;
        fullPathV1.SetExt("fcst");

        if (fullPathV1.Exists()) break;

        if (i == 99) {
            wxLogError(_("No previous/next forecast was found under %s"), fullPathV2.GetPath());
            return;
        }
    }

    // List the files in the directory
    wxArrayString files;
    wxDir::GetAllFiles(fullPathV3.GetPath(), &files);

    // Identify the corresponding forecasts
    wxArrayString accurateFiles;
    for (int i = 0; i < (int)files.GetCount(); i++) {
        wxFileName fileNameCheck(files[i]);

        if (fileNameCheck.GetFullName().Contains(prefixFileName)) {
            accurateFiles.Add(files[i]);
        }
    }

    // Open the forecasts
    m_forecastManager->ClearForecasts();
    if (!OpenForecast(accurateFiles)) {
        wxLogError(_("Failed to open the forecasts."));
        return;
    }

    // Refresh view
    m_forecastViewer->Redraw();
    UpdateHeaderTexts();
    UpdatePanelCaptionAll();
    UpdatePanelAnalogDates();
}

bool asFrameViewer::OpenForecast(const wxArrayString& names) {
    wxBusyCursor wait;

    if (names.GetCount() == 0) return false;

    Freeze();

    // Close plots
    bool continueClosing = true;
    while (continueClosing) {
        continueClosing = false;

        wxWindow* framePlotsTimeseries = wxWindow::FindWindowById(asWINDOW_PLOTS_TIMESERIES);
        if (framePlotsTimeseries != nullptr) {
            wxASSERT(framePlotsTimeseries);
            framePlotsTimeseries->SetId(0);
            framePlotsTimeseries->Destroy();
            continueClosing = true;
        }

        wxWindow* framePlotsDistributions = wxWindow::FindWindowById(asWINDOW_PLOTS_DISTRIBUTIONS);
        if (framePlotsDistributions != nullptr) {
            wxASSERT(framePlotsDistributions);
            framePlotsDistributions->SetId(0);
            framePlotsDistributions->Destroy();
            continueClosing = true;
        }

        wxWindow* frameGrid = wxWindow::FindWindowById(asWINDOW_GRID_ANALOGS);
        if (frameGrid != nullptr) {
            wxASSERT(frameGrid);
            frameGrid->SetId(0);
            frameGrid->Destroy();
            continueClosing = true;
        }

        wxTheApp->Yield();
        wxWakeUpIdle();
    }

    // Open files
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Enter();
#endif
    for (int i = 0; i < names.GetCount(); i++) {
        if (i == 0) {
            wxString dir = names.Item(i);
            wxUniChar dirSep = DS.GetChar(0);
            dir = dir.BeforeLast(dirSep);
            if (dir.Length() > 10) {
                dir = dir.Left(dir.Length() - 10);
                m_forecastManager->AddDirectoryPastForecasts(dir);
            }
        }

        bool doRefresh = false;
        if (i == names.GetCount() - 1) {
            doRefresh = true;
        }

        bool successOpen = m_forecastManager->Open(names.Item(i), doRefresh);
        if (!successOpen) {
            wxLogError(_("A forecast file could not be opened (%s)."), names.Item(i));
#if defined(__WIN32__)
            m_critSectionViewerLayerManager.Leave();
#endif
            m_viewerLayerManager->Reload();
            Thaw();
            return false;
        }
    }
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Leave();
#endif

    UpdateLeadTimeSwitch();

    m_leadTimeSwitcher->SetLeadTime(m_forecastViewer->GetLeadTimeIndex());
    m_leadTimeSwitcher->SetForecastSelection(m_forecastViewer->GetMethodSelection(),
                                             m_forecastViewer->GetForecastSelection());

    Thaw();

    return true;
}

void asFrameViewer::OnKeyDown(wxKeyEvent& event) {
    m_keyBoardState = wxKeyboardState(event.ControlDown(), event.ShiftDown(), event.AltDown(), event.MetaDown());
    if (m_keyBoardState.GetModifiers() != wxMOD_CMD) {
        event.Skip();
        return;
    }

    const vrDisplayTool* tool = m_displayCtrl->GetTool();
    if (!tool) {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_IN) {
        m_displayCtrl->SetToolZoomOut();
    }
    event.Skip();
}

void asFrameViewer::OnKeyUp(wxKeyEvent& event) {
    if (m_keyBoardState.GetModifiers() != wxMOD_CMD) {
        event.Skip();
        return;
    }

    const vrDisplayTool* tool = m_displayCtrl->GetTool();
    if (!tool) {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_OUT || tool->GetID() == wxID_ZOOM_IN) {
        m_displayCtrl->SetToolZoom();
    }
    event.Skip();
}

void asFrameViewer::OnToolSelect(wxCommandEvent& event) {
    m_displayCtrl->SetToolDefault();
}

void asFrameViewer::OnToolZoomIn(wxCommandEvent& event) {
    m_displayCtrl->SetToolZoom();
}

void asFrameViewer::OnToolZoomOut(wxCommandEvent& event) {
    m_displayCtrl->SetToolZoomOut();
}

void asFrameViewer::OnToolPan(wxCommandEvent& event) {
    m_displayCtrl->SetToolPan();
}

void asFrameViewer::OnToolZoomToFit(wxCommandEvent& event) {
    // Fit to the forecasts layer
    FitExtentToForecasts();
}

void asFrameViewer::FitExtentToForecasts() {
    wxBusyCursor wait;

    vrLayerVector* layer = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - specific") + ".memory");

    if (layer != nullptr) {
        wxASSERT(layer);

        // Get the forecast layer extent
        vrRealRect extent;
        layer->GetExtent(extent);

        // Add a margin
        wxDouble width = extent.GetRight() - extent.GetLeft();
        wxDouble height = extent.GetTop() - extent.GetBottom();
        wxDouble marginFactor = 0.05;
        extent.SetLeft(extent.GetLeft() - marginFactor * width);
        extent.SetRight(extent.GetRight() + marginFactor * width);
        extent.SetBottom(extent.GetBottom() - marginFactor * height);
        extent.SetTop(extent.GetTop() + marginFactor * height);

        // Force new extent
        m_viewerLayerManager->InitializeExtent(extent);
    } else {
        wxLogError(_("The forecasts layer was not found."));
    }

    ReloadViewerLayerManager();
}

void asFrameViewer::OnMoveLayer(wxCommandEvent& event) {
    wxBusyCursor wait;

    // Check that more than 1 layer
    if (m_viewerLayerManager->GetCount() <= 1) {
        wxLogError(_("Moving layer not possible with less than 2 layers"));
        return;
    }

    // Get selection
    int iOldPos = m_panelSidebarGisLayers->GetTocCtrl()->GetSelection();
    if (iOldPos == wxNOT_FOUND) {
        wxLogError(_("No layer selected, select a layer first"));
        return;
    }

    // Contextual menu
    wxMenu posMenu;
    posMenu.SetTitle(_("Move layer to following position"));
    for (int i = 0; i < m_viewerLayerManager->GetCount(); i++) {
        posMenu.Append(
            asID_MENU_POPUP_LAYER + i,
            asStrF("%d - %s", i + 1, m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetDisplayName().GetFullName()));
    }
    wxPoint pos = wxGetMousePosition();

    int iNewID = GetPopupMenuSelectionFromUser(posMenu, ScreenToClient(pos));
    if (iNewID == wxID_NONE) return;

    int iNewPos = iNewID - asID_MENU_POPUP_LAYER;
    if (iNewPos == iOldPos) return;

#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Enter();
#endif
    m_viewerLayerManager->Move(iOldPos, iNewPos);
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Leave();
#endif

    m_workspace.SetHasChanged(true);
}

void asFrameViewer::OnToolAction(wxCommandEvent& event) {
    // Get event
    auto msg = static_cast<vrDisplayToolMessage*>(event.GetClientData());
    wxASSERT(msg);

    if (msg->m_evtType == vrEVT_TOOL_ZOOM) {
        // Get rectangle
        vrCoordinate* coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        coord->ConvertFromPixels(msg->m_rect, realRect);
        wxASSERT(realRect.IsOk());

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        // Moving view
#if defined(__WIN32__)
        auto thread = new asThreadViewerLayerManagerZoomIn(m_viewerLayerManager, &m_critSectionViewerLayerManager,
                                                           fittedRect);
        ThreadsManager().AddThread(thread);
#else
        m_viewerLayerManager->Zoom(fittedRect);
#endif
    } else if (msg->m_evtType == vrEVT_TOOL_ZOOMOUT) {
        // Get rectangle
        vrCoordinate* coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        coord->ConvertFromPixels(msg->m_rect, realRect);
        wxASSERT(realRect.IsOk());

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        // Moving view
#if defined(__WIN32__)
        auto thread = new asThreadViewerLayerManagerZoomOut(m_viewerLayerManager, &m_critSectionViewerLayerManager,
                                                            fittedRect);
        ThreadsManager().AddThread(thread);
#else
        m_viewerLayerManager->ZoomOut(fittedRect);
#endif
    } else if (msg->m_evtType == vrEVT_TOOL_SELECT) {
        // If no forecast open
        if (m_forecastManager->GetMethodsNb() == 0) {
            wxDELETE(msg);
            return;
        }

        // Transform screen coordinates to real coordinates
        vrCoordinate* coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);
        wxPoint clickedPos = msg->m_position;
        if (clickedPos != wxDefaultPosition) {
            wxPoint2DDouble realClickedPos;
            coord->ConvertFromPixels(clickedPos, realClickedPos);

            // Create a polygon to select the stations
            OGRPolygon polygon;
            OGRLinearRing linRing;

            // Buffer around the clicked point. Get extent to select according to the scale.
            double ratioBuffer = 0.025;  // 50px for the symbols 1000px for the display area
            vrRealRect actExtent = coord->GetExtent();
            int width = actExtent.GetSize().GetWidth();
            double bufferSize = width * ratioBuffer;

            linRing.addPoint(realClickedPos.m_x - bufferSize, realClickedPos.m_y - bufferSize);
            linRing.addPoint(realClickedPos.m_x - bufferSize, realClickedPos.m_y + bufferSize);
            linRing.addPoint(realClickedPos.m_x + bufferSize, realClickedPos.m_y + bufferSize);
            linRing.addPoint(realClickedPos.m_x + bufferSize, realClickedPos.m_y - bufferSize);
            linRing.addPoint(realClickedPos.m_x - bufferSize, realClickedPos.m_y - bufferSize);

            polygon.addRing(&linRing);

            // Get layer
            vrLayerVector* layer = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - specific") + ".memory");

            if (layer != nullptr) {
                // Search features
                wxArrayLong stationsClose;
                layer->SearchFeatures(&polygon, stationsClose);

                // Allow only one selection
                wxArrayLong station;
                if (stationsClose.Count() > 0) {
                    station.Add(stationsClose.Item(0));
                    int stationItem = stationsClose.Item(0);
                    OGRFeature* feature = layer->GetFeature(stationItem);
                    auto stationRow = (int)feature->GetFieldAsDouble(0);

                    if (stationRow >= 0) {
                        m_panelSidebarStationsList->GetChoiceCtrl()->Select(stationRow);
                    }
                    DrawPlotStation(stationRow);
                } else {
                    // Search on the other (not specific) forecast layer
                    vrLayerVector* layerOther = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - other") + ".memory");
                    if (layerOther != nullptr) {
                        // Search features
                        layerOther->SearchFeatures(&polygon, stationsClose);

                        // Allow only one selection
                        if (stationsClose.Count() > 0) {
                            station.Add(stationsClose.Item(0));
                            int stationItem = stationsClose.Item(0);
                            OGRFeature* feature = layerOther->GetFeature(stationItem);
                            auto stationRow = (int)feature->GetFieldAsDouble(0);

                            if (stationRow >= 0) {
                                m_panelSidebarStationsList->GetChoiceCtrl()->Select(stationRow);
                            }
                            DrawPlotStation(stationRow);
                        }
                    }
                }
                layer->SetSelectedIDs(station);

            } else {
                wxLogError(_("The desired layer was not found."));
            }

            ReloadViewerLayerManager();
        }

    } else if (msg->m_evtType == vrEVT_TOOL_PAN) {
        // Get rectangle
        vrCoordinate* coord = m_viewerLayerManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        wxPoint movedPos = msg->m_position;
        wxPoint2DDouble movedRealPt;
        if (!coord->ConvertFromPixels(movedPos, movedRealPt)) {
            wxLogError(_("Error converting point : %d, %d to real coordinate"), movedPos.x, movedPos.y);
            wxDELETE(msg);
            return;
        }

        vrRealRect actExtent = coord->GetExtent();
        actExtent.MoveLeftTopTo(movedRealPt);
        coord->SetExtent(actExtent);
        ReloadViewerLayerManager();
    } else {
        wxLogError("Operation not yet supported.");
    }

    wxDELETE(msg);
}

void asFrameViewer::OnStationSelection(wxCommandEvent& event) {
    wxBusyCursor wait;

    // Get selection
    int choice = event.GetInt();

    // If no forecast open
    if (m_forecastManager->GetMethodsNb() == 0) {
        return;
    }

    // Display on the map when only the specific layer exists
    vrLayerVector* layer = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - specific") + ".memory");
    vrLayerVector* layerOther = (vrLayerVector*)m_layerManager->GetLayer(_("Forecast - other") + ".memory");
    if (layer && !layerOther) {
        // Set selection
        wxArrayLong station;
        station.Add(choice);
        layer->SetSelectedIDs(station);
    }

    DrawPlotStation(choice);

    ReloadViewerLayerManager();
}

void asFrameViewer::OnChangeLeadTime(wxCommandEvent& event) {
    wxBusyCursor wait;

    Freeze();

    m_forecastViewer->ChangeLeadTime(event.GetInt());
    m_leadTimeSwitcher->SetLeadTime(m_forecastViewer->GetLeadTimeIndex());

    UpdatePanelAnalogDates();
    UpdatePanelCaptionAll();

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit(m_scrolledWindowOptions);
    Layout();

    Thaw();
}

void asFrameViewer::OnForecastClear(wxCommandEvent& event) {
    if (m_panelSidebarForecasts != nullptr) {
        m_panelSidebarForecasts->ClearForecasts();
    }
}

void asFrameViewer::OnForecastRatioSelectionChange(wxCommandEvent& event) {
    wxBusyCursor wait;

    m_forecastViewer->SetForecastDisplay(event.GetInt());

    UpdatePanelCaptionColorbar();
}

void asFrameViewer::OnForecastForecastSelectionChange(wxCommandEvent& event) {
    wxBusyCursor wait;

    Freeze();

    auto message = (asMessageForecastChoice*)event.GetClientData();

    m_forecastViewer->SetForecast(message->GetMethodRow(), message->GetForecastRow());

    if (m_leadTimeSwitcher) {
        m_leadTimeSwitcher->SetForecastSelection(m_forecastViewer->GetMethodSelection(),
                                                 m_forecastViewer->GetForecastSelection());
        m_leadTimeSwitcher->SetLeadTime(m_forecastViewer->GetLeadTimeIndex());
    }

    UpdateHeaderTexts();
    UpdatePanelCaptionAll();
    UpdatePanelAnalogDates();
    UpdatePanelStationsList();

    m_scrolledWindowOptions->Layout();
    m_sizerScrolledWindow->Fit(m_scrolledWindowOptions);
    Layout();

    wxDELETE(message);

    Thaw();
}

void asFrameViewer::OnForecastForecastSelectFirst(wxCommandEvent& event) {
    m_panelSidebarForecasts->GetForecastsCtrl()->SelectFirst();
}

void asFrameViewer::OnForecastQuantileSelectionChange(wxCommandEvent& event) {
    wxBusyCursor wait;

    m_forecastViewer->SetQuantile(event.GetInt());
}

void asFrameViewer::DrawPlotStation(int stationRow) {
    wxBusyCursor wait;

    m_forecastViewer->LoadPastForecast();

    // Get data
    int methodRow = m_forecastViewer->GetMethodSelection();
    int forecastRow = m_forecastViewer->GetForecastSelection();

    if (forecastRow < 0)  // Aggregator
    {
        forecastRow = m_forecastManager->GetForecastRowSpecificForStationRow(methodRow, stationRow);
    }

    auto framePlotStation = new asFramePlotTimeSeries(this, methodRow, forecastRow, stationRow, m_forecastManager);

    if (g_ppiScaleDc > 1) {
        wxSize frameSize = framePlotStation->GetSize();
        frameSize.x *= g_ppiScaleDc;
        frameSize.y *= g_ppiScaleDc;
        framePlotStation->SetSize(frameSize);
    }

    framePlotStation->Layout();
    framePlotStation->Init();
    framePlotStation->Plot();
    framePlotStation->Show();
}

void asFrameViewer::OnForecastNewAdded(wxCommandEvent& event) {
    wxBusyCursor wait;

    m_panelSidebarForecasts->Update();

    if (event.GetString().IsSameAs("last")) {
        m_forecastViewer->FixForecastSelection();

        float previousDate = m_forecastViewer->GetLeadTimeDate();
        m_forecastViewer->SetLeadTimeDate(previousDate);

        m_panelSidebarAlarms->Update();
    }
}

void asFrameViewer::ReloadViewerLayerManager() {
    wxBusyCursor wait;

    m_viewerLayerManager->Reload();

    /* Not sure there is any way to make it safe with threads.
    #if defined (__WIN32__)
        asThreadViewerLayerManagerReload *thread = new asThreadViewerLayerManagerReload(m_viewerLayerManager,
    &m_critSectionViewerLayerManager); ThreadsManager().AddThread(thread); #else m_viewerLayerManager->Reload();
    #endif*/
}

void asFrameViewer::UpdateHeaderTexts() {
    // Set header text
    wxString dateForecast = asTime::GetStringTime(m_forecastManager->GetLeadTimeOrigin(), "DD.MM.YYYY HH");
    wxString dateStr = asStrF(_("Forecast of the %sh"), dateForecast);
    m_staticTextForecastDate->SetLabel(dateStr);

    wxString forecastName;
    if (m_forecastViewer->GetForecastSelection() < 0) {
        forecastName = m_forecastManager->GetMethodName(m_forecastViewer->GetMethodSelection());
    } else {
        forecastName = m_forecastManager->GetForecastName(m_forecastViewer->GetMethodSelection(),
                                                          m_forecastViewer->GetForecastSelection());
    }

    m_staticTextForecast->SetLabel(forecastName);

    m_panelTop->Layout();
    m_panelTop->Refresh();
}

void asFrameViewer::UpdatePanelCaptionAll() {
    if (m_forecastViewer->GetLeadTimeIndex() >= 0) {
        m_panelSidebarCaptionForecastDots->Show();
        m_panelSidebarCaptionForecastRing->Hide();

        m_panelSidebarCaptionForecastDots->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());
    } else {
        m_panelSidebarCaptionForecastDots->Hide();
        m_panelSidebarCaptionForecastRing->Show();

        m_panelSidebarCaptionForecastRing->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());

        int methodRow = m_forecastViewer->GetMethodSelection();
        int forecastRow = m_forecastViewer->GetForecastSelection();
        if (forecastRow < 0) {
            forecastRow = 0;
        }

        asResultsForecast* forecast = m_forecastManager->GetForecast(methodRow, forecastRow);
        a1f dates = forecast->GetTargetDates();
        if (forecast->IsSubDaily()) {
            a1f datesClipped = dates.head(dates.size() - 1);
            m_panelSidebarCaptionForecastRing->SetDates(datesClipped);
        } else {
            m_panelSidebarCaptionForecastRing->SetDates(dates);
        }
    }
}

void asFrameViewer::UpdatePanelCaptionColorbar() {
    m_panelSidebarCaptionForecastDots->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());

    m_panelSidebarCaptionForecastRing->SetColorbarMax(m_forecastViewer->GetLayerMaxValue());
}

void asFrameViewer::UpdatePanelAnalogDates() {
    if (m_forecastViewer->GetLeadTimeIndex() < 0 || m_forecastViewer->GetForecastSelection() < 0) {
        m_panelSidebarAnalogDates->Hide();
        return;
    }

    m_panelSidebarAnalogDates->Show();

    asResultsForecast* forecast = m_forecastManager->GetForecast(m_forecastViewer->GetMethodSelection(),
                                                                 m_forecastViewer->GetForecastSelection());
    a1f arrayDate = forecast->GetAnalogsDates(m_forecastViewer->GetLeadTimeIndex());
    a1f arrayCriteria = forecast->GetAnalogsCriteria(m_forecastViewer->GetLeadTimeIndex());
    m_panelSidebarAnalogDates->SetChoices(arrayDate, arrayCriteria, forecast->GetDateFormatting());
}

void asFrameViewer::UpdatePanelStationsList() {
    int methodRow = m_forecastViewer->GetMethodSelection();
    int forecastRow = m_forecastViewer->GetForecastSelection();
    if (forecastRow < 0) {
        forecastRow = 0;
    }

    m_panelSidebarStationsList->Show();

    wxArrayString arrayStation = m_forecastManager->GetStationNamesWithHeights(methodRow, forecastRow);
    m_panelSidebarStationsList->SetChoices(arrayStation);
}

void asFrameViewer::UpdateRecentFiles() {
    wxASSERT(m_fileHistory);

    for (int i = 0; i < m_fileHistory->GetCount(); ++i) {
        wxString filePath = m_fileHistory->GetHistoryFile(i);
        if (!wxFileExists(filePath)) {
            m_fileHistory->RemoveFileFromHistory(i);
            --i;
        }
    }
}

void asFrameViewer::SetRecentFiles() {
    wxConfigBase* config = wxFileConfig::Get();
    config->SetPath("/Recent");

    wxMenuItem* menuItem = m_menuBar->FindItem(asID_MENU_RECENT);
    if (menuItem->IsSubMenu()) {
        wxMenu* menu = menuItem->GetSubMenu();
        if (menu) {
            m_fileHistory->Load(*config);
            UpdateRecentFiles();
            m_fileHistory->UseMenu(menu);
            m_fileHistory->AddFilesToMenu(menu);
        }
    }

    config->SetPath("..");
}

void asFrameViewer::SaveRecentFiles() {
    wxASSERT(m_fileHistory);
    wxConfigBase* config = wxFileConfig::Get();
    config->SetPath("/Recent");

    m_fileHistory->Save(*config);

    config->SetPath("..");
}
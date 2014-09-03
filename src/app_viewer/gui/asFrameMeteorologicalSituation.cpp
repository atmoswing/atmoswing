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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#include "asFrameMeteorologicalSituation.h"

#include "AtmoswingAppViewer.h"
#if defined (__WIN32__)
    #include "asThreadsManager.h"
    #include "asThreadViewerLayerManagerReload.h"
    #include "asThreadViewerLayerManagerZoomIn.h"
    #include "asThreadViewerLayerManagerZoomOut.h"
#endif
#include "asFrameAbout.h"
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


BEGIN_EVENT_TABLE(asFrameMeteorologicalSituation, wxFrame)
    EVT_MENU(wxID_EXIT,  asFrameMeteorologicalSituation::OnQuit)
/*    EVT_MENU(wxID_ABOUT, asFrameForecast::OnAbout)
    EVT_MENU(wxID_OPEN, asFrameForecast::OnOpenLayer)
    EVT_MENU(wxID_REMOVE, asFrameForecast::OnCloseLayer)
    EVT_MENU (wxID_INFO, asFrameForecast::OnShowLog)*/
    EVT_MENU (asID_ZOOM_IN, asFrameMeteorologicalSituation::OnToolZoomIn)
    EVT_MENU (asID_ZOOM_OUT, asFrameMeteorologicalSituation::OnToolZoomOut)
    EVT_MENU (asID_ZOOM_FIT, asFrameMeteorologicalSituation::OnToolZoomToFit)
    EVT_MENU (asID_PAN, asFrameMeteorologicalSituation::OnToolPan)/*
    EVT_MENU (vlID_MOVE_LAYER, asFrameForecast::OnMoveLayer)
*/
    EVT_KEY_DOWN(asFrameMeteorologicalSituation::OnKeyDown)
    EVT_KEY_UP(asFrameMeteorologicalSituation::OnKeyUp)

    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOM, asFrameMeteorologicalSituation::OnToolAction)
    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOMOUT, asFrameMeteorologicalSituation::OnToolAction)
    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_SELECT, asFrameMeteorologicalSituation::OnToolAction)
    EVT_COMMAND(wxID_ANY, vrEVT_TOOL_PAN, asFrameMeteorologicalSituation::OnToolAction)

    EVT_COMMAND(wxID_ANY, asEVT_ACTION_PRESET_SELECTION_CHANGED, asFrameMeteorologicalSituation::OnPresetSelection)
END_EVENT_TABLE()

/* vroomDropFiles */

vroomDropFilesMeteoSituation::vroomDropFilesMeteoSituation(asFrameMeteorologicalSituation * parent){
    wxASSERT(parent);
    m_LoaderFrame = parent;
}


bool vroomDropFilesMeteoSituation::OnDropFiles(wxCoord x, wxCoord y,
                                 const wxArrayString & filenames){
    if (filenames.GetCount() == 0) return false;

    m_LoaderFrame->OpenLayers(filenames);
    return true;
}


asFrameMeteorologicalSituation::asFrameMeteorologicalSituation( wxWindow* parent, wxWindowID id )
:
asFrameMeteorologicalSituationVirtual( parent, id )
{
    // Toolbar
    m_ToolBar->AddTool( asID_OPEN, wxT("Open"), img_open, img_open, wxITEM_NORMAL, _("Open layer"), _("Open a layer"), NULL );
    m_ToolBar->AddSeparator();
    m_ToolBar->AddTool( asID_ZOOM_IN, wxT("Zoom in"), img_map_zoom_in, img_map_zoom_in, wxITEM_NORMAL, _("Zoom in"), _("Zoom in"), NULL );
    m_ToolBar->AddTool( asID_ZOOM_OUT, wxT("Zoom out"), img_map_zoom_out, img_map_zoom_out, wxITEM_NORMAL, _("Zoom out"), _("Zoom out"), NULL );
    m_ToolBar->AddTool( asID_PAN, wxT("Pan"), img_map_move, img_map_move, wxITEM_NORMAL, _("Pan the map"), _("Move the map by panning"), NULL );
    m_ToolBar->AddTool( asID_ZOOM_FIT, wxT("Fit"), img_map_fit, img_map_fit, wxITEM_NORMAL, _("Zoom to visible layers"), _("Zoom view to the full extent of all visible layers"), NULL );
    m_ToolBar->Realize();

    // VroomGIS controls
    m_DisplayCtrl = new vrViewerDisplay( m_PanelGIS, wxID_ANY, wxColour(120,120,120));
    m_SizerGIS->Add( m_DisplayCtrl, 1, wxEXPAND, 5 );
    m_PanelGIS->Layout();
    
    // Presets panel
    m_PanelSidebarMeteoSituation = new asPanelSidebarMeteoSituation( m_ScrolledWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarMeteoSituation->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarMeteoSituation, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Gis panel
    m_PanelSidebarGisLayers = new asPanelSidebarGisLayers( m_ScrolledWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarGisLayers->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarGisLayers, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    m_ScrolledWindow->Layout();
    m_SizerScrolledWindow->Fit( m_ScrolledWindow );

    // Connect Events
    m_DisplayCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameMeteorologicalSituation::OnRightClick ), NULL, this );
    m_DisplayCtrl->Connect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameMeteorologicalSituation::OnKeyDown), NULL, this);
    m_DisplayCtrl->Connect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameMeteorologicalSituation::OnKeyUp), NULL, this);

    // DND
    m_PanelSidebarGisLayers->SetDropTarget(new vroomDropFilesMeteoSituation(this));

    // VroomGIS
    m_LayerManager = new vrLayerManager();
    m_ViewerLayerManager = new vrViewerLayerManager(m_LayerManager, this, m_DisplayCtrl , m_PanelSidebarGisLayers->GetTocCtrl());

    // Restore frame position and size
    wxConfigBase *pConfig = wxFileConfig::Get();
    int minHeight = 450, minWidth = 800;
    int x = pConfig->Read("/MeteoSituationFrame/x", 50),
        y = pConfig->Read("/MeteoSituationFrame/y", 50),
        w = pConfig->Read("/MeteoSituationFrame/w", minWidth),
        h = pConfig->Read("/MeteoSituationFrame/h", minHeight);
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
    pConfig->Read("/MeteoSituationFrame/Maximize", &doMaximize);
    Maximize(doMaximize);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameMeteorologicalSituation::~asFrameMeteorologicalSituation()
{
    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/MeteoSituationFrame/MeteoSituationFrameGisLayers", !m_PanelSidebarGisLayers->IsReduced());

    // Save the frame position
    bool doMaximize = IsMaximized();
    pConfig->Write("/MeteoSituationFrame/Maximize", doMaximize);
    if (!doMaximize)
    {
        int x, y, w, h;
        GetClientSize(&w, &h);
        GetPosition(&x, &y);
        pConfig->Write("/MeteoSituationFrame/x", (long) x);
        pConfig->Write("/MeteoSituationFrame/y", (long) y);
        pConfig->Write("/MeteoSituationFrame/w", (long) w);
        pConfig->Write("/MeteoSituationFrame/h", (long) h);
    }

    // Disconnect Events
    m_DisplayCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFrameMeteorologicalSituation::OnRightClick ), NULL, this );
    m_DisplayCtrl->Disconnect( wxEVT_KEY_DOWN, wxKeyEventHandler(asFrameMeteorologicalSituation::OnKeyDown), NULL, this);
    m_DisplayCtrl->Disconnect( wxEVT_KEY_UP, wxKeyEventHandler(asFrameMeteorologicalSituation::OnKeyUp), NULL, this);

    // Don't delete m_ViewerLayerManager, will be deleted by the manager
    wxDELETE(m_LayerManager);
}

void asFrameMeteorologicalSituation::Init()
{
    // Reduce some panels
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool display = true;
    pConfig->Read("/MeteoSituationFrame/MeteoSituationFrameGisLayers", &display, false);
    if (!display)
    {
        m_PanelSidebarGisLayers->ReducePanel();
        m_PanelSidebarGisLayers->Layout();
    }

    // Add the presets list
    wxArrayString presets;
    presets.Add(_("Select a preset..."));
    presets.Add(_("Sea surface temperature"));
    presets.Add(_("Air temperature"));
    presets.Add(_("Sea level pressure"));
    presets.Add(_("Cloud cover"));
    presets.Add(_("Precipitation"));
    presets.Add(_("Snow"));
    presets.Add(_("Wind"));
    m_PanelSidebarMeteoSituation->SetChoices(presets);

    // Set the default tool
    m_DisplayCtrl->SetToolDefault();

    m_ScrolledWindow->Layout();

    Layout();
    Refresh();
}

void asFrameMeteorologicalSituation::OnQuit( wxCommandEvent& event )
{
    event.Skip();
}

bool asFrameMeteorologicalSituation::OpenLayers (const wxArrayString & names)
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

void asFrameMeteorologicalSituation::OnOpenLayer(wxCommandEvent & event)
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

void asFrameMeteorologicalSituation::OnCloseLayer(wxCommandEvent & event)
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

void asFrameMeteorologicalSituation::OnKeyDown(wxKeyEvent & event)
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

void asFrameMeteorologicalSituation::OnKeyUp(wxKeyEvent & event)
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

void asFrameMeteorologicalSituation::OnToolZoomIn (wxCommandEvent & event)
{
    m_DisplayCtrl->SetToolZoom();
}

void asFrameMeteorologicalSituation::OnToolZoomOut (wxCommandEvent & event)
{
    m_DisplayCtrl->SetToolZoomOut();
}

void asFrameMeteorologicalSituation::OnToolPan (wxCommandEvent & event)
{
    m_DisplayCtrl->SetToolPan();
}

void asFrameMeteorologicalSituation::OnToolZoomToFit (wxCommandEvent & event)
{
    // Fit to all layers
    m_ViewerLayerManager->ZoomToFit(true);
    ReloadViewerLayerManager();
}

void asFrameMeteorologicalSituation::OnMoveLayer (wxCommandEvent & event)
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
        posMenu.Append(asID_MENU_POPUP_LAYER_METEO + i,
                         wxString::Format("%d - %s",i+1,
                                          m_ViewerLayerManager->GetRenderer(i)->GetLayer()->GetDisplayName().GetFullName()));
    }
    wxPoint pos = wxGetMousePosition();

    int iNewID = GetPopupMenuSelectionFromUser(posMenu, ScreenToClient(pos));
    if (iNewID == wxID_NONE) return;

    int iNewPos = iNewID - asID_MENU_POPUP_LAYER_METEO;
    if (iNewPos == iOldPos) return;

    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Enter();
    #endif
    m_ViewerLayerManager->Move(iOldPos, iNewPos);
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif
}

void asFrameMeteorologicalSituation::OnToolAction (wxCommandEvent & event)
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

void asFrameMeteorologicalSituation::ReloadViewerLayerManager( )
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

void asFrameMeteorologicalSituation::OnPresetSelection (wxCommandEvent & event)
{
    

    wxString basePath = "D:\\\_DEV\\AtmoSwing 1.6\\data\\";
    wxFileName fullPath(basePath);



    int selection = event.GetInt();

    wxArrayString paths;
    wxArrayString types; 
    VectorInt transparencies;
    VectorInt widths;
    std::vector < wxColour > lineColors;
    std::vector < wxColour > fillColors;

    // Set attributes
    switch (selection)
    {
        case 0: // Select a preset...
            break;

        case 1: // Sea surface temperature
            // WMS
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("NOAA-sst.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // Countries
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("countries.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(100,100,100,1));
            break;

        case 2: // Air temperature
            // WMS
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("OWM-temperature.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // Countries
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("countries.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(0,0,0,0));
            break;

        case 3: // Sea level pressure
            // WMS
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("OWM-pressure.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // WMS contours
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("OWM-pressurecntr.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // Countries
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("countries.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(0,0,0,0));
            break;

        case 4: // Cloud cover
            // Continents
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("continents.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(100,100,100,1));
            // WMS
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("OWM-cloud.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // Countries
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("countries.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(0,0,0,0));
            break;

        case 5: // Precipitation
            // Continents
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("continents.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(100,100,100,1));
            // WMS
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("OWM-precipitation.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // Countries
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("countries.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(0,0,0,0));
            break;

        case 6: // Snow
            // Continents
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("continents.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(100,100,100,1));
            // WMS
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("OWM-snow.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // Countries
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("countries.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(0,0,0,0));
            break;

        case 7: // Wind
            // WMS
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("wms");
            fullPath.AppendDir("meteo");
            fullPath.SetFullName("OWM-wind.xml");
            paths.Add(fullPath.GetFullPath());
            types.Add("wms");
            transparencies.push_back(0);
            widths.push_back(0);
            lineColors.push_back(wxColour(0,0,0,0));
            fillColors.push_back(wxColour(0,0,0,0));
            // Countries
            fullPath = wxFileName(basePath);
            fullPath.AppendDir("layers");
            fullPath.SetFullName("countries.shp");
            paths.Add(fullPath.GetFullPath());
            types.Add("vector");
            transparencies.push_back(0);
            widths.push_back(1);
            lineColors.push_back(wxColour(0,0,0,1));
            fillColors.push_back(wxColour(0,0,0,0));
            break;

        default:
            asLogError(_("Undefined preset."));
    }

    wxASSERT(paths.GetCount() == types.GetCount());
    wxASSERT(paths.GetCount() == transparencies.size());
    wxASSERT(paths.GetCount() == widths.size());
    wxASSERT(paths.GetCount() == lineColors.size());
    wxASSERT(paths.GetCount() == fillColors.size());

    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Enter();
    #endif

    m_ViewerLayerManager->FreezeBegin();

    // Save extent
    vrRealRect extent;
    if (m_ViewerLayerManager->GetCount()>0)
    {
        vrCoordinate * myCoord = m_ViewerLayerManager->GetDisplay()->GetCoordinate(); 
        wxASSERT(myCoord);
        extent = myCoord->GetExtent();
    }

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

    // Open the layers
    for (int i=0; i<paths.GetCount(); i++)
    {
        if (wxFileName::FileExists(paths.Item(i)))
        {
            if (m_LayerManager->Open(wxFileName(paths.Item(i))))
            {
                if (types.Item(i).IsSameAs("raster"))
                {
                    vrRenderRaster* render = new vrRenderRaster();
                    render->SetTransparency(transparencies[i]);

                    vrLayer* layer = m_LayerManager->GetLayer( wxFileName(paths.Item(i)));
                    wxASSERT(layer);
                    m_ViewerLayerManager->Add(-1, layer, render, NULL);
                }
                else if (types.Item(i).IsSameAs("vector"))
                {
                    vrRenderVector* render = new vrRenderVector();
                    render->SetTransparency(transparencies[i]);
                    render->SetSize(widths[i]);
                    render->SetColorPen(lineColors[i]);
                    if (fillColors[i].Alpha()==0)
                    {
                        render->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
                    }
                    else
                    {
                        render->SetColorBrush(fillColors[i]);
                    }

                    vrLayer* layer = m_LayerManager->GetLayer( wxFileName(paths.Item(i)));
                    wxASSERT(layer);
                    m_ViewerLayerManager->Add(-1, layer, render, NULL);
                }
                else if (types.Item(i).IsSameAs("wms"))
                {
                    vrRenderRaster* render = new vrRenderRaster();
                    render->SetTransparency(transparencies[i]);

                    vrLayer* layer = m_LayerManager->GetLayer( wxFileName(paths.Item(i)));
                    wxASSERT(layer);
                    m_ViewerLayerManager->Add(-1, layer, render, NULL);
                }
                else
                {
                    asLogError(wxString::Format(_("The GIS layer type %s does not correspond to allowed values."), types.Item(i).c_str()));
                }
            }
            else
            {
                asLogWarning(wxString::Format(_("The file %s cound not be opened."), paths.Item(i).c_str()));
            }
        }
        else
        {
            asLogWarning(wxString::Format(_("The file %s cound not be found."), paths.Item(i).c_str()));
        }
    }
    
    if (extent.IsOk())
    {
        m_ViewerLayerManager->InitializeExtent(extent);
    }

    ReloadViewerLayerManager();

    m_ViewerLayerManager->FreezeEnd();

    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif
}
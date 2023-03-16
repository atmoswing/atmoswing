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
 * Portions Copyright 2022-2023 Pascal Horton, Terranum.
 */

#include "asFramePredictors.h"

#include "asFramePreferencesViewer.h"
#if defined(__WIN32__)
#include "asThreadViewerLayerManagerReload.h"
#include "asThreadViewerLayerManagerZoomIn.h"
#include "asThreadViewerLayerManagerZoomOut.h"
#include "asThreadsManager.h"
#endif
#include "asResults.h"
#include "img_toolbar.h"

BEGIN_EVENT_TABLE(asFramePredictors, wxFrame)

END_EVENT_TABLE()

asFramePredictors::asFramePredictors(wxWindow* parent, int selectedForecast, asForecastManager* forecastManager,
                                     wxWindowID id)
    : asFramePredictorsVirtual(parent, id) {
    m_forecastManager = forecastManager;
asFramePredictors::asFramePredictors(wxWindow* parent, asForecastManager* forecastManager, asWorkspace* workspace,
                                     int selectedForecast, wxWindowID id)
    : asFramePredictorsVirtual(parent, id),
      m_forecastManager(forecastManager),
      m_workspace(workspace)
{
    // Toolbar
    m_toolBar->AddTool(asID_ZOOM_IN, wxT("Zoom in"), *_img_map_zoom_in, *_img_map_zoom_in, wxITEM_NORMAL, _("Zoom in"),
                       _("Zoom in"), nullptr);
    m_toolBar->AddTool(asID_ZOOM_OUT, wxT("Zoom out"), *_img_map_zoom_out, *_img_map_zoom_out, wxITEM_NORMAL, _("Zoom out"),
                       _("Zoom out"), nullptr);
    m_toolBar->AddTool(asID_PAN, wxT("Pan"), *_img_map_move, *_img_map_move, wxITEM_NORMAL, _("Pan the map"),
                       _("Move the map by panning"), nullptr);
    m_toolBar->AddTool(asID_ZOOM_FIT, wxT("Fit"), *_img_map_fit, *_img_map_fit, wxITEM_NORMAL, _("Zoom to visible layers"),
                       _("Zoom view to the full extent of all visible layers"), nullptr);
    m_toolBar->AddSeparator();
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), *_img_preferences, *_img_preferences, wxITEM_NORMAL,
                       _("Preferences"), _("Preferences"), nullptr);
    m_toolBar->Realize();

    // VroomGIS controls
    m_displayCtrlLeft = new vrViewerDisplay(m_panelGISLeft, wxID_ANY, wxColour(255, 255, 255));
    m_displayCtrlRight = new vrViewerDisplay(m_panelGISRight, wxID_ANY, wxColour(255, 255, 255));
    m_sizerGISLeft->Add(m_displayCtrlLeft, 1, wxEXPAND | wxALL, 0);
    m_sizerGISRight->Add(m_displayCtrlRight, 1, wxEXPAND | wxALL, 0);
    m_panelGIS->Layout();
    m_layerManager = new vrLayerManager();
    m_viewerLayerManagerLeft = new vrViewerLayerManager(m_layerManager, this, m_displayCtrlLeft, m_tocCtrlLeft);
    m_viewerLayerManagerRight = new vrViewerLayerManager(m_layerManager, this, m_displayCtrlRight, m_tocCtrlRight);

    // Viewer
    m_predictorsManager = new asPredictorsManager();
    m_predictorsViewer = new asPredictorsRenderer(this, m_layerManager, m_predictorsManager, m_viewerLayerManagerLeft,
                                                  m_viewerLayerManagerRight, m_checkListPredictors);

    // Menus
    m_menuTools->AppendCheckItem(asID_SET_SYNCRO_MODE, "Syncronize tools",
                                 "When set to true, browsing is syncronized on all display");
    m_menuTools->Check(asID_SET_SYNCRO_MODE, m_syncroTool);

    // Connect Events
    m_displayCtrlLeft->Connect(wxEVT_RIGHT_DOWN, wxMouseEventHandler(asFramePredictors::OnRightClick), nullptr, this);
    m_displayCtrlLeft->Connect(wxEVT_KEY_DOWN, wxKeyEventHandler(asFramePredictors::OnKeyDown), nullptr, this);
    m_displayCtrlLeft->Connect(wxEVT_KEY_UP, wxKeyEventHandler(asFramePredictors::OnKeyUp), nullptr, this);
    this->Connect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFramePredictors::OpenFramePreferences));

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFramePredictors::~asFramePredictors() {
    // Disconnect Events
    m_displayCtrlLeft->Disconnect(wxEVT_RIGHT_DOWN, wxMouseEventHandler(asFramePredictors::OnRightClick), nullptr, this);
    m_displayCtrlLeft->Disconnect(wxEVT_KEY_DOWN, wxKeyEventHandler(asFramePredictors::OnKeyDown), nullptr, this);
    m_displayCtrlLeft->Disconnect(wxEVT_KEY_UP, wxKeyEventHandler(asFramePredictors::OnKeyUp), nullptr, this);
    this->Disconnect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                     wxCommandEventHandler(asFramePredictors::OpenFramePreferences));

    wxDELETE(m_layerManager);
}

void asFramePredictors::Init() {
    if (m_forecastManager->GetMethodsNb() > 0) {
        // Forecast list
        wxArrayString arrayForecasts = m_forecastManager->GetAllForecastNamesWxArray();
        m_choiceForecast->Set(arrayForecasts);
        m_choiceForecast->Select(m_selectedForecast);

        m_selectedTargetDate = 0;
        m_selectedAnalogDate = 0;
    InitExtent();
}

void asFramePredictors::InitExtent() {
    // Desired extent
    vrRealRect desiredExtent;
    m_viewerLayerManagerLeft->InitializeExtent(desiredExtent);
    m_viewerLayerManagerRight->InitializeExtent(desiredExtent);
}
void asFramePredictors::OnKeyDown(wxKeyEvent& event) {
    m_KeyBoardState = wxKeyboardState(event.ControlDown(), event.ShiftDown(), event.AltDown(), event.MetaDown());
    if (m_KeyBoardState.GetModifiers() != wxMOD_CMD) {
        event.Skip();
        return;
    }

    const vrDisplayTool* tool = m_displayCtrlLeft->GetTool();
    if (!tool) {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_IN) {
        m_displayCtrlLeft->SetToolZoomOut();
        m_displayCtrlRight->SetToolZoomOut();
    }
    event.Skip();
}

void asFramePredictors::OnKeyUp(wxKeyEvent& event) {
    if (m_KeyBoardState.GetModifiers() != wxMOD_CMD) {
        event.Skip();
        return;
    }

    const vrDisplayTool* tool = m_displayCtrlLeft->GetTool();
    if (!tool) {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_OUT || tool->GetID() == wxID_ZOOM_IN) {
        m_displayCtrlLeft->SetToolZoom();
        m_displayCtrlRight->SetToolZoom();
    }
    event.Skip();
}

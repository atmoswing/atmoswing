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
#include <wx/colour.h>
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
    OpenDefaultLayers();
}

void asFramePredictors::InitExtent() {
    // Desired extent
    vrRealRect desiredExtent;
    m_viewerLayerManagerLeft->InitializeExtent(desiredExtent);
    m_viewerLayerManagerRight->InitializeExtent(desiredExtent);
}
void asFramePredictors::OpenDefaultLayers() {
    // Default paths
    wxConfigBase* pConfig = wxFileConfig::Get();
    wxString dirData = asConfig::GetDataDir() + "share";
    if (!wxDirExists(dirData)) {
        dirData = asConfig::GetDataDir() + ".." + DS + "share";
    }

    wxString gisData = dirData + DS + "atmoswing" + DS + "gis" + DS + "shapefiles";

    wxString continentsFilePath = pConfig->Read("/GIS/LayerContinentsFilePath", gisData + DS + "continents.shp");
    wxString countriesFilePath = pConfig->Read("/GIS/LayerCountriesFilePath", gisData + DS + "countries.shp");
    wxString latLongFilePath = pConfig->Read("/GIS/LayerLatLongFilePath", gisData + DS + "latlong.shp");
    wxString geogridFilePath = pConfig->Read("/GIS/LayerGeogridFilePath", gisData + DS + "geogrid.shp");

    // Try to open layers
    m_viewerLayerManagerLeft->FreezeBegin();
    m_viewerLayerManagerRight->FreezeBegin();
    vrLayer* layer;

    // Continents
    if (wxFileName::FileExists(continentsFilePath)) {
        if (m_layerManager->Open(wxFileName(continentsFilePath))) {
            long continentsTransp = pConfig->ReadLong("/GIS/LayerContinentsTransp", 50);
            long continentsColor = pConfig->ReadLong("/GIS/LayerContinentsColor", (long)0x99999999);
            wxColour colorContinents;
            colorContinents.SetRGB((wxUint32)continentsColor);
            long continentsSize = pConfig->ReadLong("/GIS/LayerContinentsSize", 1);
            bool continentsVisibility = pConfig->ReadBool("/GIS/LayerContinentsVisibility", true);

            vrRenderVector* renderContinents1 = new vrRenderVector();
            renderContinents1->SetTransparency(continentsTransp);
            renderContinents1->SetColorPen(colorContinents);
            renderContinents1->SetColorBrush(colorContinents);
            renderContinents1->SetBrushStyle(wxBRUSHSTYLE_SOLID);
            renderContinents1->SetSize(continentsSize);
            vrRenderVector* renderContinents2 = new vrRenderVector();
            renderContinents2->SetTransparency(continentsTransp);
            renderContinents2->SetColorPen(colorContinents);
            renderContinents2->SetColorBrush(colorContinents);
            renderContinents2->SetBrushStyle(wxBRUSHSTYLE_SOLID);
            renderContinents2->SetSize(continentsSize);

            layer = m_layerManager->GetLayer(wxFileName(continentsFilePath));
            wxASSERT(layer);
            m_viewerLayerManagerLeft->Add(-1, layer, renderContinents1, nullptr, continentsVisibility);
            m_viewerLayerManagerRight->Add(-1, layer, renderContinents2, nullptr, continentsVisibility);
        } else {
            wxLogWarning(_("The Continents layer file %s cound not be opened."), continentsFilePath.c_str());
        }
    } else {
        wxLogWarning(_("The Continents layer file %s cound not be found."), continentsFilePath.c_str());
    }

    // Countries
    if (wxFileName::FileExists(countriesFilePath)) {
        if (m_layerManager->Open(wxFileName(countriesFilePath))) {
            long countriesTransp = pConfig->ReadLong("/GIS/LayerCountriesTransp", 0);
            long countriesColor = pConfig->ReadLong("/GIS/LayerCountriesColor", (long)0x77999999);
            wxColour colorCountries;
            colorCountries.SetRGB((wxUint32)countriesColor);
            long countriesSize = pConfig->ReadLong("/GIS/LayerCountriesSize", 1);
            bool countriesVisibility = pConfig->ReadBool("/GIS/LayerCountriesVisibility", true);

            vrRenderVector* renderCountries1 = new vrRenderVector();
            renderCountries1->SetTransparency(countriesTransp);
            renderCountries1->SetColorPen(colorCountries);
            renderCountries1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderCountries1->SetSize(countriesSize);
            vrRenderVector* renderCountries2 = new vrRenderVector();
            renderCountries2->SetTransparency(countriesTransp);
            renderCountries2->SetColorPen(colorCountries);
            renderCountries2->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderCountries2->SetSize(countriesSize);

            layer = m_layerManager->GetLayer(wxFileName(countriesFilePath));
            wxASSERT(layer);
            m_viewerLayerManagerLeft->Add(-1, layer, renderCountries1, nullptr, countriesVisibility);
            m_viewerLayerManagerRight->Add(-1, layer, renderCountries2, nullptr, countriesVisibility);
        } else {
            wxLogWarning(_("The Countries layer file %s cound not be opened."), countriesFilePath.c_str());
        }
    } else {
        wxLogWarning(_("The Countries layer file %s cound not be found."), countriesFilePath.c_str());
    }

    // LatLong
    if (wxFileName::FileExists(latLongFilePath)) {
        if (m_layerManager->Open(wxFileName(latLongFilePath))) {
            long latLongTransp = pConfig->ReadLong("/GIS/LayerLatLongTransp", 80);
            long latLongColor = pConfig->ReadLong("/GIS/LayerLatLongColor", (long)0xff999999);
            wxColour colorLatLong;
            colorLatLong.SetRGB((wxUint32)latLongColor);
            long latLongSize = pConfig->ReadLong("/GIS/LayerLatLongSize", 1);
            bool latLongVisibility = pConfig->ReadBool("/GIS/LayerLatLongVisibility", true);

            vrRenderVector* renderLatLong1 = new vrRenderVector();
            renderLatLong1->SetTransparency(latLongTransp);
            renderLatLong1->SetColorPen(colorLatLong);
            renderLatLong1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderLatLong1->SetSize(latLongSize);
            vrRenderVector* renderLatLong2 = new vrRenderVector();
            renderLatLong2->SetTransparency(latLongTransp);
            renderLatLong2->SetColorPen(colorLatLong);
            renderLatLong2->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderLatLong2->SetSize(latLongSize);

            layer = m_layerManager->GetLayer(wxFileName(latLongFilePath));
            wxASSERT(layer);
            m_viewerLayerManagerLeft->Add(-1, layer, renderLatLong1, nullptr, latLongVisibility);
            m_viewerLayerManagerRight->Add(-1, layer, renderLatLong2, nullptr, latLongVisibility);
        } else {
            wxLogWarning(_("The LatLong layer file %s cound not be opened."), latLongFilePath.c_str());
        }
    } else {
        wxLogWarning(_("The LatLong layer file %s cound not be found."), latLongFilePath.c_str());
    }

    // Geogrid
    if (wxFileName::FileExists(geogridFilePath)) {
        if (m_layerManager->Open(wxFileName(geogridFilePath))) {
            long geogridTransp = pConfig->ReadLong("/GIS/LayerGeogridTransp", 80);
            long geogridColor = pConfig->ReadLong("/GIS/LayerGeogridColor", (long)0xff999999);
            wxColour colorGeogrid;
            colorGeogrid.SetRGB((wxUint32)geogridColor);
            long geogridSize = pConfig->ReadLong("/GIS/LayerGeogridSize", 2);
            bool geogridVisibility = pConfig->ReadBool("/GIS/LayerGeogridVisibility", false);

            vrRenderVector* renderGeogrid1 = new vrRenderVector();
            renderGeogrid1->SetTransparency(geogridTransp);
            renderGeogrid1->SetColorPen(colorGeogrid);
            renderGeogrid1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderGeogrid1->SetSize(geogridSize);
            vrRenderVector* renderGeogrid2 = new vrRenderVector();
            renderGeogrid2->SetTransparency(geogridTransp);
            renderGeogrid2->SetColorPen(colorGeogrid);
            renderGeogrid2->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderGeogrid2->SetSize(geogridSize);

            layer = m_layerManager->GetLayer(wxFileName(geogridFilePath));
            wxASSERT(layer);
            m_viewerLayerManagerLeft->Add(-1, layer, renderGeogrid1, nullptr, geogridVisibility);
            m_viewerLayerManagerRight->Add(-1, layer, renderGeogrid2, nullptr, geogridVisibility);
        } else {
            wxLogWarning(_("The Geogrid layer file %s cound not be opened."), geogridFilePath.c_str());
        }
    } else {
        wxLogWarning(_("The Geogrid layer file %s cound not be found."), geogridFilePath.c_str());
    }

    m_viewerLayerManagerLeft->FreezeEnd();
    m_viewerLayerManagerRight->FreezeEnd();
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

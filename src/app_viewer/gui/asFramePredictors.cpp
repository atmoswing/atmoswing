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

#include "asPredictorsManager.h"
#include "img_misc.h"
#include "img_toolbar.h"

BEGIN_EVENT_TABLE(asFramePredictors, wxFrame)
EVT_MENU(wxID_OPEN, asFramePredictors::OnOpenLayer)
EVT_MENU(wxID_REMOVE, asFramePredictors::OnCloseLayer)
EVT_MENU(asID_ZOOM_IN, asFramePredictors::OnToolZoomIn)
EVT_MENU(asID_ZOOM_OUT, asFramePredictors::OnToolZoomOut)
EVT_MENU(asID_ZOOM_FIT, asFramePredictors::OnToolZoomToFit)
EVT_MENU(asID_PAN, asFramePredictors::OnToolPan)
EVT_MENU(asID_CROSS_MARKER, asFramePredictors::OnToolSight)
EVT_MENU(asID_SET_SYNCRO_MODE, asFramePredictors::OnSyncroToolSwitch)

EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOM, asFramePredictors::OnToolAction)
EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOMOUT, asFramePredictors::OnToolAction)
EVT_COMMAND(wxID_ANY, vrEVT_TOOL_PAN, asFramePredictors::OnToolAction)
EVT_COMMAND(wxID_ANY, vrEVT_TOOL_SIGHT, asFramePredictors::OnToolAction)

END_EVENT_TABLE()

/* vroomDropFilesPredictors */

vroomDropFilesPredictors::vroomDropFilesPredictors(asFramePredictors* parent) {
    wxASSERT(parent);
    m_LoaderFrame = parent;
}

bool vroomDropFilesPredictors::OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) {
    if (filenames.GetCount() == 0) return false;

    m_LoaderFrame->OpenLayers(filenames);
    return true;
}

asFramePredictors::asFramePredictors(wxWindow* parent, asForecastManager* forecastManager, asWorkspace* workspace,
                                     int methodRow, int forecastRow, wxWindowID id)
    : asFramePredictorsVirtual(parent, id),
      m_forecastManager(forecastManager),
      m_workspace(workspace),
      m_selectedMethod(methodRow),
      m_selectedForecast(forecastRow),
      m_syncroTool(true),
      m_displayPanelLeft(true),
      m_displayPanelRight(true),
      m_selectedTargetDate(-1),
      m_selectedAnalogDate(-1),
      m_selectedPredictor(-1)
{
    m_selectedForecast = wxMax(m_selectedForecast, 0);

    // Toolbar
    m_toolBar->AddTool(asID_ZOOM_IN, wxT("Zoom in"), *_img_map_zoom_in, *_img_map_zoom_in, wxITEM_NORMAL, _("Zoom in"),
                       _("Zoom in"), nullptr);
    m_toolBar->AddTool(asID_ZOOM_OUT, wxT("Zoom out"), *_img_map_zoom_out, *_img_map_zoom_out, wxITEM_NORMAL,
                       _("Zoom out"), _("Zoom out"), nullptr);
    m_toolBar->AddTool(asID_PAN, wxT("Pan"), *_img_map_move, *_img_map_move, wxITEM_NORMAL, _("Pan the map"),
                       _("Move the map by panning"), nullptr);
    m_toolBar->AddTool(asID_ZOOM_FIT, wxT("Fit"), *_img_map_fit, *_img_map_fit, wxITEM_NORMAL,
                       _("Zoom to visible layers"), _("Zoom view to the full extent of all visible layers"), nullptr);
    m_toolBar->AddTool(asID_CROSS_MARKER, wxT("Marker overlay"), *_img_map_cross, *_img_map_cross, wxITEM_NORMAL,
                       _("Display a cross marker overlay"), _("Display a cross marker overlay on both frames"),
                       nullptr);
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), *_img_preferences, *_img_preferences, wxITEM_NORMAL,
                       _("Preferences"), _("Preferences"), nullptr);
    m_toolBar->Realize();

    // VroomGIS controls
    m_displayCtrlLeft = new vrViewerDisplay(m_panelGISLeft, wxID_ANY, wxColour(255, 255, 255));
    m_displayCtrlRight = new vrViewerDisplay(m_panelGISRight, wxID_ANY, wxColour(255, 255, 255));
    m_sizerGISLeft->Add(m_displayCtrlLeft, 1, wxEXPAND | wxALL, 0);
    m_sizerGISRight->Add(m_displayCtrlRight, 1, wxEXPAND | wxALL, 0);
    m_panelGIS->Layout();
    m_tocCtrlLeft = new vrViewerTOCList(m_scrolledWindowOptions, wxID_ANY);
    m_tocCtrlRight = new vrViewerTOCList(m_scrolledWindowOptions, wxID_ANY);
    m_sizerScrolledWindow->Insert(7, m_tocCtrlLeft->GetControl(), 1, wxEXPAND, 0);
    m_sizerScrolledWindow->Add(m_tocCtrlRight->GetControl(), 1, wxEXPAND, 0);
    m_sizerScrolledWindow->Fit(m_scrolledWindowOptions);

    m_layerManager = new vrLayerManager();
    m_viewerLayerManagerLeft = new vrViewerLayerManager(m_layerManager, this, m_displayCtrlLeft, m_tocCtrlLeft);
    m_viewerLayerManagerRight = new vrViewerLayerManager(m_layerManager, this, m_displayCtrlRight, m_tocCtrlRight);

    // Colorbars
    m_panelPredictorsColorbarLeft = new asPanelPredictorsColorbar(m_panelColorbarLeft, wxID_ANY, wxDefaultPosition,
                                                                  wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelPredictorsColorbarLeft->Layout();
    m_sizerColorbarLeft->Add(m_panelPredictorsColorbarLeft, 1, wxEXPAND, 0);
    m_panelColorbarLeft->Layout();

    m_panelPredictorsColorbarRight = new asPanelPredictorsColorbar(m_panelColorbarRight, wxID_ANY, wxDefaultPosition,
                                                                   wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    m_panelPredictorsColorbarRight->Layout();
    m_sizerColorbarRight->Add(m_panelPredictorsColorbarRight, 1, wxEXPAND, 0);
    m_panelColorbarRight->Layout();

    // Viewer
    m_predictorsManagerTarget = new asPredictorsManager(m_listPredictors, m_workspace, true);
    m_predictorsManagerAnalog = new asPredictorsManager(m_listPredictors, m_workspace);
    m_predictorsRenderer = new asPredictorsRenderer(this, m_layerManager, m_predictorsManagerTarget,
                                                    m_predictorsManagerAnalog, m_viewerLayerManagerLeft,
                                                    m_viewerLayerManagerRight);
    m_predictorsRenderer->LinkToColorbars(m_panelPredictorsColorbarLeft, m_panelPredictorsColorbarRight);

    // Menus
    m_menuTools->AppendCheckItem(asID_SET_SYNCRO_MODE, "Synchronize tools",
                                 "When set to true, browsing is synchronized on all display");
    m_menuTools->Check(asID_SET_SYNCRO_MODE, m_syncroTool);

    // Connect Events
    m_displayCtrlLeft->Connect(wxEVT_RIGHT_DOWN, wxMouseEventHandler(asFramePredictors::OnRightClick), nullptr, this);
    m_displayCtrlLeft->Connect(wxEVT_KEY_DOWN, wxKeyEventHandler(asFramePredictors::OnKeyDown), nullptr, this);
    m_displayCtrlLeft->Connect(wxEVT_KEY_UP, wxKeyEventHandler(asFramePredictors::OnKeyUp), nullptr, this);
    this->Connect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFramePredictors::OpenFramePreferences));

    // DND
    m_scrolledWindowOptions->SetDropTarget(new vroomDropFilesPredictors(this));

    // Bitmap
    m_bpButtonSwitchRight->SetBitmapLabel(*_img_arrow_right);
    m_bpButtonSwitchLeft->SetBitmapLabel(*_img_arrow_left);

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
        m_selectedTargetDate = 0;
        m_selectedAnalogDate = 0;
        UpdateMethodsList();
    }

    // GIS
    InitExtent();
    OpenDefaultLayers();
}

void asFramePredictors::UpdateMethodsList() {
    wxArrayString methods = m_forecastManager->GetMethodNamesWxArray();
    m_choiceMethod->Set(methods);
    m_selectedMethod = wxMin(m_selectedMethod, int(methods.Count()) - 1);
    m_choiceMethod->Select(m_selectedMethod);
    UpdateForecastList();
}

void asFramePredictors::UpdateForecastList() {
    wxArrayString forecasts = m_forecastManager->GetForecastNamesWxArray(m_selectedMethod);
    m_choiceForecast->Set(forecasts);
    m_selectedForecast = wxMin(m_selectedForecast, int(forecasts.Count()) - 1);
    m_choiceForecast->Select(m_selectedForecast);
    m_selectedPredictor = 0;
    UpdatePredictorsProperties();
    UpdatePredictorsList();
    UpdateTargetDatesList();
}

void asFramePredictors::UpdatePredictorsList() {
    asResultsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    vwxs predictorDataIds = forecast->GetPredictorDataIdsOper();
    vf predictorLevels = forecast->GetPredictorLevels();
    vf predictorHours = forecast->GetPredictorHours();
    wxArrayString dataListString;

    for (int i = 0; i < predictorDataIds.size(); ++i) {
        wxASSERT(predictorLevels.size() > i);
        wxASSERT(predictorHours.size() > i);
        if (int(predictorLevels[i]) == 0) {
            dataListString.Add(asStrF("%s %dh", predictorDataIds[i], int(predictorHours[i])));
        } else {
            dataListString.Add(asStrF("%s %d %dh", predictorDataIds[i], int(predictorLevels[i]), int(predictorHours[i])));
        }
    }

    m_listPredictors->Clear();
    m_listPredictors->Set(dataListString);
    m_listPredictors->Layout();
}

void asFramePredictors::UpdatePredictorsProperties() {
    asResultsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    m_predictorsManagerTarget->SetForecastDate(forecast->GetLeadTimeOrigin());
    m_predictorsManagerTarget->SetForecastTimeStepHours(forecast->GetForecastTimeStepHours());
    m_predictorsManagerTarget->SetLeadTimeNb(forecast->GetTargetDatesLength());
    m_predictorsManagerTarget->SetDatasetIds(forecast->GetPredictorDatasetIdsOper());
    m_predictorsManagerTarget->SetDataIds(forecast->GetPredictorDataIdsOper());
    m_predictorsManagerTarget->SetLevels(forecast->GetPredictorLevels());
    m_predictorsManagerTarget->SetHours(forecast->GetPredictorHours());
    m_predictorsManagerAnalog->SetDatasetIds(forecast->GetPredictorDatasetIdsArchive());
    m_predictorsManagerAnalog->SetDataIds(forecast->GetPredictorDataIdsArchive());
    m_predictorsManagerAnalog->SetLevels(forecast->GetPredictorLevels());
    m_predictorsManagerAnalog->SetHours(forecast->GetPredictorHours());
}

void asFramePredictors::UpdateTargetDatesList() {
    wxArrayString dates = m_forecastManager->GetTargetDatesWxArray(m_selectedMethod, m_selectedForecast);
    m_choiceTargetDates->Set(dates);
    m_selectedTargetDate = wxMin(m_selectedTargetDate, int(dates.Count()) - 1);
    m_choiceTargetDates->Select(m_selectedTargetDate);
    UpdateAnalogDatesList();
}

void asFramePredictors::UpdateAnalogDatesList() {
    asResultsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    a1f analogDates = forecast->GetAnalogsDates(m_selectedTargetDate);
    wxArrayString arrayAnalogDates;
    wxString format = forecast->GetDateFormatting();
    int rank = 0;
    for (float analogDate : analogDates) {
        rank++;
        wxString label = asStrF("%d - %s", rank, asTime::GetStringTime(analogDate, format));
        arrayAnalogDates.Add(label);
    }
    m_choiceAnalogDates->Set(arrayAnalogDates);
    m_selectedAnalogDate = wxMin(m_selectedAnalogDate, int(arrayAnalogDates.Count()) - 1);
    m_choiceAnalogDates->Select(m_selectedAnalogDate);
}

void asFramePredictors::InitExtent() {
    vrRealRect desiredExtent = getDesiredExtent();

    m_viewerLayerManagerLeft->InitializeExtent(desiredExtent);
    m_viewerLayerManagerRight->InitializeExtent(desiredExtent);
}

void asFramePredictors::OpenFramePreferences(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesViewer(this, m_workspace, asWINDOW_PREFERENCES);
    frame->Fit();
    frame->Show();
}

void asFramePredictors::OnSwitchRight(wxCommandEvent& event) {
    if (!m_displayPanelRight) return;

    Freeze();

    if (m_displayPanelLeft) {
        m_sizerGIS->Hide(m_panelRight, true);
        m_displayPanelRight = false;
    } else {
        m_sizerGIS->Show(m_panelLeft, true);
        m_sizerGIS->Show(m_panelRight, true);
        m_displayPanelLeft = true;
        m_displayPanelRight = true;
    }

    m_sizerGIS->Fit(m_panelGIS);
    Layout();
    Refresh();
    Thaw();
}

void asFramePredictors::OnSwitchLeft(wxCommandEvent& event) {
    if (!m_displayPanelLeft) return;

    Freeze();

    if (m_displayPanelRight) {
        m_sizerGIS->Hide(m_panelLeft, true);
        m_displayPanelLeft = false;
    } else {
        m_sizerGIS->Show(m_panelLeft, true);
        m_sizerGIS->Show(m_panelRight, true);
        m_displayPanelLeft = true;
        m_displayPanelRight = true;
    }

    m_sizerGIS->Fit(m_panelGIS);
    Layout();
    Refresh();
    Thaw();
}

void asFramePredictors::OnPredictorSelectionChange(wxCommandEvent& event) {
    m_selectedPredictor = event.GetInt();
    m_predictorsManagerTarget->NeedsDataReload();
    m_predictorsManagerAnalog->NeedsDataReload();
    UpdateLayers();
}

void asFramePredictors::OnMethodChange(wxCommandEvent& event) {
    m_selectedMethod = event.GetInt();
    m_predictorsManagerTarget->NeedsDataReload();
    m_predictorsManagerAnalog->NeedsDataReload();
    UpdateForecastList();
    UpdateLayers();
}

void asFramePredictors::OnForecastChange(wxCommandEvent& event) {
    m_selectedForecast = event.GetInt();
    m_predictorsManagerTarget->NeedsDataReload();
    m_predictorsManagerAnalog->NeedsDataReload();
    UpdateTargetDatesList();
    UpdateLayers();
}

void asFramePredictors::OnTargetDateChange(wxCommandEvent& event) {
    m_selectedTargetDate = event.GetInt();
    m_predictorsManagerTarget->NeedsDataReload();
    m_predictorsManagerAnalog->NeedsDataReload();
    UpdateAnalogDatesList();
    UpdateLayers();
}

void asFramePredictors::OnAnalogDateChange(wxCommandEvent& event) {
    m_selectedAnalogDate = event.GetInt();
    m_predictorsManagerTarget->NeedsDataReload();
    m_predictorsManagerAnalog->NeedsDataReload();
    UpdateLayers();
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

            auto renderContinents1 = new vrRenderVector();
            renderContinents1->SetTransparency(continentsTransp);
            renderContinents1->SetColorPen(colorContinents);
            renderContinents1->SetColorBrush(colorContinents);
            renderContinents1->SetBrushStyle(wxBRUSHSTYLE_SOLID);
            renderContinents1->SetSize(continentsSize);
            auto renderContinents2 = new vrRenderVector();
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

    // LatLong
    if (wxFileName::FileExists(latLongFilePath)) {
        if (m_layerManager->Open(wxFileName(latLongFilePath))) {
            long latLongTransp = pConfig->ReadLong("/GIS/LayerLatLongTransp", 80);
            long latLongColor = pConfig->ReadLong("/GIS/LayerLatLongColor", (long)0xff999999);
            wxColour colorLatLong;
            colorLatLong.SetRGB((wxUint32)latLongColor);
            long latLongSize = pConfig->ReadLong("/GIS/LayerLatLongSize", 1);
            bool latLongVisibility = pConfig->ReadBool("/GIS/LayerLatLongVisibility", true);

            auto renderLatLong1 = new vrRenderVector();
            renderLatLong1->SetTransparency(latLongTransp);
            renderLatLong1->SetColorPen(colorLatLong);
            renderLatLong1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderLatLong1->SetSize(latLongSize);
            auto renderLatLong2 = new vrRenderVector();
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

            auto renderGeogrid1 = new vrRenderVector();
            renderGeogrid1->SetTransparency(geogridTransp);
            renderGeogrid1->SetColorPen(colorGeogrid);
            renderGeogrid1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderGeogrid1->SetSize(geogridSize);
            auto renderGeogrid2 = new vrRenderVector();
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

    // Countries
    if (wxFileName::FileExists(countriesFilePath)) {
        if (m_layerManager->Open(wxFileName(countriesFilePath))) {
            long countriesTransp = pConfig->ReadLong("/GIS/LayerCountriesTransp", 0);
            long countriesColor = pConfig->ReadLong("/GIS/LayerCountriesColor", (long)0x77999999);
            wxColour colorCountries;
            colorCountries.SetRGB((wxUint32)countriesColor);
            long countriesSize = pConfig->ReadLong("/GIS/LayerCountriesSize", 1);
            bool countriesVisibility = pConfig->ReadBool("/GIS/LayerCountriesVisibility", true);

            auto renderCountries1 = new vrRenderVector();
            renderCountries1->SetTransparency(countriesTransp);
            renderCountries1->SetColorPen(colorCountries);
            renderCountries1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderCountries1->SetSize(countriesSize);
            auto renderCountries2 = new vrRenderVector();
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

    m_viewerLayerManagerLeft->FreezeEnd();
    m_viewerLayerManagerRight->FreezeEnd();
}

bool asFramePredictors::OpenLayers(const wxArrayString& names) {
    // Open files
    for (unsigned int i = 0; i < names.GetCount(); i++) {
        if (!m_layerManager->Open(wxFileName(names.Item(i)))) {
            wxLogError(_("The layer could not be opened."));
            return false;
        }
    }

// Get files
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Enter();
#endif
    m_viewerLayerManagerLeft->FreezeBegin();
    m_viewerLayerManagerRight->FreezeBegin();
    for (unsigned int i = 0; i < names.GetCount(); i++) {
        vrLayer* layer = m_layerManager->GetLayer(wxFileName(names.Item(i)));
        wxASSERT(layer);

        // Add files to the viewer
        m_viewerLayerManagerLeft->Add(1, layer, nullptr);
        m_viewerLayerManagerRight->Add(1, layer, nullptr);
    }
    m_viewerLayerManagerLeft->FreezeEnd();
    m_viewerLayerManagerRight->FreezeEnd();
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Leave();
#endif
    return true;
}

void asFramePredictors::OnOpenLayer(wxCommandEvent& event) {
    vrDrivers drivers;
    wxFileDialog myFileDlg(this, _("Select GIS layers"), wxEmptyString, wxEmptyString, drivers.GetWildcards(),
                           wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_MULTIPLE | wxFD_CHANGE_DIR);

    wxArrayString pathsFileName;

    // Try to open files
    if (myFileDlg.ShowModal() == wxID_OK) {
        myFileDlg.GetPaths(pathsFileName);
        wxASSERT(pathsFileName.GetCount() > 0);

        OpenLayers(pathsFileName);
    }
}

void asFramePredictors::OnCloseLayer(wxCommandEvent& event) {
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Enter();
#endif

    wxArrayString layersName;
    for (int i = 0; i < m_viewerLayerManagerLeft->GetCount(); i++) {
        vrRenderer* renderer = m_viewerLayerManagerLeft->GetRenderer(i);
        wxASSERT(renderer);
        layersName.Add(renderer->GetLayer()->GetDisplayName().GetFullName());
    }

    if (layersName.IsEmpty()) {
        wxLogError("No layer opened, nothing to close.");
#if defined(__WIN32__)
        m_critSectionViewerLayerManager.Leave();
#endif
        return;
    }

    wxMultiChoiceDialog choiceDlg(this, "Select Layer(s) to close.", "Close layer(s)", layersName);
    if (choiceDlg.ShowModal() != wxID_OK) {
#if defined(__WIN32__)
        m_critSectionViewerLayerManager.Leave();
#endif
        return;
    }

    wxArrayInt layerToRemoveIndex = choiceDlg.GetSelections();
    if (layerToRemoveIndex.IsEmpty()) {
        wxLogWarning(_("Nothing selected, no layer will be closed."));
#if defined(__WIN32__)
        m_critSectionViewerLayerManager.Leave();
#endif
        return;
    }

    // Removing layer(s)
    m_viewerLayerManagerLeft->FreezeBegin();
    m_viewerLayerManagerRight->FreezeBegin();

    for (int j = (signed)layerToRemoveIndex.GetCount() - 1; j >= 0; j--) {
        // Remove from viewer manager (TOC and Display)
        vrRenderer* rendererLeft = m_viewerLayerManagerLeft->GetRenderer(layerToRemoveIndex.Item(j));
        vrLayer* layer = rendererLeft->GetLayer();
        wxASSERT(rendererLeft);
        m_viewerLayerManagerLeft->Remove(rendererLeft);

        vrRenderer* rendererRight = m_viewerLayerManagerRight->GetRenderer(layerToRemoveIndex.Item(j));
        wxASSERT(rendererRight);
        m_viewerLayerManagerRight->Remove(rendererRight);

        // Close layer (not used anymore);
        m_layerManager->Close(layer);
    }
    m_viewerLayerManagerLeft->FreezeEnd();
    m_viewerLayerManagerRight->FreezeEnd();
#if defined(__WIN32__)
    m_critSectionViewerLayerManager.Leave();
#endif
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

void asFramePredictors::OnSyncroToolSwitch(wxCommandEvent& event) {
    m_syncroTool = GetMenuBar()->IsChecked(asID_SET_SYNCRO_MODE);
}

void asFramePredictors::OnToolZoomIn(wxCommandEvent& event) {
    m_displayCtrlLeft->SetToolZoom();
    m_displayCtrlRight->SetToolZoom();
}

void asFramePredictors::OnToolZoomOut(wxCommandEvent& event) {
    m_displayCtrlLeft->SetToolZoomOut();
    m_displayCtrlRight->SetToolZoomOut();
}

void asFramePredictors::OnToolPan(wxCommandEvent& event) {
    m_displayCtrlLeft->SetToolPan();
    m_displayCtrlRight->SetToolPan();
}

void asFramePredictors::OnToolSight(wxCommandEvent& event) {
    m_displayCtrlLeft->SetToolSight();
    m_displayCtrlRight->SetToolSight();
}

void asFramePredictors::OnToolZoomToFit(wxCommandEvent& event) {
    vrRealRect desiredExtent = getDesiredExtent();

    if (m_displayPanelLeft) {
        m_viewerLayerManagerLeft->InitializeExtent(desiredExtent);
        ReloadViewerLayerManagerLeft();
    }
    if (m_displayPanelRight) {
        m_viewerLayerManagerRight->InitializeExtent(desiredExtent);
        ReloadViewerLayerManagerRight();
    }
}

vrRealRect asFramePredictors::getDesiredExtent() const {
    vf extent = m_forecastManager->GetMaxExtent();
    float width = extent[1] - extent[0];
    float height = extent[2] - extent[3];
    float marginWidth = 0.5f * width;
    float marginHeight = 0.5f * height;

    vrRealRect desiredExtent;
    desiredExtent.m_x = extent[0] - marginWidth;
    desiredExtent.m_width = width + 2 * marginWidth;
    desiredExtent.m_y = extent[3] + marginHeight;
    desiredExtent.m_height = height - 2 * marginHeight;

    return desiredExtent;
}

void asFramePredictors::OnToolAction(wxCommandEvent& event) {
    auto msg = static_cast<vrDisplayToolMessage*>(event.GetClientData());
    wxASSERT(msg);

    if (msg->m_evtType == vrEVT_TOOL_ZOOM) {
        // Get rectangle
        vrCoordinate* coord = msg->m_parentManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        coord->ConvertFromPixels(msg->m_rect, realRect);
        wxASSERT(realRect.IsOk());

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        if (!m_syncroTool) {
#if defined(__WIN32__)
            auto thread = new asThreadViewerLayerManagerZoomIn(msg->m_parentManager, &m_critSectionViewerLayerManager,
                                                               fittedRect);
            ThreadsManager().AddThread(thread);
#else
            msg->m_parentManager->Zoom(fittedRect);
#endif
        } else {
            if (m_displayPanelLeft) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomIn(m_viewerLayerManagerLeft,
                                                                   &m_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                m_viewerLayerManagerLeft->Zoom(fittedRect);
#endif
            }
            if (m_displayPanelRight) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomIn(m_viewerLayerManagerRight,
                                                                   &m_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                m_viewerLayerManagerRight->Zoom(fittedRect);
#endif
            }
        }
    } else if (msg->m_evtType == vrEVT_TOOL_ZOOMOUT) {
        // Getting rectangle
        vrCoordinate* coord = msg->m_parentManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        // Get real rectangle
        vrRealRect realRect;
        coord->ConvertFromPixels(msg->m_rect, realRect);
        wxASSERT(realRect.IsOk());

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);
        wxASSERT(fittedRect.IsOk());

        if (!m_syncroTool) {
#if defined(__WIN32__)
            auto thread = new asThreadViewerLayerManagerZoomOut(msg->m_parentManager, &m_critSectionViewerLayerManager,
                                                                fittedRect);
            ThreadsManager().AddThread(thread);
#else
            msg->m_parentManager->ZoomOut(fittedRect);
#endif
        } else {
            if (m_displayPanelLeft) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomOut(m_viewerLayerManagerLeft,
                                                                    &m_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                m_viewerLayerManagerLeft->ZoomOut(fittedRect);
#endif
            }
            if (m_displayPanelRight) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomOut(m_viewerLayerManagerRight,
                                                                    &m_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                m_viewerLayerManagerRight->ZoomOut(fittedRect);
#endif
            }
        }
    } else if (msg->m_evtType == vrEVT_TOOL_PAN) {
        vrCoordinate* coord = msg->m_parentManager->GetDisplay()->GetCoordinate();
        wxASSERT(coord);

        wxPoint movedPos = msg->m_position;
        wxPoint2DDouble myMovedRealPt;
        if (!coord->ConvertFromPixels(movedPos, myMovedRealPt)) {
            wxLogError("Error converting point : %d, %d to real coordinate", movedPos.x, movedPos.y);
            wxDELETE(msg);
            return;
        }

        vrRealRect actExtent = coord->GetExtent();
        actExtent.MoveLeftTopTo(myMovedRealPt);

        if (!m_syncroTool) {
            coord->SetExtent(actExtent);
            msg->m_parentManager->Reload();
            ReloadViewerLayerManagerLeft();
            ReloadViewerLayerManagerRight();
        } else {
            if (m_displayPanelLeft) {
                m_viewerLayerManagerLeft->GetDisplay()->GetCoordinate()->SetExtent(actExtent);
                ReloadViewerLayerManagerLeft();
            }
            if (m_displayPanelRight) {
                m_viewerLayerManagerRight->GetDisplay()->GetCoordinate()->SetExtent(actExtent);
                ReloadViewerLayerManagerRight();
            }
        }

    } else if (msg->m_evtType == vrEVT_TOOL_SIGHT) {
        vrViewerLayerManager* invertedMgr = m_viewerLayerManagerLeft;
        if (invertedMgr == msg->m_parentManager) {
            invertedMgr = m_viewerLayerManagerRight;
        }

        switch (msg->m_mouseStatus) {
            case vrMOUSE_DOWN:
            case vrMOUSE_MOVE: {
                wxClientDC dc(invertedMgr->GetDisplay());
                wxDCOverlay overlayDc(m_overlay, &dc);
                overlayDc.Clear();
                dc.SetPen(*wxRED_PEN);
                dc.CrossHair(msg->m_position);
            } break;
            case vrMOUSE_UP: {
                wxClientDC dc(invertedMgr->GetDisplay());
                wxDCOverlay overlayDc(m_overlay, &dc);
                overlayDc.Clear();
            }
                m_overlay.Reset();
                break;
            case vrMOUSE_UNKNOWN:
                wxLogError("Operation not recognized.");
                break;
        }
    } else {
        wxLogError("Operation not yet supported.");
    }

    wxDELETE(msg);
}

void asFramePredictors::UpdateLayers() {
    // Check that elements are selected
    if ((m_selectedMethod == -1) || (m_selectedForecast == -1) || (m_selectedTargetDate == -1) ||
        (m_selectedAnalogDate == -1) || (m_selectedPredictor == -1)) {
        return;
    }

    // Get dates
    asResultsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    a1f targetDates = forecast->GetTargetDates();
    double targetDate = targetDates[m_selectedTargetDate];
    a1f analogDates = forecast->GetAnalogsDates(m_selectedTargetDate);
    double analogDate = analogDates[m_selectedAnalogDate];

    // Get domain
    if (forecast->GetPredictorLonMin().size() == 0) {
        wxLogError(_("Only forecasts of AtmoSwing 3+ can be visualized here."));
        return;
    }
    vf domain;
    domain.push_back(forecast->GetPredictorLonMin()[m_selectedPredictor]);
    domain.push_back(forecast->GetPredictorLonMax()[m_selectedPredictor]);
    domain.push_back(forecast->GetPredictorLatMin()[m_selectedPredictor]);
    domain.push_back(forecast->GetPredictorLatMax()[m_selectedPredictor]);

    m_predictorsManagerTarget->SetDate(targetDate);
    m_predictorsManagerAnalog->SetDate(analogDate);
    m_predictorsRenderer->Redraw(domain);
}

void asFramePredictors::ReloadViewerLayerManagerLeft() {
#if defined(__WIN32__)
    auto thread = new asThreadViewerLayerManagerReload(m_viewerLayerManagerLeft, &m_critSectionViewerLayerManager);
    ThreadsManager().AddThread(thread);
#else
    m_viewerLayerManagerLeft->Reload();
#endif
}

void asFramePredictors::ReloadViewerLayerManagerRight() {
#if defined(__WIN32__)
    auto thread = new asThreadViewerLayerManagerReload(m_viewerLayerManagerRight, &m_critSectionViewerLayerManager);
    ThreadsManager().AddThread(thread);
#else
    m_viewerLayerManagerRight->Reload();
#endif
}

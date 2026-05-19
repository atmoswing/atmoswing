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

#include "asBitmaps.h"
#include "asFramePreferencesViewer.h"
#if defined(__WIN32__)
#include "asThreadViewerLayerManagerReload.h"
#include "asThreadViewerLayerManagerZoomIn.h"
#include "asThreadViewerLayerManagerZoomOut.h"
#include "asThreadsManager.h"
#endif

#include <proj.h>
#include <wx/colour.h>
#include <wx/filename.h>

#include "asPredictorsManager.h"

BEGIN_EVENT_TABLE(asFramePredictors, wxFrame)
EVT_MENU(wxID_OPEN, asFramePredictors::OnOpenLayer)
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

vroomDropFilesPredictors::vroomDropFilesPredictors(asFramePredictors* parent) {
    wxASSERT(parent);
    _loaderFrame = parent;
}

bool vroomDropFilesPredictors::OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) {
    if (filenames.GetCount() == 0) return false;

    _loaderFrame->OpenLayers(filenames);
    return true;
}

asFramePredictors::asFramePredictors(wxWindow* parent, asForecastManager* forecastManager, asWorkspace* workspace,
                                     int methodRow, int forecastRow, wxWindowID id)
    : asFramePredictorsVirtual(parent, id),
      _forecastManager(forecastManager),
      _workspace(workspace),
      _selectedMethod(methodRow),
      _selectedForecast(forecastRow),
      _selectedTargetDate(-1),
      _selectedAnalogDate(-1),
      _selectedPredictor(-1),
      _syncroTool(true),
      _displayPanelLeft(true),
      _displayPanelRight(true) {
    this->SetLabel(_("Predictors overview"));

    _selectedForecast = wxMax(_selectedForecast, 0);

    // Toolbar
    _toolBar->AddTool(asID_ZOOM_IN, wxT("Zoom in"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_ZOOM_IN), wxNullBitmap,
                       wxITEM_NORMAL, _("Zoom in"), _("Zoom in"), nullptr);
    _toolBar->AddTool(asID_ZOOM_OUT, wxT("Zoom out"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_ZOOM_OUT),
                       wxNullBitmap, wxITEM_NORMAL, _("Zoom out"), _("Zoom out"), nullptr);
    _toolBar->AddTool(asID_PAN, wxT("Pan"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_MOVE), wxNullBitmap,
                       wxITEM_NORMAL, _("Pan the map"), _("Move the map by panning"), nullptr);
    _toolBar->AddTool(asID_ZOOM_FIT, wxT("Fit"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_FIT), wxNullBitmap,
                       wxITEM_NORMAL, _("Zoom to visible layers"),
                       _("Zoom view to the full extent of all visible layers"), nullptr);
    _toolBar->AddTool(asID_CROSS_MARKER, wxT("Marker overlay"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::MAP_CROSS),
                       wxNullBitmap, wxITEM_NORMAL, _("Display a cross marker overlay"),
                       _("Display a cross marker overlay on both frames"), nullptr);
    _toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::PREFERENCES),
                       wxNullBitmap, wxITEM_NORMAL, _("Preferences"), _("Preferences"), nullptr);
    _toolBar->Realize();

    // VroomGIS controls
    _displayCtrlLeft = new vrViewerDisplay(_panelGISLeft, wxID_ANY, wxColour(255, 255, 255));
    _displayCtrlRight = new vrViewerDisplay(_panelGISRight, wxID_ANY, wxColour(255, 255, 255));
    _sizerGISLeft->Add(_displayCtrlLeft, 1, wxEXPAND | wxALL, 0);
    _sizerGISRight->Add(_displayCtrlRight, 1, wxEXPAND | wxALL, 0);
    _panelGIS->Layout();
    _tocCtrlLeft = new vrViewerTOCList(_scrolledWindowOptions, wxID_ANY);
    _tocCtrlRight = new vrViewerTOCList(_scrolledWindowOptions, wxID_ANY);
    _sizerScrolledWindow->Insert(7, _tocCtrlLeft->GetControl(), 1, wxEXPAND, 0);
    _sizerScrolledWindow->Add(_tocCtrlRight->GetControl(), 1, wxEXPAND, 0);
    _sizerScrolledWindow->Fit(_scrolledWindowOptions);

    _layerManager = new vrLayerManager();
    _viewerLayerManagerLeft = new vrViewerLayerManager(_layerManager, this, _displayCtrlLeft, _tocCtrlLeft);
    _viewerLayerManagerRight = new vrViewerLayerManager(_layerManager, this, _displayCtrlRight, _tocCtrlRight);

    // Colorbars
    _panelPredictorsColorbarLeft = new asPanelPredictorsColorbar(_panelColorbarLeft, wxID_ANY, wxDefaultPosition,
                                                                  wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    _panelPredictorsColorbarLeft->Layout();
    _sizerColorbarLeft->Add(_panelPredictorsColorbarLeft, 1, wxEXPAND, 0);
    _panelColorbarLeft->Layout();

    _panelPredictorsColorbarRight = new asPanelPredictorsColorbar(_panelColorbarRight, wxID_ANY, wxDefaultPosition,
                                                                   wxDefaultSize, wxNO_BORDER | wxTAB_TRAVERSAL);
    _panelPredictorsColorbarRight->Layout();
    _sizerColorbarRight->Add(_panelPredictorsColorbarRight, 1, wxEXPAND, 0);
    _panelColorbarRight->Layout();

    // Viewer
    _predictorsManagerTarget = new asPredictorsManager(_workspace, true);
    _predictorsManagerAnalog = new asPredictorsManager(_workspace);
    _predictorsRenderer = new asPredictorsRenderer(this, _layerManager, _predictorsManagerTarget,
                                                    _predictorsManagerAnalog, _viewerLayerManagerLeft,
                                                    _viewerLayerManagerRight);
    _predictorsRenderer->LinkToColorbars(_panelPredictorsColorbarLeft, _panelPredictorsColorbarRight);

    // Menus
    _menuTools->AppendCheckItem(asID_SET_SYNCRO_MODE, _("Synchronize tools"),
                                 _("When set to true, browsing is synchronized on all display"));
    _menuTools->Check(asID_SET_SYNCRO_MODE, _syncroTool);

    // Connect Events
    _displayCtrlLeft->Connect(wxEVT_RIGHT_DOWN, wxMouseEventHandler(asFramePredictors::OnRightClick), nullptr, this);
    _displayCtrlLeft->Connect(wxEVT_KEY_DOWN, wxKeyEventHandler(asFramePredictors::OnKeyDown), nullptr, this);
    _displayCtrlLeft->Connect(wxEVT_KEY_UP, wxKeyEventHandler(asFramePredictors::OnKeyUp), nullptr, this);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFramePredictors::OpenFramePreferences, this, asID_PREFERENCES);

    // DND
    _scrolledWindowOptions->SetDropTarget(new vroomDropFilesPredictors(this));

    // Bitmap
    _bpButtonSwitchRight->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::ARROW_RIGHT, wxSize(10, 20)));
    _bpButtonSwitchLeft->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::ARROW_LEFT, wxSize(10, 20)));

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFramePredictors::~asFramePredictors() {
    // Disconnect Events
    _displayCtrlLeft->Disconnect(wxEVT_RIGHT_DOWN, wxMouseEventHandler(asFramePredictors::OnRightClick), nullptr, this);
    _displayCtrlLeft->Disconnect(wxEVT_KEY_DOWN, wxKeyEventHandler(asFramePredictors::OnKeyDown), nullptr, this);
    _displayCtrlLeft->Disconnect(wxEVT_KEY_UP, wxKeyEventHandler(asFramePredictors::OnKeyUp), nullptr, this);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFramePredictors::OpenFramePreferences, this, asID_PREFERENCES);

    wxDELETE(_layerManager);
}

void asFramePredictors::Init() {
    if (_forecastManager->GetMethodsNb() > 0) {
        _selectedTargetDate = 0;
        _selectedAnalogDate = 0;
        UpdateMethodsList();
    }

    // GIS
    InitExtent();
    OpenDefaultLayers();
}

void asFramePredictors::UpdateMethodsList() {
    wxArrayString methods = _forecastManager->GetMethodNamesWxArray();
    _choiceMethod->Set(methods);
    _selectedMethod = wxMin(_selectedMethod, int(methods.Count()) - 1);
    _choiceMethod->Select(_selectedMethod);
    UpdateForecastList();
}

void asFramePredictors::UpdateForecastList() {
    wxArrayString forecasts = _forecastManager->GetForecastNamesWxArray(_selectedMethod);
    _choiceForecast->Set(forecasts);
    _selectedForecast = wxMin(_selectedForecast, int(forecasts.Count()) - 1);
    _choiceForecast->Select(_selectedForecast);
    _selectedPredictor = 0;
    UpdatePredictorsProperties();
    UpdatePredictorsList();
    UpdateTargetDatesList();
}

void asFramePredictors::UpdatePredictorsList() {
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);
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

    _listPredictors->Clear();
    _listPredictors->Set(dataListString);
    _listPredictors->Layout();
}

void asFramePredictors::UpdatePredictorsProperties() {
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);

    _predictorsManagerTarget->SetForecastDate(forecast->GetLeadTimeOrigin());
    _predictorsManagerTarget->SetForecastTimeStepHours(forecast->GetForecastTimeStepHours());
    _predictorsManagerTarget->SetDatasetIds(forecast->GetPredictorDatasetIdsOper());
    _predictorsManagerTarget->SetDataIds(forecast->GetPredictorDataIdsOper());
    _predictorsManagerTarget->SetLevels(forecast->GetPredictorLevels());
    _predictorsManagerTarget->SetHours(forecast->GetPredictorHours());
    _predictorsManagerAnalog->SetDatasetIds(forecast->GetPredictorDatasetIdsArchive());
    _predictorsManagerAnalog->SetDataIds(forecast->GetPredictorDataIdsArchive());
    _predictorsManagerAnalog->SetLevels(forecast->GetPredictorLevels());
    _predictorsManagerAnalog->SetHours(forecast->GetPredictorHours());
}

void asFramePredictors::UpdateTargetDatesList() {
    wxArrayString dates = _forecastManager->GetTargetDatesWxArray(_selectedMethod, _selectedForecast);
    _choiceTargetDates->Set(dates);
    _selectedTargetDate = wxMin(_selectedTargetDate, int(dates.Count()) - 1);
    _choiceTargetDates->Select(_selectedTargetDate);
    UpdateAnalogDatesList();
}

void asFramePredictors::UpdateAnalogDatesList() {
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);
    a1f analogDates = forecast->GetAnalogsDates(_selectedTargetDate);
    wxArrayString arrayAnalogDates;
    wxString format = forecast->GetDateFormatting();
    int rank = 0;
    for (float analogDate : analogDates) {
        rank++;
        wxString label = asStrF("%d - %s", rank, asTime::GetStringTime(analogDate, format));
        arrayAnalogDates.Add(label);
    }
    _choiceAnalogDates->Set(arrayAnalogDates);
    _selectedAnalogDate = wxMin(_selectedAnalogDate, int(arrayAnalogDates.Count()) - 1);
    _choiceAnalogDates->Select(_selectedAnalogDate);
}

void asFramePredictors::InitExtent() {
    vrRealRect desiredExtent = GetDesiredExtent();

    _viewerLayerManagerLeft->InitializeExtent(desiredExtent);
    _viewerLayerManagerRight->InitializeExtent(desiredExtent);
}

void asFramePredictors::OpenFramePreferences(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesViewer(this, _workspace, asWINDOW_PREFERENCES);
    frame->Fit();
    frame->Show();
}

void asFramePredictors::SwitchPanelRight() {
    if (!_displayPanelRight) return;

    Freeze();

    if (_displayPanelLeft) {
        _sizerGIS->Hide(_panelRight, true);
        _displayPanelRight = false;
    } else {
        _sizerGIS->Show(_panelLeft, true);
        _sizerGIS->Show(_panelRight, true);
        _displayPanelLeft = true;
        _displayPanelRight = true;
    }

    _sizerGIS->Fit(_panelGIS);
    Layout();
    Refresh();
    Thaw();
}

void asFramePredictors::OnSwitchRight(wxCommandEvent& event) {
    SwitchPanelRight();
}

void asFramePredictors::SwitchPanelLeft() {
    if (!_displayPanelLeft) return;

    Freeze();

    if (_displayPanelRight) {
        _sizerGIS->Hide(_panelLeft, true);
        _displayPanelLeft = false;
    } else {
        _sizerGIS->Show(_panelLeft, true);
        _sizerGIS->Show(_panelRight, true);
        _displayPanelLeft = true;
        _displayPanelRight = true;
    }

    _sizerGIS->Fit(_panelGIS);
    Layout();
    Refresh();
    Thaw();
}

void asFramePredictors::OnSwitchLeft(wxCommandEvent& event) {
    SwitchPanelLeft();
}

void asFramePredictors::OnPredictorSelectionChange(wxCommandEvent& event) {
    _selectedPredictor = event.GetInt();
    _predictorsManagerTarget->NeedsDataReload();
    _predictorsManagerAnalog->NeedsDataReload();
    UpdateLayers();
}

void asFramePredictors::OnMethodChange(wxCommandEvent& event) {
    _selectedMethod = event.GetInt();
    _predictorsManagerTarget->NeedsDataReload();
    _predictorsManagerAnalog->NeedsDataReload();
    UpdateForecastList();
    UpdateLayers();
}

void asFramePredictors::OnForecastChange(wxCommandEvent& event) {
    _selectedForecast = event.GetInt();
    _predictorsManagerTarget->NeedsDataReload();
    _predictorsManagerAnalog->NeedsDataReload();
    UpdateTargetDatesList();
    UpdateLayers();
}

void asFramePredictors::OnTargetDateChange(wxCommandEvent& event) {
    _selectedTargetDate = event.GetInt();
    _predictorsManagerTarget->NeedsDataReload();
    _predictorsManagerAnalog->NeedsDataReload();
    UpdateAnalogDatesList();
    UpdateLayers();
}

void asFramePredictors::OnAnalogDateChange(wxCommandEvent& event) {
    _selectedAnalogDate = event.GetInt();
    _predictorsManagerTarget->NeedsDataReload();
    _predictorsManagerAnalog->NeedsDataReload();
    UpdateLayers();
}

void asFramePredictors::OpenDefaultLayers() {
    // Default paths
    wxConfigBase* pConfig = wxFileConfig::Get();
    wxString dirData = asConfig::GetShareDir();
    wxString gisData = dirData + DS + "atmoswing" + DS + "gis" + DS + "shapefiles";

    wxString continentsFilePath = pConfig->Read("/GIS/LayerContinentsFilePath", gisData + DS + "continents.shp");
    wxString countriesFilePath = pConfig->Read("/GIS/LayerCountriesFilePath", gisData + DS + "countries.shp");
    wxString latLongFilePath = pConfig->Read("/GIS/LayerLatLongFilePath", gisData + DS + "latlong.shp");
    wxString geogridFilePath = pConfig->Read("/GIS/LayerGeogridFilePath", gisData + DS + "geogrid.shp");

    // Try to open layers
    _viewerLayerManagerLeft->FreezeBegin();
    _viewerLayerManagerRight->FreezeBegin();
    vrLayer* layer;

    // Continents
    if (wxFileName::FileExists(continentsFilePath)) {
        if (_layerManager->Open(wxFileName(continentsFilePath))) {
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

            layer = _layerManager->GetLayer(wxFileName(continentsFilePath));
            wxASSERT(layer);
            _viewerLayerManagerLeft->Add(-1, layer, renderContinents1, nullptr, continentsVisibility);
            _viewerLayerManagerRight->Add(-1, layer, renderContinents2, nullptr, continentsVisibility);
        } else {
            wxLogError(_("The Continents layer file %s cound not be opened."), continentsFilePath.c_str());
        }
    } else {
        wxLogError(_("The Continents layer file %s cound not be found."), continentsFilePath.c_str());
    }

    // LatLong
    if (wxFileName::FileExists(latLongFilePath)) {
        if (_layerManager->Open(wxFileName(latLongFilePath))) {
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

            layer = _layerManager->GetLayer(wxFileName(latLongFilePath));
            wxASSERT(layer);
            _viewerLayerManagerLeft->Add(-1, layer, renderLatLong1, nullptr, latLongVisibility);
            _viewerLayerManagerRight->Add(-1, layer, renderLatLong2, nullptr, latLongVisibility);
        } else {
            wxLogError(_("The LatLong layer file %s cound not be opened."), latLongFilePath.c_str());
        }
    } else {
        wxLogError(_("The LatLong layer file %s cound not be found."), latLongFilePath.c_str());
    }

    // Geogrid
    if (wxFileName::FileExists(geogridFilePath)) {
        if (_layerManager->Open(wxFileName(geogridFilePath))) {
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

            layer = _layerManager->GetLayer(wxFileName(geogridFilePath));
            wxASSERT(layer);
            _viewerLayerManagerLeft->Add(-1, layer, renderGeogrid1, nullptr, geogridVisibility);
            _viewerLayerManagerRight->Add(-1, layer, renderGeogrid2, nullptr, geogridVisibility);
        } else {
            wxLogError(_("The Geogrid layer file %s cound not be opened."), geogridFilePath.c_str());
        }
    } else {
        wxLogError(_("The Geogrid layer file %s cound not be found."), geogridFilePath.c_str());
    }

    // Countries
    if (wxFileName::FileExists(countriesFilePath)) {
        if (_layerManager->Open(wxFileName(countriesFilePath))) {
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

            layer = _layerManager->GetLayer(wxFileName(countriesFilePath));
            wxASSERT(layer);
            _viewerLayerManagerLeft->Add(-1, layer, renderCountries1, nullptr, countriesVisibility);
            _viewerLayerManagerRight->Add(-1, layer, renderCountries2, nullptr, countriesVisibility);
        } else {
            wxLogError(_("The Countries layer file %s cound not be opened."), countriesFilePath.c_str());
        }
    } else {
        wxLogError(_("The Countries layer file %s cound not be found."), countriesFilePath.c_str());
    }

    _viewerLayerManagerLeft->FreezeEnd();
    _viewerLayerManagerRight->FreezeEnd();
}

bool asFramePredictors::OpenLayers(const wxArrayString& names) {
    // Open files
    for (unsigned int i = 0; i < names.GetCount(); i++) {
        if (!_layerManager->Open(wxFileName(names.Item(i)))) {
            wxLogError(_("The layer could not be opened."));
            return false;
        }
    }

// Get files
#if defined(__WIN32__)
    _critSectionViewerLayerManager.Enter();
#endif
    _viewerLayerManagerLeft->FreezeBegin();
    _viewerLayerManagerRight->FreezeBegin();
    for (unsigned int i = 0; i < names.GetCount(); i++) {
        vrLayer* layer = _layerManager->GetLayer(wxFileName(names.Item(i)));
        wxASSERT(layer);

        // Add files to the viewer
        _viewerLayerManagerLeft->Add(1, layer, nullptr);
        _viewerLayerManagerRight->Add(1, layer, nullptr);
    }
    _viewerLayerManagerLeft->FreezeEnd();
    _viewerLayerManagerRight->FreezeEnd();
#if defined(__WIN32__)
    _critSectionViewerLayerManager.Leave();
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

void asFramePredictors::OnKeyDown(wxKeyEvent& event) {
    _keyBoardState = wxKeyboardState(event.ControlDown(), event.ShiftDown(), event.AltDown(), event.MetaDown());
    if (_keyBoardState.GetModifiers() != wxMOD_CMD) {
        event.Skip();
        return;
    }

    const vrDisplayTool* tool = _displayCtrlLeft->GetTool();
    if (!tool) {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_IN) {
        _displayCtrlLeft->SetToolZoomOut();
        _displayCtrlRight->SetToolZoomOut();
    }
    event.Skip();
}

void asFramePredictors::OnKeyUp(wxKeyEvent& event) {
    if (_keyBoardState.GetModifiers() != wxMOD_CMD) {
        event.Skip();
        return;
    }

    const vrDisplayTool* tool = _displayCtrlLeft->GetTool();
    if (!tool) {
        event.Skip();
        return;
    }

    if (tool->GetID() == wxID_ZOOM_OUT || tool->GetID() == wxID_ZOOM_IN) {
        _displayCtrlLeft->SetToolZoom();
        _displayCtrlRight->SetToolZoom();
    }
    event.Skip();
}

void asFramePredictors::OnSyncroToolSwitch(wxCommandEvent& event) {
    _syncroTool = GetMenuBar()->IsChecked(asID_SET_SYNCRO_MODE);
}

void asFramePredictors::OnToolZoomIn(wxCommandEvent& event) {
    _displayCtrlLeft->SetToolZoom();
    _displayCtrlRight->SetToolZoom();
}

void asFramePredictors::OnToolZoomOut(wxCommandEvent& event) {
    _displayCtrlLeft->SetToolZoomOut();
    _displayCtrlRight->SetToolZoomOut();
}

void asFramePredictors::OnToolPan(wxCommandEvent& event) {
    _displayCtrlLeft->SetToolPan();
    _displayCtrlRight->SetToolPan();
}

void asFramePredictors::OnToolSight(wxCommandEvent& event) {
    _displayCtrlLeft->SetToolSight();
    _displayCtrlRight->SetToolSight();
}

void asFramePredictors::OnToolZoomToFit(wxCommandEvent& event) {
    vrRealRect desiredExtent = GetDesiredExtent();

    if (_displayPanelLeft) {
        _viewerLayerManagerLeft->InitializeExtent(desiredExtent);
        ReloadViewerLayerManagerLeft();
    }
    if (_displayPanelRight) {
        _viewerLayerManagerRight->InitializeExtent(desiredExtent);
        ReloadViewerLayerManagerRight();
    }
}

vrRealRect asFramePredictors::GetDesiredExtent() const {
    vf extent = _forecastManager->GetMaxExtent();
    float width = extent[1] - extent[0];
    float height = extent[2] - extent[3];
    float marginWidth = 0.5f * width;
    float marginHeight = 0.5f * height;

    vrRealRect desiredExtent;
    desiredExtent._x = extent[0] - marginWidth;
    desiredExtent._width = width + 2 * marginWidth;
    desiredExtent._y = extent[3] + marginHeight;
    desiredExtent._height = height - 2 * marginHeight;

    return desiredExtent;
}

void asFramePredictors::OnToolAction(wxCommandEvent& event) {
    auto msg = static_cast<vrDisplayToolMessage*>(event.GetClientData());
    wxASSERT(msg);

    vrRealRect realRect;

    if (msg->_evtType == vrEVT_TOOL_ZOOM) {
        // Get rectangle
        vrCoordinate* coord = msg->_parentManager->GetDisplay()->GetCoordinate();

        // Get real rectangle
        coord->ConvertFromPixels(msg->_rect, realRect);

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);

        if (!_syncroTool) {
#if defined(__WIN32__)
            auto thread = new asThreadViewerLayerManagerZoomIn(msg->_parentManager, &_critSectionViewerLayerManager,
                                                               fittedRect);
            ThreadsManager().AddThread(thread);
#else
            msg->_parentManager->Zoom(fittedRect);
#endif
        } else {
            if (_displayPanelLeft) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomIn(_viewerLayerManagerLeft,
                                                                   &_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                _viewerLayerManagerLeft->Zoom(fittedRect);
#endif
            }
            if (_displayPanelRight) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomIn(_viewerLayerManagerRight,
                                                                   &_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                _viewerLayerManagerRight->Zoom(fittedRect);
#endif
            }
        }
    } else if (msg->_evtType == vrEVT_TOOL_ZOOMOUT) {
        // Getting rectangle
        vrCoordinate* coord = msg->_parentManager->GetDisplay()->GetCoordinate();

        // Get real rectangle
        coord->ConvertFromPixels(msg->_rect, realRect);

        // Get fitted rectangle
        vrRealRect fittedRect = coord->GetRectFitted(realRect);

        if (!_syncroTool) {
#if defined(__WIN32__)
            auto thread = new asThreadViewerLayerManagerZoomOut(msg->_parentManager, &_critSectionViewerLayerManager,
                                                                fittedRect);
            ThreadsManager().AddThread(thread);
#else
            msg->_parentManager->ZoomOut(fittedRect);
#endif
        } else {
            if (_displayPanelLeft) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomOut(_viewerLayerManagerLeft,
                                                                    &_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                _viewerLayerManagerLeft->ZoomOut(fittedRect);
#endif
            }
            if (_displayPanelRight) {
#if defined(__WIN32__)
                auto thread = new asThreadViewerLayerManagerZoomOut(_viewerLayerManagerRight,
                                                                    &_critSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
#else
                _viewerLayerManagerRight->ZoomOut(fittedRect);
#endif
            }
        }
    } else if (msg->_evtType == vrEVT_TOOL_PAN) {
        vrCoordinate* coord = msg->_parentManager->GetDisplay()->GetCoordinate();

        wxPoint movedPos = msg->_position;
        wxPoint2DDouble myMovedRealPt;
        if (!coord->ConvertFromPixels(movedPos, myMovedRealPt)) {
            wxLogError("Error converting point : %d, %d to real coordinate", movedPos.x, movedPos.y);
            wxDELETE(msg);
            return;
        }

        realRect = coord->GetExtent();
        realRect.MoveLeftTopTo(myMovedRealPt);

        if (!_syncroTool) {
            coord->SetExtent(realRect);
            msg->_parentManager->Reload();
            ReloadViewerLayerManagerLeft();
            ReloadViewerLayerManagerRight();
        } else {
            if (_displayPanelLeft) {
                _viewerLayerManagerLeft->GetDisplay()->GetCoordinate()->SetExtent(realRect);
                ReloadViewerLayerManagerLeft();
            }
            if (_displayPanelRight) {
                _viewerLayerManagerRight->GetDisplay()->GetCoordinate()->SetExtent(realRect);
                ReloadViewerLayerManagerRight();
            }
        }

    } else if (msg->_evtType == vrEVT_TOOL_SIGHT) {
        vrViewerLayerManager* invertedMgr = _viewerLayerManagerLeft;
        if (invertedMgr == msg->_parentManager) {
            invertedMgr = _viewerLayerManagerRight;
        }

        switch (msg->_mouseStatus) {
            case vrMOUSE_DOWN:
            case vrMOUSE_MOVE: {
                wxClientDC dc(invertedMgr->GetDisplay());
                wxDCOverlay overlayDc(_overlay, &dc);
                overlayDc.Clear();
                dc.SetPen(*wxRED_PEN);
                dc.CrossHair(msg->_position);
            } break;
            case vrMOUSE_UP: {
                wxClientDC dc(invertedMgr->GetDisplay());
                wxDCOverlay overlayDc(_overlay, &dc);
                overlayDc.Clear();
            }
                _overlay.Reset();
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
    if ((_selectedMethod == -1) || (_selectedForecast == -1) || (_selectedTargetDate == -1) ||
        (_selectedAnalogDate == -1) || (_selectedPredictor == -1)) {
        return;
    }

    // Get dates
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);
    a1f targetDates = forecast->GetTargetDates();
    double targetDate = targetDates[_selectedTargetDate];
    a1f analogDates = forecast->GetAnalogsDates(_selectedTargetDate);
    double analogDate = analogDates[_selectedAnalogDate];

    // Get domain
    if (forecast->GetPredictorLonMin().size() == 0) {
        wxLogError(_("Only forecasts of AtmoSwing 3+ can be visualized here."));
        return;
    }
    vf domain;
    domain.push_back(forecast->GetPredictorLonMin()[_selectedPredictor]);
    domain.push_back(forecast->GetPredictorLonMax()[_selectedPredictor]);
    domain.push_back(forecast->GetPredictorLatMin()[_selectedPredictor]);
    domain.push_back(forecast->GetPredictorLatMax()[_selectedPredictor]);

    Coo location = GetStationsMeanCoordinatesWgs84(forecast);

    _predictorsManagerTarget->SetDate(targetDate);
    _predictorsManagerAnalog->SetDate(analogDate);
    _predictorsRenderer->Redraw(domain, location, _listPredictors->GetSelection());
}

Coo asFramePredictors::GetStationsMeanCoordinatesWgs84(asResultsForecast* forecast) {
    Coo location = {0, 0};
    wxString coordSys = forecast->GetCoordinateSystem();
    if (!coordSys.IsEmpty()) {
        location = forecast->GetStationsMeanCoordinates();

        // Define coordinate reference systems
        const char* epsgOrig = coordSys;
        const char* epsgWgs84 = "EPSG:4326";

        // Create Proj projections for the coordinate systems
        PJ* pj = proj_create_crs_to_crs(PJ_DEFAULT_CTX, epsgOrig, epsgWgs84, nullptr);

        // Check if the coordinate reference systems were created successfully
        if (pj == nullptr) {
            wxLogError(_("Failed to create transformation object (from %s)."), coordSys);
            return {0, 0};
        }

        // Convert the coordinates
        PJ_COORD a = proj_coord(location.x, location.y, 0, 0);
        PJ_COORD b = proj_trans(pj, PJ_FWD, a);
        location.x = b.v[1];
        location.y = b.v[0];

        proj_destroy(pj);
    }

    return location;
}

void asFramePredictors::ReloadViewerLayerManagerLeft() {
#if defined(__WIN32__)
    auto thread = new asThreadViewerLayerManagerReload(_viewerLayerManagerLeft, &_critSectionViewerLayerManager);
    ThreadsManager().AddThread(thread);
#else
    _viewerLayerManagerLeft->Reload();
#endif
}

void asFramePredictors::ReloadViewerLayerManagerRight() {
#if defined(__WIN32__)
    auto thread = new asThreadViewerLayerManagerReload(_viewerLayerManagerRight, &_critSectionViewerLayerManager);
    ThreadsManager().AddThread(thread);
#else
    _viewerLayerManagerRight->Reload();
#endif
}

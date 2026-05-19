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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asFramePlotDistributions.h"

#include "asForecastManager.h"

BEGIN_EVENT_TABLE(asFramePlotDistributions, wxFrame)
EVT_CLOSE(asFramePlotDistributions::OnClose)
END_EVENT_TABLE()

asFramePlotDistributions::asFramePlotDistributions(wxWindow* parent, int methodRow, int forecastRow,
                                                   asForecastManager* forecastManager, wxWindowID id)
    : asFramePlotDistributionsVirutal(parent, id),
      _forecastManager(forecastManager),
      _selectedMethod(methodRow),
      _selectedForecast(forecastRow),
      _selectedStation(0),
      _selectedDate(0),
      _xmaxPredictands(0) {
    SetLabel(_("Distribution plots"));

    forecastRow = wxMax(forecastRow, 0);

    _panelPlotPredictands = new asPanelPlot(_panelPredictandsRight);
    _panelPlotPredictands->GetPlotCtrl()->HideScrollBars();
    _panelPlotPredictands->Layout();
    _sizerPlotPredictands->Add(_panelPlotPredictands, 1, wxALL | wxEXPAND, 0);
    _sizerPlotPredictands->Fit(_panelPredictandsRight);

    _panelPlotCriteria = new asPanelPlot(_panelCriteria);
    _panelPlotCriteria->GetPlotCtrl()->HideScrollBars();
    _panelPlotCriteria->Layout();
    _sizerPlotCriteria->Add(_panelPlotCriteria, 1, wxALL | wxEXPAND, 0);
    _sizerPlotCriteria->Fit(_panelCriteria);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif

    Layout();
}

asFramePlotDistributions::~asFramePlotDistributions() {}

void asFramePlotDistributions::OnClose(wxCloseEvent& evt) {
    // Save checked layers
    wxConfigBase* pConfig = wxFileConfig::Get();
    bool doPlotAllAnalogsPoints = _checkListTocPredictands->IsChecked(AllAnalogsPoints);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotAllAnalogsPoints", doPlotAllAnalogsPoints);
    bool doPlotAllAnalogsCurve = _checkListTocPredictands->IsChecked(AllAnalogsCurve);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotAllAnalogsCurve", doPlotAllAnalogsCurve);
    bool doPlotBestAnalogs10Points = _checkListTocPredictands->IsChecked(BestAnalogs10Points);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Points", doPlotBestAnalogs10Points);
    bool doPlotBestAnalogs10Curve = _checkListTocPredictands->IsChecked(BestAnalogs10Curve);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Curve", doPlotBestAnalogs10Curve);
    bool doPlotBestAnalogs5Points = _checkListTocPredictands->IsChecked(BestAnalogs5Points);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Points", doPlotBestAnalogs5Points);
    bool doPlotBestAnalogs5Curve = _checkListTocPredictands->IsChecked(BestAnalogs5Curve);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Curve", doPlotBestAnalogs5Curve);
    bool doPlotAllReturnPeriods = _checkListTocPredictands->IsChecked(AllReturnPeriods);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotAllReturnPeriods", doPlotAllReturnPeriods);
    bool doPlotClassicReturnPeriod = _checkListTocPredictands->IsChecked(ClassicReturnPeriod);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotClassicReturnPeriod", doPlotClassicReturnPeriod);
    bool doPlotClassicQuantiles = _checkListTocPredictands->IsChecked(ClassicQuantiles);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotClassicQuantiles", doPlotClassicQuantiles);

    evt.Skip();
}

void asFramePlotDistributions::Init() {
    if (_selectedForecast < 0) {
        _selectedForecast = 0;
    }

    // Forecast list
    RebuildChoiceForecast();

    // Dates list
    wxArrayString arrayDates = _forecastManager->GetTargetDatesWxArray(_selectedMethod, _selectedForecast);
    _choiceDate->Set(arrayDates);
    _choiceDate->Select(_selectedDate);

    // Stations list
    wxArrayString arrayStation = _forecastManager->GetStationNamesWithHeights(_selectedMethod, _selectedForecast);
    _choiceStation->Set(arrayStation);
    _choiceStation->Select(_selectedStation);

    InitPredictandsCheckListBox();
    InitPredictandsPlotCtrl();
    InitCriteriaPlotCtrl();
}

void asFramePlotDistributions::RebuildChoiceForecast() {
    // Reset forecast list
    wxArrayString arrayForecasts = _forecastManager->GetCombinedForecastNamesWxArray();
    _choiceForecast->Set(arrayForecasts);
    int linearIndex = _forecastManager->GetLinearIndex(_selectedMethod, _selectedForecast);
    _choiceForecast->Select(linearIndex);

    // Highlight the specific forecasts
    for (int methodRow = 0; methodRow < _forecastManager->GetMethodsNb(); methodRow++) {
        int stationId =
            _forecastManager->GetForecast(_selectedMethod, _selectedForecast)->GetStationId(_selectedStation);
        int forecastRow = _forecastManager->GetForecastRowSpecificForStationId(methodRow, stationId);
        int index = _forecastManager->GetLinearIndex(methodRow, forecastRow);
        wxString val = "* " + _choiceForecast->GetString(index) + " *";
        _choiceForecast->SetString(index, val);
    }
}

void asFramePlotDistributions::OnChoiceForecastChange(wxCommandEvent& event) {
    int linearIndex = event.GetInt();
    _selectedMethod = _forecastManager->GetMethodRowFromLinearIndex(linearIndex);
    _selectedForecast = _forecastManager->GetForecastRowFromLinearIndex(linearIndex);

    // Dates list
    wxArrayString arrayDates = _forecastManager->GetTargetDatesWxArray(_selectedMethod, _selectedForecast);
    _choiceDate->Set(arrayDates);
    if (arrayDates.size() <= _selectedDate) {
        _selectedDate = 0;
    }
    _choiceDate->Select(_selectedDate);

    // Stations list
    wxArrayString arrayStation = _forecastManager->GetStationNamesWithHeights(_selectedMethod, _selectedForecast);
    _choiceStation->Set(arrayStation);
    if (arrayStation.size() <= _selectedStation) {
        _selectedStation = 0;
    }
    _choiceStation->Select(_selectedStation);

    Plot();
}

void asFramePlotDistributions::OnChoiceStationChange(wxCommandEvent& event) {
    _selectedStation = event.GetInt();

    RebuildChoiceForecast();

    PlotPredictands();  // Doesn't change for criteria
}

void asFramePlotDistributions::OnChoiceDateChange(wxCommandEvent& event) {
    _selectedDate = event.GetInt();

    Plot();
}

void asFramePlotDistributions::InitPredictandsCheckListBox() {
    wxArrayString checkList;

    checkList.Add(_("Quantiles 90%, 60%, 20%"));
    checkList.Add(_("All analogs (points)"));
    checkList.Add(_("All analogs (curve)"));
    checkList.Add(_("10 best analogs (points)"));
    checkList.Add(_("10 best analogs (curve)"));
    checkList.Add(_("5 best analogs (points)"));
    checkList.Add(_("5 best analogs (curve)"));
    checkList.Add(_("10 year return period"));
    checkList.Add(_("All return periods"));

    _checkListTocPredictands->Set(checkList);
}

void asFramePlotDistributions::InitPredictandsPlotCtrl() {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Set the axis lables
    plotctrl->SetShowXAxisLabel(true);
    plotctrl->SetShowYAxisLabel(true);
    plotctrl->SetXAxisLabel(_("Precipitation [mm]"));
    plotctrl->SetYAxisLabel(_("Cumulative frequency"));
    plotctrl->SetYAxisTicksWidth(25);

    // Legend
    plotctrl->SetKeyBottom(true);

    // Title
    plotctrl->SetShowPlotTitle(true);
    plotctrl->SetPlotTitle(_("Analogs precipitation distribution"));
    wxFont titleFont = plotctrl->GetPlotTitleFont();
    titleFont.SetPointSize(titleFont.GetPointSize() + 2);
    plotctrl->SetPlotTitleFont(titleFont);

    // Set the grid color
    wxColour gridColor(240, 240, 240);
    plotctrl->SetGridColour(gridColor);

    // Open layers defined in the preferences
    wxConfigBase* pConfig = wxFileConfig::Get();
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotAllAnalogsPoints", false))
        _checkListTocPredictands->Check(AllAnalogsPoints);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotAllAnalogsCurve", true))
        _checkListTocPredictands->Check(AllAnalogsCurve);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Points", false))
        _checkListTocPredictands->Check(BestAnalogs10Points);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Curve", true))
        _checkListTocPredictands->Check(BestAnalogs10Curve);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Points", true))
        _checkListTocPredictands->Check(BestAnalogs5Points);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Curve", false))
        _checkListTocPredictands->Check(BestAnalogs5Curve);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotAllReturnPeriods", false))
        _checkListTocPredictands->Check(AllReturnPeriods);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotClassicReturnPeriod", true))
        _checkListTocPredictands->Check(ClassicReturnPeriod);
    if (pConfig->ReadBool("/PlotsDistributionsPredictands/DoPlotClassicQuantiles", true))
        _checkListTocPredictands->Check(ClassicQuantiles);
}

void asFramePlotDistributions::InitCriteriaPlotCtrl() {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotCriteria->GetPlotCtrl();

    // Set the axis lables
    plotctrl->SetShowXAxisLabel(true);
    plotctrl->SetShowYAxisLabel(true);
    plotctrl->SetXAxisLabel(_("Analogues"));
    plotctrl->SetYAxisLabel(_("Criteria of analogy"));
    plotctrl->SetYAxisTicksWidth(25);

    // Title
    plotctrl->SetShowPlotTitle(true);
    plotctrl->SetPlotTitle(_("Criteria distribution"));
    wxFont titleFont = plotctrl->GetPlotTitleFont();
    titleFont.SetPointSize(titleFont.GetPointSize() + 2);
    plotctrl->SetPlotTitleFont(titleFont);

    // Set the grid color
    wxColour gridColor(240, 240, 240);
    plotctrl->SetGridColour(gridColor);
}

void asFramePlotDistributions::OnTocSelectionChange(wxCommandEvent& event) {
    PlotPredictands();
}

bool asFramePlotDistributions::Plot() {
    if (_forecastManager->GetMethodsNb() < 1) return false;
    if (!PlotPredictands()) return false;

    return PlotCriteria();
}

bool asFramePlotDistributions::PlotPredictands() {
    wxBusyCursor wait;

    if (_forecastManager->GetMethodsNb() < 1) return false;

    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Check that there is no NaNs
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);
    a1f analogs = forecast->GetAnalogsValuesRaw(_selectedDate, _selectedStation);
    if (asHasNaN(&analogs[0], &analogs[analogs.size() - 1])) {
        wxLogError(_("The forecast contains NaNs. Plotting has been canceled."));
        return false;
    }

    // Clear previous curves
    int curvesNb = plotctrl->GetCurveCount();
    for (int i = curvesNb - 1; i >= 0; i--) {
        wxPlotData* plotData = plotctrl->GetDataCurve(i);
        if (plotData) {
            plotctrl->DeleteCurve(plotData);
        }
    }

    // Clear previous markers
    plotctrl->ClearMarkers();

    // Set a first threshold for the zoom
    _xmaxPredictands = 50;

    // Get curves to plot
    bool DoPlotAllAnalogsPoints = false;
    bool DoPlotAllAnalogsCurve = false;
    bool DoPlotBestAnalogs10Points = false;
    bool DoPlotBestAnalogs10Curve = false;
    bool DoPlotBestAnalogs5Points = false;
    bool DoPlotBestAnalogs5Curve = false;
    bool DoPlotAllReturnPeriods = false;
    bool DoPlotClassicReturnPeriod = false;
    bool DoPlotClassicQuantiles = false;

    for (int curve = 0; curve <= 8; curve++) {
        if (_checkListTocPredictands->IsChecked(curve)) {
            switch (curve) {
                case (AllAnalogsPoints):
                    DoPlotAllAnalogsPoints = true;
                    break;
                case (AllAnalogsCurve):
                    DoPlotAllAnalogsCurve = true;
                    break;
                case (BestAnalogs10Points):
                    DoPlotBestAnalogs10Points = true;
                    break;
                case (BestAnalogs10Curve):
                    DoPlotBestAnalogs10Curve = true;
                    break;
                case (BestAnalogs5Points):
                    DoPlotBestAnalogs5Points = true;
                    break;
                case (BestAnalogs5Curve):
                    DoPlotBestAnalogs5Curve = true;
                    break;
                case (AllReturnPeriods):
                    DoPlotAllReturnPeriods = true;
                    break;
                case (ClassicReturnPeriod):
                    DoPlotClassicReturnPeriod = true;
                    break;
                case (ClassicQuantiles):
                    DoPlotClassicQuantiles = true;
                    break;
                default:
                    wxLogError(_("The option was not found."));
            }
        }
    }

    if (DoPlotAllAnalogsPoints) PlotAllAnalogsPoints();
    if (DoPlotBestAnalogs10Points) PlotBestAnalogsPoints(10);
    if (DoPlotBestAnalogs5Points) PlotBestAnalogsPoints(5);
    if (DoPlotClassicQuantiles) PlotClassicQuantiles();
    if (DoPlotAllAnalogsCurve) PlotAllAnalogsCurve();
    if (DoPlotBestAnalogs10Curve) PlotBestAnalogsCurve(10);
    if (DoPlotBestAnalogs5Curve) PlotBestAnalogsCurve(5);
    if (forecast->HasReferenceValues()) {
        if (DoPlotClassicReturnPeriod) PlotReturnPeriod(10);
        if (DoPlotAllReturnPeriods) PlotAllReturnPeriods();
    }

    // Set the view rectangle (wxRect2DDouble(x, y, w, h))
    wxRect2DDouble currentView(0, 0, _xmaxPredictands * 1.1, 1);
    plotctrl->SetViewRect(currentView);

    // Redraw
    plotctrl->Redraw(wxPLOTCTRL_REDRAW_PLOT);

    return true;
}

void asFramePlotDistributions::ResetExtent(wxCommandEvent& event) {
    // Set the view rectangle (wxRect2DDouble(x, y, w, h))
    wxRect2DDouble currentView(0, 0, _xmaxPredictands * 1.1, 1);
    _panelPlotPredictands->GetPlotCtrl()->SetViewRect(currentView);

    // Redraw
    _panelPlotPredictands->GetPlotCtrl()->Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

bool asFramePlotDistributions::PlotCriteria() {
    wxBusyCursor wait;

    if (_forecastManager->GetMethodsNb() < 1) return false;

    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotCriteria->GetPlotCtrl();

    // Check that there is no NaNs
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);
    a1f criteria = forecast->GetAnalogsCriteria(_selectedDate);
    if (asHasNaN(&criteria[0], &criteria[criteria.size() - 1])) {
        wxLogError(_("The forecast criteria contains NaNs. Plotting has been canceled."));
        return false;
    }

    // Clear previous curves
    int curvesNb = plotctrl->GetCurveCount();
    for (int i = curvesNb - 1; i >= 0; i--) {
        wxPlotData* plotData = plotctrl->GetDataCurve(i);
        if (plotData) {
            plotctrl->DeleteCurve(plotData);
        }
    }

    // Clear previous markers
    plotctrl->ClearMarkers();

    // Plot curves
    PlotCriteriaCurve();

    // Get min/max of the criteria
    auto critMin = (float)999999999.0, critMax = (float)0.0;
    for (int i = 0; i < forecast->GetTargetDatesLength(); i++) {
        a1f tmpCriteria = forecast->GetAnalogsCriteria(i);
        if (tmpCriteria[0] < critMin) critMin = tmpCriteria[0];
        if (tmpCriteria[tmpCriteria.size() - 1] > critMax) critMax = tmpCriteria[tmpCriteria.size() - 1];
    }

    // Set the view rectangle (wxRect2DDouble(x, y, w, h))
    wxRect2DDouble currentView(1, critMin, forecast->GetAnalogsNumber(_selectedDate) - 1, critMax - critMin);
    plotctrl->SetViewRect(currentView);

    // Redraw
    plotctrl->Redraw(wxPLOTCTRL_REDRAW_PLOT);

    return true;
}

void asFramePlotDistributions::PlotAllReturnPeriods() {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Get return periods
    a1f retPeriods = _forecastManager->GetForecast(_selectedMethod, _selectedForecast)->GetReferenceAxis();

    for (int i = retPeriods.size() - 1; i >= 0; i--) {
        if (std::abs(retPeriods[i] - 2.33) < 0.1) continue;

        // Get precipitation value
        float val =
            _forecastManager->GetForecast(_selectedMethod, _selectedForecast)->GetReferenceValue(_selectedStation, i);

        // Color (from yellow to red)
        float ratio = (float)i / (float)(retPeriods.size() - 1);
        wxGenericPen pen(wxGenericColour(255, 255 - ratio * 255, 0), 2);

        // Markers -> cannot add legend entries
        // wxPlotMarker marker;
        // marker.CreateHorizLineMarker(val, pen);
        // plotctrl->AddMarker(marker);

        // Store max val
        if (val > _xmaxPredictands) _xmaxPredictands = val;

        // Create plot data
        wxPlotData plotData;
        plotData.Create(2);
        if (std::abs(retPeriods[i] - 2.33) < 0.1) {
            plotData.SetFilename(asStrF("P%3.2f", retPeriods[i]));
        } else {
            auto roundedVal = (int)asRound(retPeriods[i]);
            plotData.SetFilename(asStrF("P%d", roundedVal));
        }
        plotData.SetValue(0, val, -1);
        plotData.SetValue(1, val, 2);

        // Check and add to the plot
        if (plotData.Ok()) {
            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);

            plotData.SetDrawSymbols(false);
            plotData.SetDrawLines(true);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        } else {
            wxLogError(_("The return periods couldn't be added to the plot"));
        }

        plotData.Destroy();
    }
}

void asFramePlotDistributions::PlotReturnPeriod(int returnPeriod) {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Get return periods
    a1f retPeriods = _forecastManager->GetForecast(_selectedMethod, _selectedForecast)->GetReferenceAxis();

    // Find the value 10
    int index = asFind(&retPeriods[0], &retPeriods[retPeriods.size() - 1], returnPeriod);

    if ((index != asNOT_FOUND) && (index != asOUT_OF_RANGE)) {
        // Get precipitation value
        float val = _forecastManager->GetForecast(_selectedMethod, _selectedForecast)
                        ->GetReferenceValue(_selectedStation, index);

        // Color (red)
        wxGenericPen pen(wxGenericColour(255, 0, 0), 2);

        // Lines
        wxPlotMarker marker;
        marker.CreateVertLineMarker(val, pen);
        plotctrl->AddMarker(marker);

        // Store max val
        if (val > _xmaxPredictands) _xmaxPredictands = val;
    } else {
        wxLogError(_("The 10 year return period was not found in the data."));
    }
}

void asFramePlotDistributions::PlotAllAnalogsPoints() {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);

    // Get the total number of points
    a1f analogs = forecast->GetAnalogsValuesRaw(_selectedDate, _selectedStation);
    asSortArray(&analogs[0], &analogs[analogs.size() - 1], Asc);
    int nbPoints = analogs.size();

    // Cumulative frequency
    a1f F = asGetCumulativeFrequency(nbPoints);

    // Create plot data
    wxPlotData plotData;
    plotData.Create(nbPoints);
    int counter = 0;
    for (int iAnalog = 0; iAnalog < analogs.size(); iAnalog++) {
        plotData.SetValue(counter, analogs[iAnalog], F(iAnalog));
        counter++;

        // Store max val
        if (analogs[iAnalog] > _xmaxPredictands) _xmaxPredictands = analogs[iAnalog];
    }

    // Check and add to the plot
    if (plotData.Ok()) {
        wxPen pen(wxColour(180, 180, 180), 1);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);
        // wxPlotSymbol_Type : wxPLOTSYMBOL_ELLIPSE, wxPLOTSYMBOL_RECTANGLE, wxPLOTSYMBOL_CROSS, wxPLOTSYMBOL_PLUS,
        // wxPLOTSYMBOL_MAXTYPE
        plotData.SetSymbol(wxPLOTSYMBOL_CROSS, wxPLOTPEN_NORMAL, 5, 5, &pen, nullptr);

        plotData.SetDrawSymbols(true);
        plotData.SetDrawLines(false);

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    } else {
        wxLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

void asFramePlotDistributions::PlotAllAnalogsCurve() {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);

    // Get the total number of points
    a1f analogs = forecast->GetAnalogsValuesRaw(_selectedDate, _selectedStation);
    asSortArray(&analogs[0], &analogs[analogs.size() - 1], Asc);
    int nbPoints = analogs.size();

    // Cumulative frequency
    a1f F = asGetCumulativeFrequency(nbPoints);

    // Create plot data
    wxPlotData plotData;
    plotData.Create(nbPoints);
    int counter = 0;
    for (int iAnalog = 0; iAnalog < analogs.size(); iAnalog++) {
        plotData.SetValue(counter, analogs[iAnalog], F(iAnalog));
        counter++;

        // Store max val
        if (analogs[iAnalog] > _xmaxPredictands) _xmaxPredictands = analogs[iAnalog];
    }

    // Check and add to the plot
    if (plotData.Ok()) {
        wxPen pen(wxColour(0, 0, 240), 2);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);

        plotData.SetDrawSymbols(false);
        plotData.SetDrawLines(true);
        plotData.SetFilename(_("All analogs"));

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    } else {
        wxLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

void asFramePlotDistributions::PlotBestAnalogsPoints(int analogsNb) {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);

    // Extract best analogs
    a1f analogsAll = forecast->GetAnalogsValuesRaw(_selectedDate, _selectedStation);
    int nbPoints = wxMin((int)analogsAll.size(), analogsNb);
    a1f analogs = analogsAll.head(nbPoints);
    a1f ranks = a1f::LinSpaced(nbPoints, 0, nbPoints - 1);
    asSortArrays(&analogs[0], &analogs[analogs.size() - 1], &ranks[0], &ranks[ranks.size() - 1], Asc);

    // Cumulative frequency
    a1f F = asGetCumulativeFrequency(nbPoints);

    // Create plot data
    for (int iAnalog = 0; iAnalog < analogs.size(); iAnalog++) {
        wxPlotData plotData;
        plotData.Create(1);

        plotData.SetValue(0, analogs[iAnalog], F(iAnalog));

        // Check and add to the plot
        if (plotData.Ok()) {
            // Color (from yellow to red)
            float ratio = ranks[iAnalog] / (float)(nbPoints - 1);
            wxPen pen(wxColor(255, ratio * 255, 0), 2);

            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);
            // wxPlotSymbol_Type : wxPLOTSYMBOL_ELLIPSE, wxPLOTSYMBOL_RECTANGLE, wxPLOTSYMBOL_CROSS, wxPLOTSYMBOL_PLUS,
            // wxPLOTSYMBOL_MAXTYPE
            plotData.SetSymbol(wxPLOTSYMBOL_CROSS, wxPLOTPEN_NORMAL, 9, 9, &pen, nullptr);

            plotData.SetDrawSymbols(true);
            plotData.SetDrawLines(false);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        } else {
            wxLogError(_("The analogs data couldn't be added to the plot"));
        }

        plotData.Destroy();

        // Store max val
        if (analogs[iAnalog] > _xmaxPredictands) _xmaxPredictands = analogs[iAnalog];
    }
}

void asFramePlotDistributions::PlotBestAnalogsCurve(int analogsNb) {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);

    // Extract best analogs
    a1f analogsAll = forecast->GetAnalogsValuesRaw(_selectedDate, _selectedStation);
    int nbPoints = wxMin((int)analogsAll.size(), analogsNb);
    a1f analogs = analogsAll.head(nbPoints);
    asSortArray(&analogs[0], &analogs[analogs.size() - 1], Asc);

    // Cumulative frequency
    a1f F = asGetCumulativeFrequency(nbPoints);

    // Create plot data
    wxPlotData plotData;
    plotData.Create(nbPoints);
    int counter = 0;
    for (int iAnalog = 0; iAnalog < analogs.size(); iAnalog++) {
        plotData.SetValue(counter, analogs[iAnalog], F(iAnalog));
        counter++;

        // Store max val
        if (analogs[iAnalog] > _xmaxPredictands) _xmaxPredictands = analogs[iAnalog];
    }

    // Check and add to the plot
    if (plotData.Ok()) {
        wxPen pen(wxColour(180, 0, 180), 1);

        if (analogsNb == 5) {
            pen.SetColour(wxColour(0, 112, 0));
            plotData.SetFilename(_("5 first analogs"));
        } else if (analogsNb == 10) {
            pen.SetColour(wxColour(0, 228, 0));
            plotData.SetFilename(_("10 first analogs"));
        }

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);

        plotData.SetDrawSymbols(false);
        plotData.SetDrawLines(true);

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    } else {
        wxLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

void asFramePlotDistributions::PlotClassicQuantiles() {
    // Quantiles
    a1f pc(3);
    pc << 0.2f, 0.6f, 0.9f;

    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);
    a1f analogs = forecast->GetAnalogsValuesRaw(_selectedDate, _selectedStation);

    // Loop over the quantiles
    for (int iPc = 0; iPc < pc.size(); iPc++) {
        float thisQuantile = pc[iPc];

        // Create plot data
        wxPlotData plotData;
        plotData.Create(1);
        float pcVal = asGetValueForQuantile(analogs, thisQuantile);
        plotData.SetValue(0, pcVal, thisQuantile);

        // Store max val
        if (pcVal > _xmaxPredictands) _xmaxPredictands = pcVal;

        // Check and add to the plot
        if (plotData.Ok()) {
            wxPen pen(wxColour(100, 100, 100), 2, wxPENSTYLE_SOLID);

            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);
            plotData.SetSymbol(wxPLOTSYMBOL_CIRCLE, wxPLOTPEN_NORMAL, 15, 15, &pen, wxTRANSPARENT_BRUSH);

            plotData.SetDrawSymbols(true);
            plotData.SetDrawLines(false);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        } else {
            wxLogError(_("The quantiles couldn't be added to the plot"));
        }

        plotData.Destroy();
    }
}

void asFramePlotDistributions::PlotCriteriaCurve() {
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = _panelPlotCriteria->GetPlotCtrl();

    // Get forecast
    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);

    // Get the criteria
    a1f criteria = forecast->GetAnalogsCriteria(_selectedDate);
    a1f indices = a1f::LinSpaced(criteria.size(), 1, criteria.size());  // LinSpaced(size, low, high)

    // Create plot data
    wxPlotData plotData;
    plotData.Create(criteria.size());
    for (int iAnalog = 0; iAnalog < criteria.size(); iAnalog++) {
        plotData.SetValue(iAnalog, indices[iAnalog], criteria[iAnalog]);
    }

    // Check and add to the plot
    if (plotData.Ok()) {
        wxPen pen(wxColour(0, 240, 240), 2);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);

        plotData.SetDrawSymbols(false);
        plotData.SetDrawLines(true);

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    } else {
        wxLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

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

#include "asFramePlotTimeSeries.h"

#include "asForecastManager.h"
#include "asResultsAnalogsForecast.h"
#include "asFileAscii.h"

BEGIN_EVENT_TABLE(asFramePlotTimeSeries, wxFrame)
    EVT_CLOSE(asFramePlotTimeSeries::OnClose)
END_EVENT_TABLE()

asFramePlotTimeSeries::asFramePlotTimeSeries(wxWindow *parent, int selectedMethod, int selectedForecast,
                                             int selectedStation, asForecastManager *forecastManager, wxWindowID id)
        : asFramePlotTimeSeriesVirtual(parent, id)
{
    m_maxVal = 100;

    int paneMinSize = (int) (m_splitter->GetMinimumPaneSize() * g_ppiScaleDc);
    m_splitter->SetMinimumPaneSize(paneMinSize);

    m_selectedStation = selectedStation;
    m_selectedMethod = selectedMethod;
    m_selectedForecast = selectedForecast;
    m_forecastManager = forecastManager;

    m_panelPlot = new asPanelPlot(m_panelRight);
    m_panelPlot->Layout();
    m_sizerPlot->Add(m_panelPlot, 1, wxALL | wxEXPAND, 0);
    m_sizerPlot->Fit(m_panelRight);

    m_staticTextStationName->SetLabel(
            forecastManager->GetStationNameWithHeight(m_selectedMethod, m_selectedForecast, m_selectedStation));
    wxFont titleFont = m_staticTextStationName->GetFont();
    titleFont.SetPointSize(titleFont.GetPointSize() + 2);
    m_staticTextStationName->SetFont(titleFont);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif

    Layout();
}

asFramePlotTimeSeries::~asFramePlotTimeSeries()
{

}

void asFramePlotTimeSeries::OnClose(wxCloseEvent &evt)
{
    // Save checked layers
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool doPlotAllQuantiles = m_checkListToc->IsChecked(AllQuantiles);
    pConfig->Write("/PlotsTimeSeries/DoPlotAllQuantiles", doPlotAllQuantiles);
    bool doPlotAllAnalogs = m_checkListToc->IsChecked(AllAnalogs);
    pConfig->Write("/PlotsTimeSeries/DoPlotAllAnalogs", doPlotAllAnalogs);
    bool doPlotBestAnalogs10 = m_checkListToc->IsChecked(BestAnalogs10);
    pConfig->Write("/PlotsTimeSeries/DoPlotBestAnalogs10", doPlotBestAnalogs10);
    bool doPlotBestAnalogs5 = m_checkListToc->IsChecked(BestAnalogs5);
    pConfig->Write("/PlotsTimeSeries/DoPlotBestAnalogs5", doPlotBestAnalogs5);
    bool doPlotAllReturnPeriods = m_checkListToc->IsChecked(AllReturnPeriods);
    pConfig->Write("/PlotsTimeSeries/DoPlotAllReturnPeriods", doPlotAllReturnPeriods);
    bool doPlotClassicReturnPeriod = m_checkListToc->IsChecked(ClassicReturnPeriod);
    pConfig->Write("/PlotsTimeSeries/DoPlotClassicReturnPeriod", doPlotClassicReturnPeriod);
    bool doPlotClassicQuantiles = m_checkListToc->IsChecked(ClassicQuantiles);
    pConfig->Write("/PlotsTimeSeries/DoPlotClassicQuantiles", doPlotClassicQuantiles);
    bool doPlotPreviousForecasts = m_checkListToc->IsChecked(PreviousForecasts);
    pConfig->Write("/PlotsTimeSeries/DoPlotPreviousForecasts", doPlotPreviousForecasts);
    //bool doPlotInterpretation = m_checkListToc->IsChecked(Interpretation);
    //pConfig->Write("/PlotsTimeSeries/DoPlotInterpretation", doPlotInterpretation);

    evt.Skip();
}

void asFramePlotTimeSeries::Init()
{
    InitCheckListBox();
    InitPlotCtrl();
}

void asFramePlotTimeSeries::InitCheckListBox()
{
    wxArrayString checkList;

    checkList.Add(_("3 quantiles"));
    checkList.Add(_("All quantiles"));
    checkList.Add(_("All analogs"));
    checkList.Add(_("10 best analogs"));
    checkList.Add(_("5 best analogs"));
    checkList.Add(_("10 year return period"));
    checkList.Add(_("All return periods"));
    checkList.Add(_("Previous forecasts"));
    //checkList.Add(_("Interpretation"));

    m_checkListToc->Set(checkList);

    wxArrayString listPast;
    for (int i = 0; i < m_forecastManager->GetPastForecastsNb(m_selectedMethod, m_selectedForecast); i++) {
        asResultsAnalogsForecast *forecast = m_forecastManager->GetPastForecast(m_selectedMethod, m_selectedForecast,
                                                                                i);
        listPast.Add(forecast->GetLeadTimeOriginString());
    }
    m_checkListPast->Set(listPast);

    for (int i = 0; i < m_forecastManager->GetPastForecastsNb(m_selectedMethod, m_selectedForecast); i++) {
        m_checkListPast->Check(i);
    }
}

void asFramePlotTimeSeries::InitPlotCtrl()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Set the axis lables
    plotctrl->SetShowXAxisLabel(false);
    plotctrl->SetShowYAxisLabel(true);
    plotctrl->SetYAxisLabel(_("Precipitation [mm]"));
    plotctrl->SetYAxisTicksWidth(25);

    // Legend
    plotctrl->SetShowKey(true);

    // Title
    plotctrl->SetShowPlotTitle(true);
    plotctrl->SetPlotTitle(_("Forecasted time series"));
    wxFont titleFont = plotctrl->GetPlotTitleFont();
    titleFont.SetPointSize(titleFont.GetPointSize() + 2);
    plotctrl->SetPlotTitleFont(titleFont);

    // Set the grid color
    wxColour gridColor(240, 240, 240);
    plotctrl->SetGridColour(gridColor);

    // Set the x axis
    asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    int length = forecast->GetTargetDatesLength();
    Array1DFloat dates = forecast->GetTargetDates();
    m_leadTimes.resize(length);
    for (int i = 0; i < length; i++) {
        m_leadTimes[i] = dates[i];
    }

    // Add a large vertical line at present time
    wxGenericPen p(gridColor, 5);
    wxPlotMarker m;
    m.CreateVertLineMarker(floor(forecast->GetLeadTimeOrigin()), p);
    plotctrl->AddMarker(m);

    // Freeze the ticks of the x axis and set dates label
    plotctrl->SetFixXAxisTickStep(1);
    plotctrl->SetXAxisTickType(wxPLOTCTRL_DATE_DDMM_FROMMJD);

    // Open layers defined in the preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool doPlotAllQuantiles;
    pConfig->Read("/PlotsTimeSeries/DoPlotAllQuantiles", &doPlotAllQuantiles, false);
    if (doPlotAllQuantiles)
        m_checkListToc->Check(AllQuantiles);
    bool doPlotAllAnalogs;
    pConfig->Read("/PlotsTimeSeries/DoPlotAllAnalogs", &doPlotAllAnalogs, true);
    if (doPlotAllAnalogs)
        m_checkListToc->Check(AllAnalogs);
    bool doPlotBestAnalogs10;
    pConfig->Read("/PlotsTimeSeries/DoPlotBestAnalogs10", &doPlotBestAnalogs10, true);
    if (doPlotBestAnalogs10)
        m_checkListToc->Check(BestAnalogs10);
    bool doPlotBestAnalogs5;
    pConfig->Read("/PlotsTimeSeries/DoPlotBestAnalogs5", &doPlotBestAnalogs5, false);
    if (doPlotBestAnalogs5)
        m_checkListToc->Check(BestAnalogs5);
    bool doPlotAllReturnPeriods;
    pConfig->Read("/PlotsTimeSeries/DoPlotAllReturnPeriods", &doPlotAllReturnPeriods, false);
    if (doPlotAllReturnPeriods)
        m_checkListToc->Check(AllReturnPeriods);
    bool doPlotClassicReturnPeriod;
    pConfig->Read("/PlotsTimeSeries/DoPlotClassicReturnPeriod", &doPlotClassicReturnPeriod, true);
    if (doPlotClassicReturnPeriod)
        m_checkListToc->Check(ClassicReturnPeriod);
    bool doPlotClassicQuantiles;
    pConfig->Read("/PlotsTimeSeries/DoPlotClassicQuantiles", &doPlotClassicQuantiles, true);
    if (doPlotClassicQuantiles)
        m_checkListToc->Check(ClassicQuantiles);
    bool doPlotPreviousForecasts;
    pConfig->Read("/PlotsTimeSeries/DoPlotPreviousForecasts", &doPlotPreviousForecasts, true);
    if (doPlotPreviousForecasts)
        m_checkListToc->Check(PreviousForecasts);
    //bool doPlotInterpretation;
    //pConfig->Read("/PlotsTimeSeries/DoPlotInterpretation", &doPlotInterpretation, true);
    //if (doPlotInterpretation) m_checkListToc->Check(Interpretation);
}

void asFramePlotTimeSeries::OnTocSelectionChange(wxCommandEvent &event)
{
    Plot();
}

void asFramePlotTimeSeries::OnExportTXT(wxCommandEvent &event) const
{
    wxString stationName = m_forecastManager->GetStationName(m_selectedMethod, m_selectedForecast, m_selectedStation);
    wxString forecastName = m_forecastManager->GetForecastName(m_selectedMethod, m_selectedForecast);
    wxString date = asTime::GetStringTime(m_forecastManager->GetLeadTimeOrigin(), "YYYY.MM.DD hh");
    wxString filename = wxString::Format("%sh - %s - %s", date, forecastName, stationName);

    wxFileDialog dialog(this, wxT("Save file as"), wxEmptyString, filename, wxT("Text files (*.txt)|*.txt"),
                        wxFD_SAVE | wxFD_OVERWRITE_PROMPT);

    if (dialog.ShowModal() == wxID_OK) {
        asFileAscii file(dialog.GetPath(), asFile::Write);
        file.Open();

        // Add header
        file.AddLineContent(wxString::Format("Forecast of the %sh",
                                             asTime::GetStringTime(m_forecastManager->GetLeadTimeOrigin(),
                                                                   "DD.MM.YYYY hh")));
        file.AddLineContent(wxString::Format("Forecast: %s", forecastName));
        file.AddLineContent(wxString::Format("Station: %s", stationName));
        file.AddLineContent();

        // Quantiles
        Array1DFloat pc(11);
        pc << 1, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0;

        // Get forecast
        asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

        // Set lead times
        file.AddLineContent("Quantiles:");
        wxString leadTimes = "\t";
        for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
            leadTimes.Append(wxString::Format("%s\t", asTime::GetStringTime(m_leadTimes[i_leadtime], "DD.MM")));
        }
        file.AddLineContent(leadTimes);

        // Loop over the quantiles to display as polygons
        for (int i_pc = 0; i_pc < pc.size(); i_pc++) {
            float thisQuantile = pc[i_pc];

            wxString quantilesStr = wxString::Format("%f\t", thisQuantile);

            for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
                Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
                float pcVal = asTools::GetValueForQuantile(analogs, thisQuantile);

                quantilesStr.Append(wxString::Format("%f\t", pcVal));
            }

            file.AddLineContent(quantilesStr);
        }
        file.AddLineContent();

        // Set best analogs values
        file.AddLineContent("Best analogs values:");
        file.AddLineContent(leadTimes);

        // Loop over the quantiles to display as polygons
        for (int rk = 0; rk < 10; rk++) {
            wxString rankStr = wxString::Format("%d\t", rk + 1);

            for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
                Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
                rankStr.Append(wxString::Format("%f\t", analogs[rk]));
            }

            file.AddLineContent(rankStr);
        }
        file.AddLineContent();

        // Set best analogs values
        file.AddLineContent("Best analogs dates:");
        file.AddLineContent(leadTimes);

        // Loop over the quantiles to display as polygons
        for (int rk = 0; rk < 10; rk++) {
            wxString rankStr = wxString::Format("%d\t", rk + 1);

            for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
                Array1DFloat dates = forecast->GetAnalogsDates(i_leadtime);
                rankStr.Append(asTime::GetStringTime(dates[rk], "DD.MM.YYYY") + "\t");
            }

            file.AddLineContent(rankStr);
        }
        file.AddLineContent();

        // All traces
        file.AddLineContent("All traces:");

        asResultsAnalogsForecast *oldestForecast = m_forecastManager->GetPastForecast(m_selectedMethod,
                                                                                      m_selectedForecast,
                                                                                      m_forecastManager->GetPastForecastsNb(
                                                                                              m_selectedMethod,
                                                                                              m_selectedForecast) - 1);
        float leadtimeStart = oldestForecast->GetTargetDates()[0];
        float leadtimeEnd = forecast->GetTargetDates()[forecast->GetTargetDatesLength() - 1];

        wxLogMessage(asTime::GetStringTime(leadtimeStart));
        wxLogMessage(asTime::GetStringTime(leadtimeEnd));

        Array1DFloat leadtimes = Array1DFloat::LinSpaced(leadtimeEnd - leadtimeStart + 1, leadtimeStart, leadtimeEnd);

        wxString allLeadtimesStr = "\t";
        for (int i_lt = 0; i_lt < leadtimes.size(); i_lt++) {
            allLeadtimesStr.Append(wxString::Format("%s\t", asTime::GetStringTime(leadtimes[i_lt], "DD.MM")));
        }

        Array1DFloat pcAll(4);
        pcAll << 0.9f, 0.6f, 0.5f, 0.2f;

        for (int i_pc = 0; i_pc < pcAll.size(); i_pc++) {
            file.AddLineContent();
            file.AddLineContent(wxString::Format("Quantile %f:", pcAll[i_pc]));
            file.AddLineContent(allLeadtimesStr);

            for (int past = 0;
                 past < m_forecastManager->GetPastForecastsNb(m_selectedMethod, m_selectedForecast); past++) {
                asResultsAnalogsForecast *pastForecast = m_forecastManager->GetPastForecast(m_selectedMethod,
                                                                                        m_selectedForecast, past);
                Array1DFloat dates = pastForecast->GetTargetDates();
                wxString currentLine = asTime::GetStringTime(pastForecast->GetLeadTimeOrigin(), "DD.MM") + "\t";

                for (int i_leadtime = 0; i_leadtime < pastForecast->GetTargetDatesLength(); i_leadtime++) {
                    Array1DFloat analogs = pastForecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
                    float pcVal = asTools::GetValueForQuantile(analogs, pcAll[i_pc]);

                    if (i_leadtime == 0) {
                        int index = asTools::SortedArraySearch(&leadtimes[0], &leadtimes[leadtimes.size() - 1],
                                                               dates[i_leadtime]);

                        if (index > 0) {
                            for (int i = 0; i < index; i++) {
                                currentLine.Append("\t");
                            }
                        }
                    }

                    currentLine.Append(wxString::Format("%f\t", pcVal));
                }

                file.AddLineContent(currentLine);
            }
        }

        file.Close();
    }
}

void asFramePlotTimeSeries::OnExportSVG(wxCommandEvent &event) const
{
    m_panelPlot->ExportSVG();
}

void asFramePlotTimeSeries::OnPreview(wxCommandEvent &event)
{
    m_panelPlot->PrintPreview();
}

void asFramePlotTimeSeries::OnPrint(wxCommandEvent &event)
{
    m_panelPlot->Print();
}

bool asFramePlotTimeSeries::Plot()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Check that there is no NaNs
    asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
        Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
        if (asTools::HasNaN(&analogs[0], &analogs[analogs.size() - 1])) {
            asLogError(_("The forecast contains NaNs. Plotting has been canceled."));
            return false;
        }
    }

    // Clear previous curves
    int curvesNb = plotctrl->GetCurveCount();
    for (int i = curvesNb - 1; i >= 0; i--) {
        wxPlotData *plotData = plotctrl->GetDataCurve(i);
        if (plotData) {
            plotctrl->DeleteCurve(plotData);
        }
    }

    // Clear previous markers
    plotctrl->ClearMarkers();

    // Add a large vertical line at present time
    wxGenericPen p(wxColour(200, 200, 200), 5);
    wxPlotMarker m;
    m.CreateVertLineMarker(floor(forecast->GetLeadTimeOrigin()), p);
    plotctrl->AddMarker(m);

    // Set a first threshold for the zoom
    m_maxVal = 50;

    // Get curves to plot
    bool DoPlotAllQuantiles = false;
    bool DoPlotAllAnalogs = false;
    bool DoPlotBestAnalogs10 = false;
    bool DoPlotBestAnalogs5 = false;
    bool DoPlotAllReturnPeriods = false;
    bool DoPlotClassicReturnPeriod = false;
    bool DoPlotClassicQuantiles = false;
    bool DoPlotPreviousForecasts = false;
    bool DoPlotInterpretation = false;

    for (int curve = 0; curve < 8; curve++) {
        if (m_checkListToc->IsChecked(curve)) {
            switch (curve) {
                case (AllQuantiles):
                    DoPlotAllQuantiles = true;
                    break;
                case (AllAnalogs):
                    DoPlotAllAnalogs = true;
                    break;
                case (BestAnalogs10):
                    DoPlotBestAnalogs10 = true;
                    break;
                case (BestAnalogs5):
                    DoPlotBestAnalogs5 = true;
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
                case (PreviousForecasts):
                    DoPlotPreviousForecasts = true;
                    break;
                case (Interpretation):
                    DoPlotInterpretation = true;
                    break;
                default:
                    asLogError(_("The option was not found."));

            }
        }
    }

    if (DoPlotAllQuantiles)
        PlotAllQuantiles();
    if (DoPlotAllAnalogs)
        PlotAllAnalogs();
    if (DoPlotBestAnalogs10)
        PlotBestAnalogs(10);
    if (DoPlotBestAnalogs5)
        PlotBestAnalogs(5);
    if (forecast->HasReferenceValues()) {
        if (DoPlotAllReturnPeriods)
            PlotAllReturnPeriods();
        if (DoPlotClassicReturnPeriod)
            PlotReturnPeriod(10);
    }
    if (DoPlotPreviousForecasts)
        PlotPastForecasts();
    if (DoPlotClassicQuantiles)
        PlotClassicQuantiles();
    if (DoPlotInterpretation)
        PlotInterpretation();

    // Set the view rectangle
    wxRect2DDouble view(m_leadTimes[0] - 2.5, 0, m_leadTimes.size() + 2, m_maxVal * 1.1);
    plotctrl->SetViewRect(view);

    // Redraw
    plotctrl->Redraw(wxPLOTCTRL_REDRAW_PLOT);

    return true;
}

void asFramePlotTimeSeries::PlotAllReturnPeriods()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get return periods
    Array1DFloat retPeriods = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceAxis();

    for (int i = retPeriods.size() - 1; i >= 0; i--) {
        if (std::abs(retPeriods[i] - 2.33) < 0.1)
            continue;

        // Get precipitation value
        float val = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceValue(
                m_selectedStation, i);

        // Color (from yellow to red)
        float ratio = (float) i / (float) (retPeriods.size() - 1);
        wxGenericPen pen(wxGenericColour(255, 255 - ratio * 255, 0), 2);

        // Markers -> cannot add legend entries
        //wxPlotMarker marker;
        //marker.CreateHorizLineMarker(val, pen);
        //plotctrl->AddMarker(marker);

        // Store max val
        if (val > m_maxVal)
            m_maxVal = val;

        // Create plot data
        wxPlotData plotData;
        plotData.Create(2);
        if (std::abs(retPeriods[i] - 2.33) < 0.1) {
            plotData.SetFilename(wxString::Format("P%3.2f", retPeriods[i]));
        } else {
            int roundedVal = (int) asTools::Round(retPeriods[i]);
            plotData.SetFilename(wxString::Format("P%d", roundedVal));
        }
        plotData.SetValue(0, m_leadTimes[0] - 10, val);
        plotData.SetValue(1, m_leadTimes[m_leadTimes.size() - 1] + 10, val);

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
            asLogError(_("The return periods couldn't be added to the plot"));
        }

        plotData.Destroy();
    }

}

void asFramePlotTimeSeries::PlotReturnPeriod(int returnPeriod)
{
    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get return periods
    Array1DFloat retPeriods = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceAxis();

    // Find the value 10
    int index = asTools::SortedArraySearch(&retPeriods[0], &retPeriods[retPeriods.size() - 1], returnPeriod);

    if ((index != asNOT_FOUND) && (index != asOUT_OF_RANGE)) {
        // Get precipitation value
        float val = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceValue(
                m_selectedStation, index);

        // Color (red)
        wxGenericPen pen(wxGenericColour(255, 0, 0), 2);

        // Lines
        wxPlotMarker marker;
        marker.CreateHorizLineMarker(val, pen);
        plotctrl->AddMarker(marker);

        // Store max val
        if (val > m_maxVal)
            m_maxVal = val;
    } else {
        asLogError(_("The 10 year return period was not found in the data."));
    }
}

void asFramePlotTimeSeries::PlotAllAnalogs()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Get the total number of points
    int nbPoints = 0;
    for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
        Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
        for (int i_analog = 0; i_analog < analogs.size(); i_analog++) {
            nbPoints++;
        }
    }

    // Create plot data
    wxPlotData plotData;
    plotData.Create(nbPoints);
    int counter = 0;
    for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
        Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
        for (int i_analog = 0; i_analog < analogs.size(); i_analog++) {
            plotData.SetValue(counter, m_leadTimes[i_leadtime], analogs[i_analog]);
            counter++;

            // Store max val
            if (analogs[i_analog] > m_maxVal)
                m_maxVal = analogs[i_analog];
        }
    }

    // Check and add to the plot
    if (plotData.Ok()) {
        wxPen pen(wxColour(180, 180, 180), 1);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);
        // wxPlotSymbol_Type : wxPLOTSYMBOL_ELLIPSE, wxPLOTSYMBOL_RECTANGLE, wxPLOTSYMBOL_CROSS, wxPLOTSYMBOL_PLUS, wxPLOTSYMBOL_MAXTYPE
        plotData.SetSymbol(wxPLOTSYMBOL_CROSS, wxPLOTPEN_NORMAL, 5, 5, &pen, NULL);

        plotData.SetDrawSymbols(true);
        plotData.SetDrawLines(false);

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    } else {
        asLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

void asFramePlotTimeSeries::PlotBestAnalogs(int pointsNb)
{
    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Loop over the analogs to set the color (from the less important to the best)
    for (int i_analog = pointsNb - 1; i_analog >= 0; i_analog--) {
        // Get the total number of points
        int nbPoints = 0;
        for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
            Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
            if (analogs.size() > i_analog)
                nbPoints++;
        }

        // Create plot data
        wxPlotData plotData;
        plotData.Create(nbPoints);
        int counter = 0;
        for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
            Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
            if (analogs.size() > i_analog) {
                plotData.SetValue(counter, m_leadTimes[i_leadtime], analogs[i_analog]);
                counter++;

                // Store max val
                if (analogs[i_analog] > m_maxVal)
                    m_maxVal = analogs[i_analog];
            }
        }

        // Check and add to the plot
        if (plotData.Ok()) {
            // Color (from red to yellow)
            float ratio = (float) i_analog / (float) (pointsNb - 1);
            wxPen pen(wxColor(255, ratio * 255, 0), 2);

            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);
            // wxPlotSymbol_Type : wxPLOTSYMBOL_ELLIPSE, wxPLOTSYMBOL_RECTANGLE, wxPLOTSYMBOL_CROSS, wxPLOTSYMBOL_PLUS, wxPLOTSYMBOL_MAXTYPE
            plotData.SetSymbol(wxPLOTSYMBOL_CROSS, wxPLOTPEN_NORMAL, 9, 9, &pen, NULL);

            plotData.SetDrawSymbols(true);
            plotData.SetDrawLines(false);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        } else {
            asLogError(_("The analogs data couldn't be added to the plot"));
        }

        plotData.Destroy();
    }
}

void asFramePlotTimeSeries::PlotClassicQuantiles()
{
    // Quantiles
    Array1DFloat pc(3);
    pc << 0.9f, 0.6f, 0.2f;
    std::vector<wxColour> colours;
    colours.push_back(wxColour(0, 0, 175));
    colours.push_back(wxColour(0, 83, 255));
    colours.push_back(wxColour(0, 226, 255));

    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Loop over the quantiles
    for (int i_pc = 0; i_pc < pc.size(); i_pc++) {
        float thisQuantile = pc[i_pc];

        // Create plot data
        wxPlotData plotData;
        plotData.Create(m_leadTimes.size());
        int quantileRounded = (int) (asTools::Round(thisQuantile * 100.0));
        plotData.SetFilename(wxString::Format("Quantile %d", quantileRounded));
        int counter = 0;
        for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
            Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
            float pcVal = asTools::GetValueForQuantile(analogs, thisQuantile);
            plotData.SetValue(counter, m_leadTimes[i_leadtime], pcVal);
            counter++;

            // Store max val
            if (pcVal > m_maxVal)
                m_maxVal = pcVal;
        }

        // Check and add to the plot
        if (plotData.Ok()) {
            wxPen pen(colours[i_pc], 2, wxPENSTYLE_SOLID);

            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);

            plotData.SetDrawSymbols(false);
            plotData.SetDrawLines(true);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        } else {
            asLogError(_("The quantiles couldn't be added to the plot"));
        }

        plotData.Destroy();
    }
}

void asFramePlotTimeSeries::PlotPastForecasts()
{
    for (int past = 0; past < m_forecastManager->GetPastForecastsNb(m_selectedMethod, m_selectedForecast); past++) {
        if (m_checkListPast->IsChecked(past)) {
            PlotPastForecast(past);
        }
    }
}

void asFramePlotTimeSeries::PlotPastForecast(int i)
{
    // Quantiles
    Array1DFloat pc(3);
    pc << 0.9f, 0.6f, 0.2f;
    std::vector<wxColour> colours;
    colours.push_back(wxColour(152, 152, 222));
    colours.push_back(wxColour(152, 187, 255));
    colours.push_back(wxColour(153, 243, 254));

    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast *forecast = m_forecastManager->GetPastForecast(m_selectedMethod, m_selectedForecast, i);
    int length = forecast->GetTargetDatesLength();
    Array1DFloat dates = forecast->GetTargetDates();

    // Loop over the quantiles
    for (int i_pc = 0; i_pc < pc.size(); i_pc++) {
        float thisQuantile = pc[i_pc];

        // Create plot data
        wxPlotData plotData;
        plotData.Create(length);
        //int quantileRounded = (int)(asTools::Round(thisQuantile*100.0));
        int counter = 0;
        for (int i_leadtime = 0; i_leadtime < length; i_leadtime++) {
            Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
            float pcVal = asTools::GetValueForQuantile(analogs, thisQuantile);
            plotData.SetValue(counter, dates[i_leadtime], pcVal);
            counter++;

            // Store max val
            if (pcVal > m_maxVal)
                m_maxVal = pcVal;
        }

        // Check and add to the plot
        if (plotData.Ok()) {
            wxPen pen(colours[i_pc], 1, wxPENSTYLE_SOLID);

            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);

            plotData.SetDrawSymbols(false);
            plotData.SetDrawLines(true);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        } else {
            asLogError(_("The quantiles couldn't be added to the plot"));
        }

        plotData.Destroy();
    }
}

void asFramePlotTimeSeries::PlotAllQuantiles()
{
    // Quantiles
    Array1DFloat pcUp(5);
    pcUp << 1, 0.9f, 0.8f, 0.7f, 0.6f;
    Array1DFloat pcDown(5);
    pcDown << 0, 0.1f, 0.2f, 0.3f, 0.4f;
    float pcMid = 0.5f;
    std::vector<wxColour> colours;
    colours.push_back(wxColour(252, 252, 252));
    colours.push_back(wxColour(220, 220, 220));
    colours.push_back(wxColour(200, 200, 200));
    colours.push_back(wxColour(150, 150, 150));
    colours.push_back(wxColour(100, 100, 100));
    wxColour colourMid = wxColour(50, 50, 50);

    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Loop over the quantiles to display as polygons
    for (int i_pc = 0; i_pc < pcUp.size(); i_pc++) {
        float thisQuantileUp = pcUp[i_pc];
        float thisQuantileDown = pcDown[i_pc];

        // Create plot data
        wxPlotData plotData;
        plotData.Create(2 * m_leadTimes.size() + 1);
        int counter = 0;
        float bkpVal = 0;
        // Left to right
        for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
            Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
            float pcVal = asTools::GetValueForQuantile(analogs, thisQuantileUp);
            plotData.SetValue(counter, m_leadTimes[i_leadtime], pcVal);
            counter++;
            if (i_leadtime == 0)
                bkpVal = pcVal;

            // Store max val
            if (pcVal > m_maxVal)
                m_maxVal = pcVal;
        }
        // Right to left
        for (int i_leadtime = m_leadTimes.size() - 1; i_leadtime >= 0; i_leadtime--) {
            Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
            float pcVal = asTools::GetValueForQuantile(analogs, thisQuantileDown);
            plotData.SetValue(counter, m_leadTimes[i_leadtime], pcVal);
            counter++;
        }
        // Close the polygon
        plotData.SetValue(counter, m_leadTimes[0], bkpVal);

        // Check and add to the plot
        if (plotData.Ok()) {
            wxPen pen(colours[i_pc], 1, wxPENSTYLE_SOLID);
            wxBrush brush(colours[i_pc], wxBRUSHSTYLE_SOLID);

            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);
            plotData.SetBrush(brush);

            plotData.SetDrawSymbols(false);
            plotData.SetDrawLines(false);
            plotData.SetDrawPolygon(true);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        } else {
            asLogError(_("The quantiles couldn't be added to the plot"));
        }

        plotData.Destroy();

    }

    // Set the quantile to display as line
    float thisQuantile = pcMid;

    // Create plot data
    wxPlotData plotData;
    plotData.Create(m_leadTimes.size());
    int counter = 0;
    for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
        Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
        float pcVal = asTools::GetValueForQuantile(analogs, thisQuantile);
        plotData.SetValue(counter, m_leadTimes[i_leadtime], pcVal);
        counter++;
    }

    // Check and add to the plot
    if (plotData.Ok()) {
        wxPen pen(colourMid, 2, wxPENSTYLE_SOLID);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);

        plotData.SetDrawSymbols(false);
        plotData.SetDrawLines(true);
        plotData.SetDrawPolygon(false);

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    } else {
        asLogError(_("The quantiles couldn't be added to the plot"));
    }

    plotData.Destroy();

}

void asFramePlotTimeSeries::PlotInterpretation()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl *plotctrl = m_panelPlot->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Create plot data
    wxPlotData plotData;
    plotData.Create(m_leadTimes.size());
    int counter = 0;

    for (unsigned int i_leadtime = 0; i_leadtime < m_leadTimes.size(); i_leadtime++) {
        // Process quantiles
        Array1DFloat analogs = forecast->GetAnalogsValuesGross(i_leadtime, m_selectedStation);
        float pc30 = asTools::GetValueForQuantile(analogs, 0.3f);
        float pc60 = asTools::GetValueForQuantile(analogs, 0.6f);
        float pc90 = asTools::GetValueForQuantile(analogs, 0.9f);

        // Follow the rules
        float val = 0;
        if (pc60 == 0) // if quantile 60% is null, there will be no rain
        {
            val = 0;
        } else if (pc30 > 0) // if quantile 30% is not null, it's gonna rain
        {
            val = pc90;
        } else {
            val = pc60;
        }

        plotData.SetValue(counter, m_leadTimes[i_leadtime], val);
        counter++;

        // Store max val
        if (val > m_maxVal)
            m_maxVal = val;
    }

    // Check and add to the plot
    if (plotData.Ok()) {
        wxPen pen(wxColour(213, 0, 163), 2, wxPENSTYLE_SHORT_DASH);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);

        plotData.SetDrawSymbols(false);
        plotData.SetDrawLines(true);

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    } else {
        asLogError(_("The interpretation curve couldn't be added to the plot"));
    }

    plotData.Destroy();

}


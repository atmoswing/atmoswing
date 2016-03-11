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
#include "asResultsAnalogsForecast.h"

BEGIN_EVENT_TABLE(asFramePlotDistributions, wxFrame)
    EVT_CLOSE(asFramePlotDistributions::OnClose)
END_EVENT_TABLE()

asFramePlotDistributions::asFramePlotDistributions( wxWindow* parent, int methodRow, int forecastRow, asForecastManager *forecastManager, wxWindowID id )
:
asFramePlotDistributionsVirutal( parent, id )
{
	forecastRow = wxMax(forecastRow, 0);

	int paneMinSize = m_splitterPredictands->GetMinimumPaneSize() * g_ppiScaleDc;
	m_splitterPredictands->SetMinimumPaneSize(paneMinSize);

    m_forecastManager = forecastManager;
    m_selectedMethod = methodRow;
    m_selectedForecast = forecastRow;
    m_selectedStation = 0;
    m_selectedDate = 0;
    m_xmaxPredictands = 0;

    m_panelPlotPredictands = new asPanelPlot( m_panelPredictandsRight );
    m_panelPlotPredictands->Layout();
    m_sizerPlotPredictands->Add( m_panelPlotPredictands, 1, wxALL|wxEXPAND, 0 );
    m_sizerPlotPredictands->Fit( m_panelPredictandsRight );

    m_panelPlotCriteria = new asPanelPlot( m_panelCriteria );
    m_panelPlotCriteria->Layout();
    m_sizerPlotCriteria->Add( m_panelPlotCriteria, 1, wxALL|wxEXPAND, 0 );
    m_sizerPlotCriteria->Fit( m_panelCriteria );

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif

    Layout();
}

asFramePlotDistributions::~asFramePlotDistributions()
{

}

void asFramePlotDistributions::OnClose( wxCloseEvent& evt )
{
    // Save checked layers
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool doPlotAllAnalogsPoints = m_checkListTocPredictands->IsChecked(AllAnalogsPoints);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotAllAnalogsPoints", doPlotAllAnalogsPoints);
    bool doPlotAllAnalogsCurve = m_checkListTocPredictands->IsChecked(AllAnalogsCurve);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotAllAnalogsCurve", doPlotAllAnalogsCurve);
    bool doPlotBestAnalogs10Points = m_checkListTocPredictands->IsChecked(BestAnalogs10Points);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Points", doPlotBestAnalogs10Points);
    bool doPlotBestAnalogs10Curve = m_checkListTocPredictands->IsChecked(BestAnalogs10Curve);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Curve", doPlotBestAnalogs10Curve);
    bool doPlotBestAnalogs5Points = m_checkListTocPredictands->IsChecked(BestAnalogs5Points);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Points", doPlotBestAnalogs5Points);
    bool doPlotBestAnalogs5Curve = m_checkListTocPredictands->IsChecked(BestAnalogs5Curve);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Curve", doPlotBestAnalogs5Curve);
    bool doPlotAllReturnPeriods = m_checkListTocPredictands->IsChecked(AllReturnPeriods);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotAllReturnPeriods", doPlotAllReturnPeriods);
    bool doPlotClassicReturnPeriod = m_checkListTocPredictands->IsChecked(ClassicReturnPeriod);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotClassicReturnPeriod", doPlotClassicReturnPeriod);
    bool doPlotClassicQuantiles = m_checkListTocPredictands->IsChecked(ClassicQuantiles);
    pConfig->Write("/PlotsDistributionsPredictands/DoPlotClassicQuantiles", doPlotClassicQuantiles);

    evt.Skip();
}

void asFramePlotDistributions::Init()
{
    // Forecast list
	RebuildChoiceForecast();

    // Dates list
    wxArrayString arrayDates = m_forecastManager->GetLeadTimes(m_selectedMethod, m_selectedForecast);
    m_choiceDate->Set(arrayDates);
    m_choiceDate->Select(m_selectedDate);

    // Stations list
    wxArrayString arrayStation = m_forecastManager->GetStationNamesWithHeights(m_selectedMethod, m_selectedForecast);
    m_choiceStation->Set(arrayStation);
    m_choiceStation->Select(m_selectedStation);

    InitPredictandsCheckListBox();
    InitPredictandsPlotCtrl();
    InitCriteriaPlotCtrl();
}

void asFramePlotDistributions::RebuildChoiceForecast()
{
	// Reset forecast list
	wxArrayString arrayForecasts = m_forecastManager->GetAllForecastNamesWxArray();
	m_choiceForecast->Set(arrayForecasts);
	int linearIndex = m_forecastManager->GetLinearIndex(m_selectedMethod, m_selectedForecast);
	m_choiceForecast->Select(linearIndex);

	// Highlight the specific forecasts
	for (int methodRow = 0; methodRow < m_forecastManager->GetMethodsNb(); methodRow++)
	{
		int stationId = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetStationId(m_selectedStation);
		int forecastRow = m_forecastManager->GetForecastRowSpecificForStationId(methodRow, stationId);
		int index = m_forecastManager->GetLinearIndex(methodRow, forecastRow);
		wxString val = " --> " + m_choiceForecast->GetString(index) + " <-- ";
		m_choiceForecast->SetString(index, val);
	}
}

void asFramePlotDistributions::OnChoiceForecastChange( wxCommandEvent& event )
{
    int linearIndex = event.GetInt();
    m_selectedMethod = m_forecastManager->GetMethodRowFromLinearIndex(linearIndex);
    m_selectedForecast = m_forecastManager->GetForecastRowFromLinearIndex(linearIndex);

    // Dates list
    wxArrayString arrayDates = m_forecastManager->GetLeadTimes(m_selectedMethod, m_selectedForecast);
    m_choiceDate->Set(arrayDates);
    if (arrayDates.size()<=(unsigned)m_selectedDate)
    {
        m_selectedDate = 0;
    }
    m_choiceDate->Select(m_selectedDate);

    // Stations list
    wxArrayString arrayStation = m_forecastManager->GetStationNamesWithHeights(m_selectedMethod, m_selectedForecast);
    m_choiceStation->Set(arrayStation);
    if (arrayStation.size()<=(unsigned)m_selectedStation)
    {
        m_selectedStation = 0;
    }
    m_choiceStation->Select(m_selectedStation);

    Plot();
}

void asFramePlotDistributions::OnChoiceStationChange( wxCommandEvent& event )
{
    m_selectedStation = event.GetInt();

	RebuildChoiceForecast();

    PlotPredictands(); // Doesn't change for criteria
}

void asFramePlotDistributions::OnChoiceDateChange( wxCommandEvent& event )
{
    m_selectedDate = event.GetInt();

    Plot();
}

void asFramePlotDistributions::InitPredictandsCheckListBox()
{
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

    m_checkListTocPredictands->Set(checkList);
}

void asFramePlotDistributions::InitPredictandsPlotCtrl()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

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
    titleFont.SetPointSize(titleFont.GetPointSize()+2);
    plotctrl->SetPlotTitleFont(titleFont);

    // Set the grid color
    wxColour gridColor(240,240,240);
    plotctrl->SetGridColour(gridColor);

    // Open layers defined in the preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool doPlotAllAnalogsPoints;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotAllAnalogsPoints", &doPlotAllAnalogsPoints, false);
    if (doPlotAllAnalogsPoints) m_checkListTocPredictands->Check(AllAnalogsPoints);
    bool doPlotAllAnalogsCurve;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotAllAnalogsCurve", &doPlotAllAnalogsCurve, true);
    if (doPlotAllAnalogsCurve) m_checkListTocPredictands->Check(AllAnalogsCurve);
    bool doPlotBestAnalogs10Points;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Points", &doPlotBestAnalogs10Points, false);
    if (doPlotBestAnalogs10Points) m_checkListTocPredictands->Check(BestAnalogs10Points);
    bool doPlotBestAnalogs10Curve;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotBestAnalogs10Curve", &doPlotBestAnalogs10Curve, true);
    if (doPlotBestAnalogs10Curve) m_checkListTocPredictands->Check(BestAnalogs10Curve);
    bool doPlotBestAnalogs5Points;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Points", &doPlotBestAnalogs5Points, true);
    if (doPlotBestAnalogs5Points) m_checkListTocPredictands->Check(BestAnalogs5Points);
    bool doPlotBestAnalogs5Curve;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotBestAnalogs5Curve", &doPlotBestAnalogs5Curve, false);
    if (doPlotBestAnalogs5Curve) m_checkListTocPredictands->Check(BestAnalogs5Curve);
    bool doPlotAllReturnPeriods;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotAllReturnPeriods", &doPlotAllReturnPeriods, false);
    if (doPlotAllReturnPeriods) m_checkListTocPredictands->Check(AllReturnPeriods);
    bool doPlotClassicReturnPeriod;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotClassicReturnPeriod", &doPlotClassicReturnPeriod, true);
    if (doPlotClassicReturnPeriod) m_checkListTocPredictands->Check(ClassicReturnPeriod);
    bool doPlotClassicQuantiles;
    pConfig->Read("/PlotsDistributionsPredictands/DoPlotClassicQuantiles", &doPlotClassicQuantiles, true);
    if (doPlotClassicQuantiles) m_checkListTocPredictands->Check(ClassicQuantiles);
}

void asFramePlotDistributions::InitCriteriaPlotCtrl()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotCriteria->GetPlotCtrl();

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
    titleFont.SetPointSize(titleFont.GetPointSize()+2);
    plotctrl->SetPlotTitleFont(titleFont);

    // Set the grid color
    wxColour gridColor(240,240,240);
    plotctrl->SetGridColour(gridColor);
}

void asFramePlotDistributions::OnTocSelectionChange( wxCommandEvent& event )
{
    PlotPredictands();
}

bool asFramePlotDistributions::Plot()
{
    if (m_forecastManager->GetMethodsNb()<1) return false;
    if (!PlotPredictands()) return false;
    if (!PlotCriteria()) return false;
    return true;
}

bool asFramePlotDistributions::PlotPredictands()
{
    if (m_forecastManager->GetMethodsNb()<1) return false;

    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Check that there is no NaNs
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    Array1DFloat analogs = forecast->GetAnalogsValuesGross(m_selectedDate, m_selectedStation);
    if (asTools::HasNaN(&analogs[0], &analogs[analogs.size()-1]))
    {
        asLogError(_("The forecast contains NaNs. Plotting has been canceled."));
        return false;
    }

    // Clear previous curves
    int curvesNb = plotctrl->GetCurveCount();
    for (int i=curvesNb-1; i>=0; i--)
    {
        wxPlotData *plotData = plotctrl->GetDataCurve(i);
        if (plotData)
        {
            plotctrl->DeleteCurve(plotData);
        }
    }

    // Clear previous markers
    plotctrl->ClearMarkers();

    // Set a first threshold for the zoom
    m_xmaxPredictands = 50;

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

    for (int curve=0; curve<=8; curve++)
    {
        if(m_checkListTocPredictands->IsChecked(curve))
        {
            switch (curve)
            {
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
                asLogError(_("The option was not found."));

            }
        }
    }

    if (DoPlotAllAnalogsPoints)
        PlotAllAnalogsPoints();
    if (DoPlotBestAnalogs10Points)
        PlotBestAnalogsPoints(10);
    if (DoPlotBestAnalogs5Points)
        PlotBestAnalogsPoints(5);
    if (DoPlotClassicQuantiles)
        PlotClassicQuantiles();
    if (DoPlotAllAnalogsCurve)
        PlotAllAnalogsCurve();
    if (DoPlotBestAnalogs10Curve)
        PlotBestAnalogsCurve(10);
    if (DoPlotBestAnalogs5Curve)
        PlotBestAnalogsCurve(5);
    if(forecast->HasReferenceValues())
    {
        if (DoPlotClassicReturnPeriod)
            PlotReturnPeriod(10);
        if (DoPlotAllReturnPeriods)
            PlotAllReturnPeriods();
    }

    // Set the view rectangle (wxRect2DDouble(x, y, w, h))
    wxRect2DDouble currentView(0, 0, m_xmaxPredictands*1.1, 1);
    plotctrl->SetViewRect(currentView);

    // Redraw
    plotctrl->Redraw(wxPLOTCTRL_REDRAW_PLOT);

    return true;
}

bool asFramePlotDistributions::PlotCriteria()
{
    if (m_forecastManager->GetMethodsNb()<1) return false;

    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotCriteria->GetPlotCtrl();

    // Check that there is no NaNs
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    Array1DFloat criteria = forecast->GetAnalogsCriteria(m_selectedDate);
    if (asTools::HasNaN(&criteria[0], &criteria[criteria.size()-1]))
    {
        asLogError(_("The forecast criteria contains NaNs. Plotting has been canceled."));
        return false;
    }

    // Clear previous curves
    int curvesNb = plotctrl->GetCurveCount();
    for (int i=curvesNb-1; i>=0; i--)
    {
        wxPlotData *plotData = plotctrl->GetDataCurve(i);
        if (plotData)
        {
            plotctrl->DeleteCurve(plotData);
        }
    }

    // Clear previous markers
    plotctrl->ClearMarkers();

    // Plot curves
    PlotCriteriaCurve();

    // Get min/max of the criteria
    float critMin = (float)999999999.0, critMax = (float)0.0;
    for (int i=0; i<forecast->GetTargetDatesLength(); i++)
    {
        Array1DFloat tmpCriteria = forecast->GetAnalogsCriteria(i);
        if (tmpCriteria[0]<critMin) critMin = tmpCriteria[0];
        if (tmpCriteria[tmpCriteria.size()-1]>critMax) critMax = tmpCriteria[tmpCriteria.size()-1];
    }

    // Set the view rectangle (wxRect2DDouble(x, y, w, h))
    wxRect2DDouble currentView(1, critMin, forecast->GetAnalogsNumber(m_selectedDate)-1, critMax-critMin);
    plotctrl->SetViewRect(currentView);

    // Redraw
    plotctrl->Redraw(wxPLOTCTRL_REDRAW_PLOT);

    return true;
}

void asFramePlotDistributions::PlotAllReturnPeriods()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Get return periods
    Array1DFloat retPeriods = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceAxis();

    for (int i=retPeriods.size()-1; i>=0; i--)
    {
        if (std::abs(retPeriods[i]-2.33)<0.1) continue;

        // Get precipitation value
        float val = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceValue(m_selectedStation, i);

        // Color (from yellow to red)
        float ratio = (float)i/(float)(retPeriods.size()-1);
        wxGenericPen pen(wxGenericColour(255,255-ratio*255,0), 2);

        // Markers -> cannot add legend entries
        //wxPlotMarker marker;
        //marker.CreateHorizLineMarker(val, pen);
        //plotctrl->AddMarker(marker);

        // Store max val
        if (val>m_xmaxPredictands) m_xmaxPredictands = val;

        // Create plot data
        wxPlotData plotData;
        plotData.Create(2);
        if (std::abs(retPeriods[i]-2.33)<0.1)
        {
            plotData.SetFilename(wxString::Format("P%3.2f",retPeriods[i]));
        }
        else
        {
            int roundedVal = (int)asTools::Round(retPeriods[i]);
            plotData.SetFilename(wxString::Format("P%d",roundedVal));
        }
        plotData.SetValue(0, val, -1);
        plotData.SetValue(1, val, 2);

        // Check and add to the plot
        if (plotData.Ok())
        {
            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);

            plotData.SetDrawSymbols(false);
            plotData.SetDrawLines(true);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        }
        else
        {
            asLogError(_("The return periods couldn't be added to the plot"));
        }

        plotData.Destroy();
    }

}

void asFramePlotDistributions::PlotReturnPeriod(int returnPeriod)
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Get return periods
    Array1DFloat retPeriods = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceAxis();

    // Find the value 10
    int index = asTools::SortedArraySearch(&retPeriods[0], &retPeriods[retPeriods.size()-1], returnPeriod);

    if ( (index!=asNOT_FOUND) && (index!=asOUT_OF_RANGE) )
    {
        // Get precipitation value
        float val = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetReferenceValue(m_selectedStation, index);

        // Color (red)
        wxGenericPen pen(wxGenericColour(255,0,0), 2);

        // Lines
        wxPlotMarker marker;
        marker.CreateVertLineMarker(val, pen);
        plotctrl->AddMarker(marker);

        // Store max val
        if (val>m_xmaxPredictands) m_xmaxPredictands = val;
    }
    else
    {
        asLogError(_("The 10 year return period was not found in the data."));
    }
}

void asFramePlotDistributions::PlotAllAnalogsPoints()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Get the total number of points
    Array1DFloat analogs = forecast->GetAnalogsValuesGross(m_selectedDate, m_selectedStation);
    asTools::SortArray(&analogs[0], &analogs[analogs.size()-1], Asc);
    int nbPoints = analogs.size();

    // Cumulative frequency
    Array1DFloat F = asTools::GetCumulativeFrequency(nbPoints);

    // Create plot data
    wxPlotData plotData;
    plotData.Create(nbPoints);
    int counter=0;
    for (int i_analog=0; i_analog<analogs.size(); i_analog++)
    {
        plotData.SetValue(counter, analogs[i_analog], F(i_analog));
        counter++;

        // Store max val
        if (analogs[i_analog]>m_xmaxPredictands) m_xmaxPredictands = analogs[i_analog];
    }

    // Check and add to the plot
    if (plotData.Ok())
    {
        wxPen pen(wxColour(180,180,180), 1);

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
    }
    else
    {
        asLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

void asFramePlotDistributions::PlotAllAnalogsCurve()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Get the total number of points
    Array1DFloat analogs = forecast->GetAnalogsValuesGross(m_selectedDate, m_selectedStation);
    asTools::SortArray(&analogs[0], &analogs[analogs.size()-1], Asc);
    int nbPoints = analogs.size();

    // Cumulative frequency
    Array1DFloat F = asTools::GetCumulativeFrequency(nbPoints);

    // Create plot data
    wxPlotData plotData;
    plotData.Create(nbPoints);
    int counter=0;
    for (int i_analog=0; i_analog<analogs.size(); i_analog++)
    {
        plotData.SetValue(counter, analogs[i_analog], F(i_analog));
        counter++;

        // Store max val
        if (analogs[i_analog]>m_xmaxPredictands) m_xmaxPredictands = analogs[i_analog];
    }

    // Check and add to the plot
    if (plotData.Ok())
    {
        wxPen pen(wxColour(0,0,240), 2);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);

        plotData.SetDrawSymbols(false);
        plotData.SetDrawLines(true);
        plotData.SetFilename(_("All analogs"));

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    }
    else
    {
        asLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

void asFramePlotDistributions::PlotBestAnalogsPoints(int analogsNb)
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Extract best analogs
    Array1DFloat analogsAll = forecast->GetAnalogsValuesGross(m_selectedDate, m_selectedStation);
    int nbPoints = wxMin((int)analogsAll.size(), analogsNb);
    Array1DFloat analogs = analogsAll.head(nbPoints);
    Array1DFloat ranks = Array1DFloat::LinSpaced(nbPoints,0,nbPoints-1);
    asTools::SortArrays(&analogs[0], &analogs[analogs.size()-1], &ranks[0], &ranks[ranks.size()-1], Asc);

    // Cumulative frequency
    Array1DFloat F = asTools::GetCumulativeFrequency(nbPoints);

    // Create plot data
    for (int i_analog=0; i_analog<analogs.size(); i_analog++)
    {
        wxPlotData plotData;
        plotData.Create(1);

        plotData.SetValue(0, analogs[i_analog], F(i_analog));

        // Check and add to the plot
        if (plotData.Ok())
        {
            // Color (from yellow to red)
            float ratio = ranks[i_analog]/(float)(nbPoints-1);
            wxPen pen(wxColor(255,ratio*255,0), 2);

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
        }
        else
        {
            asLogError(_("The analogs data couldn't be added to the plot"));
        }

        plotData.Destroy();

        // Store max val
        if (analogs[i_analog]>m_xmaxPredictands) m_xmaxPredictands = analogs[i_analog];
    }
}

void asFramePlotDistributions::PlotBestAnalogsCurve(int analogsNb)
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Extract best analogs
    Array1DFloat analogsAll = forecast->GetAnalogsValuesGross(m_selectedDate, m_selectedStation);
    int nbPoints = wxMin((int)analogsAll.size(), analogsNb);
    Array1DFloat analogs = analogsAll.head(nbPoints);
    asTools::SortArray(&analogs[0], &analogs[analogs.size()-1], Asc);

    // Cumulative frequency
    Array1DFloat F = asTools::GetCumulativeFrequency(nbPoints);

    // Create plot data
    wxPlotData plotData;
    plotData.Create(nbPoints);
    int counter=0;
    for (int i_analog=0; i_analog<analogs.size(); i_analog++)
    {
        plotData.SetValue(counter, analogs[i_analog], F(i_analog));
        counter++;

        // Store max val
        if (analogs[i_analog]>m_xmaxPredictands) m_xmaxPredictands = analogs[i_analog];
    }

    // Check and add to the plot
    if (plotData.Ok())
    {
        wxPen pen(wxColour(180,0,180), 1);

        if(analogsNb==5)
        {
            pen.SetColour(wxColour(0,112,0));
            plotData.SetFilename(_("5 first analogs"));
        }
        else if (analogsNb==10)
        {
            pen.SetColour(wxColour(0,228,0));
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
    }
    else
    {
        asLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}

void asFramePlotDistributions::PlotClassicQuantiles()
{
    // Quantiles
    Array1DFloat pc(3);
    pc << 0.2f, 0.6f, 0.9f;

    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotPredictands->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    Array1DFloat analogs = forecast->GetAnalogsValuesGross(m_selectedDate, m_selectedStation);

    // Loop over the quantiles
    for (int i_pc=0; i_pc<pc.size(); i_pc++)
    {
        float thisQuantile = pc[i_pc];

        // Create plot data
        wxPlotData plotData;
        plotData.Create(1);
        float pcVal = asTools::GetValueForQuantile(analogs, thisQuantile);
        plotData.SetValue(0, pcVal, thisQuantile);

        // Store max val
        if (pcVal>m_xmaxPredictands) m_xmaxPredictands = pcVal;

        // Check and add to the plot
        if (plotData.Ok())
        {
            wxPen pen(wxColour(100,100,100), 2, wxPENSTYLE_SOLID);

            // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
            plotData.SetPen(wxPLOTPEN_NORMAL, pen);
            plotData.SetSymbol(wxPLOTSYMBOL_CIRCLE, wxPLOTPEN_NORMAL, 15, 15, &pen, wxTRANSPARENT_BRUSH);

            plotData.SetDrawSymbols(true);
            plotData.SetDrawLines(false);

            // Add the curve
            bool select = false;
            bool send_event = false;
            plotctrl->AddCurve(plotData, select, send_event);
        }
        else
        {
            asLogError(_("The quantiles couldn't be added to the plot"));
        }

        plotData.Destroy();
    }
}

void asFramePlotDistributions::PlotCriteriaCurve()
{
    // Get a pointer to the plotctrl
    wxPlotCtrl* plotctrl = m_panelPlotCriteria->GetPlotCtrl();

    // Get forecast
    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);

    // Get the criteria
    Array1DFloat criteria = forecast->GetAnalogsCriteria(m_selectedDate);
    Array1DFloat indices = Array1DFloat::LinSpaced(criteria.size(), 1, criteria.size()); //LinSpaced(size, low, high)

    // Create plot data
    wxPlotData plotData;
    plotData.Create(criteria.size());
    for (int i_analog=0; i_analog<criteria.size(); i_analog++)
    {
        plotData.SetValue(i_analog, indices[i_analog], criteria[i_analog]);
    }

    // Check and add to the plot
    if (plotData.Ok())
    {
        wxPen pen(wxColour(0,240,240), 2);

        // wxPlotPen_Type : wxPLOTPEN_NORMAL, wxPLOTPEN_ACTIVE, wxPLOTPEN_SELECTED, wxPLOTPEN_MAXTYPE
        plotData.SetPen(wxPLOTPEN_NORMAL, pen);

        plotData.SetDrawSymbols(false);
        plotData.SetDrawLines(true);

        // Add the curve
        bool select = false;
        bool send_event = false;
        plotctrl->AddCurve(plotData, select, send_event);
    }
    else
    {
        asLogError(_("The analogs data couldn't be added to the plot"));
    }

    plotData.Destroy();
}
